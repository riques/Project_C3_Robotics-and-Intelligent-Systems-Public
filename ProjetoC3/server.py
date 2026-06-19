"""
server.py — Backend Flask para integração com Groq (C3)
Recebe os dados de detecção do SSD MobileNet (C2) e retorna análise da LLM.
"""

import os
import json
import time
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

# ─────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
CORS(app)  # Permite chamadas cross-origin do front-end

# Logger estruturado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Cliente Groq — lê a chave do .env
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Modelo escolhido e justificativa (ver README)
MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────
# Engenharia de Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Você é um especialista em visão computacional e sistemas inteligentes.
Seu papel é analisar os resultados de uma detecção de objetos feita pelo algoritmo SSD MobileNet
e gerar um relatório técnico claro, objetivo e útil.

Sempre responda em **português brasileiro**.

Seu relatório deve conter:
1. **Resumo Executivo** — o que foi detectado, em 2-3 frases diretas.
2. **Análise por Objeto** — para cada classe detectada, comente: quantidade, confiança média e relevância contextual.
3. **Padrões e Anomalias** — identifique padrões relevantes (ex: muitos objetos do mesmo tipo, baixa confiança generalizada, ausência de objetos esperados).
4. **Recomendações** — sugira ajustes ao pipeline de detecção ou ações práticas baseadas nos dados.
5. **Índice de Qualidade da Cena** — dê uma nota de 0 a 10 para a qualidade da detecção, com justificativa em uma frase.

Seja técnico mas acessível. Não repita os dados brutos; interprete-os."""


def construir_user_prompt(dados_c2: dict) -> str:
    """Constrói o prompt de usuário a partir dos dados de detecção da C2."""
    payload = json.dumps(dados_c2, ensure_ascii=False, indent=2)
    return f"""Analise os dados de detecção abaixo, produzidos pelo sistema SSD MobileNet em tempo real.

```json
{payload}
```

Gere o relatório técnico conforme as instruções do seu papel."""


# ─────────────────────────────────────────────
# Rotas
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    """Health check simples."""
    return jsonify({"status": "online", "modelo": MODEL})


@app.route("/analisar", methods=["POST"])
def analisar():
    """
    Recebe JSON com resultados de detecção da C2 e retorna análise da LLM.
    
    Corpo esperado:
    {
      "deteccoes": [...],       // lista de objetos detectados
      "total_objetos": int,
      "fonte": "foto" | "webcam",
      "timestamp": "ISO-8601"
    }
    """
    dados = request.get_json(silent=True)
    if not dados:
        return jsonify({"erro": "Corpo JSON inválido ou ausente."}), 400

    # O payload vem estruturado pelo construirPayloadC2() do front-end
    totais = dados.get("totais", {})
    fonte  = dados.get("metadados", {}).get("fonte", "desconhecida")

    logger.info("Requisição recebida: %d objeto(s) detectado(s) via %s",
                totais.get("objetos_detectados", 0),
                fonte)

    # Verifica se há detecções para analisar (chave correta do payload da C2)
    deteccoes = dados.get("detecções_brutas", dados.get("deteccoes", []))
    if not deteccoes:
        return jsonify({
            "analise": "Nenhum objeto foi detectado na cena. "
                       "Tente ajustar o ângulo da câmera, a iluminação ou o limiar de confiança.",
            "modelo_utilizado": MODEL,
            "tokens_consumidos": 0,
            "latencia_ms": 0,
        })

    prompt_usuario = construir_user_prompt(dados)
    inicio = time.monotonic()

    try:
        resposta = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt_usuario},
            ],
            temperature=0.2,      # Baixa para respostas analíticas e reproduzíveis
            max_tokens=1024,      # Limite para evitar respostas excessivamente longas
        )

        latencia_ms = int((time.monotonic() - inicio) * 1000)
        analise     = resposta.choices[0].message.content
        tokens      = resposta.usage.total_tokens if resposta.usage else 0

        logger.info("Resposta Groq: %d tokens, %d ms de latência", tokens, latencia_ms)

        return jsonify({
            "analise":          analise,
            "modelo_utilizado": MODEL,
            "tokens_consumidos": tokens,
            "latencia_ms":      latencia_ms,
            "timestamp":        datetime.now().isoformat(),
        })

    except Exception as erro:
        # Falha na LLM não deve derrubar o sistema
        logger.error("Erro na chamada Groq: %s", str(erro))
        return jsonify({
            "erro":    f"Falha na chamada à LLM: {str(erro)}",
            "analise": None,
        }), 500


if __name__ == "__main__":
    chave = os.environ.get("GROQ_API_KEY")
    if not chave:
        logger.warning("⚠️  GROQ_API_KEY não encontrada no .env — as chamadas à LLM falharão.")
    else:
        logger.info("✅  GROQ_API_KEY carregada com sucesso.")
    
    logger.info("🚀  Servidor rodando em http://localhost:5000")
    app.run(debug=False, port=5000)