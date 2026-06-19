# VisionAI — C3: Detecção de Objetos + Análise por LLM

**Aluno:** Henrique De Nadai Salvador e Jeferson Moraes Pereira de Sousa

**Disciplina:** Robótica e Sistemas Inteligentes  

**Unidade:** C3 — Integração com IA Generativa via Groq

---

## Visão Geral do Projeto

Este projeto é a evolução do sistema de detecção de objetos desenvolvido na C2. O sistema agora implementa um **pipeline completo** que:

1. **Captura** uma imagem via upload ou webcam
2. **Detecta** objetos com o algoritmo **SSD MobileNet** (C2)
3. **Serializa** os resultados em JSON estruturado
4. **Envia** o JSON a uma LLM via **Groq** (C3)
5. **Apresenta** a análise interpretativa da LLM na interface

```
[Câmera/Foto] → [SSD MobileNet] → [JSON] → [Backend Flask] → [Groq API] → [Análise LLM]
```

---

## Arquitetura do Sistema

```
c3-project/
├── index.html          # Interface web principal
├── style.css           # Estilo visual (tema dark tech)
├── script.js           # Lógica front-end: detecção + chamada ao backend
├── server.py           # Backend Flask: recebe JSON e chama Groq API
├── requirements.txt    # Dependências Python
├── .env.example        # Template de variáveis de ambiente
├── .env                # ⚠ NÃO commitar — contém a API Key
└── .gitignore
```

---

## Tecnologias Utilizadas

| Componente | Tecnologia |
|---|---|
| Front-end | HTML5 · CSS3 · JavaScript (ES2022) |
| Detecção de objetos | TensorFlow.js · COCO-SSD (SSD MobileNet) |
| Backend | Python 3 · Flask · Flask-CORS |
| LLM | Groq API · Llama 3.3 70B Versatile |
| Gestão de credenciais | python-dotenv |

---

## Como Executar

### Pré-requisitos
- Python 3.10+
- pip
- Conta gratuita no [Groq Console](https://console.groq.com)
- Extensão **Live Server** no VS Code (para o front-end)

### Passo 1 — Obter a API Key do Groq
1. Acesse [console.groq.com/keys](https://console.groq.com/keys)
2. Crie uma nova API Key
3. Guarde-a em local seguro

### Passo 2 — Configurar variáveis de ambiente
```bash
# Copie o template
cp .env.example .env

# Edite o arquivo .env e substitua pela sua chave real
GROQ_API_KEY=gsk_sua_chave_aqui
```

### Passo 3 — Instalar dependências Python
```bash
pip install -r requirements.txt
```

### Passo 4 — Iniciar o backend Flask
```bash
python server.py
```
O servidor estará disponível em `http://localhost:5000`.

### Passo 5 — Abrir o front-end
- Abra a pasta no **VS Code**
- Clique com botão direito em `index.html` → **"Open with Live Server"**
- O projeto abrirá no navegador em `http://localhost:5500` (ou porta similar)

> ⚠ **Importante:** O Live Server é necessário por causa das políticas CORS dos navegadores ao carregar modelos TensorFlow.js localmente. Não abra o `index.html` diretamente com duplo clique.

---

## Pipeline de Integração C2 → C3

### 1. Serialização do resultado da C2

Após a detecção pelo SSD MobileNet, os resultados são estruturados em um JSON padronizado:

```json
{
  "metadados": {
    "algoritmo": "SSD MobileNet (COCO-SSD)",
    "fonte": "foto",
    "timestamp": "2026-06-01T14:32:10.000Z",
    "threshold_pct": 50
  },
  "totais": {
    "objetos_detectados": 3,
    "classes_únicas": 2
  },
  "resumo_por_classe": [
    {
      "classe": "person",
      "quantidade": 2,
      "confiança_média_pct": 87.5,
      "confiança_min_pct": 82.1,
      "confiança_max_pct": 92.9
    }
  ],
  "detecções_brutas": [ ... ]
}
```

### 2. Engenharia de Prompt

**System Prompt** — define o papel da LLM como especialista em visão computacional:
```
Você é um especialista em visão computacional e sistemas inteligentes.
Seu papel é analisar os resultados de uma detecção de objetos feita pelo 
algoritmo SSD MobileNet e gerar um relatório técnico claro, objetivo e útil.

O relatório deve conter:
1. Resumo Executivo
2. Análise por Objeto
3. Padrões e Anomalias
4. Recomendações
5. Índice de Qualidade da Cena (0–10)
```

**User Prompt** — injeta o JSON da C2:
```
Analise os dados de detecção abaixo, produzidos pelo sistema SSD MobileNet em tempo real.
[JSON]
Gere o relatório técnico conforme as instruções do seu papel.
```

**Parâmetros:**
- `temperature: 0.2` — baixa para respostas analíticas e reproduzíveis
- `max_tokens: 1024` — limite para evitar respostas excessivamente longas

### 3. Chamada à API Groq

O backend Flask recebe o JSON via POST em `/analisar`, chama a API Groq e retorna:
```json
{
  "analise": "## Resumo Executivo\n...",
  "modelo_utilizado": "llama-3.3-70b-versatile",
  "tokens_consumidos": 487,
  "latencia_ms": 823,
  "timestamp": "2026-06-01T14:32:11.000Z"
}
```

### 4. Apresentação ao usuário

A interface exibe lado a lado:
- **Esquerda:** imagem com bounding boxes + JSON bruto da C2
- **Direita:** análise da LLM em Markdown + métricas da chamada (modelo, tokens, latência)

---

## Modelo LLM Escolhido: Llama 3.3 70B Versatile

**Justificativa da escolha:**

| Critério | Llama 3.3 70B | llama-3.1-8b | Mixtral 8x7B |
|---|---|---|---|
| Qualidade analítica | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Velocidade (Groq) | ~800ms | ~200ms | ~500ms |
| Contexto disponível | 128K tokens | 128K tokens | 32K tokens |
| Português | Excelente | Bom | Bom |

O `llama-3.3-70b-versatile` apresentou a melhor qualidade de análise nos testes, com respostas estruturadas, interpretações precisas e recomendações relevantes. A latência de ~800ms no Groq é aceitável para o caso de uso (análise sob demanda, não em tempo real).

---

## Análise Crítica da Qualidade das Respostas

### Pontos fortes observados:
- Identifica corretamente padrões de distribuição espacial (ex: múltiplos objetos da mesma classe)
- Detecta anomalias como baixa confiança generalizada
- Recomendações práticas e contextualizadas ao domínio de visão computacional
- Estrutura consistente com o formato solicitado no system prompt

### Limitações identificadas:
- A LLM não tem acesso à imagem em si, apenas ao JSON — algumas análises ficam genéricas
- Objetos com nomes em inglês (cat, dog, person) são mencionados em inglês no JSON; a LLM traduz corretamente mas poderia haver inconsistências
- Com `temperature=0.2`, respostas são previsíveis mas podem ser repetitivas para cenas similares
- O modelo não distingue contexto semântico (ex: não sabe se "bottle" é de água ou de vinho)

### Melhorias futuras:
- Enviar um recorte da imagem como base64 junto ao JSON (multimodal)
- Cachear respostas para cenas com detecções idênticas
- Implementar histórico de análises para comparação temporal

---

## Tratamento de Erros

O sistema é resiliente a falhas na LLM:
- Falhas de rede ou timeout são capturadas e exibidas sem derrubar o sistema
- A detecção de objetos continua funcionando mesmo sem o backend
- O usuário recebe mensagem clara de erro com instrução de resolução
- Logs detalhados no painel lateral para depuração

---

## Segurança de Credenciais

- A `GROQ_API_KEY` é armazenada exclusivamente no arquivo `.env` (ignorado pelo `.gitignore`)
- O back-end Flask nunca expõe a chave ao front-end
- O front-end não tem acesso à chave; toda comunicação com o Groq passa pelo servidor

---

## Prompts Utilizados (Documentação da Arquitetura)

### System Prompt (server.py)
Papel: especialista em visão computacional  
Estrutura de resposta: 5 seções fixas (Resumo, Análise, Padrões, Recomendações, Índice)  
Idioma: português brasileiro  
Tom: técnico mas acessível

### User Prompt (server.py — `construir_user_prompt()`)
Conteúdo: dados JSON da C2 em bloco de código  
Instrução: análise técnica conforme o papel definido  
Temperatura: 0.2 (analítica, baixa criatividade)  
Max tokens: 1024

---

## Histórico de Versões

| Versão | Descrição |
|---|---|
| C1 | Classificação de Gatos vs Cachorros com Teachable Machine + TensorFlow.js |
| C2 | Detecção de objetos em tempo real com SSD MobileNet (COCO-SSD) |
| C3 | Integração com LLM (Groq) — análise interpretativa dos resultados da detecção |
