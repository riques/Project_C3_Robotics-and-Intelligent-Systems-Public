/**
 * script.js — VisionAI C3
 * Pipeline: Captura → SSD MobileNet (C2) → Groq LLM (C3) → Interface
 *
 * Arquitetura:
 *  - detectarObjetos()   : encapsula o pipeline da C2 e retorna JSON padronizado
 *  - chamarGroq()        : envia o JSON ao backend Flask, que chama a API Groq
 *  - renderizarAnalise() : exibe a resposta da LLM com formatação Markdown simples
 */

"use strict";

// ─────────────────────────────────────────────────────────────
// Constantes
// ─────────────────────────────────────────────────────────────

/** URL do backend Flask. Trocar para produção se necessário. */
const BACKEND_URL = "http://localhost:5000/analisar";

/** Limiar de confiança padrão (0–1). Controlado pelo slider. */
let THRESHOLD = 0.50;

/** Paleta de cores para bounding boxes por classe. */
const CLASS_COLORS = {
  person: "#00d4aa", car: "#58a6ff", dog: "#d29922", cat: "#f0883e",
  chair: "#bc8cff", bottle: "#39d353", laptop: "#ff7b72", cell_phone: "#ff79c6",
  book: "#a5d6ff", cup: "#ffd700", bicycle: "#ff9580", bird: "#7ee787",
  _default: "#e6edf3",
};

// ─────────────────────────────────────────────────────────────
// Estado da aplicação
// ─────────────────────────────────────────────────────────────

let model          = null;
let webcamStream   = null;
let webcamAtiva    = false;
let animFrameId    = null;
let ultimoPayloadC2 = null;   // Último resultado da C2 para re-envio à LLM

// ─────────────────────────────────────────────────────────────
// Utilitários de UI
// ─────────────────────────────────────────────────────────────

function log(mensagem, tipo = "info") {
  const container = document.getElementById("logContainer");
  const entrada   = document.createElement("span");
  const hora      = new Date().toLocaleTimeString("pt-BR");
  entrada.className = `log-entry log-${tipo}`;
  entrada.textContent = `[${hora}] ${mensagem}`;
  container.appendChild(entrada);
  container.scrollTop = container.scrollHeight;
}

function limparLog() {
  document.getElementById("logContainer").innerHTML = "";
  log("Log limpo.");
}

function setPipelineStep(etapa) {
  const steps = document.querySelectorAll(".pipeline-step");
  steps.forEach((el, i) => {
    el.classList.remove("active", "done");
    if (i + 1 < etapa)  el.classList.add("done");
    if (i + 1 === etapa) el.classList.add("active");
  });
}

function setStatus(texto) {
  document.getElementById("statusText").textContent = texto;
}

function setBadge(id, texto, classe) {
  const el = document.getElementById(id);
  el.textContent = texto;
  el.className = `badge ${classe}`;
}

function obterCor(classe) {
  return CLASS_COLORS[classe] ?? CLASS_COLORS._default;
}

/** Converte texto Markdown simples (negrito, código, headings, listas) em HTML. */
function markdownParaHTML(texto) {
  return texto
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm,  "<h2>$1</h2>")
    .replace(/^# (.+)$/gm,   "<h1>$1</h1>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/^[\*\-] (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>)/gs, m => `<ul>${m}</ul>`)
    .replace(/\n\n/g, "</p><p>")
    .replace(/^(?!<[hul])/gm, "")
    .replace(/^(.+)$/gm, m => m.startsWith("<") ? m : `<p>${m}</p>`)
    .replace(/<\/ul><ul>/g, "")
    .replace(/---/g, "<hr>")
    .trim();
}

/** Copia o JSON da C2 para a área de transferência. */
function copiarJSON() {
  if (!ultimoPayloadC2) return;
  navigator.clipboard.writeText(JSON.stringify(ultimoPayloadC2, null, 2))
    .then(() => log("JSON copiado para área de transferência.", "ok"))
    .catch(() => log("Erro ao copiar JSON.", "warn"));
}

// ─────────────────────────────────────────────────────────────
// Carregamento do modelo (C2)
// ─────────────────────────────────────────────────────────────

async function carregarModelo() {
  log("Carregando modelo SSD MobileNet…");
  setPipelineStep(1);
  try {
    model = await cocoSsd.load();
    log("✓ Modelo SSD MobileNet carregado.", "ok");
    setBadge("model-status", "Modelo Pronto", "badge-ready");
    setStatus("Modelo pronto. Carregue uma imagem ou ligue a webcam.");
  } catch (erro) {
    log(`Erro ao carregar modelo: ${erro.message}`, "error");
    setBadge("model-status", "Erro no Modelo", "badge-loading");
  }
}

// ─────────────────────────────────────────────────────────────
// Controle de threshold
// ─────────────────────────────────────────────────────────────

function updateThreshold(valor) {
  THRESHOLD = valor / 100;
  document.getElementById("thresholdValue").textContent = `${valor}%`;
  log(`Confiança mínima ajustada para ${valor}%.`);
}

// ─────────────────────────────────────────────────────────────
// Upload de foto (C2)
// ─────────────────────────────────────────────────────────────

document.getElementById("uploadFoto").addEventListener("change", async (evento) => {
  const arquivo = evento.target.files[0];
  if (!arquivo) return;
  if (!model) { log("Modelo ainda está carregando. Aguarde.", "warn"); return; }

  pararWebcam();
  setPipelineStep(2);
  setStatus("Processando imagem…");

  const img = document.getElementById("imgEstatica");
  img.src = URL.createObjectURL(arquivo);

  document.getElementById("mediaPlaceholder").style.display = "none";
  document.getElementById("webcamVideo").style.display = "none";
  img.style.display = "block";

  img.onload = async () => {
    log(`Imagem carregada: ${arquivo.name} (${(arquivo.size / 1024).toFixed(0)} KB)`);
    await detectarEProcessar(img, "foto");
  };
});

// ─────────────────────────────────────────────────────────────
// Webcam (C2)
// ─────────────────────────────────────────────────────────────

async function toggleWebcam() {
  if (webcamAtiva) {
    pararWebcam();
  } else {
    await iniciarWebcam();
  }
}

async function iniciarWebcam() {
  if (!model) { log("Modelo ainda está carregando. Aguarde.", "warn"); return; }

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    const video  = document.getElementById("webcamVideo");
    video.srcObject = webcamStream;
    video.style.display = "block";
    document.getElementById("imgEstatica").style.display  = "none";
    document.getElementById("mediaPlaceholder").style.display = "none";
    webcamAtiva = true;
    document.getElementById("btnWebcam").textContent = "⏹ Parar Webcam";

    log("Webcam iniciada. Iniciando detecção em tempo real.", "ok");
    setPipelineStep(2);

    video.onloadedmetadata = () => {
      const canvas  = document.getElementById("overlayCanvas");
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
      detectarLoopWebcam(video);
    };
  } catch (err) {
    log(`Erro ao acessar webcam: ${err.message}`, "error");
  }
}

function pararWebcam() {
  webcamAtiva = false;
  if (animFrameId) cancelAnimationFrame(animFrameId);
  if (webcamStream) { webcamStream.getTracks().forEach(t => t.stop()); webcamStream = null; }
  document.getElementById("webcamVideo").style.display = "none";
  document.getElementById("btnWebcam").innerHTML = `
    <svg viewBox="0 0 20 20" fill="currentColor" width="16"><path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zm12.553 1.106A1 1 0 0014 8v4a1 1 0 00.553.894l2 1A1 1 0 0018 13V7a1 1 0 00-1.447-.894l-2 1z"/></svg>
    Ligar Webcam`;
  log("Webcam encerrada.");
}

// Loop de detecção da webcam — chama Groq a cada ~60 frames (~2s a 30fps)
let frameCount = 0;
async function detectarLoopWebcam(video) {
  if (!webcamAtiva) return;

  const predicoes   = await model.detect(video);
  const filtradas   = predicoes.filter(p => p.score >= THRESHOLD);

  desenharBoundingBoxes(video, filtradas);
  setStatus(`${filtradas.length} objeto(s) detectado(s) em tempo real`);

  // Chama Groq a cada 60 frames para não sobrecarregar
  frameCount++;
  if (frameCount % 60 === 0 && filtradas.length > 0) {
    const payload = construirPayloadC2(filtradas, "webcam");
    await chamarGroqComPayload(payload);
  }

  animFrameId = requestAnimationFrame(() => detectarLoopWebcam(video));
}

// ─────────────────────────────────────────────────────────────
// Núcleo do Pipeline C2 — detectarEProcessar
// ─────────────────────────────────────────────────────────────

async function detectarEProcessar(elemento, fonte) {
  // — Etapa 1: Detecção (C2)
  log("Executando SSD MobileNet…");
  const predicoes  = await model.detect(elemento);
  const filtradas  = predicoes.filter(p => p.score >= THRESHOLD);

  log(`Detecção concluída: ${filtradas.length} objeto(s) (threshold=${(THRESHOLD * 100).toFixed(0)}%)`, "ok");

  // — Etapa 2: Desenhar bounding boxes (C2)
  desenharBoundingBoxes(elemento, filtradas);
  setStatus(`${filtradas.length} objeto(s) detectado(s)`);

  // — Etapa 3: Construir payload padronizado
  const payload = construirPayloadC2(filtradas, fonte);
  ultimoPayloadC2 = payload;

  // — Exibir JSON bruto
  document.getElementById("rawOutputSection").style.display = "block";
  document.getElementById("rawJSON").textContent = JSON.stringify(payload, null, 2);
  document.getElementById("reanaliseSection").style.display = "block";

  // — Etapa 4: Enviar à LLM (C3)
  await chamarGroqComPayload(payload);
}

// ─────────────────────────────────────────────────────────────
// Estruturação de dados C2 → JSON padronizado
// ─────────────────────────────────────────────────────────────

/**
 * Transforma as predições brutas do SSD MobileNet em um JSON
 * estruturado e rico em contexto para ser enviado à LLM.
 */
function construirPayloadC2(predicoes, fonte) {
  // Agrupa por classe para análise estatística
  const porClasse = {};
  predicoes.forEach(p => {
    if (!porClasse[p.class]) porClasse[p.class] = { quantidade: 0, confianças: [] };
    porClasse[p.class].quantidade++;
    porClasse[p.class].confianças.push(+(p.score * 100).toFixed(1));
  });

  const resumoClasses = Object.entries(porClasse).map(([classe, dados]) => {
    const confs = dados.confianças;
    const media = (confs.reduce((a, b) => a + b, 0) / confs.length).toFixed(1);
    return {
      classe,
      quantidade: dados.quantidade,
      confiança_média_pct: parseFloat(media),
      confiança_min_pct:   Math.min(...confs),
      confiança_max_pct:   Math.max(...confs),
    };
  }).sort((a, b) => b.quantidade - a.quantidade);

  return {
    metadados: {
      algoritmo:       "SSD MobileNet (COCO-SSD)",
      fonte:           fonte,
      timestamp:       new Date().toISOString(),
      threshold_pct:   +(THRESHOLD * 100).toFixed(0),
    },
    totais: {
      objetos_detectados: predicoes.length,
      classes_únicas:     Object.keys(porClasse).length,
    },
    resumo_por_classe: resumoClasses,
    detecções_brutas: predicoes.map(p => ({
      classe:          p.class,
      confiança_pct:   +(p.score * 100).toFixed(1),
      bbox:            {
        x: Math.round(p.bbox[0]),
        y: Math.round(p.bbox[1]),
        largura:  Math.round(p.bbox[2]),
        altura:   Math.round(p.bbox[3]),
      },
    })),
  };
}

// ─────────────────────────────────────────────────────────────
// Renderização de bounding boxes (C2)
// ─────────────────────────────────────────────────────────────

function desenharBoundingBoxes(elemento, predicoes) {
  const canvas  = document.getElementById("overlayCanvas");
  const ctx     = canvas.getContext("2d");

  // Ajusta canvas às dimensões reais do elemento renderizado
  const rect      = elemento.getBoundingClientRect();
  const natW      = elemento.naturalWidth  || elemento.videoWidth  || rect.width;
  const natH      = elemento.naturalHeight || elemento.videoHeight || rect.height;
  const dispW     = elemento.offsetWidth   || rect.width;
  const dispH     = elemento.offsetHeight  || rect.height;
  const scaleX    = dispW / natW;
  const scaleY    = dispH / natH;

  canvas.width  = dispW;
  canvas.height = dispH;
  ctx.clearRect(0, 0, dispW, dispH);

  predicoes.forEach(({ class: classe, score, bbox }) => {
    const [x, y, w, h] = bbox;
    const cor          = obterCor(classe);
    const cx = x * scaleX, cy = y * scaleY, cw = w * scaleX, ch = h * scaleY;

    // Caixa
    ctx.strokeStyle = cor;
    ctx.lineWidth   = 2.5;
    ctx.strokeRect(cx, cy, cw, ch);

    // Fundo da label
    const rotulo    = `${classe}  ${(score * 100).toFixed(0)}%`;
    ctx.font        = "bold 12px Inter, sans-serif";
    const largText  = ctx.measureText(rotulo).width + 12;
    const altBox    = 22;
    const yLabel    = cy > altBox + 2 ? cy - altBox : cy;

    ctx.fillStyle   = cor;
    ctx.fillRect(cx - 1, yLabel, largText, altBox);

    // Texto
    ctx.fillStyle   = "#000";
    ctx.fillText(rotulo, cx + 5, yLabel + 14);
  });
}

// ─────────────────────────────────────────────────────────────
// Integração Groq LLM (C3)
// ─────────────────────────────────────────────────────────────

/** Chamado pelo botão "Reanalisar" — usa o último payload disponível. */
async function chamarGroq() {
  if (!ultimoPayloadC2) {
    log("Nenhum dado de detecção disponível para enviar à LLM.", "warn");
    return;
  }
  await chamarGroqComPayload(ultimoPayloadC2);
}

/** Núcleo da integração C3: envia payload ao backend e exibe análise. */
async function chamarGroqComPayload(payload) {
  setPipelineStep(3);
  setBadge("llm-badge", "Consultando Groq…", "badge-thinking");

  // Exibe loader
  document.getElementById("analysisPlaceholder").style.display = "none";
  document.getElementById("analysisContent").style.display     = "none";
  document.getElementById("analysisLoader").style.display      = "flex";

  log(`Enviando ${payload.totais.objetos_detectados} detecção(ões) ao Groq (${payload.metadados.fonte})…`, "llm");

  const inicio = Date.now();

  try {
    const resposta = await fetch(BACKEND_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    if (!resposta.ok) {
      const erro = await resposta.json().catch(() => ({ erro: `HTTP ${resposta.status}` }));
      throw new Error(erro.erro || `HTTP ${resposta.status}`);
    }

    const dados       = await resposta.json();
    const latenciaMs  = Date.now() - inicio;

    // Atualiza métricas
    document.getElementById("llmMeta").style.display  = "grid";
    document.getElementById("metaModelo").textContent  =
      (dados.modelo_utilizado || "—").replace("llama-", "Llama ").replace("-versatile", "");
    document.getElementById("metaTokens").textContent  = dados.tokens_consumidos ?? "—";
    document.getElementById("metaLatencia").textContent = `${dados.latencia_ms ?? latenciaMs} ms`;

    log(`✓ Resposta Groq: ${dados.tokens_consumidos} tokens, ${dados.latencia_ms} ms`, "llm");

    renderizarAnalise(dados.analise);
    setPipelineStep(4);
    setBadge("llm-badge", "Análise Concluída", "badge-done");

  } catch (erro) {
    // Falha na LLM não quebra o sistema — apenas informa o usuário
    log(`Erro na chamada Groq: ${erro.message}`, "error");
    setBadge("llm-badge", "Erro na LLM", "badge-loading");

    document.getElementById("analysisLoader").style.display  = "none";
    document.getElementById("analysisContent").style.display = "block";
    document.getElementById("analysisContent").innerHTML = `
      <p style="color:var(--danger)">
        <strong>⚠ Não foi possível obter análise da LLM.</strong><br>
        ${erro.message}<br><br>
        Verifique se o servidor Flask está rodando em <code>http://localhost:5000</code>
        e se o arquivo <code>.env</code> contém a <code>GROQ_API_KEY</code> correta.
      </p>`;
    setPipelineStep(2); // Volta para etapa de detecção
  }
}

/** Renderiza a análise da LLM com formatação Markdown. */
function renderizarAnalise(texto) {
  const contentEl = document.getElementById("analysisContent");
  contentEl.innerHTML = markdownParaHTML(texto);

  document.getElementById("analysisLoader").style.display  = "none";
  document.getElementById("analysisPlaceholder").style.display = "none";
  contentEl.style.display = "block";
}

// ─────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────

carregarModelo();
