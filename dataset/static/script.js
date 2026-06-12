/* ── DermaScan AI — script.js ───────────────────────────── */

const imageInput  = document.getElementById("imageInput");
const dropzone    = document.getElementById("dropzone");
const dzIdle      = document.getElementById("dzIdle");
const previewImg  = document.getElementById("previewImg");
const predictBtn  = document.getElementById("predictBtn");

const resultIdle  = document.getElementById("resultIdle");
const resultError = document.getElementById("resultError");
const resultBody  = document.getElementById("resultBody");
const errorMsg    = document.getElementById("errorMsg");

const resultEmoji    = document.getElementById("resultEmoji");
const resultName     = document.getElementById("resultName");
const resultSeverity = document.getElementById("resultSeverity");
const donutArc       = document.getElementById("donutArc");
const donutPct       = document.getElementById("donutPct");
const top3List       = document.getElementById("top3List");
const causesList     = document.getElementById("causesList");
const symptomsList   = document.getElementById("symptomsList");
const suggestionsList= document.getElementById("suggestionsList");

let selectedFile = null;

/* ── Drag-and-drop ────────────────────────────────────────── */
dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("over"); });
["dragleave","dragend"].forEach(ev => dropzone.addEventListener(ev, () => dropzone.classList.remove("over")));
dropzone.addEventListener("drop", e => {
  e.preventDefault(); dropzone.classList.remove("over");
  const f = e.dataTransfer?.files?.[0];
  if (f && f.type.startsWith("image/")) handleFile(f);
});

/* ── File input ───────────────────────────────────────────── */
imageInput.addEventListener("change", () => {
  const f = imageInput.files?.[0];
  if (f) handleFile(f);
});

function handleFile(file) {
  selectedFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewImg.classList.remove("hidden");
  dzIdle.classList.add("hidden");
  predictBtn.disabled = false;
  showIdle();
}

/* ── Analyse ──────────────────────────────────────────────── */
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  predictBtn.classList.add("loading");
  predictBtn.disabled = true;
  showIdle();

  const fd = new FormData();
  fd.append("image", selectedFile);

  try {
    const res  = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();

    if (!res.ok || data.error) { showError(data.error || `Server error ${res.status}`); return; }
    showResult(data);
  } catch (err) {
    showError("Could not reach the server. Is the Flask app running?");
  } finally {
    predictBtn.classList.remove("loading");
    predictBtn.disabled = false;
  }
});

/* ── Display helpers ──────────────────────────────────────── */

function showIdle() {
  resultIdle.classList.remove("hidden");
  resultError.classList.add("hidden");
  resultBody.classList.add("hidden");
}

function showError(msg) {
  resultIdle.classList.add("hidden");
  resultBody.classList.add("hidden");
  resultError.classList.remove("hidden");
  errorMsg.textContent = msg;
}

function showResult(data) {
  resultIdle.classList.add("hidden");
  resultError.classList.add("hidden");
  resultBody.classList.remove("hidden");

  const info = data.info || {};

  /* Hero */
  resultEmoji.textContent    = info.emoji   || "🔬";
  resultName.textContent     = data.prediction;
  resultSeverity.textContent = info.severity || "Unknown";

  /* Donut confidence */
  const pct = Math.min(100, Math.max(0, data.confidence));
  const circumference = 213.6;
  const offset = circumference - (pct / 100) * circumference;
  requestAnimationFrame(() => {
    donutArc.style.strokeDashoffset = offset;
    donutArc.style.transition = "stroke-dashoffset 1s cubic-bezier(.22,1,.36,1)";
    donutPct.textContent = pct.toFixed(1) + "%";
  });

  /* Top 3 */
  top3List.innerHTML = "";
  (data.top3 || []).forEach((item, i) => {
    const w = Math.min(100, item.confidence).toFixed(1);
    const li = document.createElement("li");
    li.className = "top3-item";
    li.innerHTML = `
      <span class="top3-rank">${i + 1}</span>
      <div class="top3-name-wrap">
        <span class="top3-name">${item.label}</span>
        <div class="top3-bar-track"><div class="top3-bar-fill" style="width:0%"></div></div>
      </div>
      <span class="top3-pct">${w}%</span>
    `;
    top3List.appendChild(li);
    requestAnimationFrame(() => {
      li.querySelector(".top3-bar-fill").style.width = w + "%";
    });
  });

  /* Info lists */
  buildList(causesList,      info.causes      || ["No information available."]);
  buildList(symptomsList,    info.symptoms    || ["No information available."]);
  buildList(suggestionsList, info.suggestions || ["Consult a dermatologist."]);

  /* Reset to first tab */
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));
  document.querySelector('[data-tab="causes"]').classList.add("active");
  document.getElementById("tab-causes").classList.remove("hidden");
}

function buildList(el, items) {
  el.innerHTML = "";
  items.forEach(text => {
    const li = document.createElement("li");
    li.textContent = text;
    el.appendChild(li);
  });
}

/* ── Tabs ─────────────────────────────────────────────────── */
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.remove("hidden");
  });
});