/* ── DermaScan AI — script.js ───────────────────────────── */

const imageInput     = document.getElementById("imageInput");
const dropzone       = document.getElementById("dropzone");
const dzIdle         = document.getElementById("dzIdle");
const previewImg     = document.getElementById("previewImg");
const predictBtn     = document.getElementById("predictBtn");

const resultIdle     = document.getElementById("resultIdle");
const resultError    = document.getElementById("resultError");
const resultBody     = document.getElementById("resultBody");
const errorMsg       = document.getElementById("errorMsg");

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
let _modelReady  = false;
let _pollTimer   = null;

/* ══════════════════════════════════════════════════════════
   MODEL STATUS BANNER
   Polls /health every 5 s until the model is ready.
   Shows a fixed toast at the bottom of the screen.
══════════════════════════════════════════════════════════ */

function _getBanner() {
  let el = document.getElementById("_modelBanner");
  if (!el) {
    el = document.createElement("div");
    el.id = "_modelBanner";
    el.style.cssText = [
      "position:fixed", "bottom:1.25rem", "left:50%",
      "transform:translateX(-50%)",
      "background:rgba(15,15,25,0.93)",
      "color:#e2e8f0",
      "padding:0.55rem 1.2rem",
      "border-radius:999px",
      "font-size:0.83rem",
      "font-weight:500",
      "box-shadow:0 4px 24px rgba(0,0,0,0.45)",
      "backdrop-filter:blur(10px)",
      "-webkit-backdrop-filter:blur(10px)",
      "z-index:9999",
      "display:flex",
      "align-items:center",
      "gap:0.5rem",
      "transition:opacity 0.4s ease",
      "pointer-events:none",
    ].join(";");
    document.body.appendChild(el);
  }
  return el;
}

function _showBanner(icon, text, borderColor) {
  const b = _getBanner();
  b.style.opacity = "1";
  b.style.borderLeft = `3px solid ${borderColor}`;
  b.innerHTML = `<span style="font-size:1rem">${icon}</span><span>${text}</span>`;
}

function _hideBanner() {
  const b = document.getElementById("_modelBanner");
  if (!b) return;
  b.style.opacity = "0";
  setTimeout(() => { if (b.parentNode) b.parentNode.removeChild(b); }, 450);
}

async function _pollHealth() {
  try {
    const res  = await fetch("/health");
    const data = await res.json();

    if (data.status === "ready") {
      _modelReady = true;
      clearInterval(_pollTimer);
      _pollTimer = null;
      _showBanner("✅", "Model ready — you can analyse now!", "#22c55e");
      setTimeout(_hideBanner, 3000);
      // Enable button if a file is already selected
      if (selectedFile) predictBtn.disabled = false;
      return;
    }

    if (data.status === "failed") {
      clearInterval(_pollTimer);
      _pollTimer = null;
      const reason = data.error || "unknown error";
      _showBanner("❌", `Model failed to load: ${reason}`, "#ef4444");
      return;
    }

    // Still loading
    _showBanner("⏳", "Model loading on server, please wait…", "#f59e0b");

  } catch (_) {
    _showBanner("🔄", "Connecting to server…", "#94a3b8");
  }
}

// Poll immediately on page load, then every 5 seconds
_pollHealth();
_pollTimer = setInterval(_pollHealth, 5000);


/* ══════════════════════════════════════════════════════════
   DRAG-AND-DROP
══════════════════════════════════════════════════════════ */
dropzone.addEventListener("dragover", e => {
  e.preventDefault();
  dropzone.classList.add("over");
});
["dragleave", "dragend"].forEach(ev =>
  dropzone.addEventListener(ev, () => dropzone.classList.remove("over"))
);
dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("over");
  const f = e.dataTransfer?.files?.[0];
  if (f && f.type.startsWith("image/")) handleFile(f);
});


/* ══════════════════════════════════════════════════════════
   FILE INPUT
══════════════════════════════════════════════════════════ */
imageInput.addEventListener("change", () => {
  const f = imageInput.files?.[0];
  if (f) handleFile(f);
});

function handleFile(file) {
  selectedFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewImg.classList.remove("hidden");
  dzIdle.classList.add("hidden");
  // Only enable the button when the model is confirmed ready
  predictBtn.disabled = !_modelReady;
  showIdle();
}


/* ══════════════════════════════════════════════════════════
   ANALYSE BUTTON
══════════════════════════════════════════════════════════ */
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  // Extra guard — in case button was somehow enabled early
  if (!_modelReady) {
    showError("Model is still loading. Please wait for the ✅ banner and try again.");
    return;
  }

  predictBtn.classList.add("loading");
  predictBtn.disabled = true;
  showIdle();

  const fd = new FormData();
  fd.append("image", selectedFile);

  try {
    const res  = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();

    if (res.status === 503) {
      // Model not ready yet — restart polling
      _modelReady = false;
      showError("Model is still loading. Please wait for the ✅ banner and try again.");
      if (!_pollTimer) _pollTimer = setInterval(_pollHealth, 5000);
      _pollHealth();
      return;
    }

    if (!res.ok || data.error) {
      showError(data.error || `Server error ${res.status}`);
      return;
    }

    showResult(data);

  } catch (_) {
    showError("Could not reach the server. Check your connection and try again.");
  } finally {
    predictBtn.classList.remove("loading");
    predictBtn.disabled = false;
  }
});


/* ══════════════════════════════════════════════════════════
   DISPLAY HELPERS
══════════════════════════════════════════════════════════ */

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

  // Hero
  resultEmoji.textContent    = info.emoji    || "🔬";
  resultName.textContent     = data.prediction;
  resultSeverity.textContent = info.severity || "Unknown";

  // Donut confidence ring
  const pct          = Math.min(100, Math.max(0, data.confidence));
  const circumference = 213.6;
  const offset        = circumference - (pct / 100) * circumference;
  requestAnimationFrame(() => {
    donutArc.style.strokeDashoffset = offset;
    donutArc.style.transition = "stroke-dashoffset 1s cubic-bezier(.22,1,.36,1)";
    donutPct.textContent = pct.toFixed(1) + "%";
  });

  // Top 3 predictions with animated bars
  top3List.innerHTML = "";
  (data.top3 || []).forEach((item, i) => {
    const w  = Math.min(100, item.confidence).toFixed(1);
    const li = document.createElement("li");
    li.className = "top3-item";
    li.innerHTML = `
      <span class="top3-rank">${i + 1}</span>
      <div class="top3-name-wrap">
        <span class="top3-name">${item.label}</span>
        <div class="top3-bar-track">
          <div class="top3-bar-fill" style="width:0%"></div>
        </div>
      </div>
      <span class="top3-pct">${w}%</span>
    `;
    top3List.appendChild(li);
    requestAnimationFrame(() => {
      li.querySelector(".top3-bar-fill").style.width = w + "%";
    });
  });

  // Info tab lists
  buildList(causesList,       info.causes      || ["No information available."]);
  buildList(symptomsList,     info.symptoms    || ["No information available."]);
  buildList(suggestionsList,  info.suggestions || ["Consult a dermatologist."]);

  // Reset to first tab
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


/* ══════════════════════════════════════════════════════════
   TABS
══════════════════════════════════════════════════════════ */
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.remove("hidden");
  });
});