"""
backend/model_loader.py — Standalone model loader with background threading.

This module can be imported directly for local dev or testing:
    from backend.model_loader import get_state, start_loading

The main app (main.py) calls start_loading() once at startup.
All state is thread-safe via threading.Lock().
"""

import os
import shutil
import threading

# ── Config (reads from environment, same as main.py) ──────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "skin_model.h5")

HF_REPO_ID  = os.environ.get("HF_REPO_ID", "").strip()
HF_FILENAME = "skin_model.h5"
HF_TOKEN    = os.environ.get("HF_TOKEN", "").strip() or None

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Thread-safe state ──────────────────────────────────────────────────────────
_lock         = threading.Lock()
_model        = None
_load_error   = ""
_model_status = "loading"   # "loading" | "ready" | "failed"
_thread_started = False


def _set_state(status: str, error: str = "", mdl=None):
    global _model, _load_error, _model_status
    with _lock:
        _model_status = status
        _load_error   = error
        if mdl is not None:
            _model = mdl


def get_state() -> tuple:
    """
    Returns (status, error, model).
    status is one of: "loading" | "ready" | "failed"
    """
    with _lock:
        return _model_status, _load_error, _model


def get_model():
    """Shortcut — returns the loaded Keras model or None."""
    with _lock:
        return _model


def is_ready() -> bool:
    """Returns True only when the model is loaded and ready to predict."""
    with _lock:
        return _model_status == "ready" and _model is not None


# ── Download ───────────────────────────────────────────────────────────────────
def _download_from_hf() -> bool:
    """
    Download skin_model.h5 from Hugging Face Hub.
    Requires HF_REPO_ID env var to be set on Render.
    Returns True on success, False on failure.
    """
    if not HF_REPO_ID:
        msg = "HF_REPO_ID env var is not set — add it in Render → Environment Variables"
        _set_state("failed", msg)
        print(f"❌ {msg}")
        return False

    try:
        from huggingface_hub import hf_hub_download
        print(f"⬇️  Downloading model: {HF_REPO_ID}/{HF_FILENAME}")
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=HF_TOKEN,
        )
        shutil.copy(downloaded, MODEL_PATH)
        print(f"✅ Model saved → {MODEL_PATH}")
        return True

    except Exception as exc:
        _set_state("failed", f"HuggingFace download failed: {exc}")
        print(f"❌ Download failed: {exc}")
        return False


# ── Background loader thread ───────────────────────────────────────────────────
def _load_model_thread():
    """
    Runs in a daemon thread.

    1. Downloads model from Hugging Face if not on disk.
    2. Loads model into memory using TensorFlow/Keras.

    TensorFlow is imported INSIDE this function (not at module level)
    so that importing this module does NOT trigger TF initialisation.
    This keeps Gunicorn startup fast and lets Render's health-check pass
    before the model is even downloaded.
    """
    # Step 1 — Download if missing
    if not os.path.exists(MODEL_PATH):
        ok = _download_from_hf()
        if not ok:
            return   # state already set to "failed"

    # Step 2 — Load into memory
    try:
        import tensorflow as tf   # deferred import — intentional, keeps startup fast
        print("📂 Loading model into memory…")
        mdl = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model ready — output classes: {mdl.output_shape[-1]}")
        _set_state("ready", mdl=mdl)

    except Exception as exc:
        # Delete broken/corrupt file so next deploy downloads fresh copy
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        _set_state("failed", f"Keras load error: {exc}")
        print(f"❌ Model load failed: {exc}")


def start_loading():
    """
    Starts the background model-loading thread.
    Safe to call multiple times — only starts one thread.
    Called automatically when this module is imported by main.py.
    """
    global _thread_started
    with _lock:
        if _thread_started:
            return
        _thread_started = True

    t = threading.Thread(
        target=_load_model_thread,
        daemon=True,
        name="model-loader",
    )
    t.start()
    print("🚀 Model loader thread started")


# ── Auto-start when imported ───────────────────────────────────────────────────
# Calling start_loading() here means:
#   - main.py gets the thread started the moment it imports this module
#   - No extra call needed in main.py
#   - Thread only ever starts once (guarded by _thread_started flag)
start_loading()