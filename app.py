"""
Helpa — first-aid assistant, Gradio front end. Simple, one file.

Same pattern as the NutriMama gradio app: on startup this downloads the
model + vision projector + a llama-server binary (each skipped if already
present), starts llama-server as a child process, and serves a chat UI in
front of it. No Docker, no hosting account.

Run:
    pip install -r requirements.txt
    python app.py

No system prompt is sent to the model. Helpa's training data was fixed to
NOT use one after a real bug: mixing system-tagged and system-less examples
taught the model to treat the system prompt's presence as a mode switch,
and it collapsed into echoing the user's message back when called without
one. Sending a system message here would put it back in that broken
condition, so this app deliberately never sends one.

Accepts TEXT, IMAGE, and VIDEO in the same input box. Video is NOT true
video understanding -- llama.cpp's API only has an image_url content type,
not a video one. A representative frame is extracted from the video with
OpenCV and sent as a single image; the UI says this plainly so a video test
isn't mistaken for the model watching the clip.
"""
import base64
import io
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request

import cv2
import gradio as gr
import requests
from PIL import Image

# ───────────────────────── config ──────────────────────────────────────────
REPO = "yusasif/Helpa-ADTC-challenge"
MODEL_FILE  = "Helpa-Gemma4-E2B-Q4_K_M.gguf"  # same name on HF and locally -- no rename needed
MMPROJ_FILE = "mmproj-Helpa-f16.gguf"
MODEL_DIR = os.environ.get("HELPA_MODEL_DIR", "./model")  # singular, matches the submission folder too
PORT = int(os.environ.get("LLAMA_PORT", 8098))
LLAMA_BIN   = os.environ.get("LLAMA_SERVER_BIN")
LLAMA_TAG   = "b10584"

VIDEO_EXT = {".mp4", ".mov", ".webm", ".avi", ".mkv"}

SUGGESTIONS = [
    "Someone spilled hot oil on their arm. What should I do right now, and what should I avoid?",
    "A snake bit someone on the leg. What are the first aid steps before we reach a clinic?",
    "My child is having very high fever and shivering. Could it be malaria?",
    "Ejò bu ẹnìkan jẹ ní ẹsẹ̀. Kí ni àwọn ìgbésẹ̀ tí mo gbọdọ̀ ṣe?",
]

# ───────────────────────── model bootstrap ─────────────────────────────────
def _download(remote_fname, dest):
    """remote_fname is the name on HuggingFace; dest is the local save path,
    which may use a different (branded) filename -- see MODEL_FILE."""
    if os.path.exists(dest):
        print(f"[model] {os.path.basename(dest)} already present, skipping download")
        return
    url = f"https://huggingface.co/{REPO}/resolve/main/{remote_fname}"
    print(f"[model] downloading {remote_fname} -> {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest + ".partial")
    os.rename(dest + ".partial", dest)
    print(f"[model] done: {dest}")


def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    mmproj_path = os.path.join(MODEL_DIR, MMPROJ_FILE)
    _download(MODEL_FILE, model_path)
    try:
        _download(MMPROJ_FILE, mmproj_path)
    except Exception as e:
        print(f"[model] mmproj not fetched ({e}) — photo/video input will be disabled")
        mmproj_path = None
    return model_path, mmproj_path


def ensure_llama_binary():
    if LLAMA_BIN:
        return LLAMA_BIN
    found = shutil.which("llama-server")
    if found:
        print(f"[llama.cpp] found on PATH: {found}")
        return found

    import platform, tarfile, zipfile
    sysname, machine = platform.system(), platform.machine().lower()
    if sysname == "Darwin":
        asset = "macos-arm64" if machine in ("arm64", "aarch64") else "macos-x64"
    elif sysname == "Windows":
        asset = "win-cpu-x64"
    else:
        asset = "ubuntu-arm64" if machine in ("arm64", "aarch64") else "ubuntu-x64"
    ext = "zip" if sysname == "Windows" else "tar.gz"

    cache = os.path.join(MODEL_DIR, "llama_cpp_bin")
    os.makedirs(cache, exist_ok=True)
    exe_name = "llama-server.exe" if sysname == "Windows" else "llama-server"
    for root, _, files in os.walk(cache):
        if exe_name in files:
            return os.path.join(root, exe_name)

    url = (f"https://github.com/ggml-org/llama.cpp/releases/download/"
          f"{LLAMA_TAG}/llama-{LLAMA_TAG}-bin-{asset}.{ext}")
    archive = os.path.join(cache, f"archive.{ext}")
    print(f"[llama.cpp] downloading engine ({asset})...")
    urllib.request.urlretrieve(url, archive)
    if ext == "zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(cache)
    else:
        with tarfile.open(archive) as t:
            t.extractall(cache)
    os.remove(archive)
    for root, _, files in os.walk(cache):
        if exe_name in files:
            path = os.path.join(root, exe_name)
            os.chmod(path, 0o755)
            return path
    raise RuntimeError(f"llama-server binary not found after extracting {url}")


def start_llama_server(model_path, mmproj_path):
    llama_bin = ensure_llama_binary()
    cmd = [llama_bin, "-m", model_path, "--jinja", "--reasoning", "off",
          "--host", "127.0.0.1", "--port", str(PORT), "-t", "8"]
    if mmproj_path:
        cmd += ["--mmproj", mmproj_path]
    print("[llama-server]", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                print("[llama-server] up")
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("llama-server did not come up in time")


# ───────────────────────── inference ───────────────────────────────────────
def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _extract_video_frame(path: str) -> Image.Image:
    """Grab one representative (middle) frame. This is what 'video input'
    actually means here -- see the module docstring."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read a frame from that video")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def ask(message, history):
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    note = ""

    image = None
    if files:
        path = files[0]
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXT:
            try:
                image = _extract_video_frame(path)
                note = ("\n\n*(Video input: llama.cpp only accepts still images, so this "
                        "answer is based on one frame taken from the middle of the clip, "
                        "not the full video.)*")
            except Exception as e:
                return f"⚠️ Could not read that video: {e}"
        else:
            image = Image.open(path)

    if image is not None:
        content = [{"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                  {"type": "text", "text": text or "What do you see, and what first aid applies?"}]
    else:
        content = text or "..."

    # No system message -- see module docstring. Sending one puts the model
    # back in the exact condition that used to break it.
    payload = {"messages": [{"role": "user", "content": content}],
              "temperature": 0.2, "max_tokens": 400, "stream": False}
    try:
        r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] + note
    except Exception as e:
        return f"⚠️ Could not reach the model: {e}"


# ───────────────────────── UI — simple, no marketing panel ─────────────────
def build_ui():
    with gr.Blocks(title="Helpa") as demo:
        gr.Markdown(
            "# 🩹 Helpa\n"
            "First-aid guidance, in Yorùbá or English. Type a question, or attach a "
            "photo/video — runs fully offline once the model is downloaded."
        )
        gr.ChatInterface(
            fn=ask,
            multimodal=True,
            textbox=gr.MultimodalTextbox(
                placeholder="Describe what happened, or attach a photo/video…",
                file_types=["image", "video"], file_count="single"),
            examples=[{"text": s, "files": []} for s in SUGGESTIONS],
        )
    return demo


if __name__ == "__main__":
    model_path, mmproj_path = ensure_model()
    server_proc = start_llama_server(model_path, mmproj_path)
    try:
        build_ui().launch(server_name="0.0.0.0",share =True, server_port=int(os.environ.get("PORT", 7861)))
    finally:
        server_proc.terminate()
