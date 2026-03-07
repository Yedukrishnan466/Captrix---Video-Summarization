r"""
Infer summary for one video using:
 - frame sampling
 - CLIP embeddings + trained selector MLP to pick keyframes
 - BLIP image captioning for keyframes
 - final rewrite using local Ollama LLM (if requested & available) else fallback to local summarizer (BART)

Usage:
  venv\Scripts\activate.bat
  python scripts\infer_summary.py
"""


import os
import warnings
import logging


# Reduce Hugging Face verbosity where possible
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# Disable HF symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Filter some known noisy warnings (targeted)
warnings.filterwarnings("ignore", message=".*resume_download is deprecated.*")
warnings.filterwarnings("ignore", message=".*weights_only=.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*symlinks.*")
warnings.filterwarnings("ignore", category=ResourceWarning)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# -------------------------------------------------------------------------

import argparse
import json
import subprocess
import shutil
import contextlib
import sys
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from transformers import CLIPProcessor, CLIPModel, pipeline
from brightness import calculate_brightness
import tempfile
from text_to_speech import say

# reuse MLP architecture for loading
import train_selector as ts  # expects train_selector.py in same folder

# Try to import python-side ollama generate if available; otherwise we will try CLI
try:
    from ollama import generate as ollama_generate  # some versions provide generate()
    _HAS_OLLAMA_PY = True
except Exception:
    _HAS_OLLAMA_PY = False

# -------------------------
# Helper: suppress native stderr (FFmpeg/OpenCV) for short sections
# -------------------------
@contextlib.contextmanager
def suppress_stderr():
    """
    Context manager that suppresses C-level stderr (fd 2) by redirecting it to os.devnull.
    Use around calls that invoke native libraries (e.g. cv2.VideoCapture) which print unwanted ffmpeg messages.
    """
    try:
        sys.stderr.flush()
    except Exception:
        pass

    try:
        old_stderr_fd = os.dup(sys.stderr.fileno())
    except Exception:
        # cannot duplicate; just yield (no suppression)
        yield
        return

    devnull_fd = None
    try:
        devnull_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull_fd, sys.stderr.fileno())
        yield
    finally:
        try:
            os.dup2(old_stderr_fd, sys.stderr.fileno())
        except Exception:
            pass
        try:
            os.close(old_stderr_fd)
        except Exception:
            pass
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except Exception:
                pass
        try:
            sys.stderr.flush()
        except Exception:
            pass

# -------------------------
# Utilities & core functions
# -------------------------
def dedupe_similar(captions, thresh=0.8):
    """Remove near-duplicate captions (keeps first occurrences)."""
    out = []
    for c in captions:
        if not any(SequenceMatcher(None, c, ex).ratio() > thresh for ex in out):
            out.append(c)
    return out

def sample_frames_from_video(video_path, num_samples=32):
    """
    Uniform sampling of frames from video. Returns list of PIL images and sampled indices.
    Suppresses native stderr during VideoCapture to hide ffmpeg / mmco warnings.
    """
    with suppress_stderr():
        vid = cv2.VideoCapture(str(video_path))
        if not vid.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            frames = []
            while True:
                ret, f = vid.read()
                if not ret:
                    break
                frames.append(f)
            total = len(frames)
            vid.release()
            if total == 0:
                raise RuntimeError("Video has no readable frames")
            indices = np.linspace(0, total - 1, num_samples, dtype=int)
            imgs = [Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)) for i in indices]
            return imgs, indices.tolist()

        indices = np.linspace(0, max(0, total - 1), num_samples, dtype=int)
        imgs = []
        for idx in indices:
            vid.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = vid.read()
            if not ret:
                continue
            imgs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        vid.release()
    return imgs, indices.tolist()

def select_keyframes(imgs, selector_model_path, device="cuda", top_k=4):
    """Compute CLIP image features for imgs, load selector MLP, score frames, then choose diverse top-k via KMeans."""
    if len(imgs) == 0:
        return [], np.array([])

    # load CLIP
    clip_model_name = "openai/clip-vit-base-patch32"
    clip = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = clip.get_image_features(**inputs)  # (N,D)
    feats_np = feats.cpu().numpy()

    # load selector MLP
    in_dim = feats_np.shape[1]
    model = ts.MLP(in_dim).to(device)
    # Try weights_only=True when available (silences future warnings on newer torch)
    try:
        state = torch.load(selector_model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(selector_model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(feats_np).to(device)).cpu().numpy()

    k = min(top_k, feats_np.shape[0])
    if k <= 0:
        return [], preds

    # cluster in feature space to encourage diversity
    km = KMeans(n_clusters=k, random_state=42).fit(feats_np)
    picked = []
    for c in km.cluster_centers_:
        dists = np.linalg.norm(feats_np - c[None, :], axis=1)
        candidate_idxs = np.argsort(preds)[-min(len(preds), k * 2):]
        candidate_idxs = np.unique(candidate_idxs)
        if len(candidate_idxs) == 0:
            best = int(np.argmin(dists))
        else:
            best = int(candidate_idxs[np.argmin(dists[candidate_idxs])])
        # create a temporary image to compute brightness
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                imgs[best].save(tmp.name)
                brightness = calculate_brightness(tmp.name)

            # keep only sufficiently bright frames
            if brightness >= 10:
                picked.append(best)

    picked = sorted(list(dict.fromkeys(picked)))

    return picked, preds

def clip_caption_similarity(image, caption, clip_model, clip_processor, device):
    """
    Computes cosine similarity between image and caption using CLIP.
    """
    inputs = clip_processor(
        text=[caption],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        image_embeds = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        text_embeds = clip_model.get_text_features(input_ids=inputs["input_ids"])

    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    similarity = torch.sum(image_embeds * text_embeds, dim=-1)
    return similarity.item()

def caption_images(imgs, device="cuda", num_votes=5, similarity_threshold=0.25):
    """
    BLIP-1 Large captioning with:
    - multi-caption voting
    - CLIP-based consistency check to remove hallucinations
    """

    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        CLIPProcessor, CLIPModel
    )
    from collections import Counter

    # BLIP
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)

    # CLIP
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device)

    captions = []

    for img in imgs:
        inputs = blip_processor(images=img, return_tensors="pt").to(device)

        accepted = None

        # Try multiple times to avoid hallucinations
        for attempt in range(num_votes):
            with torch.no_grad():
                out = blip_model.generate(
                    **inputs,
                    num_beams=5,
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=0.8,
                    early_stopping=True
                )

            caption = blip_processor.decode(
                out[0], skip_special_tokens=True
            ).strip()

            # CLIP consistency check
            score = clip_caption_similarity(
                img, caption, clip_model, clip_processor, device
            )

            if score >= similarity_threshold:
                accepted = caption
                break  # accept first good caption

        # Fallback if all attempts fail
        if accepted is None:
            accepted = caption  # last generated caption

        captions.append(accepted)

    return captions

# -------------------------
# Local Ollama integration (tries python client first, then CLI fallback)
# -------------------------
def _call_ollama_python_client(model, prompt, max_tokens=250):
    """
    Call python-side ollama generate (signature variations exist).
    """
    if not _HAS_OLLAMA_PY:
        raise RuntimeError("Ollama python client not available")
    try:
        resp = ollama_generate(model, prompt, max_tokens=max_tokens)
        if isinstance(resp, dict):
            for k in ("response", "text", "result", "outputs"):
                if k in resp:
                    val = resp[k]
                    if isinstance(val, (list, tuple)) and len(val) > 0:
                        return str(val[0]).strip()
                    return str(val).strip()
            return json.dumps(resp)
        return str(resp).strip()
    except TypeError as te:
        raise RuntimeError(f"Ollama python client generate failed (signature mismatch): {te}")
    except Exception as e:
        raise RuntimeError(f"Ollama python client generate failed: {e}")

def _call_ollama_cli(model, prompt, max_tokens=250, timeout=60):
    """
    Call `ollama run <model>` via subprocess (prompt passed on stdin).
    """
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ollama CLI failed: {proc.stderr.decode('utf-8', errors='ignore')}")
        out = proc.stdout.decode("utf-8", errors="ignore").strip()
        return out
    except FileNotFoundError:
        raise RuntimeError("ollama CLI not found. Did you install Ollama and add it to PATH?")
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Ollama CLI timeout: {e}")
    except Exception as e:
        raise RuntimeError(f"Ollama CLI call error: {e}")

def local_ollama_rewrite(captions, model="llama3.2:1b", max_tokens=250):
    """
    Rewrite captions using local Ollama model (python client first, CLI fallback).
    """
    if len(captions) == 0:
        return ""
    prompt = (
        "You are a helpful assistant. Given the following image captions from a video's keyframes, "
        "write a single concise, human-like paragraph summarizing the video's content. "
        "Use simple language. Add details from all these captions."
        "Do not add additional details that are not specified in the captions."
        "Mention main objects, actors, and actions without repetition.\n\n"
        "Captions:\n" + "\n".join(f"- {c}" for c in captions) + "\n\nSummary:"
    )

    if _HAS_OLLAMA_PY:
        try:
            text = _call_ollama_python_client(model, prompt, max_tokens=max_tokens)
            print("[OMODEL] Used Ollama Python client for rewrite.")
            return text
        except Exception as e:
            print(f"[WARN] Ollama python client failed: {e} (falling back to ollama CLI)")

    text = _call_ollama_cli(model, prompt, max_tokens=max_tokens)
    print("[OMODEL] Used Ollama CLI for rewrite.")
    return text

# -------------------------
# Summarization pipeline (Ollama local LLM preferred, fallback to local summarizer)
# -------------------------
def summarize_captions(captions, device_for_pipeline=0, use_local_llm=False, local_llm_model="llama3.2:1b", local_llm_max_tokens=250):
    """
    Summarize captions into a final paragraph.
    """
    if len(captions) == 0:
        return ""

    captions = dedupe_similar(captions)

    if use_local_llm:
        try:
            summary = local_ollama_rewrite(captions, model=local_llm_model, max_tokens=local_llm_max_tokens)
            if isinstance(summary, str) and len(summary.strip()) > 0:
                return summary.strip()
        except Exception as e:
            print(f"[WARN] Local Ollama rewrite failed: {e}. Falling back to local summarizer.")

    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device_for_pipeline)
    except Exception as e:
        print(f"[WARN] Could not load summarizer pipeline: {e}. Falling back to concatenation.")
        return " ".join(captions)

    input_text = " ; ".join([c.strip().rstrip('.') for c in captions])
    prompt = f"Write a single concise paragraph summarizing these captions: {input_text}"
    out = summarizer(prompt, max_length=120, min_length=25, do_sample=False)
    return out[0]["summary_text"].strip()

# -------------------------
# (Optional) helper to re-encode input using ffmpeg (not used by default)
# -------------------------
def reencode_with_ffmpeg(inpath, outpath=None, crf=18):
    """
    Re-encode video with ffmpeg to produce a clean H.264 stream.
    Returns path to cleaned file. Raises FileNotFoundError if ffmpeg not found.
    """
    p = Path(inpath)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg not found in PATH")

    out = Path(outpath) if outpath else p.with_name(p.stem + "_clean.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(p),
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        "-c:a", "copy",
        str(out)
    ]
    subprocess.run(cmd, check=True)
    return str(out)

# -------------------------
# CLI entrypoint
# -------------------------
def run_inference(
    video_path: str,
    out_dir: str,
    samples,
    keyframes
):
    # -----------------------------
    # FIXED CONFIGURATION (NO CLI)
    # -----------------------------
    SELECTOR_PATH = "models/selector.pth"
    NUM_SAMPLES = samples
    TOP_K = keyframes
    USE_LOCAL_LLM = True
    LOCAL_LLM_MODEL = "llama3:8b"
    LOCAL_LLM_MAX_TOKENS = 250
    AUTO_CLEAN = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # -----------------------------
    # Optional video cleaning
    # -----------------------------
    if AUTO_CLEAN:
        try:
            print("[INFO] Re-encoding video with ffmpeg...")
            video_path = reencode_with_ffmpeg(video_path)
        except Exception as e:
            print(f"[WARN] auto_clean failed: {e}")

    # -----------------------------
    # Frame sampling
    # -----------------------------
    imgs, _ = sample_frames_from_video(video_path, num_samples=NUM_SAMPLES)
    if len(imgs) == 0:
        raise RuntimeError("No frames extracted")

    # -----------------------------
    # Keyframe selection
    # -----------------------------
    picked, _ = select_keyframes(
        imgs,
        selector_model_path=SELECTOR_PATH,
        device=device,
        top_k=TOP_K
    )

    if len(picked) == 0:
        print("[WARN] No keyframes selected, falling back to first frame")
        picked = [0]

    selected_imgs = [imgs[i] for i in picked]

    # -----------------------------
    # Caption generation (BLIP-1)
    # -----------------------------
    try:
        captions = caption_images(selected_imgs, device=device)
    except Exception as e:
        print(f"[ERROR] captioning failed: {e}")
        captions = ["an image"] * len(selected_imgs)

    keyframe_captions = [
        {
            "frame": i,
            "caption": c
        }
        for i, c in enumerate(captions)
    ]
    print(keyframe_captions)
    # -----------------------------
    # Summarization
    # -----------------------------
    device_pipeline = 0 if device == "cuda" else -1
    summary = summarize_captions(
        captions,
        device_for_pipeline=device_pipeline,
        use_local_llm=USE_LOCAL_LLM,
        local_llm_model=LOCAL_LLM_MODEL,
        local_llm_max_tokens=LOCAL_LLM_MAX_TOKENS
    )

    print("\nFinal summary:\n", summary)

    # -----------------------------
    # Text-to-speech
    # -----------------------------

    say(summary, Path(video_path).name, out_dir)

    # -----------------------------
    # Save outputs
    # -----------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "summary.txt", "w", encoding="utf8") as f:
        f.write(summary)

    for i, img in enumerate(selected_imgs):
        img.save(out_dir / f"key_{i}.jpg")

    print(f"[DONE] Saved outputs to {out_dir}")
    return summary, keyframe_captions

if __name__ == "__main__":
    run_inference(
        video_path="sample5.mp4",
        out_dir="outputs/sample5",
        samples=64,
        keyframes=16
    )