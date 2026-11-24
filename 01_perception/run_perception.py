import base64
from concurrent.futures import ThreadPoolExecutor
import ctypes
from difflib import SequenceMatcher
import gc
import glob
import json
import os
import re
import site
import subprocess
import sys
import time
import traceback
import requests


def get_cfg(key: str, injected: str) -> str:
    env_val = os.environ.get(key, "").strip()
    if env_val:
        return env_val
    if not injected.startswith("__INJECT_"):
        return injected
    return ""


JOB_ID = get_cfg("JOB_ID", "__INJECT_JOB_ID__")
GH_REPO = get_cfg("GH_REPO", "__INJECT_GH_REPO__")
GH_TOKEN = get_cfg("GH_PAT_TOKEN", "__INJECT_GH_PAT_TOKEN__")
TG_TOKEN = get_cfg("TELEGRAM_BOT_TOKEN", "__INJECT_TELEGRAM_BOT_TOKEN__")
TG_CHAT = get_cfg("TELEGRAM_CHAT_ID", "__INJECT_TELEGRAM_CHAT_ID__")

if not JOB_ID:
    raise RuntimeError("Thiếu JOB_ID khi khởi chạy kernel.")


class GitHubSync:
    def __init__(self, repo: str, token: str, state_branch: str = "pipeline-state"):
        self.repo = repo
        self.token = token
        self.state_branch = state_branch
        self.api_base = f"https://api.github.com/repos/{repo}"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def read_json(self, path: str) -> tuple[dict, str | None]:
        url = f"{self.api_base}/contents/{path}"
        r = requests.get(url, headers=self.headers, params={"ref": self.state_branch}, timeout=30)
        if r.status_code == 404:
            return {}, None
        r.raise_for_status()
        data = r.json()
        content = base64.b64decode(data["content"]).decode("utf-8").strip()
        if not content:
            return {}, data["sha"]
        return json.loads(content), data["sha"]

    def write_json(self, path: str, payload: dict, message: str, max_retries: int = 5):
        url = f"{self.api_base}/contents/{path}"
        for attempt in range(max_retries):
            _, sha = self.read_json(path)
            raw_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            body = {
                "message": message,
                "content": base64.b64encode(raw_bytes).decode("utf-8"),
                "branch": self.state_branch,
            }
            if sha:
                body["sha"] = sha
            r = requests.put(url, headers=self.headers, json=body, timeout=30)
            if r.status_code in (200, 201):
                return
            if r.status_code == 409:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
        raise RuntimeError(f"Không thể ghi {path} sau {max_retries} lần thử.")

    def ensure_release(self, tag: str, name: str | None = None) -> dict:
        url = f"{self.api_base}/releases/tags/{tag}"
        r = requests.get(url, headers=self.headers, timeout=30)
        if r.status_code == 200:
            return r.json()
        cr = requests.post(
            f"{self.api_base}/releases",
            headers=self.headers,
            json={"tag_name": tag, "name": name or f"Artifacts - {tag}", "draft": False, "prerelease": False},
            timeout=30,
        )
        cr.raise_for_status()
        return cr.json()

    def upload_asset(self, tag: str, file_path: str):
        release = self.ensure_release(tag)
        upload_url = release["upload_url"].split("{")[0]
        filename = os.path.basename(file_path)
        for asset in release.get("assets", []):
            if asset["name"] == filename:
                requests.delete(f"{self.api_base}/releases/assets/{asset['id']}", headers=self.headers, timeout=30)
        up_headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/octet-stream"}
        with open(file_path, "rb") as f:
            r = requests.post(f"{upload_url}?name={filename}", headers=up_headers, data=f.read(), timeout=300)
        r.raise_for_status()
        return r.json().get("browser_download_url")

    def download_asset(self, tag: str, filename: str, dest_path: str):
        release = self.ensure_release(tag)
        for asset in release.get("assets", []):
            if asset["name"] == filename:
                dl_headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/octet-stream"}
                with requests.get(asset["url"], headers=dl_headers, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(dest_path, "wb") as out:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                out.write(chunk)
                return dest_path
        raise FileNotFoundError(f"Không tìm thấy asset {filename} trong release {tag}")

    def trigger_dispatch(self, event_type: str, client_payload: dict):
        r = requests.post(
            f"{self.api_base}/dispatches",
            headers=self.headers,
            json={"event_type": event_type, "client_payload": client_payload},
            timeout=30,
        )
        r.raise_for_status()


def send_telegram(bot_token: str, chat_id: str, text: str):
    if not bot_token or not chat_id:
        return
    clean_token = str(bot_token).strip()
    clean_chat = str(chat_id).strip()
    try:
        requests.post(
            f"https://api.telegram.org/bot{clean_token}/sendMessage",
            json={"chat_id": clean_chat, "text": text[:4000], "parse_mode": "HTML", "disable_web_page_preview": False},
            timeout=20,
        )
    except Exception as e:
        print(f"[WARN] Gửi Telegram thất bại: {e}")


def ensure_deps():
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "--upgrade",
            "faster-whisper", "nvidia-cublas-cu12", "nvidia-cudnn-cu12",
            "demucs", "soundfile", "numpy", "gdown", "transformers", "accelerate",
            "paddlepaddle", "paddleocr",
        ],
        check=True,
    )


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _normalize_gdrive_url(url: str) -> str:
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url) or re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return f"https://drive.google.com/uc?id={m.group(1)}"
    return url


def download_video_source(sync: GitHubSync, tag: str, source_ref: str, dest_path: str):
    ref = (source_ref or "source_input.mp4").strip()
    if ref.startswith(("http://", "https://")):
        if "drive.google.com" in ref:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
            direct_url = _normalize_gdrive_url(ref)
            subprocess.run([sys.executable, "-m", "gdown", direct_url, "-O", dest_path], check=True)
        else:
            subprocess.run(["curl", "-fSL", ref, "-o", dest_path], check=True)
    else:
        sync.download_asset(tag, ref, dest_path)


def _preload_cuda12_libs():
    search_roots = list(site.getsitepackages())
    if site.getusersitepackages():
        search_roots.append(site.getusersitepackages())

    extra_dirs = []
    for root in search_roots:
        for pattern in [
            os.path.join(root, "nvidia", "cublas", "lib", "libcublas.so.12*"),
            os.path.join(root, "nvidia", "cublas", "lib", "libcublasLt.so.12*"),
            os.path.join(root, "nvidia", "cudnn", "lib", "libcudnn*.so.9*"),
            os.path.join(root, "nvidia", "cudnn", "lib", "libcudnn*.so.8*"),
        ]:
            for so_path in sorted(glob.glob(pattern)):
                lib_dir = os.path.dirname(so_path)
                if lib_dir not in extra_dirs:
                    extra_dirs.append(lib_dir)
                try:
                    ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

    if extra_dirs:
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(extra_dirs + ([current_ld] if current_ld else []))


def estimate_pitch_gender(clip: "np.ndarray", sr: int) -> tuple[float, str]:
    import numpy as np

    if clip is None or len(clip) < sr // 2:
        return 0.0, "unknown"
    samples = clip[: min(len(clip), sr * 6)].astype(np.float32)
    samples = samples - np.mean(samples)
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < 1e-4:
        return 0.0, "unknown"

    corr = np.correlate(samples, samples, mode="full")[len(samples) - 1:]
    min_lag = max(1, int(sr / 340))
    max_lag = min(len(corr) - 1, int(sr / 75))
    if max_lag <= min_lag:
        return 0.0, "unknown"

    best_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    f0 = float(sr) / float(best_lag)
    if f0 < 162.0:
        return round(f0, 1), "male"
    return round(f0, 1), "female"


def run_whisper_asr(audio_16k: str) -> list[dict]:
    import numpy as np
    import soundfile as sf
    import torch

    audio_np, sr = sf.read(audio_16k, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=-1)

    model_size = os.environ.get("WHISPER_MODEL", "large-v3")
    segments = []

    try:
        _preload_cuda12_libs()
        from faster_whisper import WhisperModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        raw_segments, _ = model.transcribe(audio_np, beam_size=5, vad_filter=True)

        for idx, seg in enumerate(raw_segments):
            txt = (seg.text or "").strip()
            if not txt:
                continue
            start_t = round(float(seg.start), 2)
            end_t = round(max(float(seg.end), start_t + 0.5), 2)
            clip = audio_np[int(start_t * sr): int(end_t * sr)]
            f0, gender_hint = estimate_pitch_gender(clip, sr)
            segments.append({
                "id": idx,
                "start": start_t,
                "end": end_t,
                "speaker_id": "SPEAKER_00",
                "gender_hint": gender_hint,
                "pitch_hz": f0,
                "asr_text": txt,
            })

        del model
        free_gpu()
    except Exception as fw_err:
        print(f"[WARN] faster-whisper fallback sang PyTorch Whisper do: {fw_err}")
        free_gpu()

        from transformers import pipeline

        hf_model = f"openai/whisper-{model_size}" if "/" not in model_size else model_size
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=hf_model,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device=0 if torch.cuda.is_available() else -1,
        )
        result = asr_pipe(
            {"array": audio_np, "sampling_rate": sr},
            chunk_length_s=30,
            batch_size=8,
            return_timestamps=True,
        )

        for idx, chunk in enumerate(result.get("chunks", [])):
            txt = (chunk.get("text") or "").strip()
            ts = chunk.get("timestamp") or (0.0, 0.0)
            start_t = round(float(ts[0]) if ts[0] is not None else 0.0, 2)
            end_t = round(max(float(ts[1]) if ts[1] is not None else start_t + 2.0, start_t + 0.5), 2)
            if not txt:
                continue
            clip = audio_np[int(start_t * sr): int(end_t * sr)]
            f0, gender_hint = estimate_pitch_gender(clip, sr)
            segments.append({
                "id": idx,
                "start": start_t,
                "end": end_t,
                "speaker_id": "SPEAKER_00",
                "gender_hint": gender_hint,
                "pitch_hz": f0,
                "asr_text": txt,
            })

        del asr_pipe
        free_gpu()

    for s in segments:
        f0 = float(s.get("pitch_hz") or 0.0)
        if f0 > 0 and f0 < 155.0:
            s["speaker_id"] = "SPEAKER_MALE"
            s["gender_hint"] = "male"
        elif f0 >= 165.0:
            s["speaker_id"] = "SPEAKER_FEMALE"
            s["gender_hint"] = "female"
        else:
            s["speaker_id"] = "SPEAKER_00"

    return segments


def detect_subtitle_roi(video_path: str, probe_dir: str) -> tuple[float, float, float, float]:
    """Tự động dò vùng phụ đề (x1, y1, x2, y2 theo tỷ lệ 0..1) linh hoạt trên toàn khung hình."""
    import numpy as np
    from PIL import Image

    default_roi = (0.08, 0.70, 0.92, 0.95)
    os.makedirs(probe_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", "fps=0.4,scale=640:-1", os.path.join(probe_dir, "p_%04d.jpg")],
        capture_output=True,
        text=True,
    )
    imgs = sorted(f for f in os.listdir(probe_dir) if f.endswith(".jpg"))[:24]
    if not imgs:
        return default_roi

    y_accum = None
    x_accum = None
    valid_frames = 0

    for fname in imgs:
        arr = np.asarray(Image.open(os.path.join(probe_dir, fname)).convert("L"), dtype=np.float32)
        h, w = arr.shape
        if h < 40 or w < 40:
            continue

        y_min_px = int(h * 0.08)
        y_max_px = int(h * 0.96)
        x_min_px = int(w * 0.05)
        x_max_px = int(w * 0.95)
        crop = arr[y_min_px:y_max_px, x_min_px:x_max_px]
        if crop.size == 0:
            continue

        dx = np.abs(np.diff(crop, axis=1))
        active_pixels = (dx > 24.0).astype(np.float32)

        row_sig = np.mean(active_pixels, axis=1)
        col_sig = np.mean(active_pixels, axis=0)

        if y_accum is None:
            y_accum = row_sig
            x_accum = col_sig
        elif len(row_sig) == len(y_accum) and len(col_sig) == len(x_accum):
            y_accum += row_sig
            x_accum += col_sig
        valid_frames += 1

    if valid_frames == 0 or y_accum is None or x_accum is None or len(y_accum) < 10:
        return default_roi

    win_y = max(3, min(16, len(y_accum) // 14))
    y_smooth = np.convolve(y_accum, np.ones(win_y, dtype=np.float32) / win_y, mode="same")
    peak_y = int(np.argmax(y_smooth))
    if float(y_smooth[peak_y]) <= 0.015 * valid_frames:
        return default_roi

    h_total = float(len(y_accum)) / 0.88
    y_start_offset = h_total * 0.08
    band_h_px = max(16.0, h_total * 0.14)

    top_rel = max(0.0, (peak_y - band_h_px * 0.5) / float(len(y_accum)))
    bot_rel = min(1.0, (peak_y + band_h_px * 0.5) / float(len(y_accum)))

    y1 = round(max(0.05, min(0.88, (y_start_offset + top_rel * len(y_accum)) / h_total)), 3)
    y2 = round(max(y1 + 0.08, min(0.98, (y_start_offset + bot_rel * len(y_accum)) / h_total)), 3)

    win_x = max(5, min(32, len(x_accum) // 12))
    x_smooth = np.convolve(x_accum, np.ones(win_x, dtype=np.float32) / win_x, mode="same")
    x_thresh = float(np.max(x_smooth)) * 0.35
    active_cols = np.where(x_smooth >= x_thresh)[0]

    if len(active_cols) >= 4:
        w_total = float(len(x_accum)) / 0.90
        x_start_offset = w_total * 0.05
        c_left = max(0.0, float(active_cols[0]) - win_x * 0.5)
        c_right = min(float(len(x_accum)), float(active_cols[-1]) + win_x * 0.5)
        x1 = round(max(0.03, min(0.70, (x_start_offset + c_left) / w_total)), 3)
        x2 = round(max(x1 + 0.25, min(0.97, (x_start_offset + c_right) / w_total)), 3)
    else:
        x1, x2 = 0.08, 0.92

    return (x1, y1, x2, y2)


def extract_ocr_frames(video_path: str, work_dir: str) -> tuple[list[tuple[float, str]], tuple[float, float, float, float]]:
    roi = detect_subtitle_roi(video_path, os.path.join(work_dir, "roi_probe"))
    x1, y1, x2, y2 = roi
    crop_w = max(0.2, x2 - x1)
    crop_h = max(0.06, y2 - y1)

    ocr_dir = os.path.join(work_dir, "ocr_frames")
    os.makedirs(ocr_dir, exist_ok=True)
    fps = os.environ.get("OCR_FPS", "1.0")
    vf_expr = f"fps={fps},crop=iw*{crop_w:.3f}:ih*{crop_h:.3f}:iw*{x1:.3f}:ih*{y1:.3f},scale=1280:-1"

    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", vf_expr, os.path.join(ocr_dir, "f_%04d.jpg")],
        capture_output=True,
        text=True,
    )
    interval = 1.0 / max(0.1, float(fps))
    frames = []
    for fname in sorted(os.listdir(ocr_dir)):
        if not fname.endswith(".jpg"):
            continue
        try:
            idx = int(fname.split("_")[1].split(".")[0])
        except Exception:
            idx = len(frames) + 1
        ts = round((idx - 1) * interval, 2)
        frames.append((ts, os.path.join(ocr_dir, fname)))

    print(f"[INFO] Khoanh vùng phụ đề ROI: x={x1}..{x2}, y={y1}..{y2} ({len(frames)} khung hình)")
    return frames[:120], roi


def run_paddle_ocr(video_path: str, work_dir: str) -> tuple[list[dict], tuple[float, float, float, float]]:
    frames, roi = extract_ocr_frames(video_path, work_dir)
    if not frames:
        return [], roi

    ocr_records = []
    try:
        from paddleocr import PaddleOCR

        lang = os.environ.get("PADDLE_OCR_LANG", "ch")
        use_gpu = os.environ.get("PADDLE_OCR_GPU", "0").strip() == "1"

        try:
            ocr = PaddleOCR(lang=lang, use_textline_orientation=True)
        except Exception:
            try:
                ocr = PaddleOCR(lang=lang)
            except Exception:
                ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)

        for ts, img_path in frames:
            line_texts = []
            try:
                if hasattr(ocr, "predict"):
                    res = list(ocr.predict(img_path))
                else:
                    res = ocr.ocr(img_path)
            except Exception:
                continue

            if not res:
                continue

            for item in res:
                rec_texts = None
                rec_scores = None
                if isinstance(item, dict):
                    rec_texts = item.get("rec_texts")
                    rec_scores = item.get("rec_scores")
                elif hasattr(item, "get"):
                    rec_texts = item.get("rec_texts")
                    rec_scores = item.get("rec_scores")
                elif hasattr(item, "rec_texts"):
                    rec_texts = getattr(item, "rec_texts", None)
                    rec_scores = getattr(item, "rec_scores", None)

                if rec_texts is not None:
                    scores = list(rec_scores) if rec_scores is not None else [1.0] * len(rec_texts)
                    for txt_raw, conf in zip(rec_texts, scores):
                        txt = str(txt_raw).strip()
                        if txt and float(conf) >= 0.50:
                            line_texts.append(txt)
                    continue

                if isinstance(item, (list, tuple)):
                    for det in item:
                        if isinstance(det, (list, tuple)) and len(det) >= 2:
                            info = det[1]
                            if isinstance(info, (list, tuple)) and info:
                                txt = str(info[0]).strip()
                                conf = float(info[1]) if len(info) > 1 else 1.0
                                if txt and conf >= 0.50:
                                    line_texts.append(txt)

            merged = " ".join(line_texts).strip()
            if merged:
                ocr_records.append({"time": ts, "text": merged})

        del ocr
        free_gpu()
    except Exception as ocr_err:
        print(f"[WARN] PaddleOCR lỗi: {ocr_err}")
        free_gpu()

    print(f"[INFO] PaddleOCR thu được {len(ocr_records)} mốc phụ đề trong vùng ROI.")
    return ocr_records, roi


def align_and_refine_segments(segments: list[dict], ocr_records: list[dict]) -> list[dict]:
    refined = []
    verified_cnt = 0
    for seg in segments:
        s_t = float(seg.get("start", 0.0))
        e_t = float(seg.get("end", s_t + 1.0))
        asr_txt = (seg.get("asr_text") or "").strip()

        matched_ocrs = [
            r["text"].strip()
            for r in ocr_records
            if (s_t - 1.0) <= float(r["time"]) <= (e_t + 1.0) and r.get("text")
        ]
        dedup_ocrs = []
        for t in matched_ocrs:
            if not dedup_ocrs or t != dedup_ocrs[-1]:
                dedup_ocrs.append(t)
        ocr_txt = " ".join(dedup_ocrs).strip()

        best_txt = asr_txt
        source_mode = "asr_only"
        if asr_txt and ocr_txt:
            ratio = SequenceMatcher(None, asr_txt.lower(), ocr_txt.lower()).ratio()
            if ratio >= 0.50 or (len(ocr_txt) >= len(asr_txt) * 0.75 and len(ocr_txt) >= 3):
                best_txt = ocr_txt
                source_mode = "ocr_verified"
                verified_cnt += 1
            else:
                best_txt = asr_txt
                source_mode = "asr_with_ocr_hint"
        elif ocr_txt and not asr_txt:
            best_txt = ocr_txt
            source_mode = "ocr_only"
            verified_cnt += 1

        item = dict(seg)
        item["ocr_text"] = ocr_txt
        item["best_text"] = best_txt
        item["subtitle_source"] = source_mode
        refined.append(item)

    print(f"[INFO] Đối chiếu ASR+OCR: {verified_cnt}/{len(refined)} câu đối chiếu OCR.")
    return refined


def run_demucs_and_ref_voices(audio_16k: str, segments: list[dict], work_dir: str) -> tuple[str, dict]:
    import numpy as np
    import soundfile as sf
    import torch

    no_vocals_out = os.path.join(work_dir, "no_vocals.wav")
    demucs_out_dir = os.path.join(work_dir, "demucs_out")
    os.makedirs(demucs_out_dir, exist_ok=True)

    cmd = ["demucs", "-n", "htdemucs", "--two-stems", "vocals", "-o", demucs_out_dir, audio_16k]
    if not torch.cuda.is_available():
        cmd.extend(["-d", "cpu"])

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        for root, _, files in os.walk(demucs_out_dir):
            if "no_vocals.wav" in files:
                src_nv = os.path.join(root, "no_vocals.wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", src_nv, "-ac", "2", "-ar", "44100", no_vocals_out],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                break

    if not os.path.exists(no_vocals_out):
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_16k, "-ac", "2", "-ar", "44100", no_vocals_out],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    speaker_refs = {}
    try:
        data, sr = sf.read(audio_16k)
        if data.ndim > 1:
            data = data.mean(axis=-1)

        by_speaker = {}
        for seg in segments:
            spk = seg.get("speaker_id", "SPEAKER_00")
            by_speaker.setdefault(spk, []).append(seg)

        for spk, spk_segs in by_speaker.items():
            candidates = [s for s in spk_segs if (float(s["end"]) - float(s["start"])) >= 2.5] or spk_segs
            best_seg = max(
                candidates,
                key=lambda x: float(np.sqrt(np.mean(data[int(float(x["start"]) * sr): max(int(float(x["start"]) * sr) + 1, int(float(x["end"]) * sr))] ** 2)))
                if len(data) > 0 else 0.0,
                default=None,
            )
            if not best_seg:
                continue
            start_s = max(0.0, float(best_seg["start"]) + 0.15)
            end_s = min(len(data) / float(sr), max(start_s + 2.5, min(float(best_seg["end"]) - 0.15, start_s + 6.5)))
            if end_s > start_s:
                clip = data[int(start_s * sr): int(end_s * sr)].astype(np.float32)
                rms = float(np.sqrt(np.mean(clip ** 2)))
                if rms > 1e-5:
                    clip = np.clip(clip * (0.08 / rms), -0.95, 0.95)
                ref_name = f"ref_speaker_{spk}.wav"
                ref_path = os.path.join(work_dir, ref_name)
                sf.write(ref_path, clip, sr)
                f0, gender_hint = estimate_pitch_gender(clip, sr)
                speaker_refs[spk] = {
                    "ref_asset": ref_name,
                    "gender_hint": gender_hint,
                    "pitch_hz": f0,
                }
    except Exception:
        pass

    free_gpu()
    return no_vocals_out, speaker_refs


sync = GitHubSync(GH_REPO, GH_TOKEN)


def main():
    jobs_data, _ = sync.read_json("state/jobs.json")
    job = jobs_data.get("jobs", {}).get(JOB_ID)
    if not job:
        raise KeyError(f"Không tìm thấy job_id '{JOB_ID}' trong state/jobs.json")

    video_asset = (job.get("video_asset") or job.get("video_url") or "source_input.mp4").strip()
    send_telegram(TG_TOKEN, TG_CHAT, f"🎬 <b>[Stage 1 - Perception]</b> Bắt đầu job <code>{JOB_ID}</code>")

    work_dir = "/kaggle/working/work"
    os.makedirs(work_dir, exist_ok=True)
    video_path = os.path.join(work_dir, "input.mp4")
    audio_16k = os.path.join(work_dir, "audio_16k.wav")

    download_video_source(sync, JOB_ID, video_asset, video_path)
    if not os.path.exists(video_path) or os.path.getsize(video_path) < 1024:
        raise RuntimeError(f"File video tải về không hợp lệ hoặc rỗng từ nguồn: {video_asset}")

    ff_res = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", audio_16k],
        capture_output=True,
        text=True,
    )
    if ff_res.returncode != 0:
        raise RuntimeError(f"FFmpeg lỗi (code {ff_res.returncode}):\n{ff_res.stderr[-1500:]}")

    ensure_deps()

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_asr = pool.submit(run_whisper_asr, audio_16k)
        fut_ocr = pool.submit(run_paddle_ocr, video_path, work_dir)
        raw_segments = fut_asr.result()
        screen_texts, roi = fut_ocr.result()

    segments = align_and_refine_segments(raw_segments, screen_texts)
    no_vocals_path, speaker_refs = run_demucs_and_ref_voices(audio_16k, segments, work_dir)

    if os.path.exists(no_vocals_path):
        sync.upload_asset(JOB_ID, no_vocals_path)

    for fname in sorted(os.listdir(work_dir)):
        if fname.startswith("ref_speaker_") and fname.endswith(".wav"):
            sync.upload_asset(JOB_ID, os.path.join(work_dir, fname))

    ocr_summary = "; ".join(f"[{r['time']}s] {r['text']}" for r in screen_texts[:15])
    scene_context = f"Chữ/phụ đề màn hình (PaddleOCR): {ocr_summary}" if ocr_summary else ""

    stage1_payload = {
        "job_id": JOB_ID,
        "video_asset": video_asset,
        "subtitle_roi": {"x1": roi[0], "y1": roi[1], "x2": roi[2], "y2": roi[3]},
        "screen_texts": screen_texts,
        "scene_context": scene_context,
        "speaker_refs": speaker_refs,
        "segments": segments,
    }
    sync.write_json(f"state/{JOB_ID}_stage1.json", stage1_payload, f"stage1 done: {JOB_ID}")

    updated_jobs, _ = sync.read_json("state/jobs.json")
    updated_jobs.setdefault("jobs", {})[JOB_ID]["status"] = "STAGE1_DONE"
    sync.write_json("state/jobs.json", updated_jobs, f"update status STAGE1_DONE: {JOB_ID}")

    sync.trigger_dispatch("stage1_completed", {"job_id": JOB_ID})
    send_telegram(
        TG_TOKEN,
        TG_CHAT,
        f"✅ <b>[Stage 1]</b> Hoàn tất <code>{JOB_ID}</code> ({len(segments)} câu, {len(screen_texts)} OCR) → chuyển sang Stage 2.",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_telegram(TG_TOKEN, TG_CHAT, f"❌ <b>[Stage 1 Lỗi]</b> <code>{JOB_ID}</code>:\n<pre>{err}</pre>")
        raise
