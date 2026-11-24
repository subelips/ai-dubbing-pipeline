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

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


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
            "paddlepaddle", "paddleocr", "speechbrain", "scikit-learn",
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

    return segments


def apply_speaker_diarization(audio_16k: str, segments: list[dict], work_dir: str) -> list[dict]:
    """Ưu tiên pyannote.audio (nếu có HF_TOKEN), tự động fallback sang SpeechBrain ECAPA-TDNN."""
    if not segments:
        return segments

    import numpy as np
    import soundfile as sf
    import torch

    hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    pyannote_model = os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1").strip()

    # 1. Ưu tiên pyannote.audio nếu có HF_TOKEN
    if hf_token:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyannote.audio"], check=True)
            from pyannote.audio import Pipeline

            try:
                pipeline = Pipeline.from_pretrained(pyannote_model, token=hf_token)
            except TypeError:
                pipeline = Pipeline.from_pretrained(pyannote_model, use_auth_token=hf_token)

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda:0"))

            diar_kwargs = {}
            fixed_k_raw = os.environ.get("NUM_SPEAKERS", "").strip()
            max_spk_raw = os.environ.get("MAX_SPEAKERS", "").strip()
            if fixed_k_raw.isdigit() and int(fixed_k_raw) >= 1:
                diar_kwargs["num_speakers"] = int(fixed_k_raw)
            elif max_spk_raw.isdigit() and int(max_spk_raw) >= 2:
                diar_kwargs["max_speakers"] = int(max_spk_raw)

            diarization = pipeline(audio_16k, **diar_kwargs)

            turns = []
            for turn, _, spk in diarization.itertracks(yield_label=True):
                turns.append((float(turn.start), float(turn.end), str(spk)))

            del pipeline
            free_gpu()

            if turns:
                order_map = {}
                for _, _, raw_spk in sorted(turns, key=lambda x: x[0]):
                    if raw_spk not in order_map:
                        order_map[raw_spk] = f"SPEAKER_{len(order_map):02d}"

                for seg in segments:
                    s_t = float(seg.get("start", 0.0))
                    e_t = float(seg.get("end", s_t + 0.5))
                    best_spk = "SPEAKER_00"
                    best_overlap = 0.0
                    for t_s, t_e, raw_spk in turns:
                        ov = max(0.0, min(e_t, t_e) - max(s_t, t_s))
                        if ov > best_overlap:
                            best_overlap = ov
                            best_spk = order_map.get(raw_spk, "SPEAKER_00")
                    seg["speaker_id"] = best_spk

                detected_list = list(dict.fromkeys(s["speaker_id"] for s in segments))
                print(f"[INFO] pyannote.audio diarization ({pyannote_model}) phát hiện {len(detected_list)} người nói: {detected_list}")
                return segments
        except Exception as py_err:
            print(f"[WARN] pyannote.audio chưa chạy được ({py_err}), chuyển sang SpeechBrain.")
            free_gpu()

    # 2. Fallback sang SpeechBrain ECAPA-TDNN + phân cụm tâm cụm
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "speechbrain", "scikit-learn"], check=True)
        from sklearn.cluster import AgglomerativeClustering
        from speechbrain.inference.speaker import EncoderClassifier

        audio_np, sr = sf.read(audio_16k, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=-1)
        if sr != 16000:
            rs_wav = os.path.join(work_dir, "diar_16k.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_16k, "-ac", "1", "-ar", "16000", rs_wav],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            audio_np, sr = sf.read(rs_wav, dtype="float32")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        savedir = os.path.join(work_dir, "ecapa_model")
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": device},
        )

        anchor_embs = []
        anchor_indices = []
        all_embs = {}

        for idx, seg in enumerate(segments):
            s_idx = max(0, int(float(seg.get("start", 0.0)) * sr))
            e_idx = min(len(audio_np), int(float(seg.get("end", 0.0)) * sr))
            clip = audio_np[s_idx:e_idx]
            if len(clip) < int(0.45 * sr):
                continue
            wav_tensor = torch.from_numpy(clip).unsqueeze(0).to(device)
            with torch.inference_mode():
                emb = classifier.encode_batch(wav_tensor).squeeze().detach().cpu().numpy()
            norm = float(np.linalg.norm(emb))
            if norm > 1e-8:
                emb = emb / norm
            all_embs[idx] = emb
            if len(clip) >= int(1.0 * sr):
                anchor_embs.append(emb)
                anchor_indices.append(idx)

        del classifier
        free_gpu()

        if not anchor_embs and all_embs:
            anchor_indices = list(all_embs.keys())
            anchor_embs = [all_embs[i] for i in anchor_indices]

        if len(anchor_embs) >= 2:
            emb_mat = np.vstack(anchor_embs)
            fixed_k_raw = os.environ.get("NUM_SPEAKERS", "").strip()
            max_spk = max(2, int(os.environ.get("MAX_SPEAKERS", "3")))
            dist_thresh = float(os.environ.get("DIAR_DIST_THRESHOLD", "0.68"))

            if fixed_k_raw.isdigit() and int(fixed_k_raw) >= 1:
                k = min(int(fixed_k_raw), len(anchor_embs))
                clustering = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
                anchor_labels = clustering.fit_predict(emb_mat)
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    metric="cosine",
                    linkage="average",
                    distance_threshold=dist_thresh,
                )
                anchor_labels = clustering.fit_predict(emb_mat)
                uniq = list(dict.fromkeys(int(x) for x in anchor_labels))
                if len(uniq) > max_spk:
                    clustering_k = AgglomerativeClustering(n_clusters=max_spk, metric="cosine", linkage="average")
                    anchor_labels = clustering_k.fit_predict(emb_mat)

            cluster_vecs = {}
            for emb, lbl in zip(anchor_embs, anchor_labels):
                cluster_vecs.setdefault(int(lbl), []).append(emb)

            centroids = {}
            for lbl, vecs in cluster_vecs.items():
                c = np.mean(np.vstack(vecs), axis=0)
                cn = float(np.linalg.norm(c))
                centroids[lbl] = c / cn if cn > 1e-8 else c

            order_map = {}
            for lbl in anchor_labels:
                ilbl = int(lbl)
                if ilbl not in order_map:
                    order_map[ilbl] = f"SPEAKER_{len(order_map):02d}"

            assigned = {}
            for idx, emb in all_embs.items():
                best_lbl = max(centroids.keys(), key=lambda c_lbl: float(np.dot(emb, centroids[c_lbl])))
                assigned[idx] = order_map[best_lbl]

            last_spk = "SPEAKER_00"
            for idx, seg in enumerate(segments):
                if idx in assigned:
                    last_spk = assigned[idx]
                seg["speaker_id"] = last_spk

            detected_list = list(dict.fromkeys(s["speaker_id"] for s in segments))
            print(f"[INFO] SpeechBrain diarization xác định {len(detected_list)} người nói: {detected_list}")
        else:
            for seg in segments:
                seg["speaker_id"] = "SPEAKER_00"

    except Exception as diar_err:
        print(f"[WARN] SpeechBrain diarization fallback sang SPEAKER_00 do: {diar_err}")
        free_gpu()
        for seg in segments:
            seg["speaker_id"] = "SPEAKER_00"

    return segments


def _init_paddle_ocr_engine():
    from paddleocr import PaddleOCR

    lang = os.environ.get("PADDLE_OCR_LANG", "ch")
    use_gpu = os.environ.get("PADDLE_OCR_GPU", "0").strip() == "1"

    try:
        return PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception:
        try:
            return PaddleOCR(lang=lang)
        except Exception:
            return PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=use_gpu)


def _ocr_single_image(ocr, img_path: str, min_conf: float = 0.25) -> str:
    line_texts = []
    try:
        if hasattr(ocr, "predict"):
            res = list(ocr.predict(img_path))
        else:
            res = ocr.ocr(img_path)
    except Exception:
        return ""

    if not res:
        return ""

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
                if txt and float(conf) >= min_conf:
                    line_texts.append(txt)
            continue

        if isinstance(item, (list, tuple)):
            for det in item:
                if isinstance(det, (list, tuple)) and len(det) >= 2:
                    info = det[1]
                    if isinstance(info, (list, tuple)) and info:
                        txt = str(info[0]).strip()
                        conf = float(info[1]) if len(info) > 1 else 1.0
                        if txt and conf >= min_conf:
                            line_texts.append(txt)

    return " ".join(line_texts).strip()


def _enhance_contrast_image(img_path: str) -> str:
    from PIL import Image, ImageEnhance

    enh_path = img_path + "_enh.jpg"
    try:
        img = Image.open(img_path).convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(1.6)
        img.save(enh_path, quality=92)
        return enh_path
    except Exception:
        return img_path


def _extract_frames_for_roi(video_path: str, out_dir: str, roi: tuple[float, float, float, float], fps: float) -> list[tuple[float, str]]:
    os.makedirs(out_dir, exist_ok=True)
    x1, y1, x2, y2 = roi
    crop_w = max(0.2, x2 - x1)
    crop_h = max(0.10, y2 - y1)
    if crop_w >= 0.98 and crop_h >= 0.98:
        vf_expr = f"fps={fps},scale=1920:-1"
    else:
        vf_expr = f"fps={fps},crop=iw*{crop_w:.3f}:ih*{crop_h:.3f}:iw*{x1:.3f}:ih*{y1:.3f},scale=1920:-1"

    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", vf_expr, os.path.join(out_dir, "f_%04d.jpg")],
        capture_output=True,
        text=True,
    )
    interval = 1.0 / max(0.1, float(fps))
    frames = []
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith(".jpg"):
            continue
        try:
            idx = int(fname.split("_")[1].split(".")[0])
        except Exception:
            idx = len(frames) + 1
        ts = round((idx - 1) * interval, 2)
        frames.append((ts, os.path.join(out_dir, fname)))
    return frames[:150]


def run_paddle_ocr(video_path: str, work_dir: str) -> tuple[list[dict], tuple[float, float, float, float]]:
    """Quét phụ đề cứng đa vùng + tăng tương phản dự phòng khi nét chữ mờ/khó nhận."""
    candidate_rois = [
        (0.02, 0.65, 0.98, 0.99),
        (0.0, 0.0, 1.0, 1.0),
    ]
    fps = float(os.environ.get("OCR_FPS", "1.0"))
    chosen_roi = candidate_rois[0]
    ocr_records = []
    use_contrast = False

    try:
        ocr = _init_paddle_ocr_engine()
        active_frames = []

        for idx_roi, roi in enumerate(candidate_rois):
            roi_dir = os.path.join(work_dir, f"ocr_frames_r{idx_roi}")
            frames = _extract_frames_for_roi(video_path, roi_dir, roi, fps)
            if not frames:
                continue

            step = max(1, len(frames) // 12)
            probe_frames = frames[::step][:12]
            hits = sum(1 for _, p in probe_frames if _ocr_single_image(ocr, p, min_conf=0.25))
            enh_hits = 0
            if hits == 0:
                enh_hits = sum(
                    1 for _, p in probe_frames
                    if _ocr_single_image(ocr, _enhance_contrast_image(p), min_conf=0.25)
                )

            print(f"[INFO] Thử ROI {roi}: bắt được {hits} frame gốc, {enh_hits} frame tăng tương phản (trên {len(probe_frames)} mẫu).")
            if hits > 0 or enh_hits > 0 or idx_roi == len(candidate_rois) - 1:
                chosen_roi = roi
                active_frames = frames
                use_contrast = (hits == 0 and enh_hits > 0)
                break

        for ts, img_path in active_frames:
            target_img = _enhance_contrast_image(img_path) if use_contrast else img_path
            merged = _ocr_single_image(ocr, target_img, min_conf=0.25)
            if merged:
                ocr_records.append({"time": ts, "text": merged})

        del ocr
        free_gpu()
    except Exception as ocr_err:
        print(f"[WARN] PaddleOCR lỗi: {ocr_err}")
        free_gpu()

    print(f"[INFO] Khoanh vùng phụ đề ROI chọn: {chosen_roi} → {len(ocr_records)} mốc phụ đề.")
    return ocr_records, chosen_roi


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
            if (s_t - 1.2) <= float(r["time"]) <= (e_t + 1.2) and r.get("text")
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
            if ratio >= 0.45 or (len(ocr_txt) >= len(asr_txt) * 0.70 and len(ocr_txt) >= 3):
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


def run_demucs_and_ref_voices(audio_44k: str, segments: list[dict], work_dir: str) -> tuple[str, dict]:
    import numpy as np
    import soundfile as sf
    import torch

    no_vocals_out = os.path.join(work_dir, "no_vocals.wav")
    clean_vocals_out = os.path.join(work_dir, "clean_vocals.wav")
    demucs_out_dir = os.path.join(work_dir, "demucs_out")
    os.makedirs(demucs_out_dir, exist_ok=True)

    model_name = os.environ.get("DEMUCS_MODEL", "htdemucs_ft").strip() or "htdemucs_ft"
    shifts = max(1, min(5, int(os.environ.get("DEMUCS_SHIFTS", "2"))))

    cmd = [
        "demucs",
        "-n", model_name,
        "--two-stems", "vocals",
        "--shifts", str(shifts),
        "-o", demucs_out_dir,
        audio_44k,
    ]
    if not torch.cuda.is_available():
        cmd.extend(["-d", "cpu"])

    res = subprocess.run(cmd, capture_output=True, text=True)
    found_nv = False
    found_voc = False
    if res.returncode == 0:
        for root, _, files in os.walk(demucs_out_dir):
            if "no_vocals.wav" in files and not found_nv:
                src_nv = os.path.join(root, "no_vocals.wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", src_nv, "-ac", "2", "-ar", "44100", no_vocals_out],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                found_nv = True
            if "vocals.wav" in files and not found_voc:
                src_voc = os.path.join(root, "vocals.wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", src_voc, "-ac", "1", "-ar", "24000", clean_vocals_out],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                found_voc = True
    else:
        print(f"[WARN] Demucs ({model_name}) gặp lỗi, fallback sang htdemucs: {res.stderr[-400:]}")
        fb_cmd = ["demucs", "-n", "htdemucs", "--two-stems", "vocals", "-o", demucs_out_dir, audio_44k]
        if not torch.cuda.is_available():
            fb_cmd.extend(["-d", "cpu"])
        subprocess.run(fb_cmd, capture_output=True, text=True)
        for root, _, files in os.walk(demucs_out_dir):
            if "no_vocals.wav" in files and not found_nv:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", os.path.join(root, "no_vocals.wav"), "-ac", "2", "-ar", "44100", no_vocals_out],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                found_nv = True
            if "vocals.wav" in files and not found_voc:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", os.path.join(root, "vocals.wav"), "-ac", "1", "-ar", "24000", clean_vocals_out],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                found_voc = True

    if not os.path.exists(no_vocals_out):
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_44k, "-ac", "2", "-ar", "44100", no_vocals_out],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    ref_source_wav = clean_vocals_out if os.path.exists(clean_vocals_out) else audio_44k
    speaker_refs = {}
    try:
        data, sr = sf.read(ref_source_wav)
        if data.ndim > 1:
            data = data.mean(axis=-1)

        by_speaker = {}
        for seg in segments:
            spk = seg.get("speaker_id", "SPEAKER_00")
            by_speaker.setdefault(spk, []).append(seg)

        for spk, spk_segs in by_speaker.items():
            candidates = [s for s in spk_segs if (float(s["end"]) - float(s["start"])) >= 2.0] or spk_segs
            best_seg = max(
                candidates,
                key=lambda x: float(np.sqrt(np.mean(data[int(float(x["start"]) * sr): max(int(float(x["start"]) * sr) + 1, int(float(x["end"]) * sr))] ** 2)))
                if len(data) > 0 else 0.0,
                default=None,
            )
            if not best_seg:
                continue
            start_s = max(0.0, float(best_seg["start"]) + 0.15)
            end_s = min(len(data) / float(sr), max(start_s + 2.0, min(float(best_seg["end"]) - 0.15, start_s + 6.5)))
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
    audio_44k = os.path.join(work_dir, "audio_44k.wav")

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

    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "2", "-ar", "44100", audio_44k],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    ensure_deps()

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_asr = pool.submit(run_whisper_asr, audio_16k)
        fut_ocr = pool.submit(run_paddle_ocr, video_path, work_dir)
        raw_segments = fut_asr.result()
        screen_texts, roi = fut_ocr.result()

    diarized_segments = apply_speaker_diarization(audio_16k, raw_segments, work_dir)
    segments = align_and_refine_segments(diarized_segments, screen_texts)
    no_vocals_path, speaker_refs = run_demucs_and_ref_voices(audio_44k, segments, work_dir)

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
        f"✅ <b>[Stage 1]</b> Hoàn tất <code>{JOB_ID}</code> ({len(segments)} câu, {len(screen_texts)} OCR, {len(speaker_refs)} người nói) → chuyển sang Stage 2.",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_telegram(TG_TOKEN, TG_CHAT, f"❌ <b>[Stage 1 Lỗi]</b> <code>{JOB_ID}</code>:\n<pre>{err}</pre>")
        raise
