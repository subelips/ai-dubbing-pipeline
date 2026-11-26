import os

# Tắt PIR / OneDNN trên PaddlePaddle 3.x để tránh lỗi ConvertPirAttribute2RuntimeAttribute khi chạy CPU
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

import base64
from concurrent.futures import ThreadPoolExecutor
import ctypes
from difflib import SequenceMatcher
import gc
import glob
import json
import re
import site
import subprocess
import sys
import time
import traceback
import requests


def _clean_url_or_str(val: str) -> str:
    s = str(val or "").strip()
    m = re.match(r"^\[([^\]]+)\]\(([^)]+)\)$", s)
    if m:
        s = m.group(2).strip()
    return s.strip("[]()\"' ").strip()


def get_cfg(key: str, injected: str) -> str:
    env_val = _clean_url_or_str(os.environ.get(key, ""))
    if env_val:
        return env_val
    clean_inj = _clean_url_or_str(injected)
    if clean_inj and not clean_inj.startswith("__INJECT_"):
        return clean_inj
    return ""


JOB_ID = get_cfg("JOB_ID", "__INJECT_JOB_ID__")
GH_REPO = get_cfg("GH_REPO", "__INJECT_GH_REPO__")
GH_TOKEN = get_cfg("GH_PAT_TOKEN", "__INJECT_GH_PAT_TOKEN__") or get_cfg("GH_TOKEN", "__INJECT_GH_TOKEN__")
NTFY_TOPIC = get_cfg("NTFY_TOPIC", "__INJECT_NTFY_TOPIC__")
NTFY_SERVER = (get_cfg("NTFY_SERVER", "__INJECT_NTFY_SERVER__") or "https://ntfy.sh").rstrip("/")
NTFY_TOKEN = get_cfg("NTFY_TOKEN", "__INJECT_NTFY_TOKEN__")

if not JOB_ID:
    raise RuntimeError("Thiếu JOB_ID khi khởi chạy kernel.")


def send_ntfy(title: str, message: str, tags: str = "movie_camera", priority: str = "default"):
    topic = _clean_url_or_str(NTFY_TOPIC)
    server = _clean_url_or_str(NTFY_SERVER).rstrip("/") or "https://ntfy.sh"
    if not topic:
        return
    clean_msg = re.sub(r"<[^>]+>", "", str(message or "")).strip()[:3800]
    clean_title = re.sub(r"<[^>]+>", "", str(title or "Dubbing Pipeline")).strip()[:200]
    headers = {
        "Title": clean_title.encode("utf-8").decode("latin-1", errors="ignore") or "Dubbing Pipeline",
        "Tags": tags,
        "Priority": priority,
    }
    if NTFY_TOKEN:
        headers["Authorization"] = f"Bearer {_clean_url_or_str(NTFY_TOKEN)}"
    try:
        r = requests.post(f"{server}/{topic}", data=clean_msg.encode("utf-8"), headers=headers, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"[WARN] Gửi ntfy thất bại: {e}")


class GitHubSync:
    def __init__(self, repo: str, token: str, state_branch: str = "pipeline-state"):
        clean_repo = _clean_url_or_str(repo).replace("https://api.github.com/repos/", "").strip("/")
        self.repo = clean_repo
        self.token = _clean_url_or_str(token)
        self.state_branch = _clean_url_or_str(state_branch) or "pipeline-state"
        self.api_base = f"https://api.github.com/repos/{self.repo}"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def read_json(self, path: str) -> tuple[dict, str | None]:
        url = f"{self.api_base}/contents/{path.lstrip('/')}"
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
        url = f"{self.api_base}/contents/{path.lstrip('/')}"
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
        upload_url = _clean_url_or_str(release["upload_url"].split("{")[0])
        filename = os.path.basename(file_path)
        for asset in release.get("assets", []):
            if asset["name"] == filename:
                requests.delete(f"{self.api_base}/releases/assets/{asset['id']}", headers=self.headers, timeout=30)
        up_headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/octet-stream"}
        with open(file_path, "rb") as f:
            r = requests.post(f"{upload_url}?name={filename}", headers=up_headers, data=f.read(), timeout=300)
        r.raise_for_status()
        return _clean_url_or_str(r.json().get("browser_download_url", ""))

    def download_asset(self, tag: str, filename: str, dest_path: str):
        release = self.ensure_release(tag)
        for asset in release.get("assets", []):
            if asset["name"] == filename:
                dl_url = _clean_url_or_str(asset["url"])
                dl_headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/octet-stream"}
                with requests.get(dl_url, headers=dl_headers, stream=True, timeout=300) as r:
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


def _is_installed(pkg_name: str) -> bool:
    try:
        __import__(pkg_name)
        return True
    except Exception:
        return False


def ensure_deps():
    wanted = [
        ("faster_whisper", "faster-whisper"),
        ("demucs", "demucs"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("gdown", "gdown"),
        ("paddle", "paddlepaddle"),
        ("paddleocr", "paddleocr"),
    ]
    missing = [pip_name for mod_name, pip_name in wanted if not _is_installed(mod_name)]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)


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
    clean = _clean_url_or_str(url)
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", clean) or re.search(r"[?&]id=([a-zA-Z0-9_-]+)", clean)
    if m:
        return f"https://drive.google.com/uc?id={m.group(1)}"
    return clean


def download_video_source(sync: GitHubSync, tag: str, source_ref: str, dest_path: str):
    ref = _clean_url_or_str(source_ref) or "source_input.mp4"
    if ref.startswith(("http://", "https://")):
        if "drive.google.com" in ref:
            if not _is_installed("gdown"):
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)
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
        raw_segments, _ = model.transcribe(audio_np, beam_size=3, vad_filter=True)

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


def _split_segments_on_speaker_turns(
    raw_segs: list[dict],
    turns: list[tuple[float, float, str]],
    order_map: dict,
) -> list[dict]:
    out_segs = []
    next_id = 0

    for seg in raw_segs:
        s_t = float(seg.get("start", 0.0))
        e_t = max(s_t + 0.3, float(seg.get("end", s_t + 0.5)))
        dur = e_t - s_t
        pad = min(0.15, dur * 0.2)
        c_start = s_t + pad
        c_end = max(c_start + 0.1, e_t - pad)

        spk_durations: dict[str, float] = {}
        overlapping_turns: list[tuple[float, float, str]] = []
        for t_s, t_e, raw_spk in turns:
            ov = max(0.0, min(c_end, t_e) - max(c_start, t_s))
            if ov > 0.05:
                spk_name = order_map.get(raw_spk, "SPEAKER_00")
                spk_durations[spk_name] = spk_durations.get(spk_name, 0.0) + ov
                overlapping_turns.append((max(s_t, t_s), min(e_t, t_e), spk_name))

        significant_spks = [k for k, v in spk_durations.items() if v >= 0.45]
        txt = (seg.get("asr_text") or "").strip()
        words = txt.split()

        if len(significant_spks) >= 2 and len(words) >= 4 and len(overlapping_turns) >= 2:
            overlapping_turns.sort(key=lambda x: x[0])
            total_ov = sum(max(0.05, b - a) for a, b, _ in overlapping_turns)
            word_cursor = 0
            for idx_t, (sub_s, sub_e, spk_name) in enumerate(overlapping_turns):
                sub_dur = max(0.1, sub_e - sub_s)
                if idx_t == len(overlapping_turns) - 1:
                    sub_words = words[word_cursor:]
                else:
                    take_w = max(1, int(round(len(words) * (sub_dur / total_ov))))
                    sub_words = words[word_cursor: word_cursor + take_w]
                    word_cursor += take_w
                if not sub_words:
                    continue
                item = dict(seg)
                item["id"] = next_id
                item["start"] = round(sub_s, 2)
                item["end"] = round(max(sub_s + 0.4, sub_e), 2)
                item["speaker_id"] = spk_name
                item["asr_text"] = " ".join(sub_words)
                out_segs.append(item)
                next_id += 1
        else:
            best_spk = max(spk_durations.items(), key=lambda x: x[1])[0] if spk_durations else "SPEAKER_00"
            item = dict(seg)
            item["id"] = next_id
            item["speaker_id"] = best_spk
            out_segs.append(item)
            next_id += 1

    return out_segs


def _get_hf_token() -> str:
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        token = (UserSecretsClient().get_secret("HF_TOKEN") or "").strip()
    except Exception:
        token = ""
    return token


def apply_speaker_diarization(audio_16k: str, segments: list[dict], work_dir: str) -> list[dict]:
    if not segments:
        return segments

    import torch

    hf_token = _get_hf_token()
    pyannote_model = os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1").strip()
    use_pyannote = os.environ.get("USE_PYANNOTE", "1").strip() != "0"

    if hf_token and use_pyannote:
        try:
            if not _is_installed("pyannote"):
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
            ann = getattr(diarization, "speaker_diarization", diarization)
            turns = [(float(turn.start), float(turn.end), str(spk)) for turn, _, spk in ann.itertracks(yield_label=True)]
            del pipeline
            free_gpu()

            if turns:
                order_map = {}
                for _, _, raw_spk in sorted(turns, key=lambda x: x[0]):
                    if raw_spk not in order_map:
                        order_map[raw_spk] = f"SPEAKER_{len(order_map):02d}"

                refined_segs = _split_segments_on_speaker_turns(segments, turns, order_map)
                detected_list = list(dict.fromkeys(s["speaker_id"] for s in refined_segs))
                print(f"[INFO] pyannote.audio diarization ({pyannote_model}) phát hiện {len(detected_list)} người nói: {detected_list}")
                return refined_segs
        except Exception as py_err:
            print(f"[WARN] pyannote.audio không chạy được ({py_err}), dùng dự phòng cao độ/giới tính.")
            free_gpu()
    else:
        print("[INFO] Thiếu HF_TOKEN hoặc USE_PYANNOTE=0, dùng dự phòng cao độ/giới tính.")

    gender_to_spk = {"male": "SPEAKER_00", "female": "SPEAKER_01"}
    for seg in segments:
        g = str(seg.get("gender_hint", "unknown")).lower()
        seg["speaker_id"] = gender_to_spk.get(g, "SPEAKER_00")

    detected_list = list(dict.fromkeys(s["speaker_id"] for s in segments))
    print(f"[INFO] Dự phòng gán người nói tạo {len(detected_list)} nhãn: {detected_list}")
    return segments


def extract_embedded_subtitles(video_path: str, work_dir: str) -> list[dict]:
    srt_path = os.path.join(work_dir, "embedded_subs.srt")
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-map", "0:s:0", srt_path],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 or not os.path.exists(srt_path) or os.path.getsize(srt_path) < 10:
        return []

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_srt = f.read()

    pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*\n(.*?)(?=\n\s*\n|\Z)",
        re.DOTALL,
    )
    records = []
    for m in pattern.finditer(raw_srt):
        h1, m1, s1, ms1, h2, m2, s2, ms2, body = m.groups()
        t_start = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000.0
        t_end = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000.0
        clean_txt = re.sub(r"<[^>]+>", "", body).replace("\n", " ").strip()
        if clean_txt:
            mid_t = round((t_start + t_end) * 0.5, 2)
            records.append({"time": mid_t, "start": round(t_start, 2), "end": round(t_end, 2), "text": clean_txt})

    if records:
        print(f"[INFO] Phát hiện và trích xuất thành công {len(records)} dòng phụ đề mềm (softsub) trong video.")
    return records


def _init_paddle_ocr_engine():
    from paddleocr import PaddleOCR

    lang = os.environ.get("PADDLE_OCR_LANG", "ch")
    try:
        return PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception:
        return PaddleOCR(lang=lang)


def _preprocess_hardsub_image(img_path: str, mode: str = "contrast") -> str:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    out_path = f"{img_path}_{mode}.jpg"
    try:
        img = Image.open(img_path).convert("RGB")
        if mode == "contrast":
            img = ImageOps.autocontrast(img, cutoff=1)
            img = ImageEnhance.Contrast(img).enhance(1.65)
            img = img.filter(ImageFilter.SHARPEN)
        elif mode == "high_text":
            gray = ImageOps.grayscale(img)
            gray = ImageOps.autocontrast(gray, cutoff=2)
            gray = gray.filter(ImageFilter.SHARPEN)
            img = gray.convert("RGB")
        img.save(out_path, quality=92)
        return out_path
    except Exception:
        return img_path


def _ocr_single_image(ocr_engine, img_path: str, min_conf: float = 0.25) -> str:
    line_texts = []
    if hasattr(ocr_engine, "predict"):
        res = list(ocr_engine.predict(img_path))
    else:
        res = ocr_engine.ocr(img_path)

    if not res:
        return ""

    for item in res:
        payload = item
        if hasattr(item, "res") and isinstance(item.res, dict):
            payload = item.res
        elif hasattr(item, "json") and isinstance(item.json, dict):
            payload = item.json.get("res", item.json)

        rec_texts = None
        rec_scores = None
        if isinstance(payload, dict):
            rec_texts = payload.get("rec_texts")
            rec_scores = payload.get("rec_scores")
        elif hasattr(payload, "get"):
            rec_texts = payload.get("rec_texts")
            rec_scores = payload.get("rec_scores")
        elif hasattr(payload, "rec_texts"):
            rec_texts = getattr(payload, "rec_texts", None)
            rec_scores = getattr(payload, "rec_scores", None)

        if rec_texts is not None:
            scores = list(rec_scores) if rec_scores is not None else [1.0] * len(rec_texts)
            for txt_raw, conf in zip(rec_texts, scores):
                txt = str(txt_raw).strip()
                if txt and float(conf) >= min_conf:
                    line_texts.append(txt)
            continue

        if isinstance(payload, (list, tuple)):
            for det in payload:
                if isinstance(det, (list, tuple)) and len(det) >= 2:
                    info = det[1]
                    if isinstance(info, (list, tuple)) and info:
                        txt = str(info[0]).strip()
                        conf = float(info[1]) if len(info) > 1 else 1.0
                        if txt and conf >= min_conf:
                            line_texts.append(txt)

    return " ".join(line_texts).strip()


def _extract_frames_for_roi(video_path: str, out_dir: str, roi: tuple[float, float, float, float], fps: float) -> list[tuple[float, str]]:
    os.makedirs(out_dir, exist_ok=True)
    x1, y1, x2, y2 = roi
    crop_w = max(0.2, x2 - x1)
    crop_h = max(0.10, y2 - y1)
    if crop_w >= 0.98 and crop_h >= 0.98:
        vf_expr = f"fps={fps},scale=1280:-1"
    else:
        vf_expr = f"fps={fps},crop=iw*{crop_w:.3f}:ih*{crop_h:.3f}:iw*{x1:.3f}:ih*{y1:.3f},scale=1280:-1"

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
    return frames[:45]


def run_paddle_ocr(video_path: str, work_dir: str) -> tuple[list[dict], tuple[float, float, float, float]]:
    embedded = extract_embedded_subtitles(video_path, work_dir)
    if embedded:
        return embedded, (0.02, 0.70, 0.98, 0.98)

    if os.environ.get("ENABLE_OCR", "1").strip() == "0":
        print("[INFO] Đã tắt OCR theo ENABLE_OCR=0.")
        return [], (0.02, 0.70, 0.98, 0.98)

    candidate_rois = [
        (0.02, 0.58, 0.98, 0.96),
        (0.0, 0.0, 1.0, 1.0),
    ]
    fps = float(os.environ.get("OCR_FPS", "0.5"))
    chosen_roi = candidate_rois[0]
    chosen_mode = "raw"
    ocr_records = []

    try:
        ocr_engine = _init_paddle_ocr_engine()
        active_frames = []

        for idx_roi, roi in enumerate(candidate_rois):
            roi_dir = os.path.join(work_dir, f"ocr_frames_r{idx_roi}")
            frames = _extract_frames_for_roi(video_path, roi_dir, roi, fps)
            if not frames:
                continue

            step = max(1, len(frames) // 6)
            probe_frames = frames[::step][:6]

            matched_mode = None
            best_hits = 0
            for m_name in ("raw", "contrast"):
                hits = 0
                for _, p in probe_frames:
                    t_img = p if m_name == "raw" else _preprocess_hardsub_image(p, m_name)
                    try:
                        txt_res = _ocr_single_image(ocr_engine, t_img, min_conf=0.25)
                    except Exception:
                        txt_res = ""
                    if txt_res:
                        hits += 1
                if hits > best_hits:
                    best_hits = hits
                    matched_mode = m_name
                if hits >= 2:
                    break

            print(f"[INFO] [PaddleOCR] ROI {roi} | mode={matched_mode or 'raw'} | hits={best_hits}/{len(probe_frames)} mẫu.")
            if best_hits > 0 or idx_roi == len(candidate_rois) - 1:
                chosen_roi = roi
                chosen_mode = matched_mode or "raw"
                active_frames = frames
                break

        for ts, img_path in active_frames:
            t_img = img_path if chosen_mode == "raw" else _preprocess_hardsub_image(img_path, chosen_mode)
            try:
                merged = _ocr_single_image(ocr_engine, t_img, min_conf=0.25)
            except Exception:
                merged = ""
            if merged:
                ocr_records.append({"time": ts, "text": merged})

        del ocr_engine
        free_gpu()
    except Exception as ocr_err:
        print(f"[WARN] PaddleOCR lỗi: {ocr_err}")
        free_gpu()

    print(f"[INFO] Khoanh vùng phụ đề ROI chọn: {chosen_roi} (mode={chosen_mode}) → {len(ocr_records)} mốc phụ đề.")
    return ocr_records, chosen_roi


def _clean_ocr_typos(text: str) -> str:
    replacements = {
        "绐": "给",
        "从莱市场": "从菜市场",
        "莱市场": "菜市场",
        "别入": "别人",
        "谈谈": "堂堂",
        "引信埋名": "隐姓埋名",
        "匕下": "上下",
        "落荏": "落在",
        "陆氐": "陆氏",
    }
    cleaned = text
    for bad, good in replacements.items():
        cleaned = cleaned.replace(bad, good)
    return cleaned.strip()


def align_and_refine_segments(segments: list[dict], ocr_records: list[dict]) -> list[dict]:
    refined = []
    verified_cnt = 0

    sorted_segs = sorted(segments, key=lambda x: float(x.get("start", 0.0)))
    prev_end = 0.0

    for idx_s, seg in enumerate(sorted_segs):
        raw_s = max(0.0, float(seg.get("start", 0.0)))
        raw_e = max(raw_s + 0.5, float(seg.get("end", raw_s + 1.0)))

        s_t = round(max(raw_s, prev_end + 0.04 if idx_s > 0 else raw_s), 2)
        e_t = round(max(raw_e, s_t + 0.45), 2)
        prev_end = e_t

        asr_txt = (seg.get("asr_text") or "").strip()

        matched_ocrs = [
            _clean_ocr_typos(r["text"].strip())
            for r in ocr_records
            if (s_t - 0.8) <= float(r["time"]) <= (e_t + 0.8) and r.get("text")
        ]
        dedup_ocrs = []
        for t in matched_ocrs:
            if t and (not dedup_ocrs or t != dedup_ocrs[-1]):
                dedup_ocrs.append(t)
        ocr_txt = " ".join(dedup_ocrs).strip()

        best_txt = asr_txt
        source_mode = "asr_only"
        if asr_txt and ocr_txt:
            ratio = SequenceMatcher(None, asr_txt.lower(), ocr_txt.lower()).ratio()
            if ratio >= 0.55 and len(ocr_txt) >= int(len(asr_txt) * 0.80):
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
        item["id"] = idx_s
        item["start"] = s_t
        item["end"] = e_t
        item["asr_text"] = asr_txt
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
    shifts = max(1, min(3, int(os.environ.get("DEMUCS_SHIFTS", "1"))))

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
            candidates = [s for s in spk_segs if (float(s["end"]) - float(s["start"])) >= 1.5] or spk_segs
            best_seg = max(
                candidates,
                key=lambda x: float(np.sqrt(np.mean(data[int(float(x["start"]) * sr): max(int(float(x["start"]) * sr) + 1, int(float(x["end"]) * sr))] ** 2)))
                if len(data) > 0 else 0.0,
                default=None,
            )
            if not best_seg:
                continue
            start_s = max(0.0, float(best_seg["start"]) + 0.1)
            end_s = min(len(data) / float(sr), max(start_s + 1.5, min(float(best_seg["end"]) - 0.1, start_s + 6.0)))
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
    send_ntfy(f"[Stage 1] {JOB_ID}", f"Bắt đầu Perception cho job {JOB_ID}.", tags="movie_camera")

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
    scene_context = f"Chữ/phụ đề màn hình: {ocr_summary}" if ocr_summary else ""

    stage1_payload = {
        "job_id": JOB_ID,
        "video_asset": video_asset,
        "subtitle_roi": {"x1": roi[0], "y1": roi[1], "x2": roi[2], "y2": roi[3]},
        "screen_texts": screen_texts,
        "scene_context": scene_context,
        "speaker_refs": speaker_refs,
        "segments": segments,
    }
    with open(os.path.join(work_dir, f"{JOB_ID}_stage1.json"), "w", encoding="utf-8") as f:
        json.dump(stage1_payload, f, ensure_ascii=False, indent=2)

    sync.write_json(f"state/{JOB_ID}_stage1.json", stage1_payload, f"stage1 done: {JOB_ID}")

    updated_jobs, _ = sync.read_json("state/jobs.json")
    updated_jobs.setdefault("jobs", {})[JOB_ID]["status"] = "STAGE1_DONE"
    sync.write_json("state/jobs.json", updated_jobs, f"update status STAGE1_DONE: {JOB_ID}")

    sync.trigger_dispatch("stage1_completed", {"job_id": JOB_ID})
    send_ntfy(
        f"[Stage 1 Done] {JOB_ID}",
        f"Hoàn tất {len(segments)} câu, {len(screen_texts)} phụ đề, {len(speaker_refs)} người nói -> sang Stage 2.",
        tags="white_check_mark",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_ntfy(f"[Stage 1 Lỗi] {JOB_ID}", err, tags="x", priority="high")
        raise
