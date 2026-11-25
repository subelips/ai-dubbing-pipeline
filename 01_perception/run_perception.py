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

for _stream in (sys.stdout, sys.stderr):
    _reconf = getattr(_stream, "reconfigure", None)  # log realtime, không bị buffer
    if callable(_reconf):
        try:
            _reconf(line_buffering=True)
        except Exception:
            pass


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
HF_TOKEN = get_cfg("HF_TOKEN", "__INJECT_HF_TOKEN__")
if HF_TOKEN:
    # pyannote đọc token từ biến môi trường này
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", HF_TOKEN)

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


def _have_module(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _want_gpu_ocr() -> bool:
    # Mặc định TẮT: cài paddlepaddle-gpu kéo nvidia-nccl-cu12 bản khác làm torch chết
    # ("undefined symbol: ncclCommWindowDeregister") và tốn ~7 phút. Chỉ bật khi
    # thật sự muốn đánh đổi (OCR_USE_GPU=1).
    return os.environ.get("OCR_USE_GPU", "0").strip() == "1"


def _paddle_has_cuda() -> bool:
    try:
        import paddle
        return bool(paddle.is_compiled_with_cuda()) and int(paddle.device.cuda.device_count()) > 0
    except Exception:
        return False


def _ensure_paddle_gpu() -> bool:
    """Bảo đảm paddle bản GPU. Trả True nếu CUDA dùng được, False để chạy CPU."""
    if not _want_gpu_ocr():
        print("[INFO] OCR_USE_GPU=0 → dùng paddle CPU.")
        return False
    if _paddle_has_cuda():
        print("[INFO] Paddle đã có CUDA, OCR sẽ chạy trên GPU.")
        return True

    print("[INFO] Paddle đang là bản CPU → cài paddlepaddle-gpu cho OCR...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "paddlepaddle"], check=False)
    for index_url in (
        "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
        "https://www.paddlepaddle.org.cn/packages/stable/cu118/",
    ):
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "paddlepaddle-gpu==3.2.0", "-i", index_url],
            check=False,
        )
        if r.returncode == 0 and _paddle_has_cuda():
            print(f"[INFO] Cài paddlepaddle-gpu thành công từ {index_url}")
            return True
        print(f"[WARN] paddlepaddle-gpu từ {index_url} không dùng được (rc={r.returncode}).")

    print("[WARN] Không cài được paddle GPU → quay lại paddle CPU.")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "paddlepaddle",
         "-i", "https://www.paddlepaddle.org.cn/packages/stable/cpu/"],
        check=False,
    )
    return False


def ensure_deps():
    """Chỉ cài gói còn thiếu; không --upgrade để tránh phá numpy/pillow của image Kaggle.

    Bản cũ pip install --upgrade 15 gói mỗi lần chạy: mất ~6 phút và kéo numpy/pillow
    lên bản mới làm numba/paddle/moviepy lệch ABI.
    """
    wanted = [
        ("faster_whisper", "faster-whisper"),
        ("demucs", "demucs"),
        ("soundfile", "soundfile"),
        ("gdown", "gdown"),
        ("paddleocr", "paddleocr"),
        ("paddle", "paddlepaddle"),
        ("speechbrain", "speechbrain"),
        ("sklearn", "scikit-learn"),
    ]
    if os.environ.get("OCR_ENGINE", "easyocr").strip().lower() in ("easyocr",):
        # EasyOCR (torch, có sẵn GPU) là engine chính → khỏi cài paddle cho nặng
        wanted = [(m, s) for m, s in wanted if m not in ("paddleocr", "paddle")]
    missing = [spec for mod, spec in wanted if not _have_module(mod)]
    if _want_gpu_ocr():
        # _ensure_paddle_gpu() sẽ cài đúng bản paddle (gpu/cpu), khỏi cài CPU trước rồi gỡ
        missing = [m for m in missing if m != "paddlepaddle"]
    if not _have_module("nvidia.cublas"):
        missing.extend(["nvidia-cublas-cu12", "nvidia-cudnn-cu12"])
    if missing:
        print(f"[INFO] pip install các gói còn thiếu: {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    else:
        print("[INFO] Đã đủ dependency, bỏ qua pip install.")
    if os.environ.get("OCR_ENGINE", "easyocr").strip().lower() in ("paddle", "auto"):
        _ensure_paddle_gpu()


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


def _state_asset_ref(ref: str) -> bool:
    """True nếu ref trỏ vào file nằm trên nhánh pipeline-state (do web UI upload)."""
    return ref.startswith("uploads_tmp/") or (ref.startswith("state/") and ref.endswith("manifest.json"))


def _state_raw_bytes(sync: "GitHubSync", path: str) -> bytes:
    headers = dict(sync.headers)
    headers["Accept"] = "application/vnd.github.raw"
    r = requests.get(
        f"{sync.api_base}/contents/{path}",
        headers=headers,
        params={"ref": sync.state_branch},
        timeout=180,
    )
    r.raise_for_status()
    return r.content


def download_from_state(sync: "GitHubSync", ref: str, dest_path: str):
    """Ghép các mảnh upload từ web UI (manifest.json) thành 1 file video hoàn chỉnh."""
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    if ref.endswith("manifest.json"):
        manifest = json.loads(_state_raw_bytes(sync, ref).decode("utf-8"))
        parts = manifest.get("parts") or []
        if not parts:
            raise RuntimeError(f"Manifest {ref} không có danh sách parts")
        with open(dest_path, "wb") as out:
            for part in parts:
                out.write(_state_raw_bytes(sync, part))
    else:
        with open(dest_path, "wb") as out:
            out.write(_state_raw_bytes(sync, ref))
    if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 1024:
        raise RuntimeError(f"Tải video từ nhánh {sync.state_branch} thất bại: {ref}")


def download_video_source(sync: GitHubSync, tag: str, source_ref: str, dest_path: str):
    ref = (source_ref or "source_input.mp4").strip()
    if ref.startswith(("http://", "https://")):
        if "drive.google.com" in ref:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
            direct_url = _normalize_gdrive_url(ref)
            subprocess.run([sys.executable, "-m", "gdown", direct_url, "-O", dest_path], check=True)
        else:
            subprocess.run(["curl", "-fSL", ref, "-o", dest_path], check=True)
    elif _state_asset_ref(ref):
        download_from_state(sync, ref, dest_path)
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
    samples = clip[: min(len(clip), sr * 3)].astype(np.float32)
    samples = samples - np.mean(samples)
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < 1e-4:
        return 0.0, "unknown"

    # Tự tương quan bằng FFT: O(n log n). np.correlate trực tiếp là O(n^2) —
    # với đoạn 6 s @16 kHz tốn hàng chục giây cho MỖI câu thoại.
    n = len(samples)
    spec = np.fft.rfft(samples, 2 * n)
    corr = np.fft.irfft(spec * np.conj(spec))[:n]
    min_lag = max(1, int(sr / 340))
    max_lag = min(n - 1, int(sr / 75))
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


def apply_speaker_diarization(audio_16k: str, segments: list[dict], work_dir: str) -> list[dict]:
    """Ưu tiên pyannote.audio (nếu có HF_TOKEN), tự động fallback sang SpeechBrain ECAPA-TDNN."""
    if not segments:
        return segments

    import numpy as np
    import soundfile as sf
    import torch

    hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    pyannote_model = os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1").strip()

    if hf_token:
        try:
            # torchaudio mới cần torchcodec để đọc wav; thiếu nó pyannote báo thiếu file .so
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "pyannote.audio>=3.1,<4", "torchaudio", "torchcodec"],
                check=False,
            )
            try:
                from huggingface_hub import whoami

                info = whoami(token=hf_token)
                print(f"[INFO] HF token hợp lệ, user={info.get('name')!r}.")
            except Exception as tok_err:
                print(f"[WARN] Không xác thực được HF token: {tok_err}")

            from pyannote.audio import Pipeline

            model_candidates = [pyannote_model, "pyannote/speaker-diarization-3.1", "pyannote/speaker-diarization-3.0"]
            pipeline = None
            for cand in dict.fromkeys(model_candidates):
                try:
                    try:
                        pipeline = Pipeline.from_pretrained(cand, token=hf_token)
                    except TypeError:
                        pipeline = Pipeline.from_pretrained(cand, use_auth_token=hf_token)
                    print(f"[INFO] Nạp được pipeline {cand}.")
                    break
                except Exception as load_err:
                    print(f"[WARN] Không nạp được {cand}: {type(load_err).__name__}: {load_err}")
                    pipeline = None
            if pipeline is None:
                raise RuntimeError("Không nạp được pipeline pyannote nào (kiểm tra license + quyền token).")

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

                refined_segs = _split_segments_on_speaker_turns(segments, turns, order_map)
                detected_list = list(dict.fromkeys(s["speaker_id"] for s in refined_segs))
                print(f"[INFO] pyannote.audio diarization ({pyannote_model}) phát hiện {len(detected_list)} người nói: {detected_list}")
                return refined_segs
        except Exception as py_err:
            print(f"[WARN] pyannote.audio chưa chạy được ({py_err}), chuyển sang SpeechBrain.")
            free_gpu()
    else:
        print("[INFO] Không tìm thấy HF_TOKEN trong môi trường, dùng SpeechBrain diarization.")

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

        del classifier
        free_gpu()

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
                from sklearn.metrics import silhouette_score

                best_labels, best_score = None, -1.0
                max_k = min(max_spk, max(2, len(anchor_embs) - 1))
                for k_try in range(2, max_k + 1):
                    cl = AgglomerativeClustering(n_clusters=k_try, metric="cosine", linkage="average")
                    labels_try = cl.fit_predict(emb_mat)
                    if len(set(int(x) for x in labels_try)) < 2:
                        continue
                    try:
                        sc = float(silhouette_score(emb_mat, labels_try, metric="cosine"))
                    except Exception:
                        continue
                    if sc > best_score:
                        best_score, best_labels = sc, labels_try
                if best_labels is None:
                    cl = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average")
                    best_labels = cl.fit_predict(emb_mat)
                anchor_labels = best_labels
                print(f"[INFO] Chọn {len(set(int(x) for x in anchor_labels))} người nói theo silhouette={best_score:.2f}")

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
    use_gpu = os.environ.get("PADDLE_OCR_GPU", "0").strip() == "1"

    # oneDNN (mkldnn) tăng tốc CPU nhưng bản paddle trong image Kaggle hay lỗi PIR executor
    # (đã thấy "lỗi runtime executor" → rơi xuống EasyOCR, mất gần hết phụ đề).
    # Vì vậy mặc định TẮT, chỉ bật khi đặt PADDLE_OCR_MKLDNN=1.
    use_mkldnn = os.environ.get("PADDLE_OCR_MKLDNN", "0").strip() == "1"
    device = "gpu:0" if _paddle_has_cuda() else "cpu"
    print(f"[INFO] PaddleOCR device={device}")
    attempts = [
        dict(lang=lang, device=device, enable_mkldnn=use_mkldnn, use_doc_orientation_classify=False,
             use_doc_unwarping=False, use_textline_orientation=False),
        dict(lang=lang, device=device, enable_mkldnn=False, use_doc_orientation_classify=False,
             use_doc_unwarping=False, use_textline_orientation=False),
        dict(lang=lang, enable_mkldnn=False),
        dict(use_angle_cls=False, lang=lang, use_gpu=use_gpu, enable_mkldnn=False),
    ]
    last_err = None
    for kwargs in attempts:
        try:
            engine = PaddleOCR(**kwargs)
            print(f"[INFO] PaddleOCR khởi tạo với {kwargs}")
            return engine
        except Exception as exc:
            last_err = exc
    raise last_err or RuntimeError("Không khởi tạo được PaddleOCR với mọi cấu hình.")


def _init_easyocr_fallback():
    import torch

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "easyocr"], check=True)
    import easyocr

    lang_env = os.environ.get("PADDLE_OCR_LANG", "ch").strip().lower()
    lang_map = {"ch": ["ch_sim", "en"], "vi": ["vi", "en"], "en": ["en"], "korean": ["ko", "en"], "japan": ["ja", "en"]}
    langs = lang_map.get(lang_env, ["ch_sim", "en"])
    return easyocr.Reader(langs, gpu=torch.cuda.is_available(), verbose=False)


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
        img.save(out_path, quality=95)
        return out_path
    except Exception:
        return img_path


def _ocr_single_image(ocr_engine, img_path: str, is_easyocr: bool = False, min_conf: float = 0.25, debug: bool = False) -> str:
    line_texts = []
    if is_easyocr:
        try:
            res = ocr_engine.readtext(img_path, detail=1)
            for item in res:
                if len(item) >= 3 and float(item[2]) >= min_conf:
                    txt = str(item[1]).strip()
                    if txt:
                        line_texts.append(txt)
            return " ".join(line_texts).strip()
        except Exception as e:
            if debug:
                print(f"[DEBUG] EasyOCR exception ({os.path.basename(img_path)}): {e}")
            return ""

    try:
        if hasattr(ocr_engine, "predict"):
            res = list(ocr_engine.predict(img_path))
        else:
            res = ocr_engine.ocr(img_path)
    except Exception as e:
        if debug:
            print(f"[DEBUG] PaddleOCR predict exception ({os.path.basename(img_path)}): {e}")
        raise

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


def _frame_signature(img_path: str):
    """Chữ ký rất nhẹ (32x8 grayscale) để bỏ qua các khung phụ đề giống hệt nhau."""
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.convert("L").resize((32, 8)).tobytes()
    except Exception:
        return None


def _video_duration(path: str) -> float:
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
            capture_output=True,
            text=True,
        )
        return float((res.stdout or "").strip())
    except Exception:
        return 0.0


def _extract_frames_for_roi(video_path: str, out_dir: str, roi: tuple[float, float, float, float], fps: float) -> list[tuple[float, str]]:
    os.makedirs(out_dir, exist_ok=True)
    x1, y1, x2, y2 = roi
    crop_w = max(0.2, x2 - x1)
    crop_h = max(0.10, y2 - y1)
    # Ảnh càng lớn OCR càng chậm (gần như tuyến tính theo số pixel). 960 đủ nét cho phụ đề.
    max_w = int(os.environ.get("OCR_MAX_WIDTH", "960"))
    budget = max(20, int(os.environ.get("OCR_MAX_FRAMES", "150")))
    duration = _video_duration(video_path)
    if duration > 0:
        effective_fps = max(0.1, min(float(fps), budget / duration))
        if effective_fps < float(fps):
            print(f"[INFO] Video dài {duration:.0f}s → giảm OCR {fps} fps xuống {effective_fps:.2f} fps để phủ hết video (tối đa {budget} khung).")
        fps = effective_fps
    if crop_w >= 0.98 and crop_h >= 0.98:
        vf_expr = f"fps={fps},scale={max_w}:-1"
    else:
        vf_expr = f"fps={fps},crop=iw*{crop_w:.3f}:ih*{crop_h:.3f}:iw*{x1:.3f}:ih*{y1:.3f},scale={max_w}:-1"

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
    return frames[:budget]


_NOISE_PATTERNS = [
    re.compile(r"第\s*\d+\s*号"),
    re.compile(r"(网|广)[^\s]{0,4}(备|字|审)[^\s]{0,4}号"),
    re.compile(r"许可证|备案|公映|联合出品|出品方|发行方"),
    re.compile(r"https?://|www\.|\.(com|cn|net|tv)\b", re.I),
    re.compile(r"百度|腾讯|爱奇艺|优酷|芒果|抖音|快手|西瓜|微博"),
]


def _clean_ocr_text(text: str) -> str:
    """Loại chữ rác: watermark, giấy phép phim, dòng 1-2 ký tự, dòng toàn số.

    OCR hardsub hay đọc trúng bản quyền/logo (vd '（百度）网微剧备字（2026）第412924号'
    hoặc '11', 'Z', '0 o bb') và biến chúng thành phụ đề — lọc trước khi dùng.
    """
    t = " ".join(str(text or "").split())
    if not t:
        return ""
    for pat in _NOISE_PATTERNS:
        if pat.search(t):
            return ""
    cjk = re.findall(r"[\u4e00-\u9fff]", t)
    latin = re.findall(r"[A-Za-z]", t)
    digits = re.findall(r"[0-9]", t)
    visible = len(t.replace(" ", ""))
    if len(cjk) == 0 and len(latin) < 3:
        return ""
    if len(cjk) < 2 and len(latin) < 4:
        return ""
    if visible and len(digits) / visible > 0.4:
        return ""
    return t


def _extract_frame_at(video_path: str, ts: float, roi: tuple[float, float, float, float], width: int, out_path: str) -> str | None:
    x1, y1, x2, y2 = roi
    crop_w = max(0.2, x2 - x1)
    crop_h = max(0.10, y2 - y1)
    if crop_w >= 0.98 and crop_h >= 0.98:
        vf = f"scale={width}:-1"
    else:
        vf = f"fps=1,crop=iw*{crop_w:.3f}:ih*{crop_h:.3f}:iw*{x1:.3f}:ih*{y1:.3f},scale={width}:-1"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{max(0.0, ts):.3f}", "-i", video_path, "-vf", vf, "-frames:v", "1", "-q:v", "2", out_path],
        capture_output=True,
    )
    return out_path if os.path.exists(out_path) and os.path.getsize(out_path) > 512 else None


def _scan_subtitle_spans(video_path: str, work_dir: str, roi: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    """Quét khung hình ở tần số cao (mặc định 5 fps) để tìm MỌI mốc phụ đề đổi chữ.

    Khác cách lấy mẫu 1 fps (dễ bỏ sót phụ đề ngắn), ở đây:
      1. Trích khung 320px ở 5 fps;
      2. So chữ ký 32x8 giữa 2 khung liên tiếp, chênh > ngưỡng thì coi là mốc đổi;
      3. Cắt thành các span theo mốc, span quá ngắn (<0.3s) do nhiễu nén thì gộp vào span bên cạnh
         (gộp chứ không bỏ, nên không mất phụ đề).
    """
    scan_dir = os.path.join(work_dir, "ocr_scan")
    os.makedirs(scan_dir, exist_ok=True)
    for old in os.listdir(scan_dir):
        if old.endswith(".jpg"):
            os.remove(os.path.join(scan_dir, old))

    scan_fps = float(os.environ.get("OCR_SCAN_FPS", "5.0"))
    max_scan = int(os.environ.get("OCR_SCAN_MAX_FRAMES", "4000"))
    duration = _video_duration(video_path)
    if duration > 0:
        scan_fps = max(0.5, min(scan_fps, max_scan / duration))

    x1, y1, x2, y2 = roi
    crop_w = max(0.2, x2 - x1)
    crop_h = max(0.10, y2 - y1)
    if crop_w >= 0.98 and crop_h >= 0.98:
        vf = f"fps={scan_fps},scale=320:-1"
    else:
        vf = f"fps={scan_fps},crop=iw*{crop_w:.3f}:ih*{crop_h:.3f}:iw*{x1:.3f}:ih*{y1:.3f},scale=320:-1"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", vf, "-q:v", "5", os.path.join(scan_dir, "s_%05d.jpg")],
        capture_output=True,
    )
    files = sorted(f for f in os.listdir(scan_dir) if f.endswith(".jpg"))
    if not files:
        return []

    import numpy as np

    vecs = []
    for fname in files:
        sig = _frame_signature(os.path.join(scan_dir, fname))
        if sig is None:
            vecs.append(None)
        else:
            vecs.append(np.frombuffer(sig, dtype=np.uint8).astype(np.int16))

    thr = float(os.environ.get("OCR_SPAN_DIFF_THRESHOLD", "6.0"))
    min_dur = float(os.environ.get("OCR_SPAN_MIN_DUR", "0.30"))
    bounds = [0]
    for i in range(1, len(vecs)):
        a, b = vecs[i - 1], vecs[i]
        if a is None or b is None:
            bounds.append(i)
            continue
        if float(np.mean(np.abs(a - b))) > thr:
            bounds.append(i)
    bounds.append(len(files))

    interval = 1.0 / scan_fps
    segs = [[bounds[i] * interval, bounds[i + 1] * interval] for i in range(len(bounds) - 1)]

    merged: list[list[float]] = []
    for s, e in segs:
        if not merged:
            merged.append([s, e])
            continue
        prev = merged[-1]
        if (e - s) < min_dur or (prev[1] - prev[0]) < min_dur:
            prev[1] = e
        else:
            merged.append([s, e])

    return [(round(s, 2), round(e, 2)) for s, e in merged if e - s >= 0.20]


def run_paddle_ocr(video_path: str, work_dir: str) -> tuple[list[dict], tuple[float, float, float, float]]:
    embedded = extract_embedded_subtitles(video_path, work_dir)
    if embedded:
        return embedded, (0.02, 0.70, 0.98, 0.98)

    candidate_rois = [
        (0.02, 0.70, 0.98, 0.98),
        (0.02, 0.58, 0.98, 0.86),
        (0.02, 0.52, 0.98, 0.80),
        (0.02, 0.55, 0.98, 0.99),
        (0.0, 0.0, 1.0, 1.0),
    ]
    chosen_roi = candidate_rois[0]
    ocr_records: list[dict] = []
    ocr_engine = None
    is_easyocr = False
    _ocr_err_shown = False

    def _ocr_path(path: str, min_conf: float = 0.25) -> str:
        nonlocal _ocr_err_shown
        try:
            return _ocr_single_image(ocr_engine, path, is_easyocr=is_easyocr, min_conf=min_conf)
        except Exception as exc:
            if not _ocr_err_shown:
                _ocr_err_shown = True
                print(f"[WARN] OCR exception: {type(exc).__name__}: {exc}")
            return ""

    try:
        engine_pref = os.environ.get("OCR_ENGINE", "easyocr").strip().lower()
        if engine_pref in ("auto", "easyocr"):
            try:
                ocr_engine = _init_easyocr_fallback()
                is_easyocr = True
                print("[INFO] OCR engine: EasyOCR (dùng GPU nếu torch thấy CUDA).")
            except Exception as ee:
                print(f"[WARN] EasyOCR không khởi tạo được ({ee}) → chuyển PaddleOCR.")
        if ocr_engine is None:
            try:
                ocr_engine = _init_paddle_ocr_engine()
                print("[INFO] OCR engine: PaddleOCR.")
            except Exception as pe:
                print(f"[WARN] PaddleOCR không khởi tạo được ({pe}) → thử lại EasyOCR.")
                ocr_engine = _init_easyocr_fallback()
                is_easyocr = True

        # 1) Chọn ROI: chấm điểm bằng số phụ đề HỢP LỆ trên vài khung mẫu
        best_roi = None
        best_hits = -1
        probe_dir = os.path.join(work_dir, "ocr_probe")
        os.makedirs(probe_dir, exist_ok=True)
        for idx_roi, roi in enumerate(candidate_rois):
            frames = []
            duration = _video_duration(video_path) or 30.0
            n_probe = int(os.environ.get("OCR_PROBE_FRAMES", "12"))
            for k in range(n_probe):
                ts = duration * (0.05 + 0.90 * k / max(1, n_probe - 1))
                p = _extract_frame_at(video_path, ts, roi, 960, os.path.join(probe_dir, f"p_{idx_roi}_{k}.jpg"))
                if p:
                    frames.append(p)
            hits = 0
            samples = []
            raws = []
            for k_i, p in enumerate(frames):
                if k_i == 0 and idx_roi == 0:
                    raw = _ocr_single_image(ocr_engine, p, is_easyocr=is_easyocr, min_conf=0.15, debug=True)
                else:
                    raw = _ocr_path(p, min_conf=0.15)
                raws.append(raw)
                cleaned = _clean_ocr_text(raw)
                if cleaned:
                    hits += 1
                    samples.append(cleaned[:16])
            raw_preview = " | ".join((r or "")[:14] for r in raws[:3])
            print(f"[INFO] [{('EasyOCR' if is_easyocr else 'PaddleOCR')}] ROI {roi} | phụ đề hợp lệ={hits}/{len(frames)} | raw: {raw_preview}")
            if hits > best_hits:
                best_hits, best_roi = hits, roi
            if hits >= 4:
                break
        chosen_roi = best_roi or candidate_rois[0]
        print(f"[INFO] ROI chọn: {chosen_roi} (hits={max(0, best_hits)})")

        # 2) Quét span phụ đề ở tần số cao rồi OCR đúng 1 khung đại diện mỗi span
        spans = _scan_subtitle_spans(video_path, work_dir, chosen_roi)
        print(f"[INFO] Phát hiện {len(spans)} span phụ đề bằng quét {os.environ.get('OCR_SCAN_FPS', '5.0')} fps.")
        span_dir = os.path.join(work_dir, "ocr_spans")
        os.makedirs(span_dir, exist_ok=True)
        for i, (s_t, e_t) in enumerate(spans, 1):
            mid = (s_t + e_t) / 2.0
            fracs = (0.25, 0.75) if e_t - s_t >= 0.6 else (0.5,)
            samples = []
            for c_i, frac in enumerate(fracs):
                ts = s_t + (e_t - s_t) * frac
                img = _extract_frame_at(video_path, ts, chosen_roi, 960, os.path.join(span_dir, f"span_{i:05d}_{c_i}.jpg"))
                if not img:
                    continue
                txt = _clean_ocr_text(_ocr_path(img))
                if txt:
                    samples.append((ts, txt))
            if not samples:
                continue
            if len(samples) >= 2 and samples[0][1] != samples[1][1]:
                # Hai nửa span là hai phụ đề khác nhau → tách đôi tại trung điểm
                ocr_records.append({"time": round(samples[0][0], 2), "start": s_t, "end": round(mid, 2), "text": samples[0][1]})
                ocr_records.append({"time": round(samples[1][0], 2), "start": round(mid, 2), "end": e_t, "text": samples[1][1]})
            else:
                ocr_records.append({"time": round(mid, 2), "start": s_t, "end": e_t, "text": samples[0][1]})
            if i % 25 == 0 or i == len(spans):
                print(f"[INFO] OCR span {i}/{len(spans)} → {len(ocr_records)} dòng phụ đề hợp lệ...")

        # Gộp các record liền kề trùng chữ (1 phụ đề bị tách làm 2 span do fade)
        merged_records = []
        for rec in ocr_records:
            if merged_records and rec["text"] == merged_records[-1]["text"] and rec["start"] - merged_records[-1]["end"] <= 0.6:
                merged_records[-1]["end"] = rec["end"]
            else:
                merged_records.append(dict(rec))
        ocr_records = merged_records

        del ocr_engine
        free_gpu()
    except Exception as ocr_err:
        print(f"[WARN] OCR lỗi: {ocr_err}")
        free_gpu()

    print(f"[INFO] OCR xong: ROI {chosen_roi} → {len(ocr_records)} dòng phụ đề có mốc start/end.")
    return ocr_records, chosen_roi


def align_and_refine_segments(segments: list[dict], ocr_records: list[dict]) -> list[dict]:
    """Ghép phụ đề OCR (đã có start/end) vào từng câu ASR để SỬA CHÍNH TẢ.

    Nguyên tắc: chỉ dùng chữ OCR khi span phụ đề chồng lấn thời gian với câu thoại;
    nếu OCR và ASR là cùng một câu (tỉ lệ giống ≥0.5 hoặc độ dài tương đương) thì lấy
    chữ OCR vì OCR đọc đúng mặt chữ, ASR hay nghe nhầm từ đồng âm.
    """
    refined = []
    corrected_cnt = 0
    extended_cnt = 0
    kept_cnt = 0
    for seg in segments:
        s_t = float(seg.get("start", 0.0))
        e_t = float(seg.get("end", s_t + 1.0))
        asr_txt = (seg.get("asr_text") or "").strip()

        cands = []
        for r in ocr_records:
            r_s = float(r.get("start", r.get("time", 0.0)))
            r_e = float(r.get("end", r_s + 1.0))
            if r_e < s_t - 1.2 or r_s > e_t + 1.2:
                continue
            cleaned = _clean_ocr_text(r.get("text", ""))
            if not cleaned:
                continue
            ov = max(0.0, min(e_t, r_e) - max(s_t, r_s))
            ov_ratio = ov / max(0.3, e_t - s_t)
            sim = SequenceMatcher(None, asr_txt, cleaned).ratio() if asr_txt else 0.0
            # Điểm kết hợp thời gian + độ giống chữ: ASR và phụ đề thường lệch nhau ~0.5s,
            # nếu chỉ lấy overlap lớn nhất sẽ vơ phụ đề của câu bên cạnh.
            score = sim + 0.35 * min(1.0, ov_ratio)
            cands.append((score, sim, ov_ratio, cleaned))
        cands.sort(key=lambda x: -x[0])
        ocr_txt = cands[0][3] if cands else ""
        best_sim = cands[0][1] if cands else 0.0
        best_ovr = cands[0][2] if cands else 0.0

        best_txt = asr_txt
        source_mode = "asr_only"
        if asr_txt and ocr_txt:
            len_ratio = len(ocr_txt) / max(1, len(asr_txt))
            cjk_ocr = len(re.findall(r"[\u4e00-\u9fff]", ocr_txt))
            if best_sim >= 0.55 or (best_sim >= 0.35 and best_ovr >= 0.45):
                # Cùng một câu, khác mặt chữ do ASR nghe nhầm → lấy chữ OCR
                best_txt, source_mode = ocr_txt, "ocr_corrected"
                corrected_cnt += 1
            elif cjk_ocr >= 4 and len(asr_txt) > 3 and len_ratio > 1.3 and best_ovr >= 0.6:
                best_txt, source_mode = ocr_txt, "ocr_extended"
                extended_cnt += 1
            else:
                best_txt, source_mode = asr_txt, "asr_kept"
                kept_cnt += 1
        elif ocr_txt and not asr_txt:
            best_txt, source_mode = ocr_txt, "ocr_only"
            corrected_cnt += 1
        else:
            kept_cnt += 1

        item = dict(seg)
        item["ocr_text"] = ocr_txt
        item["best_text"] = best_txt
        item["subtitle_source"] = source_mode
        refined.append(item)

    print(
        f"[INFO] Đối chiếu ASR+OCR: {corrected_cnt} câu lấy chữ OCR (sửa chính tả), "
        f"{extended_cnt} câu OCR dài hơn, {kept_cnt} câu giữ ASR."
    )
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


def mark_job_failed(err_text: str):
    """Ghi FAILED vào state/jobs.json để UI và monitor thấy lỗi thay vì treo ở trạng thái cũ."""
    try:
        data, _ = sync.read_json("state/jobs.json")
        job = data.setdefault("jobs", {}).setdefault(JOB_ID, {})
        job["status"] = "FAILED_STAGE1"
        job["error"] = err_text[-800:]
        job["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        sync.write_json("state/jobs.json", data, f"FAILED stage1: {JOB_ID}")
    except Exception as exc:
        print(f"[WARN] Không ghi được trạng thái FAILED: {exc}")


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
        mark_job_failed(err)
        send_telegram(TG_TOKEN, TG_CHAT, f"❌ <b>[Stage 1 Lỗi]</b> <code>{JOB_ID}</code>:\n<pre>{err}</pre>")
        raise
