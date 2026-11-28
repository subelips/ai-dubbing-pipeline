import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import base64
import gc
import importlib
import inspect
import json
import pkgutil
import re
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


def send_ntfy(title: str, message: str, tags: str = "loud_sound", priority: str = "default"):
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

    def try_download_asset(self, tag: str, filename: str, dest_path: str) -> str:
        try:
            release = self.ensure_release(tag)
            for asset in release.get("assets", []):
                if asset["name"] == filename:
                    dl_url = _clean_url_or_str(asset["url"])
                    dl_headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/octet-stream"}
                    with requests.get(dl_url, headers=dl_headers, stream=True, timeout=300) as dl:
                        dl.raise_for_status()
                        with open(dest_path, "wb") as out:
                            for chunk in dl.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    out.write(chunk)
                    return dest_path if os.path.exists(dest_path) and os.path.getsize(dest_path) > 512 else ""
        except Exception as e:
            print(f"[WARN] Không tải được asset {filename} từ release {tag}: {e}")
        return ""

    def upload_asset(self, tag: str, file_path: str) -> str:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return ""
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

    def trigger_dispatch(self, event_type: str, client_payload: dict):
        r = requests.post(
            f"{self.api_base}/dispatches",
            headers=self.headers,
            json={"event_type": event_type, "client_payload": client_payload},
            timeout=30,
        )
        r.raise_for_status()


def _is_installed(mod_name: str) -> bool:
    try:
        __import__(mod_name)
        return True
    except Exception:
        return False


def ensure_deps():
    wanted = [
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("gdown", "gdown"),
        ("vieneu", "vieneu"),
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


def download_source_media(sync: GitHubSync, tag: str, video_asset: str, work_dir: str) -> tuple[str, str]:
    video_path = os.path.join(work_dir, "input.mp4")
    audio_path = os.path.join(work_dir, "source_audio_24k.wav")

    existing_mp4 = "/kaggle/working/work/input.mp4"
    if os.path.exists(existing_mp4) and os.path.getsize(existing_mp4) > 1024:
        video_path = existing_mp4
    else:
        ref = _clean_url_or_str(video_asset)
        if ref.startswith(("http://", "https://")):
            try:
                if "drive.google.com" in ref:
                    subprocess.run([sys.executable, "-m", "gdown", _normalize_gdrive_url(ref), "-O", video_path], check=True)
                else:
                    subprocess.run(["curl", "-fSL", ref, "-o", video_path], check=True)
            except Exception as e:
                print(f"[WARN] Tải video nguồn thất bại ({e}), thử tải asset âm thanh từ Release.")

    # Ưu tiên track vocals tách từ Stage 1 và resample chuẩn về 24kHz mono sạch cho speaker_encoder
    vocal_candidates = [
        "/kaggle/working/work/demucs_out/htdemucs/audio_16k/vocals.wav",
        os.path.join(work_dir, "vocals.wav"),
    ]
    for vc in vocal_candidates:
        if os.path.exists(vc) and os.path.getsize(vc) > 1024:
            subprocess.run(
                ["ffmpeg", "-y", "-i", vc, "-ac", "1", "-ar", "24000", audio_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            break

    if not os.path.exists(audio_path) and os.path.exists(video_path) and os.path.getsize(video_path) > 1024:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "24000", audio_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    if os.path.exists(video_path) and os.path.getsize(video_path) > 1024:
        return video_path, audio_path if os.path.exists(audio_path) else ""

    bgm_dl = sync.try_download_asset(tag, "no_vocals.wav", os.path.join(work_dir, "no_vocals.wav"))
    return "", bgm_dl


def detect_segment_expression(text_vi: str, duration_sec: float) -> dict:
    t = (text_vi or "").strip()
    chars = max(1, len(re.sub(r"\s+", "", t)))
    natural_pace = chars / max(0.35, duration_sec)

    tone = "neutral"
    if re.search(r"[!！]", t):
        tone = "emphatic"
    elif re.search(r"[?？]", t) or re.search(r"\b(sao|hả|à|ư|chứ|nhỉ|đúng không|tại sao)\b", t, re.I):
        tone = "questioning"
    elif re.search(r"\b(chó|cút|đồ|khốn|mày|tệ|bẩn|phá sản|mù|láo|xấc)\b", t, re.I):
        tone = "harsh_contempt"
    elif re.search(r"\b(cảm ơn|biết ơn|thương|nhớ|vợ ơi|em à|cẩn thận|lạnh|đau dạ dày)\b", t, re.I):
        tone = "warm_caring"
    elif re.search(r"\b(thật ra|bí mật|ẩn danh|người thừa kế|phá sản)\b", t, re.I):
        tone = "serious_grave"

    return {"tone": tone, "natural_pace": round(natural_pace, 2)}


def extract_segment_ref_clip(
    audio_samples,
    sr: int,
    start_t: float,
    end_t: float,
    out_wav_path: str,
    fallback_ref_path: str = "",
) -> str:
    """Cắt mẫu tham chiếu đúng khoảng câu để giữ cảm xúc từng câu, tự mở rộng nhẹ nếu câu ngắn."""
    import numpy as np
    import soundfile as sf

    if audio_samples is not None and len(audio_samples) > 0 and sr > 0:
        s_sec = max(0.0, float(start_t))
        e_sec = max(s_sec + 0.35, float(end_t))
        dur = e_sec - s_sec

        if dur < 1.50:
            pad = (1.60 - dur) * 0.5 + 0.15
            s_sec = max(0.0, s_sec - pad)
            e_sec = min(len(audio_samples) / float(sr), e_sec + pad)

        if (e_sec - s_sec) > 6.5:
            e_sec = s_sec + 6.5

        i0 = max(0, int(round(s_sec * sr)))
        i1 = min(len(audio_samples), int(round(e_sec * sr)))
        if i1 - i0 >= int(0.50 * sr):
            clip = audio_samples[i0:i1].astype(np.float32)
            rms = float(np.sqrt(np.mean(clip ** 2)))
            if rms >= 0.005:
                sf.write(out_wav_path, clip, sr)
                return out_wav_path

    if fallback_ref_path and os.path.exists(fallback_ref_path) and os.path.getsize(fallback_ref_path) > 512:
        return fallback_ref_path
    return ""


def _detect_engine_sample_rate(engine) -> int:
    """Dò chính xác sample rate đầu ra của VieNeuTTSv3Turbo để tránh lệch cao độ/tốc độ."""
    for attr in ("sample_rate", "sr", "output_sample_rate", "sampling_rate"):
        val = getattr(engine, attr, None)
        if isinstance(val, (int, float)) and 8000 <= int(val) <= 48000:
            return int(val)
    for sub_attr in ("codec", "audio_tokenizer", "tokenizer", "config", "model"):
        sub = getattr(engine, sub_attr, None)
        if sub is not None:
            for attr in ("sample_rate", "sr", "output_sample_rate", "sampling_rate"):
                val = getattr(sub, attr, None)
                if isinstance(val, (int, float)) and 8000 <= int(val) <= 48000:
                    return int(val)
    return 24000


def load_vieneu_v3_turbo():
    """Khởi tạo bắt buộc VieNeu-TTS-V3-Turbo (không fallback sang edge-tts)."""
    import torch
    import vieneu  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = os.environ.get("VIENEU_MODEL", "pnnbao-ump/VieNeu-TTS-v3-Turbo").strip() or "pnnbao-ump/VieNeu-TTS-v3-Turbo"

    modules_to_scan = [vieneu]
    if hasattr(vieneu, "__path__"):
        for _, modname, _ in pkgutil.walk_packages(vieneu.__path__, prefix="vieneu."):
            try:
                modules_to_scan.append(importlib.import_module(modname))
            except Exception:
                continue

    synth_methods = ("infer", "synthesize", "tts", "generate", "speak", "clone", "__call__")
    exclude_suffixes = (
        "lock", "config", "cfg", "error", "exception", "warning",
        "state", "context", "pool", "queue", "manager", "helper",
        "utils", "util", "parser", "arg", "args", "dataset", "loader",
    )

    candidate_classes = []
    for mod in modules_to_scan:
        mod_name = getattr(mod, "__name__", "")
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            low = attr_name.lower()
            if any(low.endswith(suf) or suf in low.split("_") for suf in exclude_suffixes):
                continue
            obj = getattr(mod, attr_name, None)
            if inspect.isclass(obj) and not inspect.isabstract(obj):
                methods_found = [m for m in synth_methods if callable(getattr(obj, m, None))]
                if methods_found:
                    score = 0
                    if "v3turbo" in low or ("v3" in low and "turbo" in low):
                        score += 50
                    elif "turbo" in low:
                        score += 20
                    elif "vieneu" in low or "tts" in low:
                        score += 10
                    if "gpu" in low and device == "cuda":
                        score += 5
                    score += len(methods_found)
                    candidate_classes.append((score, f"{mod_name}.{attr_name}", obj, methods_found))

    candidate_classes.sort(key=lambda x: x[0], reverse=True)
    print(f"[INFO] Ứng viên VieNeu-TTS-V3-Turbo: {[x[1] for x in candidate_classes[:6]]}")

    init_errors = []
    for _, full_name, cls, methods_found in candidate_classes:
        kw_list = [
            {"device": device},
            {},
            {"model_name": model_id, "device": device},
            {"model_id": model_id, "device": device},
            {"repo_id": model_id, "device": device},
            {"backbone_repo": model_id, "device": device},
            {"model_name": model_id},
            {"model_id": model_id},
        ]
        for kw in kw_list:
            try:
                eng = cls(**kw)
                detected_sr = _detect_engine_sample_rate(eng)
                sig_info = {}
                for m in ("prepare_reference", *methods_found):
                    fn = getattr(eng, m, None)
                    if callable(fn):
                        try:
                            sig_info[m] = str(inspect.signature(fn))
                        except Exception:
                            sig_info[m] = "(...)"
                print(
                    f"[INFO] Đã khởi tạo {full_name} (kwargs={list(kw.keys())}, sr={detected_sr}Hz) trên {device}. "
                    f"Signatures={sig_info}"
                )
                return eng, full_name, sig_info, detected_sr
            except Exception as e:
                init_errors.append(f"{full_name}({list(kw.keys())}): {type(e).__name__}: {e}")
                continue

    raise RuntimeError(
        "Không khởi tạo được VieNeu-TTS-V3-Turbo:\n" + "\n".join(init_errors[:10])
    )


def _extract_audio_from_vieneu_output(out, default_sr: int = 24000):
    import numpy as np

    if out is None:
        return None, default_sr

    if isinstance(out, str):
        return out, default_sr

    if isinstance(out, dict):
        sr = int(out.get("sample_rate") or out.get("sr") or default_sr)
        for k in ("audio", "wav", "waveform", "samples", "output", "data"):
            if k in out and out[k] is not None:
                return _extract_audio_from_vieneu_output(out[k], sr)
        return None, sr

    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            return None, default_sr
        if len(out) >= 2 and isinstance(out[1], (int, float)) and not hasattr(out[1], "__len__"):
            return _extract_audio_from_vieneu_output(out[0], int(out[1]))
        return _extract_audio_from_vieneu_output(out[0], default_sr)

    if hasattr(out, "detach") and hasattr(out, "cpu"):
        out = out.detach().cpu().numpy()

    arr = np.asarray(out, dtype=np.float32).squeeze()
    if arr.ndim == 0 or arr.size < 160:
        return None, default_sr
    return arr, default_sr


def _extract_speaker_emb_only(prep_out):
    """Chỉ lấy speaker_emb (vector float) từ prepare_reference, bỏ ref_codes để tránh xung đột ngữ âm Trung-Việt."""
    import numpy as np

    if prep_out is None:
        return None

    if isinstance(prep_out, dict):
        for k in ("speaker_emb", "spk_emb", "embedding"):
            if k in prep_out and prep_out[k] is not None:
                return prep_out[k]
        return None

    if isinstance(prep_out, (tuple, list)):
        for it in prep_out:
            if it is None:
                continue
            arr = it.detach().cpu().numpy() if (hasattr(it, "detach") and hasattr(it, "cpu")) else it
            if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
                return it
        return prep_out[0] if len(prep_out) > 0 else None

    return prep_out


def _apply_gen_params_if_supported(kw: dict, param_names: set[str], tone: str):
    temp_map = {
        "emphatic": 0.78,
        "questioning": 0.75,
        "harsh_contempt": 0.78,
        "warm_caring": 0.72,
        "serious_grave": 0.72,
        "neutral": 0.74,
    }
    gen_defaults = {
        "use_ref_codes": False,
        "temperature": temp_map.get(tone, 0.74),
        "top_k": 30,
        "top_p": 0.92,
        "repetition_penalty": 1.15,
    }
    for k, v in gen_defaults.items():
        if k in param_names and k not in kw:
            kw[k] = v


def _synthesize_single_line_vieneu_v3(
    engine,
    ref_code_cache: dict[str, object],
    text_vi: str,
    ref_audio_path: str,
    gender: str,
    speaker_id: str,
    tone: str,
    default_sr: int,
    out_wav: str,
) -> tuple[bool, list[str]]:
    import numpy as np
    import soundfile as sf

    has_ref = bool(ref_audio_path and os.path.exists(ref_audio_path) and os.path.getsize(ref_audio_path) > 512)
    ref_wav_arr = None
    ref_sr = 24000
    if has_ref:
        try:
            ref_wav_arr, ref_sr = sf.read(ref_audio_path, dtype="float32")
            if ref_wav_arr.ndim > 1:
                ref_wav_arr = ref_wav_arr.mean(axis=-1)
        except Exception:
            ref_wav_arr = None

    preset_voices = []
    for attr in ("voices", "voice_map", "available_voices", "speakers"):
        v_obj = getattr(engine, attr, None)
        if isinstance(v_obj, dict):
            preset_voices = list(v_obj.keys())
            break
        if isinstance(v_obj, (list, tuple)):
            preset_voices = list(v_obj)
            break

    is_female = str(gender).lower() == "female" or str(speaker_id).endswith("_F")
    chosen_voice = None
    if preset_voices:
        female_keys = [v for v in preset_voices if any(k in str(v).lower() for k in ("nu", "nữ", "female", "woman", "girl", "ngoc", "mai", "linh", "my"))]
        male_keys = [v for v in preset_voices if any(k in str(v).lower() for k in ("nam", "male", "man", "boy", "minh", "tung", "hung", "duy"))]
        if is_female and female_keys:
            chosen_voice = female_keys[0]
        elif not is_female and male_keys:
            chosen_voice = male_keys[1] if ("02" in str(speaker_id) and len(male_keys) >= 2) else male_keys[0]
        else:
            chosen_voice = preset_voices[0]

    fn_infer = getattr(engine, "infer", None) or getattr(engine, "synthesize", None) or getattr(engine, "tts", None)
    if not callable(fn_infer) and callable(engine):
        fn_infer = engine

    try:
        sig = inspect.signature(fn_infer)
        param_names = set(sig.parameters.keys())
    except Exception:
        param_names = set()

    call_candidates = []

    # 1. Cross-lingual: gọi prepare_reference(..., use_ref_codes=False) để chỉ trích xuất speaker_emb sạch
    if has_ref:
        for enc_name in ("prepare_reference", "encode_reference", "extract_speaker_embedding", "encode_speaker"):
            enc_fn = getattr(engine, enc_name, None)
            if callable(enc_fn):
                try:
                    enc_params = set(inspect.signature(enc_fn).parameters.keys())
                except Exception:
                    enc_params = set()

                def _call_spk_emb_only(ef=enc_fn, ename=enc_name, eparams=enc_params):
                    cache_key = f"{ename}_no_codes:{ref_audio_path}"
                    if cache_key not in ref_code_cache:
                        enc_kw = {}
                        if "use_ref_codes" in eparams:
                            enc_kw["use_ref_codes"] = False
                        if "denoise" in eparams:
                            enc_kw["denoise"] = True
                        try:
                            ref_code_cache[cache_key] = ef(ref_audio_path, **enc_kw)
                        except Exception:
                            if ref_wav_arr is not None:
                                try:
                                    ref_code_cache[cache_key] = ef(ref_wav_arr, sr=ref_sr, **enc_kw)
                                except Exception:
                                    ref_code_cache[cache_key] = ef(ref_wav_arr)
                            else:
                                raise

                    spk_emb = _extract_speaker_emb_only(ref_code_cache[cache_key])
                    kw = {}
                    if spk_emb is not None and "speaker_emb" in param_names:
                        kw["speaker_emb"] = spk_emb
                    _apply_gen_params_if_supported(kw, param_names, tone)
                    return fn_infer(text=text_vi, **kw)

                call_candidates.append((f"infer(speaker_emb only via {enc_name})", _call_spk_emb_only))

        if not param_names or "ref_audio" in param_names:
            def _call_ref_path():
                kw = {"ref_audio": ref_audio_path}
                _apply_gen_params_if_supported(kw, param_names, tone)
                return fn_infer(text=text_vi, **kw)
            call_candidates.append(("infer(text, ref_audio=path)", _call_ref_path))

    if chosen_voice is not None:
        if "voice" in param_names:
            def _call_voice(v=chosen_voice):
                kw = {"voice": v}
                _apply_gen_params_if_supported(kw, param_names, tone)
                return fn_infer(text=text_vi, **kw)
            call_candidates.append((f"infer(text, voice={chosen_voice})", _call_voice))
        if "speaker" in param_names:
            def _call_speaker(v=chosen_voice):
                kw = {"speaker": v}
                _apply_gen_params_if_supported(kw, param_names, tone)
                return fn_infer(text=text_vi, **kw)
            call_candidates.append((f"infer(text, speaker={chosen_voice})", _call_speaker))

    def _call_plain_kw():
        kw = {}
        _apply_gen_params_if_supported(kw, param_names, tone)
        return fn_infer(text=text_vi, **kw)

    call_candidates.append(("infer(text=text_vi)", _call_plain_kw))
    call_candidates.append(("infer(text_vi)", lambda: fn_infer(text_vi)))

    errors = []
    for desc, fn in call_candidates:
        try:
            out = fn()
            parsed_audio, sr_out = _extract_audio_from_vieneu_output(out, default_sr)
            if isinstance(parsed_audio, str) and os.path.exists(parsed_audio):
                data, sr_file = sf.read(parsed_audio, dtype="float32")
                if data.size >= 160:
                    sf.write(out_wav, data, sr_file)
                    return True, errors
            elif parsed_audio is not None and isinstance(parsed_audio, np.ndarray) and parsed_audio.size >= 160:
                sf.write(out_wav, parsed_audio, sr_out)
                return True, errors
            out_shape = getattr(out, "shape", None)
            out_keys = list(out.keys()) if isinstance(out, dict) else None
            errors.append(f"{desc} -> empty/unparsed (type={type(out).__name__}, shape={out_shape}, keys={out_keys})")
        except Exception as e:
            errors.append(f"{desc} -> {type(e).__name__}: {e}")
            continue

    return False, errors


def _generate_silence_clip(target_dur: float, out_wav: str, sr: int = 24000):
    import numpy as np
    import soundfile as sf

    n = max(1, int(round(max(0.25, float(target_dur or 0.5)) * sr)))
    sf.write(out_wav, np.zeros(n, dtype=np.float32), sr)


def _format_srt_time(seconds: float) -> str:
    ms_total = max(0, int(round(float(seconds or 0.0) * 1000.0)))
    hours = ms_total // 3_600_000
    rem = ms_total % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    secs = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_final_dubbed_track(
    dub_lines: list[dict],
    clip_map: dict[int, str],
    bgm_wav_path: str,
    out_mix_wav: str,
    out_srt_path: str,
    target_sr: int = 24000,
) -> str:
    """Giữ nguyên waveform gốc của VieNeu-TTS (không atempo gây méo tiếng), xếp nối tiếp tránh chồng lấn và ducking BGM."""
    import numpy as np
    import soundfile as sf

    sr = int(target_sr or 24000)

    # Tính trước độ dài thực tế của từng clip để không bị cắt cụt hoặc đè câu sau
    clip_audios: dict[int, np.ndarray] = {}
    max_timeline_sec = 5.0
    cursor_end_sec = 0.0

    for idx_out, line in enumerate(dub_lines):
        line_id = int(line.get("id", idx_out))
        start_t = max(0.0, float(line.get("start", 0.0)))
        end_t = max(start_t + 0.25, float(line.get("end", start_t + 0.8)))

        c_path = clip_map.get(line_id, "")
        if c_path and os.path.exists(c_path):
            seg_wav, seg_sr = sf.read(c_path, dtype="float32")
            if seg_wav.ndim > 1:
                seg_wav = seg_wav.mean(axis=-1)
            if seg_sr != sr and len(seg_wav) > 0:
                old_x = np.linspace(0.0, 1.0, len(seg_wav), endpoint=False)
                new_len = max(1, int(round(len(seg_wav) * sr / float(seg_sr))))
                new_x = np.linspace(0.0, 1.0, new_len, endpoint=False)
                seg_wav = np.interp(new_x, old_x, seg_wav).astype(np.float32)
            clip_audios[line_id] = seg_wav
            actual_start = max(start_t, cursor_end_sec + 0.04) if cursor_end_sec > 0.0 else start_t
            actual_end = actual_start + (len(seg_wav) / float(sr))
            cursor_end_sec = actual_end
            max_timeline_sec = max(max_timeline_sec, actual_end, end_t)
        else:
            max_timeline_sec = max(max_timeline_sec, end_t)

    total_samples = int(round((max_timeline_sec + 1.0) * sr))
    voice_track = np.zeros(total_samples, dtype=np.float32)

    srt_blocks = []
    cursor_end_sec = 0.0
    for idx_out, line in enumerate(dub_lines, start=1):
        line_id = int(line.get("id", idx_out - 1))
        start_t = max(0.0, float(line.get("start", 0.0)))
        end_t = max(start_t + 0.25, float(line.get("end", start_t + 0.8)))
        txt_vi = (line.get("text_vi") or line.get("src_text") or "").strip()

        if txt_vi:
            srt_blocks.append(
                f"{idx_out}\n"
                f"{_format_srt_time(start_t)} --> {_format_srt_time(end_t)}\n"
                f"[{line.get('speaker_id', 'SPEAKER_00')} | {line.get('tone', 'neutral')}] {txt_vi}\n"
            )

        seg_wav = clip_audios.get(line_id)
        if seg_wav is not None and len(seg_wav) > 0:
            actual_start = max(start_t, cursor_end_sec + 0.04) if cursor_end_sec > 0.0 else start_t
            i0 = int(round(actual_start * sr))
            i1 = min(total_samples, i0 + len(seg_wav))
            if i1 > i0:
                voice_track[i0:i1] += seg_wav[: (i1 - i0)]
                cursor_end_sec = i1 / float(sr)

    if bgm_wav_path and os.path.exists(bgm_wav_path):
        bgm_wav, bgm_sr = sf.read(bgm_wav_path, dtype="float32")
        if bgm_wav.ndim > 1:
            bgm_wav = bgm_wav.mean(axis=-1)
        if bgm_sr != sr and len(bgm_wav) > 0:
            old_x = np.linspace(0.0, 1.0, len(bgm_wav), endpoint=False)
            new_len = max(1, int(round(len(bgm_wav) * sr / float(bgm_sr))))
            new_x = np.linspace(0.0, 1.0, new_len, endpoint=False)
            bgm_wav = np.interp(new_x, old_x, bgm_wav).astype(np.float32)

        if len(bgm_wav) < total_samples:
            bgm_padded = np.zeros(total_samples, dtype=np.float32)
            bgm_padded[: len(bgm_wav)] = bgm_wav
            bgm_wav = bgm_padded
        else:
            bgm_wav = bgm_wav[:total_samples]

        win = max(1, int(0.05 * sr))
        abs_v = np.abs(voice_track)
        kernel = np.ones(win, dtype=np.float32) / float(win)
        env = np.convolve(abs_v, kernel, mode="same")
        has_speech = (env > 0.004).astype(np.float32)
        smooth_speech = np.convolve(
            has_speech,
            np.ones(max(1, int(0.15 * sr)), dtype=np.float32) / float(max(1, int(0.15 * sr))),
            mode="same",
        )
        bgm_gain = 0.45 - 0.30 * np.clip(smooth_speech, 0.0, 1.0)

        final_mix = np.clip(voice_track + bgm_gain * bgm_wav, -1.0, 1.0)
    else:
        final_mix = np.clip(voice_track, -1.0, 1.0)

    sf.write(out_mix_wav, final_mix, sr)
    with open(out_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_blocks).strip() + ("\n" if srt_blocks else ""))

    return out_mix_wav


def mux_dubbed_video(video_path: str, dubbed_wav: str, out_mp4: str) -> str:
    if not video_path or not os.path.exists(video_path) or not dubbed_wav or not os.path.exists(dubbed_wav):
        print(
            f"[WARN] Bỏ qua mux video do thiếu file: video={bool(video_path and os.path.exists(video_path))}, "
            f"wav={bool(dubbed_wav and os.path.exists(dubbed_wav))}"
        )
        return ""
    res = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            dubbed_wav,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            out_mp4,
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 or not os.path.exists(out_mp4):
        print(f"[WARN] ffmpeg mux lỗi (code {res.returncode}): {res.stderr[-400:]}")
        return ""
    return out_mp4


sync = GitHubSync(GH_REPO, GH_TOKEN)


def main():
    send_ntfy(
        f"[Stage 3] {JOB_ID}",
        f"Bắt đầu tổng hợp giọng nói bằng VieNeu-TTS-V3-Turbo (speaker_emb cross-lingual, giữ waveform gốc) cho job {JOB_ID}.",
        tags="loud_sound",
    )

    stage2_data, _ = sync.read_json(f"state/{JOB_ID}_stage2.json")
    dub_lines = stage2_data.get("lines") or stage2_data.get("utterances") or []
    video_asset = (stage2_data.get("video_asset") or "source_input.mp4").strip()

    work_dir = "/kaggle/working/work_s3"
    refs_dir = os.path.join(work_dir, "seg_refs")
    clips_dir = os.path.join(work_dir, "seg_clips")
    os.makedirs(refs_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)

    ensure_deps()
    import soundfile as sf

    video_path, src_audio_path = download_source_media(sync, JOB_ID, video_asset, work_dir)
    bgm_wav_path = sync.try_download_asset(JOB_ID, "no_vocals.wav", os.path.join(work_dir, "no_vocals.wav"))
    if not bgm_wav_path and os.path.exists("/kaggle/working/work/demucs_out/htdemucs/audio_16k/no_vocals.wav"):
        bgm_wav_path = "/kaggle/working/work/demucs_out/htdemucs/audio_16k/no_vocals.wav"

    speaker_fallback_refs: dict[str, str] = {}
    for line in dub_lines:
        spk = str(line.get("speaker_id") or "SPEAKER_00")
        if spk not in speaker_fallback_refs:
            ref_name = f"ref_speaker_{spk}.wav"
            local_ref = os.path.join(work_dir, ref_name)
            dl_ref = sync.try_download_asset(JOB_ID, ref_name, local_ref)
            if dl_ref:
                speaker_fallback_refs[spk] = dl_ref

    audio_samples = None
    audio_sr = 24000
    global_fallback_ref = ""
    if src_audio_path and os.path.exists(src_audio_path):
        try:
            samples, sr = sf.read(src_audio_path, dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=-1)
            audio_samples = samples
            audio_sr = int(sr)
            if len(audio_samples) >= audio_sr:
                gf_path = os.path.join(refs_dir, "global_fallback_ref.wav")
                sf.write(gf_path, audio_samples[: min(len(audio_samples), audio_sr * 4)], audio_sr)
                global_fallback_ref = gf_path
        except Exception as e:
            print(f"[WARN] Không đọc được âm thanh gốc để cắt mẫu theo phân đoạn: {e}")

    engine, engine_cls_name, sig_info, detected_sr = load_vieneu_v3_turbo()
    ref_code_cache: dict[str, object] = {}
    clip_map: dict[int, str] = {}
    enriched_lines: list[dict] = []
    cloned_count = 0
    non_empty_lines = 0
    first_fail_errors: list[str] = []

    try:
        for idx_l, line in enumerate(dub_lines):
            line_id = int(line.get("id", idx_l))
            start_t = float(line.get("start", 0.0))
            end_t = max(start_t + 0.25, float(line.get("end", start_t + 0.8)))
            target_dur = round(end_t - start_t, 2)
            spk_id = str(line.get("speaker_id") or "SPEAKER_00")
            gender = str(line.get("gender") or "female").lower()
            src_txt = (line.get("src_text") or "").strip()
            text_vi = (line.get("text_vi") or src_txt).strip()

            expr = detect_segment_expression(text_vi, target_dur)

            seg_ref_wav = os.path.join(refs_dir, f"ref_seg_{line_id:04d}.wav")
            chosen_ref = extract_segment_ref_clip(
                audio_samples=audio_samples,
                sr=audio_sr,
                start_t=start_t,
                end_t=end_t,
                out_wav_path=seg_ref_wav,
                fallback_ref_path=speaker_fallback_refs.get(spk_id, "") or global_fallback_ref,
            )

            out_clip_wav = os.path.join(clips_dir, f"dub_seg_{line_id:04d}.wav")
            ok_synth = False
            mode_used = "silence"

            if text_vi:
                non_empty_lines += 1
                ok_synth, errs = _synthesize_single_line_vieneu_v3(
                    engine=engine,
                    ref_code_cache=ref_code_cache,
                    text_vi=text_vi,
                    ref_audio_path=chosen_ref,
                    gender=gender,
                    speaker_id=spk_id,
                    tone=expr["tone"],
                    default_sr=detected_sr,
                    out_wav=out_clip_wav,
                )
                if ok_synth:
                    cloned_count += 1
                    mode_used = "vieneu_v3_turbo_spk_emb"
                elif not first_fail_errors and errs:
                    first_fail_errors = errs

            if not ok_synth:
                _generate_silence_clip(target_dur, out_clip_wav, sr=detected_sr)

            clip_map[line_id] = out_clip_wav
            enriched_lines.append({
                **line,
                "tone": expr["tone"],
                "seg_ref_audio": os.path.basename(chosen_ref) if chosen_ref else "",
                "tts_mode": mode_used,
                "dub_clip": os.path.basename(out_clip_wav),
            })
    finally:
        if engine is not None:
            del engine
        free_gpu()

    if non_empty_lines > 0 and cloned_count == 0:
        raise RuntimeError(
            f"VieNeu-TTS-V3-Turbo ({engine_cls_name}, sig={sig_info}) không sinh được audio nào (0/{non_empty_lines}).\n"
            + "\n".join(first_fail_errors[:10])
        )

    out_mix_wav = os.path.join(work_dir, f"{JOB_ID}_dubbed_mix.wav")
    out_srt_s3 = os.path.join(work_dir, f"{JOB_ID}_stage3.srt")
    build_final_dubbed_track(enriched_lines, clip_map, bgm_wav_path, out_mix_wav, out_srt_s3, target_sr=detected_sr)

    out_mp4 = os.path.join(work_dir, f"{JOB_ID}_dubbed.mp4")
    muxed_mp4 = mux_dubbed_video(video_path, out_mix_wav, out_mp4)

    mix_url = sync.upload_asset(JOB_ID, out_mix_wav)
    srt_url = sync.upload_asset(JOB_ID, out_srt_s3)
    mp4_url = sync.upload_asset(JOB_ID, muxed_mp4) if muxed_mp4 else ""

    stage3_payload = {
        "job_id": JOB_ID,
        "tts_engine": engine_cls_name,
        "sample_rate": detected_sr,
        "total_lines": len(enriched_lines),
        "vieneu_cloned_lines": cloned_count,
        "dubbed_mix_asset": os.path.basename(out_mix_wav),
        "dubbed_mix_url": mix_url,
        "stage3_srt_asset": os.path.basename(out_srt_s3),
        "stage3_srt_url": srt_url,
        "dubbed_video_asset": os.path.basename(muxed_mp4) if muxed_mp4 else "",
        "dubbed_video_url": mp4_url,
        "lines": enriched_lines,
    }
    with open(os.path.join(work_dir, f"{JOB_ID}_stage3.json"), "w", encoding="utf-8") as f:
        json.dump(stage3_payload, f, ensure_ascii=False, indent=2)

    sync.write_json(f"state/{JOB_ID}_stage3.json", stage3_payload, f"stage3 done (vieneu-v3-turbo): {JOB_ID}")

    updated_jobs, _ = sync.read_json("state/jobs.json")
    if JOB_ID in updated_jobs.get("jobs", {}):
        updated_jobs["jobs"][JOB_ID]["status"] = "DONE"
        if mp4_url:
            updated_jobs["jobs"][JOB_ID]["output_url"] = mp4_url
        sync.write_json("state/jobs.json", updated_jobs, f"update status DONE: {JOB_ID}")

    sync.trigger_dispatch("stage3_completed", {"job_id": JOB_ID})

    ntfy_lines = [
        f"Hoàn tất {cloned_count}/{len(enriched_lines)} phân đoạn bằng VieNeu-TTS-V3-Turbo ({engine_cls_name}, {detected_sr}Hz).",
    ]
    if mp4_url:
        ntfy_lines.append(f"Video lồng tiếng: {mp4_url}")
    else:
        ntfy_lines.append("Video lồng tiếng: không tạo được file MP4 (kiểm tra video nguồn).")
    if mix_url:
        ntfy_lines.append(f"Audio mix: {mix_url}")
    if srt_url:
        ntfy_lines.append(f"Phụ đề SRT: {srt_url}")

    send_ntfy(
        f"[Stage 3 Done] {JOB_ID}",
        "\n".join(ntfy_lines),
        tags="white_check_mark",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_ntfy(f"[Stage 3 Lỗi] {JOB_ID}", err, tags="x", priority="high")
        raise
