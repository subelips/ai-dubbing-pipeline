import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import base64
from concurrent.futures import ThreadPoolExecutor
import ctypes
from difflib import SequenceMatcher
import gc
import glob
import json
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


def send_ntfy(title: str, message: str, tags: str = "film_frames", priority: str = "default"):
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


def _preload_cuda12_libs():
    try:
        import nvidia  # type: ignore
        nvidia_roots = list(getattr(nvidia, "__path__", []))
    except Exception:
        nvidia_roots = []

    search_dirs = []
    for root in nvidia_roots:
        search_dirs.extend(glob.glob(os.path.join(root, "*", "lib")))
    for sp in sys.path:
        if sp and os.path.isdir(sp):
            search_dirs.extend(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))

    patterns = ["libcublasLt.so.12*", "libcublas.so.12*", "libcudnn*.so.9*"]
    loaded_dirs = []
    for d in dict.fromkeys(search_dirs):
        if not os.path.isdir(d):
            continue
        loaded_dirs.append(d)
        for pat in patterns:
            for so_path in sorted(glob.glob(os.path.join(d, pat))):
                try:
                    ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

    if loaded_dirs:
        cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
        extra = ":".join(loaded_dirs)
        os.environ["LD_LIBRARY_PATH"] = f"{extra}:{cur_ld}".strip(":")


def ensure_deps():
    wanted = [
        ("faster_whisper", "faster-whisper"),
        ("demucs", "demucs"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("gdown", "gdown"),
        ("paddle", "paddlepaddle"),
        ("paddleocr", "paddleocr"),
    ]
    missing = [pip_name for mod_name, pip_name in wanted if not _is_installed(mod_name)]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)

    try:
        ctypes.CDLL("libcublas.so.12")
    except OSError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"],
            check=False,
        )
    _preload_cuda12_libs()


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
            direct_url = _normalize_gdrive_url(ref)
            subprocess.run([sys.executable, "-m", "gdown", direct_url, "-O", dest_path], check=True)
        else:
            subprocess.run(["curl", "-fSL", ref, "-o", dest_path], check=True)
    else:
        sync.download_asset(tag, ref, dest_path)


def extract_audio_16k(video_path: str, out_wav: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", out_wav],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def separate_vocals_and_bgm(audio_16k: str, work_dir: str) -> tuple[str, str]:
    demucs_out = os.path.join(work_dir, "demucs_out")
    os.makedirs(demucs_out, exist_ok=True)
    cmd = ["demucs", "-n", "htdemucs", "--two-stems", "vocals", "-o", demucs_out, audio_16k]
    import torch
    if not torch.cuda.is_available():
        cmd.extend(["-d", "cpu"])
    subprocess.run(cmd, check=True)
    free_gpu()

    stem_dir = os.path.join(demucs_out, "htdemucs", os.path.splitext(os.path.basename(audio_16k))[0])
    vocals_wav = os.path.join(stem_dir, "vocals.wav")
    bgm_wav = os.path.join(stem_dir, "no_vocals.wav")
    if not os.path.exists(vocals_wav) or not os.path.exists(bgm_wav):
        raise FileNotFoundError("Demucs không tạo đủ vocals.wav và no_vocals.wav")
    return vocals_wav, bgm_wav


def _split_and_refine_segments(raw_segments, audio_arr, sr: int) -> list[dict]:
    import numpy as np

    # Chỉ tách câu tại dấu ngắt câu hoặc khoảng lặng thực sự >= 0.55s để không chẻ vụn từ tiếng Trung
    atomic_items: list[tuple[float, float, str]] = []
    punct_chars = set(",.?!;，。？！；、")

    for seg in raw_segments:
        txt = (seg.text or "").strip()
        if not txt:
            continue

        words = getattr(seg, "words", None) or []
        valid_words = [
            w for w in words
            if getattr(w, "start", None) is not None
            and getattr(w, "end", None) is not None
            and (getattr(w, "word", "") or "").strip()
        ]

        if not valid_words:
            atomic_items.append((float(seg.start), float(seg.end), txt))
            continue

        cur_words = [valid_words[0]]
        for w in valid_words[1:]:
            prev_w = cur_words[-1]
            prev_str = (prev_w.word or "").strip()
            gap = float(w.start) - float(prev_w.end)
            ends_with_punct = bool(prev_str) and (prev_str[-1] in punct_chars)
            cur_chars = len(re.sub(r"[\s\W_]+", "", "".join(x.word for x in cur_words)))

            if (ends_with_punct and cur_chars >= 2) or (gap >= 0.55 and cur_chars >= 2):
                clause_txt = "".join(x.word for x in cur_words).strip(" ,.?!;，。？！；、")
                if clause_txt:
                    atomic_items.append((float(cur_words[0].start), float(prev_w.end), clause_txt))
                cur_words = [w]
            else:
                cur_words.append(w)

        if cur_words:
            clause_txt = "".join(x.word for x in cur_words).strip(" ,.?!;，。？！；、")
            if clause_txt:
                atomic_items.append((float(cur_words[0].start), float(cur_words[-1].end), clause_txt))

    # Gộp các mảnh đơn âm tiết (1 ký tự) bị tách lẻ vào câu liền kề nếu khoảng cách < 0.6s
    merged_items: list[tuple[float, float, str]] = []
    i = 0
    while i < len(atomic_items):
        s_t, e_t, txt = atomic_items[i]
        clean_len = len(re.sub(r"[\s\W_]+", "", txt))
        if clean_len <= 1 and i + 1 < len(atomic_items):
            ns_t, ne_t, ntxt = atomic_items[i + 1]
            if (ns_t - e_t) < 0.65:
                merged_items.append((s_t, ne_t, f"{txt}{ntxt}"))
                i += 2
                continue
        if clean_len <= 1 and merged_items:
            ps_t, pe_t, ptxt = merged_items[-1]
            if (s_t - pe_t) < 0.65:
                merged_items[-1] = (ps_t, e_t, f"{ptxt}{txt}")
                i += 1
                continue
        merged_items.append((s_t, e_t, txt))
        i += 1

    segments = []
    for idx, (raw_s, raw_e, txt) in enumerate(merged_items):
        s_t = float(raw_s)
        e_t = float(raw_e)

        # Cắt bỏ khoảng lặng đầu/cuối dựa trên năng lượng thực tế của vocals
        i0 = max(0, int(s_t * sr))
        i1 = min(len(audio_arr), int(e_t * sr))
        if i1 - i0 > int(0.30 * sr):
            win = max(1, int(0.015 * sr))
            chunk = np.abs(audio_arr[i0:i1])
            peak = float(np.max(chunk)) if len(chunk) > 0 else 0.0
            if peak > 1e-4:
                thresh = max(1e-4, peak * 0.06)
                active_idx = np.where(chunk >= thresh)[0]
                if len(active_idx) > 0:
                    trim_s = s_t + max(0.0, (active_idx[0] - win) / float(sr))
                    trim_e = s_t + min((i1 - i0) / float(sr), (active_idx[-1] + win) / float(sr))
                    if trim_e - trim_s >= 0.25:
                        s_t, e_t = trim_s, trim_e

        char_len = max(1, len(re.sub(r"\s+", "", txt)))
        max_plausible_dur = max(1.4, char_len * 0.42 + 0.5)
        if (e_t - s_t) > max_plausible_dur:
            e_t = s_t + max_plausible_dur

        segments.append({
            "id": idx,
            "start": round(s_t, 2),
            "end": round(max(s_t + 0.25, e_t), 2),
            "speaker_id": "SPEAKER_00",
            "gender_hint": "unknown",
            "pitch_hz": None,
            "asr_raw": txt,
            "asr_text": txt,
            "ocr_text": "",
            "best_text": txt,
            "subtitle_source": "asr_only",
        })

    for k in range(len(segments) - 1):
        cur_end = segments[k]["end"]
        nxt_start = segments[k + 1]["start"]
        if cur_end > nxt_start - 0.04:
            segments[k]["end"] = round(max(segments[k]["start"] + 0.22, nxt_start - 0.04), 2)

    return segments


def transcribe_with_faster_whisper(vocals_wav: str) -> list[dict]:
    import numpy as np
    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel

    _preload_cuda12_libs()

    audio_arr, sr = sf.read(vocals_wav, dtype="float32")
    if audio_arr.ndim > 1:
        audio_arr = audio_arr.mean(axis=-1)
    if sr != 16000:
        wav_16k = vocals_wav + "_16k.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", vocals_wav, "-ac", "1", "-ar", "16000", wav_16k],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        audio_arr, _ = sf.read(wav_16k, dtype="float32")
        if audio_arr.ndim > 1:
            audio_arr = audio_arr.mean(axis=-1)

    audio_contiguous = np.ascontiguousarray(audio_arr, dtype=np.float32)
    model_size = os.environ.get("WHISPER_MODEL", "large-v3").strip() or "large-v3"
    lang = os.environ.get("SOURCE_LANG", "zh").strip() or None

    candidates = [("cuda", "float16"), ("cpu", "int8")] if torch.cuda.is_available() else [("cpu", "int8")]
    segments = []
    last_err = None

    for device, compute_type in candidates:
        model = None
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            raw_segments, _ = model.transcribe(
                audio_contiguous,
                language=lang,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=220, speech_pad_ms=70),
            )
            segments = _split_and_refine_segments(list(raw_segments), audio_contiguous, 16000)
            print(f"[INFO] Faster-Whisper ({model_size}, {device}/{compute_type}) nhận diện {len(segments)} câu thoại trọn ý.")
            break
        except Exception as e:
            last_err = e
            print(f"[WARN] Faster-Whisper trên {device}/{compute_type} lỗi ({e}), thử chế độ dự phòng...")
        finally:
            if model is not None:
                del model
            free_gpu()

    if not segments and last_err is not None:
        raise RuntimeError(f"Faster-Whisper thất bại hoàn toàn: {last_err}")

    return segments


def _extract_voice_features(samples, sr: int) -> tuple[float | None, "np.ndarray | None"]:
    import numpy as np

    if samples is None or len(samples) < max(64, sr // 8):
        return None, None
    s = samples.astype(np.float32)
    s = s - np.mean(s)
    energy = float(np.sqrt(np.mean(s ** 2)))
    if energy < 1e-4:
        return None, None

    frame_len = max(128, int(0.04 * sr))
    hop = max(64, frame_len // 2)
    min_lag = max(1, int(sr / 450.0))
    max_lag = min(frame_len - 1, int(sr / 75.0))
    frame_pitches = []
    spectra = []

    for start in range(0, len(s) - frame_len + 1, hop):
        fr = s[start: start + frame_len]
        rms = float(np.sqrt(np.mean(fr ** 2)))
        if rms < energy * 0.35:
            continue
        win_fr = fr * np.hanning(frame_len)
        spec = np.abs(np.fft.rfft(win_fr, n=512))
        log_spec = np.log1p(spec[:160])
        norm = float(np.linalg.norm(log_spec))
        if norm > 1e-6:
            spectra.append(log_spec / norm)

        corr = np.correlate(fr, fr, mode="full")[frame_len - 1:]
        if corr[0] > 0 and max_lag > min_lag:
            sub_c = corr[min_lag:max_lag]
            b_idx = int(np.argmax(sub_c)) + min_lag
            if corr[b_idx] / corr[0] >= 0.28:
                frame_pitches.append(float(sr) / float(b_idx))

    pitch_med = round(float(np.median(frame_pitches)), 1) if frame_pitches else None
    emb = np.mean(spectra, axis=0) if spectra else None
    if emb is not None:
        n_emb = float(np.linalg.norm(emb))
        if n_emb > 1e-6:
            emb = emb / n_emb
    return pitch_med, emb


def apply_speaker_diarization(vocals_wav: str, segments: list[dict], work_dir: str) -> dict:
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    import soundfile as sf

    wav_data, sr = sf.read(vocals_wav)
    if wav_data.ndim > 1:
        wav_data = wav_data.mean(axis=-1)

    embeddings: list["np.ndarray | None"] = []
    for seg in segments:
        i0 = max(0, int(float(seg["start"]) * sr))
        i1 = min(len(wav_data), int(float(seg["end"]) * sr))
        pitch, emb = _extract_voice_features(wav_data[i0:i1], sr) if i1 > i0 else (None, None)
        seg["pitch_hz"] = pitch
        embeddings.append(emb)
        if pitch is not None:
            seg["gender_hint"] = "female" if pitch >= 195.0 else "male"
        else:
            seg["gender_hint"] = "unknown"

    last_known_g = "male"
    for seg in segments:
        if seg["gender_hint"] in ("male", "female"):
            last_known_g = seg["gender_hint"]
        else:
            seg["gender_hint"] = last_known_g

    # Làm mượt giới tính cho câu ngắn nằm giữa 2 câu cùng giới tính
    if len(segments) >= 3:
        smoothed_g = [s["gender_hint"] for s in segments]
        for i in range(1, len(segments) - 1):
            if segments[i - 1]["gender_hint"] == segments[i + 1]["gender_hint"]:
                dur_i = float(segments[i]["end"]) - float(segments[i]["start"])
                if dur_i < 1.1 and segments[i]["gender_hint"] != segments[i - 1]["gender_hint"]:
                    smoothed_g[i] = segments[i - 1]["gender_hint"]
        for i, g in enumerate(smoothed_g):
            segments[i]["gender_hint"] = g

    male_valid = [(i, embeddings[i]) for i, s in enumerate(segments) if s["gender_hint"] == "male" and embeddings[i] is not None]
    male_cluster_map: dict[int, int] = {}
    if len(male_valid) >= 4:
        mat = np.vstack([e for _, e in male_valid])
        Z = linkage(mat, method="ward")
        labels = fcluster(Z, t=2, criterion="maxclust")
        first_seen: dict[int, int] = {}
        for (seg_idx, _), lbl in zip(male_valid, labels):
            lbl_int = int(lbl)
            if lbl_int not in first_seen:
                first_seen[lbl_int] = seg_idx
        ordered_lbls = sorted(first_seen.keys(), key=lambda k: first_seen[k])
        remap = {old_l: new_i for new_i, old_l in enumerate(ordered_lbls)}
        for (seg_idx, _), lbl in zip(male_valid, labels):
            male_cluster_map[seg_idx] = remap[int(lbl)]

    last_male_spk = "SPEAKER_00_M"
    for i, seg in enumerate(segments):
        if seg["gender_hint"] == "female":
            seg["speaker_id"] = "SPEAKER_01_F"
        else:
            if i in male_cluster_map:
                spk = "SPEAKER_00_M" if male_cluster_map[i] == 0 else "SPEAKER_02_M"
                last_male_spk = spk
            else:
                spk = last_male_spk
            seg["speaker_id"] = spk

    speaker_refs = {}
    spk_groups: dict[str, list[tuple[float, float]]] = {}
    for seg in segments:
        spk = seg["speaker_id"]
        spk_groups.setdefault(spk, []).append((float(seg["start"]), float(seg["end"])))

    for spk, intervals in spk_groups.items():
        chunks = []
        total_len = 0
        for t0, t1 in sorted(intervals, key=lambda x: x[1] - x[0], reverse=True):
            a0 = max(0, int(t0 * sr))
            a1 = min(len(wav_data), int(t1 * sr))
            if a1 > a0:
                chunks.append(wav_data[a0:a1])
                total_len += a1 - a0
            if total_len >= sr * 5:
                break
        ref_name = f"ref_speaker_{spk}.wav"
        ref_path = os.path.join(work_dir, ref_name)
        merged_wav = np.concatenate(chunks)[: sr * 6] if chunks else np.zeros(sr, dtype=np.float32)
        sf.write(ref_path, merged_wav, sr)

        pitches = [float(s["pitch_hz"]) for s in segments if s["speaker_id"] == spk and s.get("pitch_hz")]
        avg_pitch = round(float(np.mean(pitches)), 1) if pitches else None
        g_hint = "female" if spk.endswith("_F") else "male"
        speaker_refs[spk] = {
            "ref_asset": ref_name,
            "gender_hint": g_hint,
            "pitch_hz": avg_pitch,
        }

    print(f"[INFO] Phân tách người nói tạo {len(speaker_refs)} nhân vật: {list(speaker_refs.keys())}")
    return speaker_refs


def _extract_ocr_texts(ocr_result) -> list[str]:
    texts = []
    if not ocr_result:
        return texts
    for item in ocr_result:
        if not isinstance(item, (dict, list, tuple)):
            try:
                if hasattr(item, "res") and isinstance(item.res, dict):
                    item = item.res
                elif hasattr(item, "json") and isinstance(item.json, dict):
                    item = item.json.get("res", item.json)
                else:
                    item = dict(item)
            except Exception:
                pass

        if isinstance(item, dict):
            rec_texts = item.get("rec_texts") or []
            for t in rec_texts:
                if str(t).strip():
                    texts.append(str(t).strip())
        elif isinstance(item, (list, tuple)):
            for entry in item:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    info = entry[1]
                    if isinstance(info, (list, tuple)) and info and str(info[0]).strip():
                        texts.append(str(info[0]).strip())
    return texts


def _create_single_ocr_engine(lang: str):
    from paddleocr import PaddleOCR

    for kw in [
        dict(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False),
        dict(lang=lang),
    ]:
        try:
            return PaddleOCR(**kw)
        except TypeError:
            continue
    return PaddleOCR(lang=lang)


def _run_ocr_predict(ocr_engine, img) -> list[str]:
    if ocr_engine is None:
        return []
    try:
        if hasattr(ocr_engine, "predict"):
            res_p = list(ocr_engine.predict(img))
            got_p = _extract_ocr_texts(res_p)
            if got_p:
                return got_p
    except Exception:
        pass
    try:
        res = ocr_engine.ocr(img)
        got = _extract_ocr_texts(res)
        if got:
            return got
    except Exception:
        pass
    return []


def _prep_ocr_variants(crop):
    import cv2

    variants = [crop]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    boosted = cv2.convertScaleAbs(gray, alpha=1.6, beta=15)
    variants.append(cv2.cvtColor(boosted, cv2.COLOR_GRAY2BGR))
    # Lọc chữ phụ đề trắng sáng nổi bật
    _, white_mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    variants.append(cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR))
    return variants


def run_ocr_on_video(video_path: str, work_dir: str) -> tuple[dict, list[dict], object]:
    roi_candidates = [
        {"x1": 0.02, "y1": 0.50, "x2": 0.98, "y2": 0.95},
        {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
    ]
    try:
        import cv2

        forced_lang = os.environ.get("OCR_LANG", "").strip()
        if forced_lang:
            ocr_lang = forced_lang
        else:
            src_lang = os.environ.get("SOURCE_LANG", "zh").strip().lower()
            lang_map = {"zh": "ch", "cn": "ch", "ch": "ch", "ja": "japan", "jp": "japan", "ko": "korean", "kr": "korean", "vi": "vi", "en": "en"}
            ocr_lang = lang_map.get(src_lang, "ch")

        ocr = _create_single_ocr_engine(ocr_lang)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 60.0

        sample_times = [round(duration * k / 7.0, 2) for k in range(1, 7)]
        sampled_frames = []
        for ts in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                sampled_frames.append((ts, frame))
        cap.release()

        best_roi = roi_candidates[0]
        best_hits = -1
        best_texts: list[dict] = []

        for roi in roi_candidates:
            hits = 0
            screen_texts = []
            for ts, frame in sampled_frames:
                h, w, _ = frame.shape
                crop = frame[int(h * roi["y1"]):int(h * roi["y2"]), int(w * roi["x1"]):int(w * roi["x2"])]
                if crop.size == 0:
                    continue
                found = []
                for img_var in _prep_ocr_variants(crop):
                    found = _run_ocr_predict(ocr, img_var)
                    if found:
                        break
                if found:
                    hits += 1
                    screen_texts.append({"time": ts, "texts": found})

            if hits > best_hits:
                best_hits = hits
                best_roi = roi
                best_texts = screen_texts
            if hits >= 2:
                break

        print(f"[INFO] [PaddleOCR] lang={ocr_lang} | ROI {tuple(best_roi.values())} | hits={max(0, best_hits)}/{len(sampled_frames)} mẫu.")
        return best_roi, best_texts, ocr
    except Exception as e:
        print(f"[WARN] PaddleOCR bỏ qua: {e}")
        return roi_candidates[0], [], None


def _normalize_sub_text(txt: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(txt or ""), flags=re.UNICODE).lower()


def align_ocr_with_segments(video_path: str, roi: dict, ocr_engine, segments: list[dict]):
    if ocr_engine is None or not segments:
        return
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        corrected = 0
        ocr_enriched = 0

        for seg in segments:
            start_t = float(seg.get("start", 0.0))
            end_t = float(seg.get("end", start_t + 0.5))
            if end_t <= start_t:
                end_t = start_t + 0.5

            sample_ts = [
                start_t + 0.25 * (end_t - start_t),
                start_t + 0.50 * (end_t - start_t),
                start_t + 0.75 * (end_t - start_t),
            ]

            line_counts: dict[str, int] = {}
            line_order: list[str] = []
            for ts in sample_ts:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, ts) * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                h, w, _ = frame.shape
                crop = frame[int(h * roi["y1"]):int(h * roi["y2"]), int(w * roi["x1"]):int(w * roi["x2"])]
                if crop.size == 0:
                    continue
                found = []
                for img_var in _prep_ocr_variants(crop):
                    found = _run_ocr_predict(ocr_engine, img_var)
                    if found:
                        break

                for t in found:
                    t_clean = " ".join(str(t).split()).strip()
                    if len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7afA-Za-zÀ-ỹ0-9]", t_clean)) >= 2:
                        if t_clean not in line_counts:
                            line_order.append(t_clean)
                            line_counts[t_clean] = 0
                        line_counts[t_clean] += 1

            if not line_order:
                continue

            stable_lines = [ln for ln in line_order if line_counts[ln] >= 2]
            chosen_lines = stable_lines if stable_lines else [max(line_order, key=len)]
            ocr_joined = " ".join(chosen_lines).strip()
            seg["ocr_text"] = ocr_joined
            ocr_enriched += 1

            asr_orig = (seg.get("asr_text") or "").strip()
            norm_asr = _normalize_sub_text(asr_orig)
            norm_ocr = _normalize_sub_text(ocr_joined)

            if norm_ocr and len(norm_ocr) >= 2:
                ratio = SequenceMatcher(None, norm_asr, norm_ocr).ratio() if norm_asr else 1.0
                len_ok = (len(norm_ocr) >= max(2, int(len(norm_asr) * 0.55))) and (
                    len(norm_ocr) <= max(8, int(len(norm_asr) * 1.8))
                )
                if len_ok and (ratio >= 0.35 or not norm_asr):
                    if "asr_raw" not in seg:
                        seg["asr_raw"] = asr_orig
                    seg["asr_text"] = ocr_joined
                    seg["best_text"] = ocr_joined
                    seg["subtitle_source"] = "ocr_corrected"
                    corrected += 1
                else:
                    seg["best_text"] = asr_orig or ocr_joined
                    seg["subtitle_source"] = "asr_with_ocr_hint"

        cap.release()
        print(
            f"[INFO] Sửa ASR theo OCR: {corrected}/{len(segments)} câu ghi đè trực tiếp, "
            f"{ocr_enriched}/{len(segments)} câu có dữ liệu OCR."
        )
    except Exception as e:
        print(f"[WARN] Sửa ASR theo OCR bỏ qua: {e}")


def _format_srt_time(seconds: float) -> str:
    ms_total = max(0, int(round(float(seconds or 0.0) * 1000.0)))
    hours = ms_total // 3_600_000
    rem = ms_total % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    secs = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def export_stage1_srt(segments: list[dict], out_srt_path: str) -> str:
    blocks = []
    idx_out = 0
    for seg in segments:
        txt = (seg.get("best_text") or seg.get("asr_text") or seg.get("ocr_text") or "").strip()
        if not txt:
            continue
        idx_out += 1
        start_t = float(seg.get("start", 0.0))
        end_t = max(start_t + 0.25, float(seg.get("end", start_t + 0.8)))
        spk = seg.get("speaker_id", "SPEAKER_00")
        src_tag = seg.get("subtitle_source", "asr_only")
        blocks.append(
            f"{idx_out}\n"
            f"{_format_srt_time(start_t)} --> {_format_srt_time(end_t)}\n"
            f"[{spk} | {src_tag}] {txt}\n"
        )
    with open(out_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks).strip() + ("\n" if blocks else ""))
    print(f"[INFO] Đã xuất file kiểm định timestamp Stage 1: {out_srt_path} ({idx_out} dòng)")
    return out_srt_path


sync = GitHubSync(GH_REPO, GH_TOKEN)


def main():
    send_ntfy(f"[Stage 1] {JOB_ID}", f"Bắt đầu nhận diện (ASR + Diarization + OCR + Demucs) cho job {JOB_ID}.", tags="film_frames")
    jobs_data, _ = sync.read_json("state/jobs.json")
    job_info = (jobs_data.get("jobs") or {}).get(JOB_ID) or {}
    video_asset = (job_info.get("video_asset") or job_info.get("video_url") or "source_input.mp4").strip()

    work_dir = "/kaggle/working/work"
    os.makedirs(work_dir, exist_ok=True)
    video_path = os.path.join(work_dir, "input.mp4")
    audio_16k = os.path.join(work_dir, "audio_16k.wav")

    ensure_deps()
    download_video_source(sync, JOB_ID, video_asset, video_path)
    extract_audio_16k(video_path, audio_16k)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_demucs = pool.submit(separate_vocals_and_bgm, audio_16k, work_dir)
        fut_ocr = pool.submit(run_ocr_on_video, video_path, work_dir)
        vocals_wav, bgm_wav = fut_demucs.result()
        subtitle_roi, screen_texts, ocr_engine = fut_ocr.result()

    segments = transcribe_with_faster_whisper(vocals_wav)
    align_ocr_with_segments(video_path, subtitle_roi, ocr_engine, segments)
    speaker_refs = apply_speaker_diarization(vocals_wav, segments, work_dir)

    srt_s1_path = os.path.join(work_dir, f"{JOB_ID}_stage1.srt")
    export_stage1_srt(segments, srt_s1_path)
    sync.upload_asset(JOB_ID, srt_s1_path)

    sync.upload_asset(JOB_ID, bgm_wav)
    for ref_info in speaker_refs.values():
        ref_local = os.path.join(work_dir, ref_info["ref_asset"])
        if os.path.exists(ref_local):
            sync.upload_asset(JOB_ID, ref_local)

    stage1_payload = {
        "job_id": JOB_ID,
        "video_asset": video_asset,
        "subtitle_roi": subtitle_roi,
        "screen_texts": screen_texts,
        "scene_context": "",
        "speaker_refs": speaker_refs,
        "segments": segments,
    }
    with open(os.path.join(work_dir, f"{JOB_ID}_stage1.json"), "w", encoding="utf-8") as f:
        json.dump(stage1_payload, f, ensure_ascii=False, indent=2)

    sync.write_json(f"state/{JOB_ID}_stage1.json", stage1_payload, f"stage1 done: {JOB_ID}")

    updated_jobs, _ = sync.read_json("state/jobs.json")
    if JOB_ID in updated_jobs.get("jobs", {}):
        updated_jobs["jobs"][JOB_ID]["status"] = "STAGE1_DONE"
        sync.write_json("state/jobs.json", updated_jobs, f"update status STAGE1_DONE: {JOB_ID}")

    sync.trigger_dispatch("stage1_completed", {"job_id": JOB_ID})
    send_ntfy(
        f"[Stage 1 Done] {JOB_ID}",
        f"Hoàn tất {len(segments)} đoạn thoại, {len(speaker_refs)} người nói -> sang Stage 2.",
        tags="white_check_mark",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_ntfy(f"[Stage 1 Lỗi] {JOB_ID}", err, tags="x", priority="high")
        raise
