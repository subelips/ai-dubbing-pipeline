import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import asyncio
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
        ("edge_tts", "edge-tts"),
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
    audio_path = os.path.join(work_dir, "source_audio.wav")

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

    if os.path.exists(video_path) and os.path.getsize(video_path) > 1024:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "24000", audio_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return video_path, audio_path

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

    if natural_pace > 18.0:
        tempo_factor = min(1.28, round(natural_pace / 16.5, 2))
    elif natural_pace < 7.5 and duration_sec > 1.2:
        tempo_factor = max(0.85, round(natural_pace / 9.5, 2))
    else:
        tempo_factor = 1.0

    return {"tone": tone, "tempo_factor": tempo_factor}


def extract_segment_ref_clip(
    audio_samples,
    sr: int,
    start_t: float,
    end_t: float,
    out_wav_path: str,
    fallback_ref_path: str = "",
) -> str:
    import numpy as np
    import soundfile as sf

    if audio_samples is not None and len(audio_samples) > 0 and sr > 0:
        s_sec = max(0.0, float(start_t))
        e_sec = max(s_sec + 0.25, float(end_t))
        dur = e_sec - s_sec

        if dur < 0.90:
            pad = (0.95 - dur) * 0.5 + 0.15
            s_sec = max(0.0, s_sec - pad)
            e_sec = min(len(audio_samples) / float(sr), e_sec + pad)

        if (e_sec - s_sec) > 7.0:
            e_sec = s_sec + 7.0

        i0 = max(0, int(round(s_sec * sr)))
        i1 = min(len(audio_samples), int(round(e_sec * sr)))
        if i1 - i0 >= int(0.35 * sr):
            clip = audio_samples[i0:i1]
            rms = float(np.sqrt(np.mean(clip.astype(np.float32) ** 2)))
            if rms > 1e-4:
                sf.write(out_wav_path, clip, sr)
                return out_wav_path

    if fallback_ref_path and os.path.exists(fallback_ref_path) and os.path.getsize(fallback_ref_path) > 512:
        return fallback_ref_path
    return ""


def load_vieneu_engine():
    """Tự động quét và khởi tạo class/module của VieNeu-TTS trên mọi phiên bản cài đặt."""
    model_id = os.environ.get("VIENEU_MODEL", "pnnbao-ump/VieNeu-TTS").strip() or "pnnbao-ump/VieNeu-TTS"
    try:
        import torch
        import vieneu  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        modules_to_scan = [vieneu]
        if hasattr(vieneu, "__path__"):
            for _, modname, _ in pkgutil.walk_packages(vieneu.__path__, prefix="vieneu."):
                try:
                    modules_to_scan.append(importlib.import_module(modname))
                except Exception:
                    continue

        synth_methods = ("synthesize", "tts", "infer", "generate", "speak", "clone", "__call__")
        candidate_classes = []
        for mod in modules_to_scan:
            for attr_name in dir(mod):
                if attr_name.startswith("_"):
                    continue
                obj = getattr(mod, attr_name, None)
                if inspect.isclass(obj):
                    if any(callable(getattr(obj, m, None)) for m in synth_methods):
                        candidate_classes.append((f"{mod.__name__}.{attr_name}", obj))

        for full_name, cls in candidate_classes:
            for kw in (
                {"model_name": model_id, "device": device},
                {"model_id": model_id, "device": device},
                {"backbone_repo": model_id, "codec_repo": model_id},
                {"device": device},
                {},
            ):
                try:
                    eng = cls(**kw)
                    print(f"[INFO] Đã khởi tạo VieNeu-TTS qua {full_name} trên {device}.")
                    return eng
                except Exception:
                    continue

        # Trường hợp vieneu cung cấp hàm trực tiếp ở mức module
        for mod in modules_to_scan:
            for m in ("synthesize", "tts", "infer", "generate"):
                fn = getattr(mod, m, None)
                if callable(fn):
                    print(f"[INFO] Dùng hàm trực tiếp {mod.__name__}.{m} của VieNeu-TTS.")
                    return mod

        public_attrs = [a for a in dir(vieneu) if not a.startswith("_")]
        print(f"[WARN] Không khởi tạo được class trong vieneu (attrs={public_attrs}), chuyển sang giọng Việt neural fallback.")
        return None
    except Exception as e:
        print(f"[WARN] Lỗi nạp VieNeu-TTS ({e}), chuyển sang giọng Việt neural fallback.")
        return None


def _synthesize_single_line_vieneu(
    engine,
    text_vi: str,
    ref_audio_path: str,
    ref_text: str,
    expr: dict,
    target_dur: float,
    out_wav: str,
) -> bool:
    import numpy as np
    import soundfile as sf

    if engine is None:
        return False

    call_candidates = []
    for m_name in ("synthesize", "tts", "infer", "generate", "speak", "clone"):
        fn = getattr(engine, m_name, None)
        if callable(fn):
            call_candidates.extend([
                lambda f=fn: f(text=text_vi, ref_audio=ref_audio_path, ref_text=ref_text),
                lambda f=fn: f(text=text_vi, ref_audio_path=ref_audio_path, ref_text=ref_text),
                lambda f=fn: f(text=text_vi, prompt_speech=ref_audio_path, prompt_text=ref_text),
                lambda f=fn: f(text_vi, ref_audio_path, ref_text),
                lambda f=fn: f(text_vi, ref_audio_path),
                lambda f=fn: f(text=text_vi),
            ])
    if callable(engine):
        call_candidates.extend([
            lambda: engine(text=text_vi, ref_audio=ref_audio_path, ref_text=ref_text),
            lambda: engine(text_vi, ref_audio_path),
        ])

    for fn in call_candidates:
        try:
            out = fn()
            if out is None:
                continue
            if isinstance(out, str) and os.path.exists(out):
                data, sr = sf.read(out, dtype="float32")
                sf.write(out_wav, data, sr)
                return True
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                wav_arr, sr = out[0], int(out[1])
                arr_np = np.asarray(wav_arr, dtype=np.float32).squeeze()
                if arr_np.size > 0:
                    sf.write(out_wav, arr_np, sr)
                    return True
            arr_np = np.asarray(out, dtype=np.float32).squeeze()
            if arr_np.size > 0:
                sf.write(out_wav, arr_np, 24000)
                return True
        except Exception:
            continue

    return False


def _synthesize_vietnamese_neural_fallback(
    text_vi: str,
    gender: str,
    speaker_id: str,
    expr: dict,
    out_wav: str,
) -> bool:
    """Fallback dùng giọng tiếng Việt thật (Nam/Nữ + biến thể cao độ theo từng nhân vật & cảm xúc)."""
    try:
        import edge_tts  # type: ignore

        is_female = str(gender).lower() == "female" or str(speaker_id).endswith("_F")
        voice = "vi-VN-HoaiMyNeural" if is_female else "vi-VN-NamMinhNeural"

        tone = expr.get("tone", "neutral")
        # Phân biệt tông giọng giữa 2 nhân vật nam (SPEAKER_00_M vs SPEAKER_02_M) và sắc thái câu
        pitch_shift = "+0Hz"
        rate_shift = "+0%"
        if not is_female and "02" in str(speaker_id):
            pitch_shift = "-14Hz"
        if tone == "emphatic":
            pitch_shift = "+8Hz" if is_female else "+6Hz"
            rate_shift = "+6%"
        elif tone == "questioning":
            pitch_shift = "+6Hz"
        elif tone == "harsh_contempt":
            pitch_shift = "+4Hz" if is_female else "-8Hz"
            rate_shift = "+8%"
        elif tone == "warm_caring":
            pitch_shift = "-4Hz"
            rate_shift = "-4%"

        tmp_mp3 = out_wav + ".mp3"

        async def _run_edge():
            comm = edge_tts.Communicate(text=text_vi, voice=voice, rate=rate_shift, pitch=pitch_shift)
            await comm.save(tmp_mp3)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                new_loop = asyncio.new_event_loop()
                new_loop.run_until_complete(_run_edge())
                new_loop.close()
            else:
                loop.run_until_complete(_run_edge())
        except RuntimeError:
            asyncio.run(_run_edge())

        if os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 256:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp3, "-ac", "1", "-ar", "24000", out_wav],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.remove(tmp_mp3)
            return os.path.exists(out_wav) and os.path.getsize(out_wav) > 512
    except Exception as e:
        print(f"[WARN] Fallback edge-tts lỗi: {e}")
    return False


def _fit_audio_to_window(wav_path: str, target_dur: float, tempo_factor: float = 1.0):
    if not os.path.exists(wav_path) or target_dur <= 0.15:
        return
    try:
        import soundfile as sf
        data, sr = sf.read(wav_path, dtype="float32")
        if sr <= 0 or len(data) == 0:
            return
        cur_dur = len(data) / float(sr)
        if cur_dur > target_dur * 1.06 or abs(tempo_factor - 1.0) >= 0.05:
            ratio = max(0.75, min(1.55, (cur_dur / target_dur) * tempo_factor))
            filters = []
            rem = ratio
            while rem > 2.0:
                filters.append("atempo=2.0")
                rem /= 2.0
            while rem < 0.5:
                filters.append("atempo=0.5")
                rem /= 0.5
            filters.append(f"atempo={rem:.3f}")

            tmp_out = wav_path + ".fit.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-filter:a", ",".join(filters), tmp_out],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 512:
                os.replace(tmp_out, wav_path)
    except Exception as e:
        print(f"[WARN] Không căn chỉnh được độ dài clip {wav_path}: {e}")


def _generate_silence_clip(target_dur: float, out_wav: str):
    import numpy as np
    import soundfile as sf

    sr = 24000
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
) -> str:
    import numpy as np
    import soundfile as sf

    sr = 24000
    max_end_sec = max([float(x.get("end", 0.0)) for x in dub_lines] + [5.0])
    total_samples = int(round((max_end_sec + 1.0) * sr))
    voice_track = np.zeros(total_samples, dtype=np.float32)

    srt_blocks = []
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

            i0 = int(round(start_t * sr))
            i1 = min(total_samples, i0 + len(seg_wav))
            if i1 > i0:
                voice_track[i0:i1] += seg_wav[: (i1 - i0)]

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

        final_mix = np.clip(voice_track + 0.55 * bgm_wav, -1.0, 1.0)
    else:
        final_mix = np.clip(voice_track, -1.0, 1.0)

    sf.write(out_mix_wav, final_mix, sr)
    with open(out_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_blocks).strip() + ("\n" if srt_blocks else ""))

    return out_mix_wav


def mux_dubbed_video(video_path: str, dubbed_wav: str, out_mp4: str) -> str:
    if not video_path or not os.path.exists(video_path) or not dubbed_wav or not os.path.exists(dubbed_wav):
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
    return out_mp4 if res.returncode == 0 and os.path.exists(out_mp4) else ""


sync = GitHubSync(GH_REPO, GH_TOKEN)


def main():
    send_ntfy(
        f"[Stage 3] {JOB_ID}",
        f"Bắt đầu tổng hợp & clone từng phân đoạn bằng VieNeu-TTS cho job {JOB_ID}.",
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
    if src_audio_path and os.path.exists(src_audio_path):
        try:
            samples, sr = sf.read(src_audio_path, dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=-1)
            audio_samples = samples
            audio_sr = int(sr)
        except Exception as e:
            print(f"[WARN] Không đọc được âm thanh gốc để cắt mẫu theo phân đoạn: {e}")

    engine = load_vieneu_engine()
    clip_map: dict[int, str] = {}
    enriched_lines: list[dict] = []
    cloned_count = 0
    neural_fb_count = 0

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
                fallback_ref_path=speaker_fallback_refs.get(spk_id, ""),
            )

            out_clip_wav = os.path.join(clips_dir, f"dub_seg_{line_id:04d}.wav")
            ok_synth = False
            mode_used = "silence"

            if text_vi:
                ok_synth = _synthesize_single_line_vieneu(
                    engine=engine,
                    text_vi=text_vi,
                    ref_audio_path=chosen_ref,
                    ref_text=src_txt,
                    expr=expr,
                    target_dur=target_dur,
                    out_wav=out_clip_wav,
                )
                if ok_synth:
                    cloned_count += 1
                    mode_used = "vieneu_per_segment"
                else:
                    ok_fb = _synthesize_vietnamese_neural_fallback(
                        text_vi=text_vi,
                        gender=gender,
                        speaker_id=spk_id,
                        expr=expr,
                        out_wav=out_clip_wav,
                    )
                    if ok_fb:
                        ok_synth = True
                        neural_fb_count += 1
                        mode_used = "vietnamese_neural_fallback"

            if ok_synth:
                _fit_audio_to_window(out_clip_wav, target_dur, expr["tempo_factor"])
            else:
                _generate_silence_clip(target_dur, out_clip_wav)

            clip_map[line_id] = out_clip_wav
            enriched_lines.append({
                **line,
                "tone": expr["tone"],
                "tempo_factor": expr["tempo_factor"],
                "seg_ref_audio": os.path.basename(chosen_ref) if chosen_ref else "",
                "tts_mode": mode_used,
                "dub_clip": os.path.basename(out_clip_wav),
            })
    finally:
        if engine is not None:
            del engine
        free_gpu()

    out_mix_wav = os.path.join(work_dir, f"{JOB_ID}_dubbed_mix.wav")
    out_srt_s3 = os.path.join(work_dir, f"{JOB_ID}_stage3.srt")
    build_final_dubbed_track(enriched_lines, clip_map, bgm_wav_path, out_mix_wav, out_srt_s3)

    out_mp4 = os.path.join(work_dir, f"{JOB_ID}_dubbed.mp4")
    muxed_mp4 = mux_dubbed_video(video_path, out_mix_wav, out_mp4)

    mix_url = sync.upload_asset(JOB_ID, out_mix_wav)
    srt_url = sync.upload_asset(JOB_ID, out_srt_s3)
    mp4_url = sync.upload_asset(JOB_ID, muxed_mp4) if muxed_mp4 else ""

    stage3_payload = {
        "job_id": JOB_ID,
        "tts_engine": "vieneu_per_segment",
        "total_lines": len(enriched_lines),
        "vieneu_cloned_lines": cloned_count,
        "neural_fallback_lines": neural_fb_count,
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

    sync.write_json(f"state/{JOB_ID}_stage3.json", stage3_payload, f"stage3 done: {JOB_ID}")

    updated_jobs, _ = sync.read_json("state/jobs.json")
    if JOB_ID in updated_jobs.get("jobs", {}):
        updated_jobs["jobs"][JOB_ID]["status"] = "DONE"
        sync.write_json("state/jobs.json", updated_jobs, f"update status DONE: {JOB_ID}")

    sync.trigger_dispatch("stage3_completed", {"job_id": JOB_ID})
    send_ntfy(
        f"[Stage 3 Done] {JOB_ID}",
        f"Hoàn tất {len(enriched_lines)} phân đoạn (VieNeu clone: {cloned_count}, Neural VN: {neural_fb_count}).",
        tags="white_check_mark",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_ntfy(f"[Stage 3 Lỗi] {JOB_ID}", err, tags="x", priority="high")
        raise
