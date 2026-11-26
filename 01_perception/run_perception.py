import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import base64
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import re
import subprocess
import sys
import threading
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


def transcribe_with_faster_whisper(vocals_wav: str) -> list[dict]:
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model_size = os.environ.get("WHISPER_MODEL", "large-v3").strip() or "large-v3"
    lang = os.environ.get("SOURCE_LANG", "zh").strip() or None

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_segments, _ = model.transcribe(
        vocals_wav,
        language=lang,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=250),
    )

    segments = []
    for idx, seg in enumerate(raw_segments):
        txt = (seg.text or "").strip()
        if not txt:
            continue
        segments.append({
            "id": idx,
            "start": round(float(seg.start), 2),
            "end": round(float(seg.end), 2),
            "speaker_id": "SPEAKER_00",
            "gender_hint": "unknown",
            "pitch_hz": None,
            "asr_text": txt,
            "ocr_text": "",
            "best_text": txt,
            "subtitle_source": "asr_only",
        })

    del model
    free_gpu()
    return segments


def _estimate_pitch_hz(samples, sr: int) -> float | None:
    import numpy as np

    if samples is None or len(samples) < sr // 4:
        return None
    s = samples.astype(np.float32)
    s = s - np.mean(s)
    energy = float(np.sqrt(np.mean(s ** 2)))
    if energy < 1e-4:
        return None

    corr = np.correlate(s, s, mode="full")[len(s) - 1:]
    min_lag = max(1, int(sr / 450.0))
    max_lag = min(len(corr) - 1, int(sr / 65.0))
    if max_lag <= min_lag:
        return None
    window = corr[min_lag:max_lag]
    best_idx = int(np.argmax(window)) + min_lag
    if corr[0] <= 0 or corr[best_idx] / corr[0] < 0.25:
        return None
    return round(float(sr) / float(best_idx), 1)


def apply_speaker_diarization(audio_16k: str, segments: list[dict], work_dir: str) -> dict:
    import numpy as np
    import soundfile as sf

    wav_data, sr = sf.read(audio_16k)
    if wav_data.ndim > 1:
        wav_data = wav_data.mean(axis=-1)

    hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    used_pyannote = False

    if hf_token:
        try:
            import torch
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            diarization = pipeline(audio_16k)
            ann = getattr(diarization, "speaker_diarization", diarization)
            turns = [(float(turn.start), float(turn.end), str(spk)) for turn, _, spk in ann.itertracks(yield_label=True)]
            del pipeline
            free_gpu()

            if turns:
                for seg in segments:
                    s_t = float(seg["start"])
                    e_t = float(seg["end"])
                    best_spk = "SPEAKER_00"
                    best_ov = 0.0
                    for t_s, t_e, spk in turns:
                        ov = max(0.0, min(e_t, t_e) - max(s_t, t_s))
                        if ov > best_ov:
                            best_ov = ov
                            best_spk = spk
                    seg["speaker_id"] = best_spk
                used_pyannote = True
                print(f"[INFO] pyannote.audio diarization phát hiện {len({t[2] for t in turns})} cụm ban đầu.")
        except Exception as e:
            print(f"[WARN] pyannote.audio không chạy được ({e}), dùng dự phòng cao độ/giới tính.")

    # Đo cao độ và gán gender_hint cho từng đoạn thoại
    for seg in segments:
        i0 = max(0, int(float(seg["start"]) * sr))
        i1 = min(len(wav_data), int(float(seg["end"]) * sr))
        pitch = _estimate_pitch_hz(wav_data[i0:i1], sr) if i1 > i0 else None
        seg["pitch_hz"] = pitch
        if pitch is not None:
            seg["gender_hint"] = "female" if pitch >= 175.0 else ("male" if pitch <= 155.0 else "female")
        else:
            seg["gender_hint"] = "unknown"

    if not used_pyannote:
        cur_spk = "SPEAKER_00"
        prev_g = None
        for seg in segments:
            g = seg["gender_hint"]
            if g in ("male", "female") and prev_g in ("male", "female") and g != prev_g:
                cur_spk = "SPEAKER_01" if cur_spk == "SPEAKER_00" else "SPEAKER_00"
            if g in ("male", "female"):
                prev_g = g
            seg["speaker_id"] = cur_spk
        print(f"[INFO] Dự phòng gán người nói tạo {len({s['speaker_id'] for s in segments})} nhãn.")

    # Tách nhãn theo giới tính khi một cụm có cả giọng nam và giọng nữ (ví dụ có thêm bố vợ/nhân vật thứ 3)
    for seg in segments:
        base_spk = str(seg.get("speaker_id", "SPEAKER_00"))
        g_hint = str(seg.get("gender_hint", "unknown")).lower()
        if g_hint == "female":
            seg["speaker_id"] = f"{base_spk}_F"
        elif g_hint == "male":
            seg["speaker_id"] = f"{base_spk}_M"

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
        g_hint = "female" if spk.endswith("_F") else ("male" if spk.endswith("_M") else "unknown")
        speaker_refs[spk] = {
            "ref_asset": ref_name,
            "gender_hint": g_hint,
            "pitch_hz": avg_pitch,
        }

    return speaker_refs


def _extract_ocr_texts(ocr_result) -> list[str]:
    texts = []
    if not ocr_result:
        return texts
    for item in ocr_result:
        if isinstance(item, dict):
            rec_texts = item.get("rec_texts") or []
            for t in rec_texts:
                if str(t).strip():
                    texts.append(str(t).strip())
        elif isinstance(item, list):
            for entry in item:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    info = entry[1]
                    if isinstance(info, (list, tuple)) and info and str(info[0]).strip():
                        texts.append(str(info[0]).strip())
    return texts


def run_ocr_on_video(video_path: str, work_dir: str) -> tuple[dict, list[dict]]:
    roi = {"x1": 0.02, "y1": 0.58, "x2": 0.98, "y2": 0.96}
    try:
        import cv2
        from paddleocr import PaddleOCR

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 60.0

        sample_times = [round(duration * k / 7.0, 2) for k in range(1, 7)]
        ocr = PaddleOCR(
            lang="ch",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        hits = 0
        screen_texts = []
        for ts in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w, _ = frame.shape
            crop = frame[int(h * roi["y1"]):int(h * roi["y2"]), int(w * roi["x1"]):int(w * roi["x2"])]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            boosted = cv2.convertScaleAbs(gray, alpha=1.6, beta=15)
            boosted_bgr = cv2.cvtColor(boosted, cv2.COLOR_GRAY2BGR)

            found = _extract_ocr_texts(ocr.ocr(crop)) or _extract_ocr_texts(ocr.ocr(boosted_bgr))
            if found:
                hits += 1
                screen_texts.append({"time": ts, "texts": found})

        cap.release()
        print(f"[INFO] [PaddleOCR] ROI {tuple(roi.values())} | hits={hits}/{len(sample_times)} mẫu.")
        return roi, screen_texts
    except Exception as e:
        print(f"[WARN] PaddleOCR bỏ qua: {e}")
        return roi, []


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
        subtitle_roi, screen_texts = fut_ocr.result()

    segments = transcribe_with_faster_whisper(vocals_wav)
    speaker_refs = apply_speaker_diarization(audio_16k, segments, work_dir)

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
