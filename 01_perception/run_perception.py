import base64
import gc
import json
import os
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
    requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": text[:4000], "parse_mode": "HTML", "disable_web_page_preview": False},
        timeout=20,
    )


def ensure_deps():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "faster-whisper", "demucs", "soundfile", "gdown"],
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


def download_video_source(sync: GitHubSync, tag: str, source_ref: str, dest_path: str):
    ref = (source_ref or "source_input.mp4").strip()
    if ref.startswith(("http://", "https://")):
        if "drive.google.com" in ref:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
            subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", ref, "-O", dest_path], check=True)
        else:
            subprocess.run(["curl", "-fSL", ref, "-o", dest_path], check=True)
    else:
        sync.download_asset(tag, ref, dest_path)


def run_whisper_asr(audio_16k: str) -> list[dict]:
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model_size = os.environ.get("WHISPER_MODEL", "large-v3")

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_segments, _ = model.transcribe(audio_16k, beam_size=5, vad_filter=True)

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
            "asr_text": txt,
        })

    del model
    free_gpu()
    return segments


def run_demucs_and_ref_voice(audio_16k: str, segments: list[dict], work_dir: str) -> str:
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

    # Cắt file giọng tham chiếu đầu tiên (dài ~6-10s) cho Stage 3 clone giọng
    try:
        data, sr = sf.read(audio_16k)
        start_s, end_s = 0.0, min(8.0, len(data) / float(sr))
        for seg in segments:
            dur = float(seg["end"]) - float(seg["start"])
            if dur >= 3.0:
                start_s = float(seg["start"])
                end_s = min(float(seg["end"]), start_s + 8.0)
                break
        if end_s > start_s:
            clip = data[int(start_s * sr): int(end_s * sr)]
            ref_path = os.path.join(work_dir, "ref_speaker_SPEAKER_00.wav")
            sf.write(ref_path, clip, sr)
    except Exception:
        pass

    free_gpu()
    return no_vocals_out


def optional_qwen_vl_context(video_path: str, work_dir: str) -> str:
    if os.environ.get("ENABLE_QWEN_VL", "0").strip() != "1":
        return ""
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor

        frame_path = os.path.join(work_dir, "keyframe.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", "00:00:05", "-i", video_path, "-frames:v", "1", frame_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not os.path.exists(frame_path):
            return ""

        model_id = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        image = Image.open(frame_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Mô tả ngắn gọn bối cảnh phim, nhân vật và sắc thái cảnh này trong 2 câu."},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=120)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], out_ids)]
        desc = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        del model, processor
        free_gpu()
        return desc
    except Exception:
        free_gpu()
        return ""


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
    segments = run_whisper_asr(audio_16k)
    no_vocals_path = run_demucs_and_ref_voice(audio_16k, segments, work_dir)
    scene_context = optional_qwen_vl_context(video_path, work_dir)

    if os.path.exists(no_vocals_path):
        sync.upload_asset(JOB_ID, no_vocals_path)

    for fname in sorted(os.listdir(work_dir)):
        if fname.startswith("ref_speaker_") and fname.endswith(".wav"):
            sync.upload_asset(JOB_ID, os.path.join(work_dir, fname))

    stage1_payload = {
        "job_id": JOB_ID,
        "video_asset": video_asset,
        "scene_context": scene_context,
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
        f"✅ <b>[Stage 1]</b> Hoàn tất <code>{JOB_ID}</code> ({len(segments)} câu thoại) → chuyển sang Stage 2.",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_telegram(TG_TOKEN, TG_CHAT, f"❌ <b>[Stage 1 Lỗi]</b> <code>{JOB_ID}</code>:\n<pre>{err}</pre>")
        raise
