import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for candidate_dir in [SCRIPT_DIR, "/kaggle/src", "/kaggle/working", os.getcwd()]:
    if candidate_dir and os.path.isdir(candidate_dir) and candidate_dir not in sys.path:
        sys.path.insert(0, candidate_dir)

if not os.path.isdir(os.path.join(SCRIPT_DIR, "core")):
    for root, dirs, _ in os.walk("."):
        if "core" in dirs:
            found_dir = os.path.abspath(root)
            if found_dir not in sys.path:
                sys.path.insert(0, found_dir)
            break

import subprocess
import traceback
from dotenv import load_dotenv
from core.github_sync import GitHubSync
from core.telegram_notifier import send_telegram

load_dotenv()

JOB_ID = os.environ["JOB_ID"]
GH_REPO = os.environ["GH_REPO"]
GH_TOKEN = os.environ["GH_PAT_TOKEN"]
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

sync = GitHubSync(GH_REPO, GH_TOKEN)


def main():
    jobs_data, _ = sync.read_json("state/jobs.json")
    job = jobs_data.get("jobs", {}).get(JOB_ID)
    if not job:
        raise KeyError(f"Không tìm thấy job_id '{JOB_ID}' trong state/jobs.json")

    video_url = job["video_url"]
    send_telegram(TG_TOKEN, TG_CHAT, f"🎬 <b>[Stage 1 - Perception]</b> Bắt đầu job <code>{JOB_ID}</code>")

    work_dir = "/kaggle/working/work"
    os.makedirs(work_dir, exist_ok=True)
    video_path = os.path.join(work_dir, "input.mp4")
    audio_16k = os.path.join(work_dir, "audio_16k.wav")

    subprocess.run(["curl", "-L", "-s", video_url, "-o", video_path], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", audio_16k],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Khu vực xử lý:
    # - cuda:0: Whisper-large-v3 (faster-whisper FP16) + Demucs tách vocals / no_vocals.wav
    # - cuda:1: PaddleOCR + Qwen2.5-VL-7B phân tích khung hình & cắt ref_speaker_X.wav

    no_vocals_path = os.path.join(work_dir, "no_vocals.wav")
    if os.path.exists(no_vocals_path):
        sync.upload_asset(JOB_ID, no_vocals_path)

    for fname in sorted(os.listdir(work_dir)):
        if fname.startswith("ref_speaker_") and fname.endswith(".wav"):
            sync.upload_asset(JOB_ID, os.path.join(work_dir, fname))

    stage1_payload = {
        "job_id": JOB_ID,
        "video_url": video_url,
        "segments": [],
    }
    sync.write_json(f"state/{JOB_ID}_stage1.json", stage1_payload, f"stage1 done: {JOB_ID}")

    updated_jobs, _ = sync.read_json("state/jobs.json")
    updated_jobs.setdefault("jobs", {})[JOB_ID]["status"] = "STAGE1_DONE"
    sync.write_json("state/jobs.json", updated_jobs, f"update status STAGE1_DONE: {JOB_ID}")

    sync.trigger_dispatch("stage1_completed", {"job_id": JOB_ID})
    send_telegram(TG_TOKEN, TG_CHAT, f"✅ <b>[Stage 1]</b> Hoàn tất <code>{JOB_ID}</code> → chuyển sang Stage 2.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_telegram(TG_TOKEN, TG_CHAT, f"❌ <b>[Stage 1 Lỗi]</b> <code>{JOB_ID}</code>:\n<pre>{err}</pre>")
        raise
