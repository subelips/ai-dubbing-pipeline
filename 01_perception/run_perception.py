import os
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
    job = jobs_data["jobs"][JOB_ID]
    video_url = job["video_url"]

    send_telegram(TG_TOKEN, TG_CHAT, f"🎬 <b>[Stage 1 - Perception]</b> Bắt đầu job <code>{JOB_ID}</code>")

    os.makedirs("/kaggle/working/work", exist_ok=True)
    video_path = "/kaggle/working/work/input.mp4"
    subprocess.run(["curl", "-L", "-s", video_url, "-o", video_path], check=True)

    # Tách audio gốc 16kHz cho Whisper & giữ track gốc cho Demucs
    audio_16k = "/kaggle/working/work/audio_16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", audio_16k],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 1. Chạy Whisper-large-v3 (cuda:0) + Demucs tách BGM/SFX
    # 2. Chạy PaddleOCR + Qwen2.5-VL-7B (cuda:1) lấy ngữ cảnh khung hình & cắt ref_speaker_X.wav
    # -> Tạo segments list & upload no_vocals.wav + ref_speaker_*.wav lên Release tag = JOB_ID

    no_vocals_path = "/kaggle/working/work/no_vocals.wav"
    if os.path.exists(no_vocals_path):
        sync.upload_asset(JOB_ID, no_vocals_path)

    for fname in os.listdir("/kaggle/working/work"):
        if fname.startswith("ref_speaker_") and fname.endswith(".wav"):
            sync.upload_asset(JOB_ID, os.path.join("/kaggle/working/work", fname))

    stage1_payload = {
        "job_id": JOB_ID,
        "video_url": video_url,
        "segments": [],  # Điền kết quả {start, end, asr_text, ocr_text, visual_context}
    }
    sync.write_json(f"state/{JOB_ID}_stage1.json", stage1_payload, f"stage1 done: {JOB_ID}")

    jobs_data, _ = sync.read_json("state/jobs.json")
    jobs_data["jobs"][JOB_ID]["status"] = "STAGE1_DONE"
    sync.write_json("state/jobs.json", jobs_data, f"update status STAGE1_DONE: {JOB_ID}")

    # Kích hoạt workflow cho Stage 2
    sync.trigger_dispatch("stage1_completed", {"job_id": JOB_ID})
    send_telegram(TG_TOKEN, TG_CHAT, f"✅ <b>[Stage 1]</b> Hoàn tất <code>{JOB_ID}</code> → chuyển sang Stage 2.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err = traceback.format_exc()[-1200:]
        send_telegram(TG_TOKEN, TG_CHAT, f"❌ <b>[Stage 1 Lỗi]</b> <code>{JOB_ID}</code>:\n<pre>{err}</pre>")
        raise
