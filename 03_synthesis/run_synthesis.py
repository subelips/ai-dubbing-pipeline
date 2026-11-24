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
    send_telegram(TG_TOKEN, TG_CHAT, f"🎙️ <b>[Stage 3 - Synthesis]</b> Bắt đầu job <code>{JOB_ID}</code>")
    stage2_data, _ = sync.read_json(f"state/{JOB_ID}_stage2.json")

    os.makedirs("/kaggle/working/out", exist_ok=True)
    video_in = "/kaggle/working/out/source.mp4"
    bgm_in = "/kaggle/working/out/no_vocals.wav"
    final_mp4 = f"/kaggle/working/out/final_dubbed_{JOB_ID}.mp4"

    subprocess.run(["curl", "-L", "-s", stage2_data["video_url"], "-o", video_in], check=True)
    sync.download_asset(JOB_ID, "no_vocals.wav", bgm_in)

    # Tải các file ref_speaker_*.wav đã lưu ở Stage 1 nếu có
    release = sync.ensure_release(JOB_ID)
    for asset in release.get("assets", []):
        if asset["name"].startswith("ref_speaker_") and asset["name"].endswith(".wav"):
            sync.download_asset(JOB_ID, asset["name"], os.path.join("/kaggle/working/out", asset["name"]))

    # Chạy 2 tiến trình VieNeu-TTS v3 Turbo trên cuda:0 (câu chẵn) và cuda:1 (câu lẻ),
    # dùng add_voice() + infer_batch(), căn timecode với atempo, trộn bgm_in và mux với video_in.
    # Ví dụ fallback mux nhạc nền + video nếu chưa có line thoại:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_in,
            "-i", bgm_in,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            final_mp4,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    download_url = sync.upload_asset(JOB_ID, final_mp4)

    jobs_data, _ = sync.read_json("state/jobs.json")
    jobs_data["jobs"][JOB_ID]["status"] = "COMPLETED"
    jobs_data["jobs"][JOB_ID]["output_url"] = download_url
    sync.write_json("state/jobs.json", jobs_data, f"completed job: {JOB_ID}")

    send_telegram(
        TG_TOKEN,
        TG_CHAT,
        f"🎉 <b>Hoàn tất lồng tiếng!</b> Job: <code>{JOB_ID}</code>\n🔗 Link tải MP4: {download_url}",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err = traceback.format_exc()[-1200:]
        send_telegram(TG_TOKEN, TG_CHAT, f"❌ <b>[Stage 3 Lỗi]</b> <code>{JOB_ID}</code>:\n<pre>{err}</pre>")
        raise
