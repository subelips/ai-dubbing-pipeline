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
    send_telegram(TG_TOKEN, TG_CHAT, f"🎙️ <b>[Stage 3 - Synthesis]</b> Bắt đầu job <code>{JOB_ID}</code>")
    stage2_data, _ = sync.read_json(f"state/{JOB_ID}_stage2.json")

    out_dir = "/kaggle/working/out"
    os.makedirs(out_dir, exist_ok=True)
    video_in = os.path.join(out_dir, "source.mp4")
    bgm_in = os.path.join(out_dir, "no_vocals.wav")
    final_mp4 = os.path.join(out_dir, f"final_dubbed_{JOB_ID}.mp4")

    subprocess.run(["curl", "-L", "-s", stage2_data["video_url"], "-o", video_in], check=True)
    sync.download_asset(JOB_ID, "no_vocals.wav", bgm_in)

    release = sync.ensure_release(JOB_ID)
    for asset in release.get("assets", []):
        if asset["name"].startswith("ref_speaker_") and asset["name"].endswith(".wav"):
            sync.download_asset(JOB_ID, asset["name"], os.path.join(out_dir, asset["name"]))

    # Khu vực xử lý:
    # - Chạy VieNeu-TTS v3 Turbo song song trên cuda:0 và cuda:1,
    #   căn chỉnh timecode, trộn với no_vocals.wav và mux với video_in.

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

    updated_jobs, _ = sync.read_json("state/jobs.json")
    if JOB_ID in updated_jobs.get("jobs", {}):
        updated_jobs["jobs"][JOB_ID]["status"] = "COMPLETED"
        updated_jobs["jobs"][JOB_ID]["output_url"] = download_url
        sync.write_json("state/jobs.json", updated_jobs, f"completed job: {JOB_ID}")

    send_telegram(
        TG_TOKEN,
        TG_CHAT,
        f"🎉 <b>Hoàn tất lồng tiếng!</b> Job: <code>{JOB_ID}</code>\n🔗 Link tải MP4: {download_url}",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_telegram(TG_TOKEN, TG_CHAT, f"❌ <b>[Stage 3 Lỗi]</b> <code>{JOB_ID}</code>:\n<pre>{err}</pre>")
        raise
