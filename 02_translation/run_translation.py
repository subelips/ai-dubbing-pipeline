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
    send_telegram(TG_TOKEN, TG_CHAT, f"🌐 <b>[Stage 2 - Translation]</b> Bắt đầu job <code>{JOB_ID}</code>")
    stage1_data, _ = sync.read_json(f"state/{JOB_ID}_stage1.json")

    # Khu vực xử lý:
    # - Chạy Qwen2.5-14B-Instruct (4-bit) hợp nhất ASR + OCR + ngữ cảnh hình ảnh,
    #   gán speaker_id nhất quán và dịch sang tiếng Việt khớp thời lượng thoại.

    dub_lines = []
    for idx, seg in enumerate(stage1_data.get("segments", [])):
        dub_lines.append({
            "id": idx,
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 1.0),
            "speaker_id": seg.get("speaker_id", "SPEAKER_00"),
            "text_vi": seg.get("asr_text", ""),
        })

    stage2_payload = {
        "job_id": JOB_ID,
        "video_url": stage1_data.get("video_url", ""),
        "lines": dub_lines,
    }
    sync.write_json(f"state/{JOB_ID}_stage2.json", stage2_payload, f"stage2 done: {JOB_ID}")

    updated_jobs, _ = sync.read_json("state/jobs.json")
    if JOB_ID in updated_jobs.get("jobs", {}):
        updated_jobs["jobs"][JOB_ID]["status"] = "STAGE2_DONE"
        sync.write_json("state/jobs.json", updated_jobs, f"update status STAGE2_DONE: {JOB_ID}")

    sync.trigger_dispatch("stage2_completed", {"job_id": JOB_ID})
    send_telegram(TG_TOKEN, TG_CHAT, f"✅ <b>[Stage 2]</b> Hoàn tất <code>{JOB_ID}</code> → chuyển sang Stage 3.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_telegram(TG_TOKEN, TG_CHAT, f"❌ <b>[Stage 2 Lỗi]</b> <code>{JOB_ID}</code>:\n<pre>{err}</pre>")
        raise
