import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import base64
import gc
import json
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

    def download_asset(self, tag: str, filename: str, dest_path: str):
        r = requests.get(f"{self.api_base}/releases/tags/{tag}", headers=self.headers, timeout=30)
        r.raise_for_status()
        release = r.json()
        for asset in release.get("assets", []):
            if asset["name"] == filename:
                dl_headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/octet-stream"}
                with requests.get(asset["url"], headers=dl_headers, stream=True, timeout=300) as dl:
                    dl.raise_for_status()
                    with open(dest_path, "wb") as out:
                        for chunk in dl.iter_content(chunk_size=1024 * 1024):
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
        [
            sys.executable, "-m", "pip", "install", "-q", "--upgrade",
            "transformers", "accelerate", "bitsandbytes", "pillow", "gdown",
        ],
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


def fetch_video_for_labeling(sync: GitHubSync, tag: str, source_ref: str, dest_path: str) -> str:
    ref = (source_ref or "").strip()
    if not ref:
        return ""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        if ref.startswith(("http://", "https://")):
            if "drive.google.com" in ref:
                subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", ref, "-O", dest_path], check=True)
            else:
                subprocess.run(["curl", "-fSL", ref, "-o", dest_path], check=True)
        else:
            sync.download_asset(tag, ref, dest_path)
    except Exception as e:
        print(f"[WARN] Không tải được video để gắn nhãn ({ref}): {e}")
        return ""
    return dest_path if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024 else ""


def extract_keyframes(video_path: str, work_dir: str) -> list[str]:
    frames = []
    for idx, ts in enumerate(["00:00:03", "00:00:15", "00:00:35"]):
        out_img = os.path.join(work_dir, f"frame_{idx}.jpg")
        res = subprocess.run(
            ["ffmpeg", "-y", "-ss", ts, "-i", video_path, "-frames:v", "1", "-q:v", "3", out_img],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0 and os.path.exists(out_img) and os.path.getsize(out_img) > 512:
            frames.append(out_img)
    return frames


def load_qwen3_vl():
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    model_id = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    processor = AutoProcessor.from_pretrained(model_id)

    load_kwargs = {
        "device_map": "auto" if torch.cuda.is_available() else None,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        vram_gb = sum(
            torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())
        ) / (1024 ** 3)
        if vram_gb < 24:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["dtype"] = torch.bfloat16
    else:
        load_kwargs["dtype"] = torch.float32

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model, processor


def label_scene_with_qwen3_vl(model, processor, frame_paths: list[str], fallback_context: str) -> str:
    if not frame_paths:
        return fallback_context or ""

    import torch
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in frame_paths]
    content = [{"type": "image", "image": img} for img in images]
    content.append({
        "type": "text",
        "text": (
            "Hãy gắn nhãn ngắn gọn cho đoạn video này (tối đa 3 câu tiếng Việt): "
            "bối cảnh/không gian, số lượng và đặc điểm nhân vật/xưng hô, sắc thái cảm xúc hoặc chữ nổi bật trên màn hình."
        ),
    })

    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=180, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], out_ids)]
    desc = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    del inputs, out_ids, trimmed, images
    free_gpu()
    return desc or fallback_context


def translate_with_qwen3_vl(model, processor, segments: list[dict], scene_context: str) -> list[dict]:
    if not segments:
        return []

    import torch

    dub_lines = []
    batch_size = 6
    for start_idx in range(0, len(segments), batch_size):
        chunk = segments[start_idx: start_idx + batch_size]
        items_for_prompt = [
            {
                "id": s.get("id", start_idx + i),
                "duration_sec": round(max(0.5, float(s.get("end", 1.0)) - float(s.get("start", 0.0))), 2),
                "speaker_id": s.get("speaker_id", "SPEAKER_00"),
                "src_text": s.get("asr_text", ""),
            }
            for i, s in enumerate(chunk)
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là dịch giả lồng tiếng phim chuyên nghiệp sang tiếng Việt tự nhiên, "
                    "súc tích, đúng xưng hô theo nhãn cảnh và khớp thời lượng duration_sec. "
                    "Chỉ trả về đúng một mảng JSON các object có khóa: id, speaker_id, text_vi. Không dùng markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Nhãn cảnh phim: {scene_context or 'Phim chính kịch/đời thường'}\n"
                    f"Danh sách thoại:\n{json.dumps(items_for_prompt, ensure_ascii=False)}"
                ),
            },
        ]

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = processor(text=[text_input], return_tensors="pt")
        if torch.cuda.is_available():
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(**model_inputs, max_new_tokens=700, do_sample=False)
        gen_trimmed = [out[len(inp):] for inp, out in zip(model_inputs["input_ids"], generated_ids)]
        raw_reply = processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0].strip()

        translated_map = {}
        try:
            clean_json = raw_reply
            if "```" in clean_json:
                clean_json = clean_json.split("```json")[-1].split("```")[0].strip()
            parsed = json.loads(clean_json)
            if isinstance(parsed, list):
                for row in parsed:
                    if isinstance(row, dict) and "id" in row:
                        translated_map[row["id"]] = str(row.get("text_vi", "")).strip()
        except Exception:
            pass

        for i, s in enumerate(chunk):
            line_id = s.get("id", start_idx + i)
            vi_txt = translated_map.get(line_id) or s.get("asr_text", "")
            dub_lines.append({
                "id": line_id,
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 1.0)),
                "speaker_id": s.get("speaker_id", "SPEAKER_00"),
                "text_vi": vi_txt,
            })

        del model_inputs, generated_ids, gen_trimmed
        free_gpu()

    return dub_lines


sync = GitHubSync(GH_REPO, GH_TOKEN)


def main():
    send_telegram(TG_TOKEN, TG_CHAT, f"🌐 <b>[Stage 2 - Qwen3-VL-8B]</b> Bắt đầu job <code>{JOB_ID}</code>")
    stage1_data, _ = sync.read_json(f"state/{JOB_ID}_stage1.json")
    segments = stage1_data.get("segments", [])
    prev_context = stage1_data.get("scene_context", "")
    video_asset = (stage1_data.get("video_asset") or stage1_data.get("video_url") or "source_input.mp4").strip()

    work_dir = "/kaggle/working/work_s2"
    os.makedirs(work_dir, exist_ok=True)
    video_path = os.path.join(work_dir, "input.mp4")

    ensure_deps()
    fetched = fetch_video_for_labeling(sync, JOB_ID, video_asset, video_path)
    frames = extract_keyframes(fetched, work_dir) if fetched else []

    model, processor = load_qwen3_vl()
    try:
        scene_context = label_scene_with_qwen3_vl(model, processor, frames, prev_context)
        dub_lines = translate_with_qwen3_vl(model, processor, segments, scene_context)
    finally:
        del model, processor
        free_gpu()

    stage2_payload = {
        "job_id": JOB_ID,
        "video_asset": video_asset,
        "scene_context": scene_context,
        "lines": dub_lines,
    }
    sync.write_json(f"state/{JOB_ID}_stage2.json", stage2_payload, f"stage2 done: {JOB_ID}")

    updated_jobs, _ = sync.read_json("state/jobs.json")
    if JOB_ID in updated_jobs.get("jobs", {}):
        updated_jobs["jobs"][JOB_ID]["status"] = "STAGE2_DONE"
        sync.write_json("state/jobs.json", updated_jobs, f"update status STAGE2_DONE: {JOB_ID}")

    sync.trigger_dispatch("stage2_completed", {"job_id": JOB_ID})
    send_telegram(
        TG_TOKEN,
        TG_CHAT,
        f"✅ <b>[Stage 2]</b> Hoàn tất <code>{JOB_ID}</code> ({len(dub_lines)} câu dịch) → chuyển sang Stage 3.",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_telegram(TG_TOKEN, TG_CHAT, f"❌ <b>[Stage 2 Lỗi]</b> <code>{JOB_ID}</code>:\n<pre>{err}</pre>")
        raise
