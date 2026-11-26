import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import base64
import gc
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


def send_ntfy(title: str, message: str, tags: str = "globe_with_meridians", priority: str = "default"):
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

    def download_asset(self, tag: str, filename: str, dest_path: str):
        r = requests.get(f"{self.api_base}/releases/tags/{tag}", headers=self.headers, timeout=30)
        r.raise_for_status()
        release = r.json()
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
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("PIL", "pillow"),
        ("gdown", "gdown"),
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


def fetch_video_for_labeling(sync: GitHubSync, tag: str, source_ref: str, dest_path: str) -> str:
    ref = _clean_url_or_str(source_ref)
    if not ref:
        return ""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        if ref.startswith(("http://", "https://")):
            if "drive.google.com" in ref:
                direct_url = _normalize_gdrive_url(ref)
                subprocess.run([sys.executable, "-m", "gdown", direct_url, "-O", dest_path], check=True)
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
    for idx, ts in enumerate(["00:00:04", "00:00:25", "00:01:05"]):
        out_img = os.path.join(work_dir, f"frame_{idx}.jpg")
        res = subprocess.run(
            ["ffmpeg", "-y", "-ss", ts, "-i", video_path, "-vf", "scale=512:-1", "-frames:v", "1", "-q:v", "4", out_img],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0 and os.path.exists(out_img) and os.path.getsize(out_img) > 512:
            frames.append(out_img)
    return frames


def load_qwen3_vl():
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_id = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3-VL-4B-Instruct").strip() or "Qwen/Qwen3-VL-4B-Instruct"
    use_4bit = os.environ.get("QWEN_VL_4BIT", "0").strip() == "1"

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=128 * 28 * 28,
        max_pixels=320 * 28 * 28,
    )

    load_kwargs = {
        "device_map": "auto" if torch.cuda.is_available() else None,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        if use_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print(f"[INFO] Nạp {model_id} ở chế độ 4-bit NF4.")
        else:
            load_kwargs["dtype"] = torch.float16
            print(f"[INFO] Nạp {model_id} ở chế độ float16 (không nén 4-bit để giữ chất lượng dịch).")
    else:
        load_kwargs["dtype"] = torch.float32

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model, processor


def label_scene_and_speakers(model, processor, frame_paths: list[str], speaker_refs: dict, fallback_context: str) -> tuple[str, dict]:
    updated_speakers = {k: dict(v) for k, v in (speaker_refs or {}).items()}
    for spk_id, info in updated_speakers.items():
        pitch = float(info.get("pitch_hz") or 0.0)
        hint = str(info.get("gender_hint", "unknown")).lower()
        if pitch >= 175.0:
            info["gender"] = "female"
        elif 0.0 < pitch <= 155.0:
            info["gender"] = "male"
        elif hint in ("male", "female"):
            info["gender"] = hint

    if not frame_paths:
        for info in updated_speakers.values():
            info.setdefault("gender", "female")
        return fallback_context or "", updated_speakers

    import torch
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in frame_paths]
    content = [{"type": "image", "image": img} for img in images]
    content.append({
        "type": "text",
        "text": (
            "Hãy mô tả ngắn gọn bối cảnh và mối quan hệ nhân vật trong đoạn phim này bằng tiếng Việt "
            "(ví dụ: vợ chồng trẻ, cha vợ và con rể, đồng nghiệp công sở...). "
            "Trả về JSON thuần: {\"scene_label\": \"...\", \"speakers\": {\"SPEAKER_00\": \"male|female\"}}."
        ),
    })

    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], out_ids)]
    raw_out = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    del inputs, out_ids, trimmed, images
    free_gpu()

    scene_label = fallback_context or ""
    try:
        clean_json = raw_out.split("```json")[-1].split("```")[0].strip() if "```" in raw_out else raw_out
        parsed = json.loads(clean_json)
        if isinstance(parsed, dict):
            if parsed.get("scene_label"):
                lbl = str(parsed["scene_label"]).strip()
                scene_label = f"{lbl} | {fallback_context}".strip(" |") if fallback_context else lbl
            spk_map = parsed.get("speakers") or {}
            if isinstance(spk_map, dict):
                for spk_id, g_val in spk_map.items():
                    if updated_speakers.get(spk_id, {}).get("gender") in ("male", "female"):
                        continue
                    g_norm = str(g_val).lower().strip()
                    if g_norm in ("male", "nam"):
                        updated_speakers.setdefault(spk_id, {})["gender"] = "male"
                    elif g_norm in ("female", "nữ", "nu"):
                        updated_speakers.setdefault(spk_id, {})["gender"] = "female"
    except Exception:
        if raw_out:
            scene_label = f"{raw_out} | {fallback_context}".strip(" |") if fallback_context else raw_out

    for info in updated_speakers.values():
        info.setdefault("gender", "female")

    return scene_label, updated_speakers


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _clean_vietnamese_translation(text_vi: str, fallback_src: str = "") -> str:
    txt = " ".join(str(text_vi or "").split()).strip()
    txt = re.sub(r"[\u4e00-\u9fff]+", "", txt)
    txt = re.sub(r"\s+([,.!?;:])", r"\1", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()

    txt = re.sub(r"\bMinh (?:Nhu|Dương)\b", "Minh Dao", txt)
    txt = re.sub(r"\bLạc thị\b", "Lục thị", txt, flags=re.IGNORECASE)

    if len(txt) < 1 and fallback_src:
        return fallback_src
    return txt


def _translate_lines_batch(
    model,
    processor,
    batch_items: list[dict],
    scene_context: str,
    speakers_meta: dict,
) -> dict[int, str]:
    import torch

    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là biên kịch lồng tiếng phim tiếng Việt. "
                "Mỗi phần tử trong danh sách là một câu thoại độc lập với đúng thời lượng (duration_sec) và người nói.\n"
                "Quy tắc bắt buộc:\n"
                "1) Dịch ĐÚNG từng câu theo id của nó, KHÔNG gộp câu này sang câu khác, KHÔNG kéo dài câu vượt quá thời lượng đọc tự nhiên.\n"
                "2) Dùng prev_text và next_text chỉ để hiểu ngữ cảnh & xưng hô, tuyệt đối không dịch gộp prev_text/next_text vào text_vi của câu hiện tại.\n"
                "3) Xưng hô chuẩn: nếu gender='male' gọi '老婆' thì dịch 'Vợ ơi' / 'Em à'; giữa vợ chồng dùng 'anh - em' hoặc 'tôi - anh'; với bố vợ dùng 'bố - con'.\n"
                "4) Giữ tên riêng 'Lục thị', 'Hạ thị', 'Ngôn Thần', 'Minh Dao'; không chèn chữ Hán.\n"
                "Chỉ trả về mảng JSON thuần các object có khóa: id, text_vi."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Ngữ cảnh: {scene_context or 'Phim gia đình & công sở'}\n"
                f"Hồ sơ người nói: {json.dumps(speakers_meta, ensure_ascii=False)}\n"
                f"Danh sách câu cần dịch:\n{json.dumps(batch_items, ensure_ascii=False)}"
            ),
        },
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=1600, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], out_ids)]
    raw_reply = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    del inputs, out_ids, trimmed
    free_gpu()

    result_map: dict[int, str] = {}
    try:
        clean_json = raw_reply.split("```json")[-1].split("```")[0].strip() if "```" in raw_reply else raw_reply
        parsed = json.loads(clean_json)
        if isinstance(parsed, list):
            for row in parsed:
                if isinstance(row, dict) and "id" in row and row.get("text_vi"):
                    result_map[int(row["id"])] = str(row["text_vi"]).strip()
    except Exception:
        pass
    return result_map


def translate_with_qwen3_vl(model, processor, segments: list[dict], scene_context: str, speakers_meta: dict) -> tuple[list[dict], list[dict]]:
    if not segments:
        return [], []

    prepared_items = []
    for idx_s, s in enumerate(segments):
        seg_id = int(s.get("id", idx_s))
        spk_id = s.get("speaker_id", "SPEAKER_00")
        start_t = round(float(s.get("start", 0.0)), 2)
        end_t = round(float(s.get("end", start_t + 1.0)), 2)
        src_txt = (s.get("best_text") or s.get("asr_text") or "").strip()
        g_out = str(speakers_meta.get(spk_id, {}).get("gender") or s.get("gender_hint") or "female").lower()
        if g_out not in ("male", "female"):
            g_out = "female"

        prev_txt = ((segments[idx_s - 1].get("best_text") or segments[idx_s - 1].get("asr_text") or "").strip()) if idx_s > 0 else ""
        next_txt = ((segments[idx_s + 1].get("best_text") or segments[idx_s + 1].get("asr_text") or "").strip()) if idx_s + 1 < len(segments) else ""

        prepared_items.append({
            "id": seg_id,
            "start": start_t,
            "end": end_t,
            "duration_sec": round(max(0.3, end_t - start_t), 2),
            "speaker_id": spk_id,
            "gender": g_out,
            "prev_text": prev_txt,
            "src_text": src_txt,
            "next_text": next_txt,
        })

    translations: dict[int, str] = {}
    batch_size = 8
    for start_i in range(0, len(prepared_items), batch_size):
        sub_items = prepared_items[start_i: start_i + batch_size]
        got = _translate_lines_batch(model, processor, sub_items, scene_context, speakers_meta)
        translations.update(got)

    # Dịch bù từng câu còn thiếu hoặc còn sót chữ Hán
    for item in prepared_items:
        item_id = item["id"]
        cur = translations.get(item_id, "")
        if not cur or _has_cjk(cur):
            retry_got = _translate_lines_batch(model, processor, [item], scene_context, speakers_meta)
            if item_id in retry_got and retry_got[item_id].strip():
                translations[item_id] = retry_got[item_id].strip()

    dub_lines = []
    utterances_out = []
    for item in prepared_items:
        item_id = item["id"]
        src_txt = item["src_text"]
        vi_line = _clean_vietnamese_translation(translations.get(item_id) or src_txt, src_txt)
        line_obj = {
            "id": item_id,
            "start": item["start"],
            "end": item["end"],
            "speaker_id": item["speaker_id"],
            "gender": item["gender"],
            "src_text": src_txt,
            "text_vi": vi_line,
        }
        dub_lines.append(line_obj)
        utterances_out.append({
            **line_obj,
            "segment_ids": [item_id],
        })

    print(f"[INFO] Qwen3-VL dịch giữ nguyên timestamp {len(dub_lines)}/{len(segments)} câu thoại.")
    return utterances_out, dub_lines


sync = GitHubSync(GH_REPO, GH_TOKEN)


def main():
    model_name = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3-VL-4B-Instruct").strip() or "Qwen/Qwen3-VL-4B-Instruct"
    send_ntfy(f"[Stage 2] {JOB_ID}", f"Bắt đầu dịch giữ timestamp với {model_name}.", tags="globe_with_meridians")
    stage1_data, _ = sync.read_json(f"state/{JOB_ID}_stage1.json")
    segments = stage1_data.get("segments", [])
    prev_context = stage1_data.get("scene_context", "")
    speaker_refs = stage1_data.get("speaker_refs") or {}
    video_asset = (stage1_data.get("video_asset") or stage1_data.get("video_url") or "source_input.mp4").strip()

    work_dir = "/kaggle/working/work_s2"
    os.makedirs(work_dir, exist_ok=True)
    video_path = os.path.join(work_dir, "input.mp4")

    ensure_deps()
    fetched = fetch_video_for_labeling(sync, JOB_ID, video_asset, video_path)
    frames = extract_keyframes(fetched, work_dir) if fetched else []

    model, processor = load_qwen3_vl()
    try:
        scene_context, speakers_meta = label_scene_and_speakers(model, processor, frames, speaker_refs, prev_context)
        utterances_out, dub_lines = translate_with_qwen3_vl(model, processor, segments, scene_context, speakers_meta)
    finally:
        del model, processor
        free_gpu()

    stage2_payload = {
        "job_id": JOB_ID,
        "model": model_name,
        "video_asset": video_asset,
        "scene_context": scene_context,
        "speakers": speakers_meta,
        "utterances": utterances_out,
        "lines": dub_lines,
    }
    with open(os.path.join(work_dir, f"{JOB_ID}_stage2.json"), "w", encoding="utf-8") as f:
        json.dump(stage2_payload, f, ensure_ascii=False, indent=2)

    sync.write_json(f"state/{JOB_ID}_stage2.json", stage2_payload, f"stage2 done: {JOB_ID}")

    updated_jobs, _ = sync.read_json("state/jobs.json")
    if JOB_ID in updated_jobs.get("jobs", {}):
        updated_jobs["jobs"][JOB_ID]["status"] = "STAGE2_DONE"
        sync.write_json("state/jobs.json", updated_jobs, f"update status STAGE2_DONE: {JOB_ID}")

    sync.trigger_dispatch("stage2_completed", {"job_id": JOB_ID})
    send_ntfy(
        f"[Stage 2 Done] {JOB_ID}",
        f"Hoàn tất {len(dub_lines)} câu dịch giữ timestamp -> sang Stage 3.",
        tags="white_check_mark",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_ntfy(f"[Stage 2 Lỗi] {JOB_ID}", err, tags="x", priority="high")
        raise
