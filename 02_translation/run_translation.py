import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

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
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"[WARN] upload_asset: thiếu file hoặc file 0 byte '{file_path}'")
            return None
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
        ("transformers", "transformers>=4.49.0"),
        ("accelerate", "accelerate"),
        ("PIL", "pillow"),
        ("gdown", "gdown"),
    ]
    missing = [pip_name for mod_name, pip_name in wanted if not _is_installed(mod_name.split(".")[0])]
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


def load_qwen2_5_vl():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_id = os.environ.get("QWEN2_5_VL_MODEL", os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")).strip() or "Qwen/Qwen2.5-VL-3B-Instruct"
    use_4bit = os.environ.get("QWEN_VL_4BIT", "0").strip() == "1"

    min_pixels = int(os.environ.get("QWEN_VL_MIN_PIXELS", str(128 * 28 * 28)))
    max_pixels = int(os.environ.get("QWEN_VL_MAX_PIXELS", str(384 * 28 * 28)))

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
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
            print(f"[INFO] Nạp {model_id} ở chế độ float16.")
    else:
        load_kwargs["dtype"] = torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
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
    inputs = processor(text=[prompt], images=images, padding=True, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], out_ids)]
    raw_out = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    for img in images:
        img.close()
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
    screen_texts: list[dict],
) -> dict[int, str]:
    import torch

    compact_items = [
        {
            "id": it["id"],
            "speaker_id": it["speaker_id"],
            "gender": it["gender"],
            "duration_sec": it["duration_sec"],
            "src_text": it["src_text"],
            "context_prev": it["prev_text"],
            "context_next": it["next_text"],
        }
        for it in batch_items
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là biên kịch lồng tiếng phim tiếng Việt. "
                "Dịch TỪNG câu src_text sang tiếng Việt tự nhiên, ngắn gọn vừa thời lượng duration_sec.\n"
                "Quy tắc nghiêm ngặt:\n"
                "1) Mỗi id phải dịch ĐÚNG nghĩa của chính src_text ở id đó. TUYỆT ĐỐI KHÔNG lặp lại bản dịch của câu trước hay câu sau.\n"
                "2) context_prev và context_next chỉ dùng để đoán chủ ngữ/xưng hô, KHÔNG được dịch nội dung của context_prev/context_next vào text_vi.\n"
                "3) Xưng hô phù hợp hoàn cảnh: 'anh - em', 'tôi - cô/anh', 'bố - con' (khi nói với bố vợ/cha).\n"
                "4) Giữ tên riêng 'Lục thị', 'Hạ thị', 'Minh Dao'; không để sót chữ Hán.\n"
                "5) Trả về duy nhất mảng JSON thuần đúng số phần tử đầu vào: [{\"id\": <int>, \"text_vi\": \"...\"}]."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Bối cảnh: {scene_context or 'Phim gia đình & công sở'}\n"
                f"Danh sách {len(compact_items)} câu cần dịch riêng biệt:\n"
                f"{json.dumps(compact_items, ensure_ascii=False)}"
            ),
        },
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=900,
            do_sample=False,
            repetition_penalty=1.08,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], out_ids)]
    raw_reply = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    del inputs, out_ids, trimmed
    free_gpu()

    result_map: dict[int, str] = {}
    try:
        clean_json = raw_reply.split("```json")[-1].split("```")[0].strip() if "```" in raw_reply else raw_reply
        m_arr = re.search(r"\[.*\]", clean_json, flags=re.DOTALL)
        if m_arr:
            clean_json = m_arr.group(0)
        parsed = json.loads(clean_json)
        if isinstance(parsed, list):
            for row in parsed:
                if isinstance(row, dict) and "id" in row and row.get("text_vi"):
                    result_map[int(row["id"])] = str(row["text_vi"]).strip()
    except Exception:
        pass
    return result_map


def translate_with_qwen2_5_vl(
    model,
    processor,
    segments: list[dict],
    scene_context: str,
    speakers_meta: dict,
    screen_texts: list[dict],
) -> tuple[list[dict], list[dict]]:
    if not segments:
        return [], []

    def _get_seg_text(s_obj: dict) -> str:
        return (
            s_obj.get("best_text")
            or s_obj.get("asr_text")
            or s_obj.get("ocr_text")
            or ""
        ).strip()

    prepared_items = []
    for idx_s, s in enumerate(segments):
        seg_id = int(s.get("id", idx_s))
        spk_id = s.get("speaker_id", "SPEAKER_00")
        start_t = round(float(s.get("start", 0.0)), 2)
        end_t = round(float(s.get("end", start_t + 1.0)), 2)
        src_txt = _get_seg_text(s)
        g_out = str(speakers_meta.get(spk_id, {}).get("gender") or s.get("gender_hint") or "female").lower()
        if g_out not in ("male", "female"):
            g_out = "female"

        prev_txt = _get_seg_text(segments[idx_s - 1]) if idx_s > 0 else ""
        next_txt = _get_seg_text(segments[idx_s + 1]) if idx_s + 1 < len(segments) else ""

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
    batch_size = 4
    for start_i in range(0, len(prepared_items), batch_size):
        sub_items = prepared_items[start_i: start_i + batch_size]
        got = _translate_lines_batch(model, processor, sub_items, scene_context, speakers_meta, screen_texts)
        translations.update(got)

    # Phát hiện và dịch lại đơn lẻ các câu bị thiếu, còn chữ Hán, hoặc bị lặp giống hệt câu liền kề dù src_text khác nhau
    for idx_p, item in enumerate(prepared_items):
        item_id = item["id"]
        cur = (translations.get(item_id) or "").strip()
        is_dup_with_neighbor = False
        if idx_p > 0:
            prev_item = prepared_items[idx_p - 1]
            prev_vi = (translations.get(prev_item["id"]) or "").strip()
            if cur and prev_vi and cur.lower() == prev_vi.lower() and item["src_text"] != prev_item["src_text"]:
                is_dup_with_neighbor = True

        if not cur or _has_cjk(cur) or is_dup_with_neighbor:
            retry_got = _translate_lines_batch(model, processor, [item], scene_context, speakers_meta, screen_texts)
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

    print(f"[INFO] Qwen2.5-VL dịch giữ nguyên timestamp {len(dub_lines)}/{len(segments)} câu thoại.")
    return utterances_out, dub_lines


def _format_srt_time(seconds: float) -> str:
    ms_total = max(0, int(round(float(seconds or 0.0) * 1000.0)))
    hours = ms_total // 3_600_000
    rem = ms_total % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    secs = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def export_stage2_srt(dub_lines: list[dict], out_srt_path: str) -> str:
    blocks = []
    idx_out = 0
    for line in dub_lines:
        txt = (line.get("text_vi") or line.get("src_text") or "").strip()
        if not txt:
            continue
        idx_out += 1
        start_t = float(line.get("start", 0.0))
        end_t = max(start_t + 0.25, float(line.get("end", start_t + 0.8)))
        spk = line.get("speaker_id", "SPEAKER_00")
        gender = line.get("gender", "unknown")
        blocks.append(
            f"{idx_out}\n"
            f"{_format_srt_time(start_t)} --> {_format_srt_time(end_t)}\n"
            f"[{spk}/{gender}] {txt}\n"
        )
    with open(out_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks).strip() + ("\n" if blocks else ""))
    print(f"[INFO] Đã xuất file kiểm định timestamp Stage 2: {out_srt_path} ({idx_out} dòng)")
    return out_srt_path


sync = GitHubSync(GH_REPO, GH_TOKEN)


def main():
    model_name = os.environ.get("QWEN2_5_VL_MODEL", os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")).strip() or "Qwen/Qwen2.5-VL-3B-Instruct"
    send_ntfy(f"[Stage 2] {JOB_ID}", f"Bắt đầu dịch giữ timestamp với {model_name}.", tags="globe_with_meridians")
    stage1_data, _ = sync.read_json(f"state/{JOB_ID}_stage1.json")
    segments = stage1_data.get("segments", [])
    screen_texts = stage1_data.get("screen_texts", [])
    prev_context = stage1_data.get("scene_context", "")
    speaker_refs = stage1_data.get("speaker_refs") or {}
    video_asset = (stage1_data.get("video_asset") or stage1_data.get("video_url") or "source_input.mp4").strip()

    work_dir = "/kaggle/working/work_s2"
    os.makedirs(work_dir, exist_ok=True)
    video_path = os.path.join(work_dir, "input.mp4")

    ensure_deps()
    fetched = fetch_video_for_labeling(sync, JOB_ID, video_asset, video_path)
    frames = extract_keyframes(fetched, work_dir) if fetched else []

    model, processor = load_qwen2_5_vl()
    try:
        scene_context, speakers_meta = label_scene_and_speakers(model, processor, frames, speaker_refs, prev_context)
        utterances_out, dub_lines = translate_with_qwen2_5_vl(
            model, processor, segments, scene_context, speakers_meta, screen_texts
        )
    finally:
        del model, processor
        free_gpu()

    srt_s2_path = os.path.join(work_dir, f"{JOB_ID}_stage2.srt")
    export_stage2_srt(dub_lines, srt_s2_path)
    sync.upload_asset(JOB_ID, srt_s2_path)

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
