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
    clean_token = str(bot_token).strip()
    clean_chat = str(chat_id).strip()
    try:
        requests.post(
            f"https://api.telegram.org/bot{clean_token}/sendMessage",
            json={"chat_id": clean_chat, "text": text[:4000], "parse_mode": "HTML", "disable_web_page_preview": False},
            timeout=20,
        )
    except Exception as e:
        print(f"[WARN] Gửi Telegram thất bại: {e}")


def ensure_deps():
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "transformers", "accelerate", "pillow", "gdown",
        ],
        check=False,
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


def _normalize_gdrive_url(url: str) -> str:
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url) or re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return f"https://drive.google.com/uc?id={m.group(1)}"
    return url


def fetch_video_for_labeling(sync: GitHubSync, tag: str, source_ref: str, dest_path: str) -> str:
    ref = (source_ref or "").strip()
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
    if not frame_paths:
        for spk_id, info in updated_speakers.items():
            if "gender" not in info:
                hint = info.get("gender_hint", "unknown")
                info["gender"] = hint if hint in ("male", "female") else "female"
        return fallback_context or "", updated_speakers

    import torch
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in frame_paths]
    content = [{"type": "image", "image": img} for img in images]
    content.append({
        "type": "text",
        "text": (
            "Hãy gắn nhãn chi tiết cho đoạn video này bằng tiếng Việt ngắn gọn: "
            "1) Bối cảnh, cảm xúc và mối quan hệ xưng hô nhân vật; "
            "2) Ước lượng giới tính ('male' hoặc 'female') cho từng speaker nếu có thể. "
            f"Gợi ý thông số âm thanh từ Stage 1: {json.dumps(speaker_refs, ensure_ascii=False)}. "
            "Trả về JSON thuần dạng: {\"scene_label\": \"...\", \"speakers\": {\"SPEAKER_00\": \"male|female\"}}."
        ),
    })

    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=220, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], out_ids)]
    raw_out = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    del inputs, out_ids, trimmed, images
    free_gpu()

    scene_label = fallback_context or ""
    try:
        clean_json = raw_out
        if "```" in clean_json:
            clean_json = clean_json.split("```json")[-1].split("```")[0].strip()
        parsed = json.loads(clean_json)
        if isinstance(parsed, dict):
            if parsed.get("scene_label"):
                lbl = str(parsed["scene_label"]).strip()
                scene_label = f"{lbl} | {fallback_context}".strip(" |") if fallback_context else lbl
            spk_map = parsed.get("speakers") or {}
            if isinstance(spk_map, dict):
                for spk_id, g_val in spk_map.items():
                    g_norm = str(g_val).lower().strip()
                    if g_norm in ("male", "nam"):
                        updated_speakers.setdefault(spk_id, {})["gender"] = "male"
                    elif g_norm in ("female", "nữ", "nu"):
                        updated_speakers.setdefault(spk_id, {})["gender"] = "female"
    except Exception:
        if raw_out:
            scene_label = f"{raw_out} | {fallback_context}".strip(" |") if fallback_context else raw_out

    for spk_id, info in updated_speakers.items():
        if "gender" not in info:
            hint = info.get("gender_hint", "unknown")
            info["gender"] = hint if hint in ("male", "female") else "female"

    return scene_label, updated_speakers


def _clean_vietnamese_translation(text_vi: str, fallback_src: str) -> str:
    txt = " ".join(str(text_vi or "").split()).strip()
    txt = re.sub(r"[\u4e00-\u9fff]+", "", txt)
    txt = re.sub(r"\s+([,.!?;:])", r"\1", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()

    txt = re.sub(r"\bMinh Nhu\b", "Minh Dao", txt)
    txt = re.sub(r"\bLạc thị\b", "Lục thị", txt, flags=re.IGNORECASE)

    if len(txt) < 2 and fallback_src:
        return fallback_src
    return txt


def _split_utterance_across_segments(utterance_vi: str, seg_list: list[dict]) -> list[str]:
    if len(seg_list) == 1:
        return [utterance_vi.strip()]

    durs = [max(0.25, float(s.get("end", 1.0)) - float(s.get("start", 0.0))) for s in seg_list]
    total_d = sum(durs)
    words = utterance_vi.strip().split()
    if len(words) <= len(seg_list):
        out = []
        for i in range(len(seg_list)):
            out.append(words[i] if i < len(words) else words[-1] if words else "")
        return out

    alloc = []
    cursor = 0
    for i, d in enumerate(durs):
        if i == len(durs) - 1:
            part = words[cursor:]
        else:
            take = max(1, int(round(len(words) * (d / total_d))))
            remain_after = len(seg_list) - (i + 1)
            take = min(take, max(1, len(words) - cursor - remain_after))
            part = words[cursor: cursor + take]
            cursor += take
        alloc.append(" ".join(part).strip() or words[-1])
    return alloc


def _build_utterance_groups(segments: list[dict]) -> list[list[dict]]:
    if not segments:
        return []
    groups: list[list[dict]] = []
    cur: list[dict] = [segments[0]]
    for seg in segments[1:]:
        prev = cur[-1]
        same_spk = seg.get("speaker_id") == prev.get("speaker_id")
        gap = float(seg.get("start", 0.0)) - float(prev.get("end", 0.0))
        cur_dur = float(prev.get("end", 0.0)) - float(cur[0].get("start", 0.0))
        if same_spk and gap <= 1.2 and cur_dur <= 6.5:
            cur.append(seg)
        else:
            groups.append(cur)
            cur = [seg]
    groups.append(cur)
    return groups


def _translate_utterance_batch(model, processor, batch_groups: list[list[dict]], scene_context: str, speakers_meta: dict) -> dict[int, str]:
    import torch

    payload_items = []
    for g_idx, grp in enumerate(batch_groups):
        spk_id = grp[0].get("speaker_id", "SPEAKER_00")
        gender = speakers_meta.get(spk_id, {}).get("gender") or grp[0].get("gender_hint") or "female"
        s_t = float(grp[0].get("start", 0.0))
        e_t = float(grp[-1].get("end", s_t + 1.0))
        merged_src = " ".join((s.get("best_text") or s.get("asr_text") or "").strip() for s in grp).strip()
        payload_items.append({
            "utt_id": g_idx,
            "duration_sec": round(max(0.6, e_t - s_t), 2),
            "speaker_id": spk_id,
            "gender": gender,
            "src_text": merged_src,
        })

    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là biên kịch lồng tiếng phim tiếng Việt tự nhiên, sát nghĩa, không cắt cụt ý. "
                "Quy tắc bắt buộc:\n"
                "1) Dịch trọn vẹn từng ý thoại sang tiếng Việt tự nhiên, đủ vế, không bỏ sót ý hay cắt nửa chừng.\n"
                "2) Thống nhất tên riêng & thuật ngữ: giữ 'Lục thị', 'Hạ thị', 'Ngôn Thần', 'Minh Dao' (hoặc 'Dao Dao').\n"
                "3) Giữ xưng hô nhất quán theo ngữ cảnh nhân vật (chồng/vợ, người thân, cấp trên/cấp dưới).\n"
                "4) Không chèn chữ Hán vào câu dịch tiếng Việt.\n"
                "Chỉ trả về mảng JSON thuần các object có khóa: utt_id, text_vi."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Ngữ cảnh: {scene_context or 'Phim chính kịch gia đình & công sở'}\n"
                f"Danh sách ý thoại:\n{json.dumps(payload_items, ensure_ascii=False)}"
            ),
        },
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=1200, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], out_ids)]
    raw_reply = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    del inputs, out_ids, trimmed
    free_gpu()

    result_map: dict[int, str] = {}
    try:
        clean_json = raw_reply
        if "```" in clean_json:
            clean_json = clean_json.split("```json")[-1].split("```")[0].strip()
        parsed = json.loads(clean_json)
        if isinstance(parsed, list):
            for row in parsed:
                if isinstance(row, dict) and "utt_id" in row and row.get("text_vi"):
                    result_map[int(row["utt_id"])] = str(row["text_vi"]).strip()
    except Exception:
        pass
    return result_map


def translate_with_qwen3_vl(model, processor, segments: list[dict], scene_context: str, speakers_meta: dict) -> tuple[list[dict], list[dict]]:
    if not segments:
        return [], []

    groups = _build_utterance_groups(segments)
    utt_translations: dict[int, str] = {}

    batch_size = 5
    for start_g in range(0, len(groups), batch_size):
        sub_groups = groups[start_g: start_g + batch_size]
        got = _translate_utterance_batch(model, processor, sub_groups, scene_context, speakers_meta)
        for local_i, vi_txt in got.items():
            utt_translations[start_g + local_i] = vi_txt

        for local_i, grp in enumerate(sub_groups):
            g_id = start_g + local_i
            if not utt_translations.get(g_id):
                single_got = _translate_utterance_batch(model, processor, [grp], scene_context, speakers_meta)
                if 0 in single_got:
                    utt_translations[g_id] = single_got[0]

    utterances_out = []
    seg_text_map: dict[int, str] = {}

    for g_id, grp in enumerate(groups):
        spk_id = grp[0].get("speaker_id", "SPEAKER_00")
        gender = speakers_meta.get(spk_id, {}).get("gender") or grp[0].get("gender_hint") or "female"
        s_t = float(grp[0].get("start", 0.0))
        e_t = float(grp[-1].get("end", s_t + 1.0))
        merged_src = " ".join((s.get("best_text") or s.get("asr_text") or "").strip() for s in grp).strip()
        raw_vi = utt_translations.get(g_id) or merged_src
        clean_vi = _clean_vietnamese_translation(raw_vi, merged_src)

        seg_ids = [s.get("id", idx_s) for idx_s, s in enumerate(grp)]
        utterances_out.append({
            "id": g_id,
            "start": s_t,
            "end": e_t,
            "speaker_id": spk_id,
            "gender": gender,
            "src_text": merged_src,
            "text_vi": clean_vi,
            "segment_ids": seg_ids,
        })

        parts = _split_utterance_across_segments(clean_vi, grp)
        for s_item, p_txt in zip(grp, parts):
            seg_text_map[s_item.get("id")] = p_txt

    dub_lines = []
    for idx_s, s in enumerate(segments):
        seg_id = s.get("id", idx_s)
        spk_id = s.get("speaker_id", "SPEAKER_00")
        start_t = float(s.get("start", 0.0))
        end_t = float(s.get("end", start_t + 1.0))
        src_txt = (s.get("best_text") or s.get("asr_text") or "").strip()
        g_out = str(speakers_meta.get(spk_id, {}).get("gender") or s.get("gender_hint") or "female").lower()
        if g_out not in ("male", "female"):
            g_out = "female"

        vi_line = _clean_vietnamese_translation(seg_text_map.get(seg_id) or src_txt, src_txt)
        dub_lines.append({
            "id": seg_id,
            "start": start_t,
            "end": end_t,
            "speaker_id": spk_id,
            "gender": g_out,
            "src_text": src_txt,
            "text_vi": vi_line,
        })

    print(f"[INFO] Qwen3-VL dịch hoàn tất {len(utterances_out)} cụm ý → {len(dub_lines)}/{len(segments)} câu thoại.")
    return utterances_out, dub_lines


sync = GitHubSync(GH_REPO, GH_TOKEN)


def main():
    model_name = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3-VL-4B-Instruct").strip() or "Qwen/Qwen3-VL-4B-Instruct"
    send_telegram(TG_TOKEN, TG_CHAT, f"🌐 <b>[Stage 2 - {model_name}]</b> Bắt đầu job <code>{JOB_ID}</code>")
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
    send_telegram(
        TG_TOKEN,
        TG_CHAT,
        f"✅ <b>[Stage 2]</b> Hoàn tất <code>{JOB_ID}</code> ({len(dub_lines)} câu dịch, model <code>{model_name}</code>) → chuyển sang Stage 3.",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_telegram(TG_TOKEN, TG_CHAT, f"❌ <b>[Stage 2 Lỗi]</b> <code>{JOB_ID}</code>:\n<pre>{err}</pre>")
        raise
