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

for _stream in (sys.stdout, sys.stderr):
    _reconf = getattr(_stream, "reconfigure", None)  # log realtime, không bị buffer
    if callable(_reconf):
        try:
            _reconf(line_buffering=True)
        except Exception:
            pass


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
            if r.status_code in (409, 429, 500, 502, 503, 504):
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


def _have_module(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def ensure_deps():
    """Chỉ cài khi thiếu: Qwen3-VL cần transformers mới, còn lại tránh --upgrade vô ích."""
    qwen_ok = False
    try:
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
        qwen_ok = True
    except Exception:
        qwen_ok = False

    if qwen_ok:
        print("[INFO] transformers đã hỗ trợ Qwen3-VL, bỏ qua pip install transformers.")
    else:
        print("[INFO] Cài/nâng transformers + accelerate để hỗ trợ Qwen3-VL.")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "transformers", "accelerate"],
            check=True,
        )

    missing = [spec for mod, spec in (("gdown", "gdown"), ("PIL", "pillow")) if not _have_module(mod)]
    if os.environ.get("QWEN_VL_4BIT", "1").strip() == "1" and not _have_module("bitsandbytes"):
        missing.append("bitsandbytes")
    if missing:
        print(f"[INFO] pip install các gói còn thiếu: {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


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


def _state_asset_ref(ref: str) -> bool:
    """True nếu ref trỏ vào file nằm trên nhánh pipeline-state (do web UI upload)."""
    return ref.startswith("uploads_tmp/") or (ref.startswith("state/") and ref.endswith("manifest.json"))


def _state_raw_bytes(sync: "GitHubSync", path: str) -> bytes:
    headers = dict(sync.headers)
    headers["Accept"] = "application/vnd.github.raw"
    r = requests.get(
        f"{sync.api_base}/contents/{path}",
        headers=headers,
        params={"ref": sync.state_branch},
        timeout=180,
    )
    r.raise_for_status()
    return r.content


def download_from_state(sync: "GitHubSync", ref: str, dest_path: str):
    """Ghép các mảnh upload từ web UI (manifest.json) thành 1 file video hoàn chỉnh."""
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    if ref.endswith("manifest.json"):
        manifest = json.loads(_state_raw_bytes(sync, ref).decode("utf-8"))
        parts = manifest.get("parts") or []
        if not parts:
            raise RuntimeError(f"Manifest {ref} không có danh sách parts")
        with open(dest_path, "wb") as out:
            for part in parts:
                out.write(_state_raw_bytes(sync, part))
    else:
        with open(dest_path, "wb") as out:
            out.write(_state_raw_bytes(sync, ref))
    if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 1024:
        raise RuntimeError(f"Tải video từ nhánh {sync.state_branch} thất bại: {ref}")


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
        elif _state_asset_ref(ref):
            download_from_state(sync, ref, dest_path)
        else:
            sync.download_asset(tag, ref, dest_path)
    except Exception as e:
        print(f"[WARN] Không tải được video để gắn nhãn ({ref}): {e}")
        return ""
    return dest_path if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024 else ""


def extract_keyframes(video_path: str, work_dir: str) -> list[str]:
    frames = []
    for idx, ts in enumerate(["00:00:04", "00:00:20"]):
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

    model_id = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct").strip() or "Qwen/Qwen3-VL-8B-Instruct"
    use_4bit = os.environ.get("QWEN_VL_4BIT", "1").strip() == "1"

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
    if not frame_paths:
        return fallback_context or "", speaker_refs

    import torch
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in frame_paths]
    content = [{"type": "image", "image": img} for img in images]
    content.append({
        "type": "text",
        "text": (
            "Hãy gắn nhãn chi tiết cho đoạn video này bằng tiếng Việt ngắn gọn: "
            "1) Bối cảnh, cảm xúc và xưng hô nhân vật; "
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
    updated_speakers = {k: dict(v) for k, v in (speaker_refs or {}).items()}

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


def fit_text_to_duration(text_vi: str, duration_sec: float) -> str:
    # ~17 ký tự/giây: chấp nhận nhanh hơn nhịp đọc tự nhiên một chút để KHÔNG cắt cụt
    # nghĩa câu; stage 3 sẽ tăng tốc độ đọc (atempo ≤1.5) cho vừa slot.
    max_chars = max(8, int(round(duration_sec * 22.0)))
    txt = " ".join(text_vi.split()).strip()
    if len(txt) <= max_chars:
        return txt
    cut = txt[:max_chars]
    for sep in ["。", "！", "？", ".", "!", "?", ",", "，", ";", " "]:
        pos = cut.rfind(sep)
        if pos >= int(max_chars * 0.60):
            return cut[:pos].strip()
    return cut.strip()


def _translate_chunk_with_model(model, processor, chunk: list[dict], start_idx: int, scene_context: str, speakers_meta: dict, context_pair: tuple[list[str], list[str]] | None = None) -> dict:
    import torch

    items_for_prompt = []
    for i, s in enumerate(chunk):
        line_id = s.get("id", start_idx + i)
        spk_id = s.get("speaker_id", "SPEAKER_00")
        dur = round(max(0.6, float(s.get("end", 1.0)) - float(s.get("start", 0.0))), 2)
        src_txt = (s.get("best_text") or s.get("asr_text") or "").strip()
        ocr_txt = (s.get("ocr_text") or "").strip()
        gender = speakers_meta.get(spk_id, {}).get("gender") or s.get("gender_hint") or "female"
        items_for_prompt.append({
            "id": line_id,
            "duration_sec": dur,
            "speaker_id": spk_id,
            "gender": gender,
            "src_text": src_txt,
            "ocr_text": ocr_txt,
        })

    before, after = context_pair if context_pair else ([], [])
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là dịch giả phụ đề phim Trung → Việt chuyên nghiệp, làm lồng tiếng.\n"
                "Quy tắc bắt buộc:\n"
                "1) Dịch thoát nghĩa theo khẩu ngữ, giữ sắc thái cảm xúc; không dịch máy móc từng chữ.\n"
                "2) Giữ nguyên tên riêng/công ty, viết phiên âm Hán-Việt quen thuộc "
                "(vd 言辰 → Ngôn Thần, 夏氏集团 → Tập đoàn Hạ thị).\n"
                "3) Xưng hô theo trường gender và quan hệ trong nhãn cảnh "
                "(vợ/chồng, sếp/nhân viên, mẹ/con...); không dùng 'bạn' máy móc khi đã rõ vai.\n"
                "4) Mỗi dòng là MỘT câu thoại ngắn, dài ≤ duration_sec × 16 ký tự, "
                "không gộp nhiều dòng, không thêm chú thích, không dùng ngoặc kép hay markdown.\n"
                "5) Thành ngữ/tiếng lóng thì dịch nghĩa bóng cho tự nhiên.\n"
                "6) Giữ đại từ nhất quán: mỗi speaker_id dùng một cặp xưng hô cố định trong cả đoạn "
                "(vd chồng→vợ: 'anh–em'; vợ→chồng: 'em–anh'), không đổi qua lại 'tôi/bạn/cậu'.\n"
                "Ví dụ: [{\"id\": 0, \"src_text\": \"你的饭给狗吃了\", \"duration_sec\": 2.1, "
                "\"text_vi\": \"Cơm của anh đem cho chó ăn rồi.\"}]\n"
                "Chỉ trả về JSON array với đúng các khoá: id, speaker_id, gender, text_vi."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Nhãn cảnh & ngữ cảnh: {scene_context or 'Phim chính kịch/đời thường'}\n"
                f"Câu thoại ngay trước: {json.dumps(before, ensure_ascii=False)}\n"
                f"Câu thoại ngay sau: {json.dumps(after, ensure_ascii=False)}\n"
                f"Danh sách thoại cần dịch:\n{json.dumps(items_for_prompt, ensure_ascii=False)}"
            ),
        },
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = processor(text=[text_input], return_tensors="pt")
    if torch.cuda.is_available():
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False)
    gen_trimmed = [out[len(inp):] for inp, out in zip(model_inputs["input_ids"], generated_ids)]
    raw_reply = processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0].strip()

    del model_inputs, generated_ids, gen_trimmed
    free_gpu()

    translated_map = {}
    try:
        clean_json = raw_reply
        if "```" in clean_json:
            clean_json = clean_json.split("```json")[-1].split("```")[0].strip()
        parsed = json.loads(clean_json)
        if isinstance(parsed, list):
            for row in parsed:
                if isinstance(row, dict) and "id" in row:
                    txt = str(row.get("text_vi") or "").strip().strip('"').strip("“”")
                    row["text_vi"] = txt
                    translated_map[row["id"]] = row
    except Exception:
        pass
    return translated_map
def translate_with_qwen3_vl(model, processor, segments: list[dict], scene_context: str, speakers_meta: dict) -> list[dict]:
    if not segments:
        return []

    dub_lines = []
    seen_texts = {}
    batch_size = int(os.environ.get("TRANSLATE_CHUNK", "8"))
    for start_idx in range(0, len(segments), batch_size):
        chunk = segments[start_idx: start_idx + batch_size]
        ctx_before = [(s.get("best_text") or s.get("asr_text") or "").strip() for s in segments[max(0, start_idx - 2):start_idx]]
        ctx_after = [(s.get("best_text") or s.get("asr_text") or "").strip() for s in segments[start_idx + len(chunk): start_idx + len(chunk) + 2]]
        translated_map = _translate_chunk_with_model(
            model, processor, chunk, start_idx, scene_context, speakers_meta, (ctx_before, ctx_after)
        )

        missing_items = [
            (i, s) for i, s in enumerate(chunk)
            if not (translated_map.get(s.get("id", start_idx + i)) or {}).get("text_vi")
        ]
        if missing_items:
            retry_chunk = [s for _, s in missing_items]
            retry_map = _translate_chunk_with_model(model, processor, retry_chunk, start_idx, scene_context, speakers_meta)
            translated_map.update(retry_map)

        for i, s in enumerate(chunk):
            line_id = s.get("id", start_idx + i)
            spk_id = s.get("speaker_id", "SPEAKER_00")
            start_t = float(s.get("start", 0.0))
            end_t = float(s.get("end", start_t + 1.0))
            dur = max(0.6, end_t - start_t)
            fallback_src = (s.get("best_text") or s.get("asr_text") or "").strip()

            row_obj = translated_map.get(line_id) or {}
            raw_vi = str(row_obj.get("text_vi") or fallback_src).strip()
            # Model hay trả lặp lại bản dịch của câu trước → dịch lại riêng câu này
            if raw_vi and raw_vi in seen_texts and seen_texts[raw_vi] != fallback_src:
                retry_map = _translate_chunk_with_model(model, processor, [s], start_idx + i, scene_context, speakers_meta)
                alt = str((retry_map.get(line_id) or {}).get("text_vi") or "").strip()
                if alt and alt != raw_vi:
                    raw_vi = alt
            seen_texts[raw_vi] = fallback_src
            fitted_vi = fit_text_to_duration(raw_vi, dur)
            g_out = str(row_obj.get("gender") or speakers_meta.get(spk_id, {}).get("gender") or s.get("gender_hint") or "female").lower()
            if g_out not in ("male", "female"):
                g_out = "female"

            dub_lines.append({
                "id": line_id,
                "start": start_t,
                "end": end_t,
                "speaker_id": spk_id,
                "gender": g_out,
                "src_text": fallback_src,
                "text_vi": fitted_vi,
            })

    print(f"[INFO] Qwen3-VL dịch hoàn tất {len(dub_lines)}/{len(segments)} câu thoại.")
    return dub_lines


sync = GitHubSync(GH_REPO, GH_TOKEN)


def mark_job_failed(err_text: str):
    """Ghi FAILED vào state/jobs.json để UI và monitor thấy lỗi thay vì treo ở trạng thái cũ."""
    try:
        data, _ = sync.read_json("state/jobs.json")
        job = data.setdefault("jobs", {}).setdefault(JOB_ID, {})
        job["status"] = "FAILED_STAGE2"
        job["error"] = err_text[-800:]
        job["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        sync.write_json("state/jobs.json", data, f"FAILED stage2: {JOB_ID}")
    except Exception as exc:
        print(f"[WARN] Không ghi được trạng thái FAILED: {exc}")


def build_utterances(segments: list[dict], ocr_spans: list[dict] | None = None, max_gap: float = 0.45, max_chars: int = 60) -> list[dict]:
    """Gộp các mảnh ASR vụn thành CÂU THOẠI TRỌN VẸN trước khi dịch.

    ASR cắt câu rất vụn (đo thật: 29/54 mảnh ≤6 ký tự). Dịch từng mảnh thì câu tiếng
    Việt cũng vụn, sai nghĩa và lặp.

    Ưu tiên gộp theo PHỤ ĐỀ OCR: mỗi dòng phụ đề ~ một câu nói, nên các mảnh ASR cùng
    rơi vào một span phụ đề được gộp lại. Không có span thì gộp theo khoảng lặng.
    """
    spans = []
    for sp in (ocr_spans or []):
        try:
            s0 = float(sp.get("start") or 0.0)
            s1 = float(sp.get("end") or 0.0)
        except Exception:
            continue
        if s1 > s0:
            spans.append((s0, s1))

    def span_key(seg) -> int | None:
        s_t = float(seg.get("start", 0.0))
        e_t = float(seg.get("end", s_t + 0.5))
        best_i, best_ov = None, 0.0
        for i, (s0, s1) in enumerate(spans):
            ov = max(0.0, min(e_t, s1) - max(s_t, s0))
            if ov > best_ov:
                best_i, best_ov = i, ov
        if best_i is not None and best_ov >= min(0.30, 0.4 * max(0.3, e_t - s_t)):
            return best_i
        return None

    utts: list[dict] = []
    cur = None
    for s in segments:
        txt = (s.get("best_text") or s.get("asr_text") or "").strip()
        if not txt:
            continue
        spk = s.get("speaker_id") or "SPEAKER_00"
        key = span_key(s)

        if cur is None:
            cur = {"id": 0, "start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)),
                   "speaker_id": spk, "gender_hint": s.get("gender_hint"), "best_text": txt,
                   "ocr_text": s.get("ocr_text", ""), "segment_ids": [s.get("id")], "key": key}
            continue

        gap = float(s.get("start", 0.0)) - float(cur["end"])
        same_span = key is not None and key == cur.get("key")
        short_gap_merge = (
            key is None and cur.get("key") is None and spk == cur["speaker_id"]
            and gap <= max_gap and (len(cur["best_text"]) <= 6 or len(txt) <= 6)
            and len(cur["best_text"]) + len(txt) <= 20
        )
        if (same_span or short_gap_merge) and spk == cur["speaker_id"] and len(cur["best_text"]) + len(txt) <= max_chars:
            cur["end"] = float(s.get("end", cur["end"]))
            cur["best_text"] = f"{cur['best_text']}{txt}"
            cur["segment_ids"].append(s.get("id"))
            if key is not None:
                cur["key"] = key
        else:
            utts.append(cur)
            cur = {"id": 0, "start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)),
                   "speaker_id": spk, "gender_hint": s.get("gender_hint"), "best_text": txt,
                   "ocr_text": s.get("ocr_text", ""), "segment_ids": [s.get("id")], "key": key}
    if cur:
        utts.append(cur)

    for i, u in enumerate(utts):
        u["id"] = i
        u.pop("key", None)
    return utts


def expand_utterances_to_lines(utts: list[dict], segments: list[dict]) -> list[dict]:
    """Chia bản dịch của câu thoại về các mảnh timing gốc (để UI/tương thích cũ)."""
    seg_by_id = {s.get("id"): s for s in segments}
    lines = []
    for u in utts:
        text = (u.get("text_vi") or "").strip()
        ids = [i for i in (u.get("segment_ids") or []) if i in seg_by_id]
        if not ids:
            continue
        durs = [max(0.3, float(seg_by_id[i].get("end", 0.0)) - float(seg_by_id[i].get("start", 0.0))) for i in ids]
        total = sum(durs) or 1.0
        words = text.split()
        pos = 0
        for k, sid in enumerate(ids):
            seg = seg_by_id[sid]
            if k == len(ids) - 1:
                part = " ".join(words[pos:])
            else:
                take = max(1, int(round(len(words) * durs[k] / total))) if words else 0
                part = " ".join(words[pos:pos + take])
                pos += take
            lines.append({
                "id": sid,
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "speaker_id": u.get("speaker_id"),
                "gender": u.get("gender", "female"),
                "src_text": u.get("src_text", ""),
                "text_vi": part.strip(),
            })
    return lines


def main():
    model_name = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct").strip() or "Qwen/Qwen3-VL-8B-Instruct"
    send_telegram(TG_TOKEN, TG_CHAT, f"🌐 <b>[Stage 2 - {model_name}]</b> Bắt đầu job <code>{JOB_ID}</code>")
    stage1_data, _ = sync.read_json(f"state/{JOB_ID}_stage1.json")
    segments = stage1_data.get("segments", [])
    utterances = build_utterances(segments, stage1_data.get("screen_texts", []))
    print(f"[INFO] Gộp {len(segments)} mảnh ASR → {len(utterances)} câu thoại trọn vẹn trước khi dịch.")
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
        dub_utts = translate_with_qwen3_vl(model, processor, utterances, scene_context, speakers_meta)
    finally:
        del model, processor
        free_gpu()

    utt_map = {u.get("id"): u for u in dub_utts}
    payload_utts = []
    for u in utterances:
        tr = utt_map.get(u["id"]) or {}
        gender = str(
            tr.get("gender") or speakers_meta.get(u["speaker_id"], {}).get("gender") or u.get("gender_hint") or "female"
        ).lower()
        if gender not in ("male", "female"):
            gender = "female"
        payload_utts.append({
            "id": u["id"],
            "start": u["start"],
            "end": u["end"],
            "speaker_id": u["speaker_id"],
            "gender": gender,
            "src_text": u["best_text"],
            "text_vi": (tr.get("text_vi") or "").strip(),
            "segment_ids": u["segment_ids"],
        })
    dub_lines = expand_utterances_to_lines(payload_utts, segments)

    stage2_payload = {
        "job_id": JOB_ID,
        "model": model_name,
        "video_asset": video_asset,
        "scene_context": scene_context,
        "speakers": speakers_meta,
        "utterances": payload_utts,
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
        f"✅ <b>[Stage 2]</b> Hoàn tất <code>{JOB_ID}</code> ({len(dub_lines)} câu dịch, model <code>{model_name}</code>) → chuyển sang Stage 3.",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        mark_job_failed(err)
        send_telegram(TG_TOKEN, TG_CHAT, f"❌ <b>[Stage 2 Lỗi]</b> <code>{JOB_ID}</code>:\n<pre>{err}</pre>")
        raise
