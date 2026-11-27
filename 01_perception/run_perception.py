import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import base64
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import re
import subprocess
import sys
import time
import traceback
import zipfile
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


def send_ntfy(title: str, message: str, tags: str = "film_frames", priority: str = "default"):
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
                with requests.get(dl_url, headers=dl_headers, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(dest_path, "wb") as out:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
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
        ("demucs", "demucs"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("gdown", "gdown"),
        ("paddle", "paddlepaddle"),
        ("paddleocr", "paddleocr"),
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


def download_video_source(sync: GitHubSync, tag: str, source_ref: str, dest_path: str):
    ref = _clean_url_or_str(source_ref) or "source_input.mp4"
    if ref.startswith(("http://", "https://")):
        if "drive.google.com" in ref:
            direct_url = _normalize_gdrive_url(ref)
            subprocess.run([sys.executable, "-m", "gdown", direct_url, "-O", dest_path], check=True)
        else:
            subprocess.run(["curl", "-fSL", ref, "-o", dest_path], check=True)
    else:
        sync.download_asset(tag, ref, dest_path)


def extract_audio_16k(video_path: str, out_wav: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", out_wav],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def separate_vocals_and_bgm(audio_16k: str, work_dir: str) -> tuple[str, str]:
    demucs_out = os.path.join(work_dir, "demucs_out")
    os.makedirs(demucs_out, exist_ok=True)
    cmd = ["demucs", "-n", "htdemucs", "--two-stems", "vocals", "-o", demucs_out, audio_16k]
    import torch
    if not torch.cuda.is_available():
        cmd.extend(["-d", "cpu"])
    subprocess.run(cmd, check=True)
    free_gpu()

    stem_dir = os.path.join(demucs_out, "htdemucs", os.path.splitext(os.path.basename(audio_16k))[0])
    vocals_wav = os.path.join(stem_dir, "vocals.wav")
    bgm_wav = os.path.join(stem_dir, "no_vocals.wav")
    if not os.path.exists(vocals_wav) or not os.path.exists(bgm_wav):
        raise FileNotFoundError("Demucs không tạo đủ vocals.wav và no_vocals.wav")
    return vocals_wav, bgm_wav


def _get_dbnet_config() -> dict:
    det_model = os.environ.get("OCR_DET_MODEL", "PP-OCRv5_mobile_det").strip() or "PP-OCRv5_mobile_det"
    rec_model = os.environ.get("OCR_REC_MODEL", "PP-OCRv5_mobile_rec").strip() or "PP-OCRv5_mobile_rec"
    return {
        "det_model": det_model,
        "rec_model": rec_model,
        "dbnet_thresh": float(os.environ.get("DBNET_THRESH", "0.30")),
        "dbnet_box_thresh": float(os.environ.get("DBNET_BOX_THRESH", "0.50")),
        "dbnet_unclip_ratio": float(os.environ.get("DBNET_UNCLIP_RATIO", "1.60")),
        "dbnet_limit_side_len": int(os.environ.get("DBNET_LIMIT_SIDE_LEN", "960")),
    }


def _create_mobile_dbnet_ocr(lang: str, cfg: dict):
    from paddleocr import PaddleOCR

    # Cấu hình DBNet + model mobile cho detection và recognition
    candidate_kwargs = [
        dict(
            lang=lang,
            text_detection_model_name=cfg["det_model"],
            text_recognition_model_name=cfg["rec_model"],
            text_det_thresh=cfg["dbnet_thresh"],
            text_det_box_thresh=cfg["dbnet_box_thresh"],
            text_det_unclip_ratio=cfg["dbnet_unclip_ratio"],
            text_det_limit_side_len=cfg["dbnet_limit_side_len"],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        ),
        dict(
            lang=lang,
            text_detection_model_name="PP-OCRv4_mobile_det",
            text_recognition_model_name="PP-OCRv4_mobile_rec",
            text_det_thresh=cfg["dbnet_thresh"],
            text_det_box_thresh=cfg["dbnet_box_thresh"],
            text_det_unclip_ratio=cfg["dbnet_unclip_ratio"],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        ),
        dict(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        ),
    ]

    last_err = None
    for kw in candidate_kwargs:
        try:
            return PaddleOCR(**kw)
        except (TypeError, ValueError) as e:
            last_err = e
            continue
    if last_err is not None:
        print(f"[WARN] Dùng cấu hình PaddleOCR mặc định do: {last_err}")
    return PaddleOCR(lang=lang)


def _parse_ocr_detections(ocr_result) -> list[dict]:
    """Trích xuất danh sách {'poly': [[x,y]...], 'text': str, 'score': float} từ kết quả PaddleOCR."""
    detections: list[dict] = []
    if not ocr_result:
        return detections

    for item in ocr_result:
        res_obj = item
        if not isinstance(res_obj, dict):
            for getter in (
                lambda x: x.res,
                lambda x: x.json() if callable(getattr(x, "json", None)) else getattr(x, "json", None),
            ):
                try:
                    val = getter(res_obj)
                    if isinstance(val, dict):
                        res_obj = val.get("res", val)
                        break
                except Exception:
                    continue

        if isinstance(res_obj, dict):
            polys = res_obj.get("dt_polys") or res_obj.get("rec_polys") or res_obj.get("rec_boxes") or []
            rec_texts = res_obj.get("rec_texts") or []
            rec_scores = res_obj.get("rec_scores") or []

            count = max(len(polys), len(rec_texts))
            for i in range(count):
                poly_raw = polys[i] if i < len(polys) else None
                txt = str(rec_texts[i]).strip() if i < len(rec_texts) else ""
                sc = float(rec_scores[i]) if i < len(rec_scores) and rec_scores[i] is not None else 1.0

                pts = []
                if poly_raw is not None:
                    try:
                        arr = poly_raw.tolist() if hasattr(poly_raw, "tolist") else list(poly_raw)
                        # Nếu là box [x1, y1, x2, y2] chuyển thành 4 đỉnh polygon
                        if len(arr) == 4 and all(isinstance(v, (int, float)) for v in arr):
                            x1, y1, x2, y2 = arr
                            pts = [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]
                        else:
                            for p in arr:
                                if len(p) >= 2:
                                    pts.append([int(round(float(p[0]))), int(round(float(p[1])))])
                    except Exception:
                        pts = []

                if pts or txt:
                    detections.append({"poly": pts, "text": txt, "score": round(sc, 3)})
            continue

        if isinstance(item, (list, tuple)):
            for entry in item:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    raw_box = entry[0]
                    info = entry[1]
                    txt = str(info[0]).strip() if isinstance(info, (list, tuple)) and info else ""
                    sc = float(info[1]) if isinstance(info, (list, tuple)) and len(info) >= 2 else 1.0
                    pts = []
                    try:
                        for p in raw_box:
                            if len(p) >= 2:
                                pts.append([int(round(float(p[0]))), int(round(float(p[1])))])
                    except Exception:
                        pts = []
                    detections.append({"poly": pts, "text": txt, "score": round(sc, 3)})

    return detections


def _run_dbnet_ocr(ocr_engine, img) -> list[dict]:
    if ocr_engine is None or img is None:
        return []
    try:
        if hasattr(ocr_engine, "predict"):
            res_p = list(ocr_engine.predict(img))
            dets = _parse_ocr_detections(res_p)
            if dets:
                return dets
    except Exception as e:
        print(f"[WARN] DBNet mobile predict lỗi: {e}")
    try:
        res = ocr_engine.ocr(img)
        return _parse_ocr_detections(res)
    except Exception:
        return []


def _draw_bbox_visualization(
    full_frame,
    roi: dict,
    detections_in_crop: list[dict],
    out_vis_path: str,
    ts_sec: float,
):
    """Vẽ ROI và các bbox DBNet phát hiện được lên khung hình để kiểm tra trực quan."""
    import cv2
    import numpy as np

    vis = full_frame.copy()
    h, w, _ = vis.shape
    rx1 = int(w * roi["x1"])
    ry1 = int(h * roi["y1"])
    rx2 = int(w * roi["x2"])
    ry2 = int(h * roi["y2"])

    # Vẽ khung ROI màu vàng
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 220, 255), 2)
    cv2.putText(
        vis,
        f"t={ts_sec:.2f}s | ROI ({roi['x1']},{roi['y1']})-({roi['x2']},{roi['y2']})",
        (max(8, rx1), max(22, ry1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        2,
    )

    # Vẽ từng bbox DBNet (tọa độ quy đổi về khung hình gốc)
    for idx, det in enumerate(detections_in_crop):
        pts = det.get("poly") or []
        txt = det.get("text") or ""
        sc = det.get("score", 0.0)
        if len(pts) >= 3:
            shifted = np.array([[p[0] + rx1, p[1] + ry1] for p in pts], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [shifted], isClosed=True, color=(0, 255, 0), thickness=2)
            top_x = int(np.min(shifted[:, 0, 0]))
            top_y = int(np.min(shifted[:, 0, 1]))
            label = f"#{idx + 1} ({sc:.2f})"
            cv2.putText(
                vis,
                label,
                (max(4, top_x), max(18, top_y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 0),
                1,
            )
            if txt:
                print(f"    [bbox #{idx + 1} @ {ts_sec:.2f}s] score={sc:.2f} | text={txt}")

    cv2.imwrite(out_vis_path, vis)


def run_ocr_and_draw_bbox(video_path: str, work_dir: str) -> tuple[dict, list[dict], str]:
    roi = {"x1": 0.02, "y1": 0.52, "x2": 0.98, "y2": 0.96}
    vis_dir = os.path.join(work_dir, "bbox_vis")
    os.makedirs(vis_dir, exist_ok=True)
    zip_path = os.path.join(work_dir, f"{JOB_ID}_bbox_vis.zip")

    try:
        import cv2

        db_cfg = _get_dbnet_config()
        forced_lang = os.environ.get("OCR_LANG", "").strip()
        if forced_lang:
            ocr_lang = forced_lang
        else:
            src_lang = os.environ.get("SOURCE_LANG", "zh").strip().lower()
            lang_map = {
                "zh": "ch",
                "cn": "ch",
                "ch": "ch",
                "ja": "japan",
                "jp": "japan",
                "ko": "korean",
                "kr": "korean",
                "vi": "vi",
                "en": "en",
            }
            ocr_lang = lang_map.get(src_lang, "ch")

        ocr = _create_mobile_dbnet_ocr(ocr_lang, db_cfg)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 60.0

        num_samples = int(os.environ.get("OCR_SAMPLE_COUNT", "10"))
        num_samples = max(4, min(20, num_samples))
        sample_times = [round(duration * k / float(num_samples + 1), 2) for k in range(1, num_samples + 1)]

        hits = 0
        screen_texts: list[dict] = []
        vis_files: list[str] = []

        for idx_s, ts in enumerate(sample_times):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            h, w, _ = frame.shape
            crop = frame[int(h * roi["y1"]):int(h * roi["y2"]), int(w * roi["x1"]):int(w * roi["x2"])]
            if crop.size == 0:
                continue

            dets = _run_dbnet_ocr(ocr, crop)
            texts_found = [d["text"] for d in dets if str(d.get("text") or "").strip()]

            vis_name = f"frame_{idx_s:02d}_{ts:.2f}s.jpg"
            vis_path = os.path.join(vis_dir, vis_name)
            _draw_bbox_visualization(frame, roi, dets, vis_path, ts)
            if os.path.exists(vis_path):
                vis_files.append(vis_path)

            if dets or texts_found:
                hits += 1
                screen_texts.append({
                    "time": ts,
                    "texts": texts_found,
                    "boxes": dets,
                    "vis_image": vis_name,
                })

        cap.release()

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for vf in vis_files:
                zf.write(vf, arcname=os.path.basename(vf))

        print(
            f"[INFO] [PaddleOCR Mobile + DBNet] lang={ocr_lang} det={db_cfg['det_model']} rec={db_cfg['rec_model']} "
            f"| ROI {tuple(roi.values())} | hits={hits}/{len(sample_times)} | ảnh bbox={len(vis_files)}"
        )
        return roi, screen_texts, zip_path
    except Exception as e:
        print(f"[WARN] PaddleOCR Mobile / DBNet bỏ qua: {e}")
        return roi, [], ""


def build_ocr_segments_without_asr(screen_texts: list[dict], duration_fallback: float = 60.0) -> list[dict]:
    """Tạo danh sách đoạn tạm từ kết quả OCR (bỏ qua ASR) để xuất SRT/JSON cho Stage 2."""
    segments: list[dict] = []
    for idx, item in enumerate(screen_texts):
        ts = float(item.get("time", idx * 2.0))
        next_ts = float(screen_texts[idx + 1]["time"]) if idx + 1 < len(screen_texts) else min(duration_fallback, ts + 2.5)
        start_t = round(max(0.0, ts - 0.3), 2)
        end_t = round(max(start_t + 0.8, min(next_ts - 0.05, ts + 2.2)), 2)
        texts = [str(t).strip() for t in (item.get("texts") or []) if str(t).strip()]
        joined = " ".join(texts).strip()
        if not joined and not item.get("boxes"):
            continue
        display_txt = joined or f"[BBOX_ONLY x{len(item.get('boxes') or [])}]"
        segments.append({
            "id": idx,
            "start": start_t,
            "end": end_t,
            "speaker_id": "SPEAKER_00",
            "gender_hint": "unknown",
            "pitch_hz": None,
            "asr_raw": "",
            "asr_text": "",
            "ocr_text": joined,
            "best_text": display_txt,
            "subtitle_source": "ocr_mobile_dbnet",
            "boxes": item.get("boxes") or [],
            "vis_image": item.get("vis_image") or "",
        })
    return segments


def _format_srt_time(seconds: float) -> str:
    ms_total = max(0, int(round(float(seconds or 0.0) * 1000.0)))
    hours = ms_total // 3_600_000
    rem = ms_total % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    secs = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def export_stage1_srt(segments: list[dict], out_srt_path: str) -> str:
    blocks = []
    idx_out = 0
    for seg in segments:
        txt = (seg.get("best_text") or seg.get("ocr_text") or "").strip()
        if not txt:
            continue
        idx_out += 1
        start_t = float(seg.get("start", 0.0))
        end_t = max(start_t + 0.25, float(seg.get("end", start_t + 0.8)))
        src_tag = seg.get("subtitle_source", "ocr_mobile_dbnet")
        blocks.append(
            f"{idx_out}\n"
            f"{_format_srt_time(start_t)} --> {_format_srt_time(end_t)}\n"
            f"[{src_tag}] {txt}\n"
        )
    with open(out_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks).strip() + ("\n" if blocks else ""))
    print(f"[INFO] Đã xuất file SRT Stage 1 (OCR Mobile + DBNet, không ASR): {out_srt_path} ({idx_out} dòng)")
    return out_srt_path


sync = GitHubSync(GH_REPO, GH_TOKEN)


def main():
    send_ntfy(
        f"[Stage 1] {JOB_ID}",
        f"Bắt đầu nhận diện (PaddleOCR Mobile + DBNet bbox + Demucs, tạm bỏ ASR) cho job {JOB_ID}.",
        tags="film_frames",
    )
    jobs_data, _ = sync.read_json("state/jobs.json")
    job_info = (jobs_data.get("jobs") or {}).get(JOB_ID) or {}
    video_asset = (job_info.get("video_asset") or job_info.get("video_url") or "source_input.mp4").strip()

    work_dir = "/kaggle/working/work"
    os.makedirs(work_dir, exist_ok=True)
    video_path = os.path.join(work_dir, "input.mp4")
    audio_16k = os.path.join(work_dir, "audio_16k.wav")

    ensure_deps()
    download_video_source(sync, JOB_ID, video_asset, video_path)
    extract_audio_16k(video_path, audio_16k)

    # Chạy tuần tự Demucs rồi OCR + vẽ bbox để tránh tranh chấp tài nguyên
    vocals_wav, bgm_wav = separate_vocals_and_bgm(audio_16k, work_dir)
    subtitle_roi, screen_texts, bbox_zip_path = run_ocr_and_draw_bbox(video_path, work_dir)

    segments = build_ocr_segments_without_asr(screen_texts)

    srt_s1_path = os.path.join(work_dir, f"{JOB_ID}_stage1.srt")
    export_stage1_srt(segments, srt_s1_path)
    sync.upload_asset(JOB_ID, srt_s1_path)
    sync.upload_asset(JOB_ID, bgm_wav)
    if bbox_zip_path and os.path.exists(bbox_zip_path):
        sync.upload_asset(JOB_ID, bbox_zip_path)

    stage1_payload = {
        "job_id": JOB_ID,
        "video_asset": video_asset,
        "asr_enabled": False,
        "ocr_engine": "paddleocr_mobile_dbnet",
        "subtitle_roi": subtitle_roi,
        "screen_texts": screen_texts,
        "bbox_zip_asset": os.path.basename(bbox_zip_path) if bbox_zip_path else "",
        "scene_context": "",
        "speaker_refs": {},
        "segments": segments,
    }
    with open(os.path.join(work_dir, f"{JOB_ID}_stage1.json"), "w", encoding="utf-8") as f:
        json.dump(stage1_payload, f, ensure_ascii=False, indent=2)

    sync.write_json(f"state/{JOB_ID}_stage1.json", stage1_payload, f"stage1 done (ocr mobile+dbnet): {JOB_ID}")

    updated_jobs, _ = sync.read_json("state/jobs.json")
    if JOB_ID in updated_jobs.get("jobs", {}):
        updated_jobs["jobs"][JOB_ID]["status"] = "STAGE1_DONE"
        sync.write_json("state/jobs.json", updated_jobs, f"update status STAGE1_DONE: {JOB_ID}")

    sync.trigger_dispatch("stage1_completed", {"job_id": JOB_ID})
    send_ntfy(
        f"[Stage 1 Done] {JOB_ID}",
        f"Hoàn tất OCR Mobile + DBNet ({len(screen_texts)} khung có bbox/chữ, tạm bỏ ASR) -> sang Stage 2.",
        tags="white_check_mark",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_ntfy(f"[Stage 1 Lỗi] {JOB_ID}", err, tags="x", priority="high")
        raise
