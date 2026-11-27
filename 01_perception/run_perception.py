import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import base64
from difflib import SequenceMatcher
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

    def upload_asset(self, tag: str, file_path: str) -> str:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"[WARN] Bỏ qua upload file rỗng (0 byte): {file_path}")
            return ""
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


def _ensure_paddle_gpu():
    need_gpu_install = False
    try:
        import paddle
        if not paddle.device.is_compiled_with_cuda():
            need_gpu_install = True
    except Exception:
        need_gpu_install = True

    if need_gpu_install:
        print("[INFO] Cài đặt paddlepaddle-gpu (CUDA 12) để chạy PaddleOCR trên GPU...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "paddlepaddle"], check=False)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "paddlepaddle-gpu",
                "-i",
                "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
            ],
            check=False,
        )


def ensure_deps():
    _ensure_paddle_gpu()
    wanted = [
        ("demucs", "demucs"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("gdown", "gdown"),
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
        "min_score": float(os.environ.get("OCR_MIN_SCORE", "0.65")),
        "step_sec": max(0.4, float(os.environ.get("OCR_STEP_SEC", "0.8"))),
        "idle_jump_sec": max(1.0, float(os.environ.get("OCR_IDLE_JUMP_SEC", "2.4"))),
        "batch_size": max(1, min(16, int(os.environ.get("OCR_BATCH_SIZE", "8")))),
        "save_bbox_vis": os.environ.get("SAVE_BBOX_VIS", "0").strip() == "1",
    }


def _create_mobile_dbnet_ocr(lang: str, cfg: dict):
    import paddle
    from paddleocr import PaddleOCR

    try:
        paddle.set_flags({"FLAGS_use_mkldnn": False, "FLAGS_use_onednn": False, "FLAGS_enable_pir_api": False})
    except Exception:
        pass

    use_gpu = False
    try:
        use_gpu = bool(paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)
    except Exception:
        use_gpu = False

    if use_gpu:
        paddle.set_device("gpu:0")
    device_str = "gpu:0" if use_gpu else "cpu"

    candidate_kwargs = [
        dict(
            device=device_str,
            enable_mkldnn=False,
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
            device=device_str,
            enable_mkldnn=False,
            text_detection_model_name=cfg["det_model"],
            text_recognition_model_name=cfg["rec_model"],
            text_det_thresh=cfg["dbnet_thresh"],
            text_det_box_thresh=cfg["dbnet_box_thresh"],
            text_det_unclip_ratio=cfg["dbnet_unclip_ratio"],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        ),
        dict(
            device=device_str,
            enable_mkldnn=False,
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        ),
        dict(
            device=device_str,
            lang=lang,
        ),
    ]

    last_err = None
    for kw in candidate_kwargs:
        try:
            eng = PaddleOCR(**kw)
            print(f"[INFO] Đã khởi tạo PaddleOCR mobile với device={device_str}, compiled_cuda={use_gpu}.")
            return eng
        except (TypeError, ValueError) as e:
            last_err = e
            continue
    if last_err is not None:
        print(f"[WARN] Dùng cấu hình PaddleOCR mặc định do: {last_err}")
    return PaddleOCR(device=device_str, lang=lang)


def _parse_single_ocr_item(item, min_score: float = 0.65) -> list[dict]:
    detections: list[dict] = []
    if not item:
        return detections

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
        polys = res_obj.get("dt_polys")
        if polys is None:
            polys = res_obj.get("rec_polys")
        if polys is None:
            polys = res_obj.get("rec_boxes")
        if polys is None:
            polys = []

        rec_texts = res_obj.get("rec_texts") or []
        rec_scores = res_obj.get("rec_scores") or []

        count = max(len(polys), len(rec_texts))
        for i in range(count):
            poly_raw = polys[i] if i < len(polys) else None
            txt = str(rec_texts[i]).strip() if i < len(rec_texts) else ""
            sc = float(rec_scores[i]) if i < len(rec_scores) and rec_scores[i] is not None else 1.0
            clean_chars = re.sub(r"[\s\W_]+", "", txt, flags=re.UNICODE)
            min_len = 1 if (sc >= 0.85 and re.search(r"[\u4e00-\u9fff]", clean_chars)) else 2
            if sc < min_score or len(clean_chars) < min_len:
                continue

            pts = []
            if poly_raw is not None:
                try:
                    arr = poly_raw.tolist() if hasattr(poly_raw, "tolist") else list(poly_raw)
                    if len(arr) == 4 and all(isinstance(v, (int, float)) for v in arr):
                        x1, y1, x2, y2 = arr
                        pts = [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]
                    else:
                        for p in arr:
                            if len(p) >= 2:
                                pts.append([int(round(float(p[0]))), int(round(float(p[1])))])
                except Exception:
                    pts = []

            detections.append({"poly": pts, "text": txt, "score": round(sc, 3)})
        return detections

    if isinstance(item, (list, tuple)):
        for entry in item:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                raw_box = entry[0]
                info = entry[1]
                txt = str(info[0]).strip() if isinstance(info, (list, tuple)) and info else ""
                sc = float(info[1]) if isinstance(info, (list, tuple)) and len(info) >= 2 else 1.0
                clean_chars = re.sub(r"[\s\W_]+", "", txt, flags=re.UNICODE)
                min_len = 1 if (sc >= 0.85 and re.search(r"[\u4e00-\u9fff]", clean_chars)) else 2
                if sc < min_score or len(clean_chars) < min_len:
                    continue
                pts = []
                try:
                    for p in raw_box:
                        if len(p) >= 2:
                            pts.append([int(round(float(p[0]))), int(round(float(p[1])))])
                except Exception:
                    pts = []
                detections.append({"poly": pts, "text": txt, "score": round(sc, 3)})

    return detections


def _run_dbnet_ocr_batch(ocr_engine, crops: list, min_score: float = 0.65) -> list[list[dict]]:
    if ocr_engine is None or not crops:
        return [[] for _ in crops]
    try:
        if hasattr(ocr_engine, "predict"):
            res_p = list(ocr_engine.predict(crops))
            if len(res_p) == len(crops):
                return [_parse_single_ocr_item(it, min_score=min_score) for it in res_p]
            if res_p:
                out = [[] for _ in crops]
                for it in res_p:
                    idx_it = 0
                    if isinstance(it, dict) and "input_path" not in it and "index" in it:
                        idx_it = int(it["index"])
                    if 0 <= idx_it < len(out):
                        out[idx_it].extend(_parse_single_ocr_item(it, min_score=min_score))
                if any(out):
                    return out
    except Exception as e:
        print(f"[WARN] DBNet mobile batch predict lỗi: {e}")

    results = []
    for c in crops:
        try:
            r = ocr_engine.ocr(c)
            results.append(_parse_single_ocr_item(r, min_score=min_score))
        except Exception:
            results.append([])
    return results


def _draw_bbox_visualization(
    full_frame,
    roi: dict,
    detections_in_crop: list[dict],
    out_vis_path: str,
    ts_sec: float,
):
    import cv2
    import numpy as np

    vis = full_frame.copy()
    h, w, _ = vis.shape
    rx1 = int(w * roi["x1"])
    ry1 = int(h * roi["y1"])
    rx2 = int(w * roi["x2"])
    ry2 = int(h * roi["y2"])

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

    for idx, det in enumerate(detections_in_crop):
        pts = det.get("poly") or []
        sc = det.get("score", 0.0)
        if len(pts) >= 3:
            shifted = np.array([[p[0] + rx1, p[1] + ry1] for p in pts], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [shifted], isClosed=True, color=(0, 255, 0), thickness=2)
            top_x = int(np.min(shifted[:, 0, 0]))
            top_y = int(np.min(shifted[:, 0, 1]))
            cv2.putText(
                vis,
                f"#{idx + 1} ({sc:.2f})",
                (max(4, top_x), max(18, top_y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 0),
                1,
            )

    cv2.imwrite(out_vis_path, vis)


def run_ocr_with_jump_step(video_path: str, work_dir: str) -> tuple[dict, list[dict], str, float]:
    roi = {"x1": 0.02, "y1": 0.52, "x2": 0.98, "y2": 0.96}
    vis_dir = os.path.join(work_dir, "bbox_vis")
    zip_path = os.path.join(work_dir, f"{JOB_ID}_bbox_vis.zip")
    duration = 60.0

    try:
        import cv2

        db_cfg = _get_dbnet_config()
        save_vis = db_cfg["save_bbox_vis"]
        if save_vis:
            os.makedirs(vis_dir, exist_ok=True)

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

        step_sec = db_cfg["step_sec"]
        idle_jump_sec = db_cfg["idle_jump_sec"]
        min_score = db_cfg["min_score"]
        batch_size = db_cfg["batch_size"]

        # Lập danh sách mốc nhảy toàn video rồi chạy theo từng batch trên GPU
        jump_times: list[float] = []
        t_cursor = round(min(duration * 0.01, 0.4), 2)
        while t_cursor < duration:
            jump_times.append(t_cursor)
            t_cursor = round(t_cursor + step_sec, 2)

        hits = 0
        screen_texts: list[dict] = []
        vis_files: list[str] = []
        consecutive_empty = 0
        idx_t = 0

        while idx_t < len(jump_times):
            batch_ts: list[float] = []
            while idx_t < len(jump_times) and len(batch_ts) < batch_size:
                ts_val = jump_times[idx_t]
                # Nếu liên tiếp >= 2 mốc không có chữ, bỏ qua thêm 1 mốc để nhảy nhanh qua đoạn trống
                if consecutive_empty >= 2 and batch_ts:
                    idx_t += 1
                    consecutive_empty = 0
                    continue
                batch_ts.append(ts_val)
                idx_t += 1

            sampled = []
            for ts in batch_ts:
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                fh, fw, _ = frame.shape
                crop = frame[int(fh * roi["y1"]):int(fh * roi["y2"]), int(fw * roi["x1"]):int(fw * roi["x2"])]
                if crop.size > 0:
                    sampled.append((ts, frame, crop))

            if not sampled:
                continue

            batch_dets = _run_dbnet_ocr_batch(ocr, [x[2] for x in sampled], min_score=min_score)
            for (ts, frame, _), dets in zip(sampled, batch_dets):
                texts_found = [d["text"] for d in dets if str(d.get("text") or "").strip()]
                if dets or texts_found:
                    hits += 1
                    consecutive_empty = 0
                    vis_name = f"frame_{hits - 1:03d}_{ts:.2f}s.jpg"
                    if save_vis:
                        vis_path = os.path.join(vis_dir, vis_name)
                        _draw_bbox_visualization(frame, roi, dets, vis_path, ts)
                        if os.path.exists(vis_path):
                            vis_files.append(vis_path)
                    screen_texts.append({
                        "time": ts,
                        "texts": texts_found,
                        "boxes": dets,
                        "vis_image": vis_name if save_vis else "",
                    })
                else:
                    consecutive_empty += 1

        cap.release()

        if save_vis and vis_files:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for vf in vis_files:
                    zf.write(vf, arcname=os.path.basename(vf))

        print(
            f"[INFO] [PaddleOCR Mobile + DBNet GPU Batch={batch_size} | JumpStep={step_sec}s/Idle={idle_jump_sec}s] "
            f"lang={ocr_lang} det={db_cfg['det_model']} rec={db_cfg['rec_model']} "
            f"| mốc nhảy={len(jump_times)} | hits={hits}"
        )
        return roi, screen_texts, zip_path if (save_vis and vis_files) else "", duration
    except Exception as e:
        print(f"[WARN] PaddleOCR Mobile / DBNet bỏ qua: {e}")
        return roi, [], "", duration


def _is_same_subtitle(t1: str, t2: str) -> bool:
    s1 = re.sub(r"[\s\W_]+", "", str(t1 or ""), flags=re.UNICODE).lower()
    s2 = re.sub(r"[\s\W_]+", "", str(t2 or ""), flags=re.UNICODE).lower()
    if not s1 or not s2:
        return False
    if s1 == s2:
        return True
    return SequenceMatcher(None, s1, s2).ratio() >= 0.85


def build_ocr_segments_from_jumps(screen_texts: list[dict], duration: float = 60.0, step_sec: float = 0.8) -> list[dict]:
    grouped: list[dict] = []
    for item in sorted(screen_texts, key=lambda x: float(x.get("time", 0.0))):
        ts = float(item.get("time", 0.0))
        texts = [str(t).strip() for t in (item.get("texts") or []) if str(t).strip()]
        joined = " ".join(texts).strip()
        if not joined:
            continue

        if grouped and _is_same_subtitle(grouped[-1]["ocr_text"], joined) and (ts - grouped[-1]["_last_ts"]) <= (step_sec * 2.4):
            if len(joined) > len(grouped[-1]["ocr_text"]):
                grouped[-1]["ocr_text"] = joined
                grouped[-1]["best_text"] = joined
            grouped[-1]["_last_ts"] = ts
            grouped[-1]["end"] = round(min(duration, ts + step_sec * 0.55), 2)
            continue

        start_t = round(max(0.0, ts - step_sec * 0.35), 2)
        end_t = round(min(duration, max(start_t + 0.55, ts + step_sec * 0.55)), 2)
        grouped.append({
            "id": len(grouped),
            "start": start_t,
            "end": end_t,
            "_last_ts": ts,
            "speaker_id": "SPEAKER_00",
            "gender_hint": "unknown",
            "pitch_hz": None,
            "asr_raw": "",
            "asr_text": "",
            "ocr_text": joined,
            "best_text": joined,
            "subtitle_source": "ocr_mobile_dbnet_jump_gpu",
            "boxes": item.get("boxes") or [],
            "vis_image": item.get("vis_image") or "",
        })

    for idx, g in enumerate(grouped):
        g["id"] = idx
        g.pop("_last_ts", None)
        if idx + 1 < len(grouped) and grouped[idx]["end"] > grouped[idx + 1]["start"] - 0.04:
            grouped[idx]["end"] = round(max(grouped[idx]["start"] + 0.25, grouped[idx + 1]["start"] - 0.04), 2)

    return grouped


def _extract_voice_features(samples, sr: int) -> tuple[float | None, float]:
    import numpy as np

    if samples is None or len(samples) < max(64, sr // 8):
        return None, 0.0
    s = samples.astype(np.float32)
    s = s - np.mean(s)
    energy = float(np.sqrt(np.mean(s ** 2)))
    if energy < 1e-4:
        return None, 0.0

    frame_len = max(128, int(0.04 * sr))
    hop = max(64, frame_len // 2)
    min_lag = max(1, int(sr / 450.0))
    max_lag = min(frame_len - 1, int(sr / 75.0))
    frame_pitches = []
    centroids = []
    freqs = np.fft.rfftfreq(512, d=1.0 / sr)

    for start in range(0, len(s) - frame_len + 1, hop):
        fr = s[start: start + frame_len]
        rms = float(np.sqrt(np.mean(fr ** 2)))
        if rms < energy * 0.35:
            continue
        win_fr = fr * np.hanning(frame_len)
        spec = np.abs(np.fft.rfft(win_fr, n=512))
        s_sum = float(np.sum(spec))
        if s_sum > 1e-6:
            centroids.append(float(np.sum(freqs * spec) / s_sum))

        corr = np.correlate(fr, fr, mode="full")[frame_len - 1:]
        if corr[0] > 0 and max_lag > min_lag:
            sub_c = corr[min_lag:max_lag]
            b_idx = int(np.argmax(sub_c)) + min_lag
            if corr[b_idx] / corr[0] >= 0.28:
                frame_pitches.append(float(sr) / float(b_idx))

    pitch_med = round(float(np.median(frame_pitches)), 1) if frame_pitches else None
    cent_med = float(np.median(centroids)) if centroids else 0.0
    return pitch_med, cent_med


def align_vad_and_diarize_from_vocals(vocals_wav: str, segments: list[dict], work_dir: str) -> dict:
    import numpy as np
    import soundfile as sf

    if not segments or not os.path.exists(vocals_wav):
        return {}

    wav_data, sr = sf.read(vocals_wav)
    if wav_data.ndim > 1:
        wav_data = wav_data.mean(axis=-1)
    total_dur = len(wav_data) / float(sr)

    for idx, seg in enumerate(segments):
        prev_end = float(segments[idx - 1]["end"]) + 0.04 if idx > 0 else 0.0
        next_start = float(segments[idx + 1]["start"]) - 0.04 if idx + 1 < len(segments) else total_dur

        search_s = max(prev_end, float(seg["start"]) - 0.35)
        search_e = min(next_start, float(seg["end"]) + 0.35)
        if search_e <= search_s + 0.2:
            continue

        i0 = max(0, int(search_s * sr))
        i1 = min(len(wav_data), int(search_e * sr))
        chunk = np.abs(wav_data[i0:i1])
        peak = float(np.max(chunk)) if len(chunk) > 0 else 0.0
        if peak > 1e-4:
            win = max(1, int(0.015 * sr))
            active = np.where(chunk >= max(1e-3, peak * 0.12))[0]
            if len(active) > 0:
                new_s = round(max(prev_end, search_s + max(0.0, (active[0] - win) / float(sr))), 2)
                new_e = round(min(next_start, search_s + min((i1 - i0) / float(sr), (active[-1] + win) / float(sr))), 2)
                if new_e - new_s >= 0.30:
                    seg["start"] = new_s
                    seg["end"] = new_e

    centroids: list[float] = []
    for seg in segments:
        i0 = max(0, int(float(seg["start"]) * sr))
        i1 = min(len(wav_data), int(float(seg["end"]) * sr))
        pitch, cent = _extract_voice_features(wav_data[i0:i1], sr) if i1 > i0 else (None, 0.0)
        seg["pitch_hz"] = pitch
        centroids.append(cent)
        if pitch is not None:
            seg["gender_hint"] = "female" if pitch >= 195.0 else "male"
        else:
            seg["gender_hint"] = "unknown"

    last_known_g = "male"
    for seg in segments:
        if seg["gender_hint"] in ("male", "female"):
            last_known_g = seg["gender_hint"]
        else:
            seg["gender_hint"] = last_known_g

    male_turns: list[list[int]] = []
    cur_turn: list[int] = []
    for i, seg in enumerate(segments):
        if seg["gender_hint"] != "male":
            if cur_turn:
                male_turns.append(cur_turn)
                cur_turn = []
            continue
        if not cur_turn:
            cur_turn = [i]
        else:
            gap = float(seg["start"]) - float(segments[cur_turn[-1]]["end"])
            if gap <= 1.35:
                cur_turn.append(i)
            else:
                male_turns.append(cur_turn)
                cur_turn = [i]
    if cur_turn:
        male_turns.append(cur_turn)

    turn_scores = []
    for t_idx, idxs in enumerate(male_turns):
        valid_p = [float(segments[k]["pitch_hz"]) for k in idxs if segments[k].get("pitch_hz") and 65.0 <= float(segments[k]["pitch_hz"]) <= 185.0]
        valid_c = [centroids[k] for k in idxs if centroids[k] > 0]
        p_med = float(np.median(valid_p)) if valid_p else 108.0
        c_med = float(np.median(valid_c)) if valid_c else 1100.0
        turn_scores.append((t_idx, p_med, c_med))

    turn_speaker: dict[int, str] = {}
    if len(turn_scores) >= 2:
        p_vals = np.array([x[1] for x in turn_scores], dtype=np.float32)
        c_vals = np.array([x[2] for x in turn_scores], dtype=np.float32)
        p_norm = (p_vals - np.mean(p_vals)) / max(1.0, float(np.std(p_vals)))
        c_norm = (c_vals - np.mean(c_vals)) / max(1.0, float(np.std(c_vals)))
        combo = 0.65 * p_norm + 0.35 * c_norm
        split_thr = float(np.median(combo))
        first_is_low = combo[0] <= split_thr
        for (t_idx, _, _), sc in zip(turn_scores, combo):
            is_low = sc <= split_thr
            turn_speaker[t_idx] = "SPEAKER_00_M" if (is_low == first_is_low) else "SPEAKER_02_M"
    else:
        for t_idx, _, _ in turn_scores:
            turn_speaker[t_idx] = "SPEAKER_00_M"

    idx_to_male_spk: dict[int, str] = {}
    for t_idx, idxs in enumerate(male_turns):
        for k in idxs:
            idx_to_male_spk[k] = turn_speaker.get(t_idx, "SPEAKER_00_M")

    for i, seg in enumerate(segments):
        if seg["gender_hint"] == "female":
            seg["speaker_id"] = "SPEAKER_01_F"
        else:
            seg["speaker_id"] = idx_to_male_spk.get(i, "SPEAKER_00_M")

    speaker_refs = {}
    spk_groups: dict[str, list[tuple[float, float]]] = {}
    for seg in segments:
        spk_groups.setdefault(seg["speaker_id"], []).append((float(seg["start"]), float(seg["end"])))

    for spk, intervals in spk_groups.items():
        chunks = []
        total_len = 0
        for t0, t1 in sorted(intervals, key=lambda x: x[1] - x[0], reverse=True):
            a0 = max(0, int(t0 * sr))
            a1 = min(len(wav_data), int(t1 * sr))
            if a1 > a0:
                chunks.append(wav_data[a0:a1])
                total_len += a1 - a0
            if total_len >= sr * 5:
                break
        ref_name = f"ref_speaker_{spk}.wav"
        ref_path = os.path.join(work_dir, ref_name)
        merged_wav = np.concatenate(chunks)[: sr * 6] if chunks else np.zeros(sr, dtype=np.float32)
        sf.write(ref_path, merged_wav, sr)

        pitches = [float(s["pitch_hz"]) for s in segments if s["speaker_id"] == spk and s.get("pitch_hz")]
        avg_pitch = round(float(np.mean(pitches)), 1) if pitches else None
        g_hint = "female" if spk.endswith("_F") else "male"
        speaker_refs[spk] = {
            "ref_asset": ref_name,
            "gender_hint": g_hint,
            "pitch_hz": avg_pitch,
        }

    print(f"[INFO] Căn chỉnh VAD & phân tách người nói tạo {len(speaker_refs)} nhân vật: {list(speaker_refs.keys())}")
    return speaker_refs


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
        spk = seg.get("speaker_id", "SPEAKER_00")
        src_tag = seg.get("subtitle_source", "ocr_mobile_dbnet_jump_gpu")
        blocks.append(
            f"{idx_out}\n"
            f"{_format_srt_time(start_t)} --> {_format_srt_time(end_t)}\n"
            f"[{spk} | {src_tag}] {txt}\n"
        )
    if not blocks:
        blocks.append("1\n00:00:00,000 --> 00:00:01,000\n[NO_OCR_TEXT_DETECTED]\n")
    with open(out_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks).strip() + "\n")
    print(f"[INFO] Đã xuất file SRT Stage 1 (GPU Batch Jump OCR + VAD): {out_srt_path} ({idx_out} dòng)")
    return out_srt_path


sync = GitHubSync(GH_REPO, GH_TOKEN)


def main():
    send_ntfy(
        f"[Stage 1] {JOB_ID}",
        f"Bắt đầu nhận diện GPU Batch Jump OCR (PaddleOCR Mobile + DBNet + VAD, bỏ ASR) cho job {JOB_ID}.",
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

    vocals_wav, bgm_wav = separate_vocals_and_bgm(audio_16k, work_dir)
    subtitle_roi, screen_texts, bbox_zip_path, duration = run_ocr_with_jump_step(video_path, work_dir)
    step_sec = _get_dbnet_config()["step_sec"]
    segments = build_ocr_segments_from_jumps(screen_texts, duration=duration, step_sec=step_sec)
    speaker_refs = align_vad_and_diarize_from_vocals(vocals_wav, segments, work_dir)

    srt_s1_path = os.path.join(work_dir, f"{JOB_ID}_stage1.srt")
    export_stage1_srt(segments, srt_s1_path)
    sync.upload_asset(JOB_ID, srt_s1_path)
    sync.upload_asset(JOB_ID, bgm_wav)
    for ref_info in speaker_refs.values():
        ref_local = os.path.join(work_dir, ref_info["ref_asset"])
        if os.path.exists(ref_local):
            sync.upload_asset(JOB_ID, ref_local)
    if bbox_zip_path and os.path.exists(bbox_zip_path):
        sync.upload_asset(JOB_ID, bbox_zip_path)

    stage1_payload = {
        "job_id": JOB_ID,
        "video_asset": video_asset,
        "asr_enabled": False,
        "ocr_engine": "paddleocr_mobile_dbnet_jump_gpu",
        "ocr_step_sec": step_sec,
        "subtitle_roi": subtitle_roi,
        "screen_texts": screen_texts,
        "bbox_zip_asset": os.path.basename(bbox_zip_path) if bbox_zip_path else "",
        "scene_context": "",
        "speaker_refs": speaker_refs,
        "segments": segments,
    }
    with open(os.path.join(work_dir, f"{JOB_ID}_stage1.json"), "w", encoding="utf-8") as f:
        json.dump(stage1_payload, f, ensure_ascii=False, indent=2)

    sync.write_json(f"state/{JOB_ID}_stage1.json", stage1_payload, f"stage1 done (gpu jump ocr): {JOB_ID}")

    updated_jobs, _ = sync.read_json("state/jobs.json")
    if JOB_ID in updated_jobs.get("jobs", {}):
        updated_jobs["jobs"][JOB_ID]["status"] = "STAGE1_DONE"
        sync.write_json("state/jobs.json", updated_jobs, f"update status STAGE1_DONE: {JOB_ID}")

    sync.trigger_dispatch("stage1_completed", {"job_id": JOB_ID})
    send_ntfy(
        f"[Stage 1 Done] {JOB_ID}",
        f"Hoàn tất GPU Jump OCR + VAD ({len(segments)} đoạn, {len(speaker_refs)} người nói) -> sang Stage 2.",
        tags="white_check_mark",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_ntfy(f"[Stage 1 Lỗi] {JOB_ID}", err, tags="x", priority="high")
        raise
