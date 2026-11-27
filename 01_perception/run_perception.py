import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
        clean_name = _clean_url_or_str(filename).lstrip("/")
        base_name = os.path.basename(clean_name)
        candidates = {clean_name, base_name}

        matched = None
        for asset in release.get("assets", []):
            if asset.get("name") in candidates:
                matched = asset
                break

        if matched:
            dl_url = _clean_url_or_str(matched["url"])
            dl_headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/octet-stream"}
            with requests.get(dl_url, headers=dl_headers, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as out:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out.write(chunk)
            return dest_path

        raise FileNotFoundError(f"Không tìm thấy asset {filename} (hoặc {base_name}) trong release {tag}")

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
        ("onnxruntime", "onnxruntime"),
        ("rapidocr", "rapidocr"),
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
        return

    for local_cand in [ref, os.path.join(os.getcwd(), ref), os.path.join("/kaggle/working", ref)]:
        if local_cand and os.path.exists(local_cand) and os.path.getsize(local_cand) > 1024:
            if os.path.abspath(local_cand) != os.path.abspath(dest_path):
                subprocess.run(["cp", "-f", local_cand, dest_path], check=True)
            return

    for rel_tag in dict.fromkeys([tag, "uploads", "sources"]):
        try:
            return sync.download_asset(rel_tag, ref, dest_path)
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        f"Không tìm thấy video '{ref}' ở thư mục local lẫn GitHub Release ({tag}). "
        "Hãy kiểm tra bước upload video cục bộ trước khi trigger Stage 1."
    )


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
    return {
        "dbnet_thresh": float(os.environ.get("DBNET_THRESH", "0.32")),
        "dbnet_box_thresh": float(os.environ.get("DBNET_BOX_THRESH", "0.52")),
        "dbnet_unclip_ratio": float(os.environ.get("DBNET_UNCLIP_RATIO", "1.60")),
        "dbnet_limit_side_len": int(os.environ.get("DBNET_LIMIT_SIDE_LEN", "960")),
        "min_score": float(os.environ.get("OCR_MIN_SCORE", "0.65")),
        "step_sec": max(0.35, float(os.environ.get("OCR_STEP_SEC", "0.65"))),
        "idle_jump_sec": max(1.0, float(os.environ.get("OCR_IDLE_JUMP_SEC", "1.5"))),
        "save_bbox_vis": os.environ.get("SAVE_BBOX_VIS", "0").strip() == "1",
    }


def _create_mobile_dbnet_ocr(cfg: dict):
    from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    use_cuda = os.environ.get("OCR_USE_CUDA", "0").strip() == "1"

    engine_type = EngineType.ONNXRUNTIME
    params = {
        "Global.use_cls": False,
        "Global.text_score": cfg["min_score"],
        "Det.engine_type": engine_type,
        "Det.lang_type": LangDet.CH,
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.thresh": cfg["dbnet_thresh"],
        "Det.box_thresh": cfg["dbnet_box_thresh"],
        "Det.unclip_ratio": cfg["dbnet_unclip_ratio"],
        "Det.limit_side_len": cfg["dbnet_limit_side_len"],
        "Rec.engine_type": engine_type,
        "Rec.lang_type": LangRec.CH,
        "Rec.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "EngineConfig.onnxruntime.use_cuda": use_cuda,
    }

    try:
        ocr = RapidOCR(params=params)
        print(
            f"[INFO] Đã khởi tạo RapidOCR PP-OCRv5 Mobile + DBNet "
            f"(use_cuda={use_cuda}, step={cfg['step_sec']}s, idle={cfg['idle_jump_sec']}s, min_score={cfg['min_score']})."
        )
        return ocr
    except Exception as e:
        print(f"[WARN] Fallback RapidOCR cấu hình gọn do: {e}")
        return RapidOCR()


def _is_valid_subtitle_detection(txt: str, pts: list, crop_w: int, crop_h: int) -> bool:
    """Lọc bỏ nhãn tên/biểu ngữ góc hoặc chữ rác không phải phụ đề thoại."""
    s = str(txt or "").strip()
    if not s:
        return False
    if re.search(r"[A-Za-z]", s) and re.search(r"[\u4e00-\u9fff]", s) and len(s) <= 8:
        return False
    if len(pts) >= 4 and crop_w > 0 and crop_h > 0:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        if bh > bw * 1.15 and min(xs) > crop_w * 0.65:
            return False
    return True


def _run_dbnet_ocr_single(ocr_engine, crop, min_score: float = 0.65) -> list[dict]:
    if ocr_engine is None or crop is None:
        return []
    try:
        res = ocr_engine(crop)
    except Exception:
        return []

    crop_h, crop_w = crop.shape[:2]
    boxes = getattr(res, "boxes", None)
    txts = getattr(res, "txts", None)
    scores = getattr(res, "scores", None)

    if boxes is None and isinstance(res, tuple) and len(res) >= 1 and res[0]:
        boxes = [x[0] for x in res[0]]
        txts = [x[1] for x in res[0]]
        scores = [x[2] for x in res[0]]

    if not txts:
        return []

    detections: list[dict] = []
    for i, txt_raw in enumerate(txts):
        txt = str(txt_raw or "").strip()
        sc = float(scores[i]) if (scores is not None and i < len(scores) and scores[i] is not None) else 1.0
        clean_chars = re.sub(r"[\s\W_]+", "", txt, flags=re.UNICODE)
        if not clean_chars or sc < min_score:
            continue
        min_len = 1 if (sc >= 0.85 and re.search(r"[\u4e00-\u9fff]", clean_chars)) else 2
        if len(clean_chars) < min_len:
            continue

        pts = []
        if boxes is not None and i < len(boxes) and boxes[i] is not None:
            try:
                arr = boxes[i].tolist() if hasattr(boxes[i], "tolist") else list(boxes[i])
                for p in arr:
                    if len(p) >= 2:
                        pts.append([int(round(float(p[0]))), int(round(float(p[1])))])
            except Exception:
                pts = []

        if not _is_valid_subtitle_detection(txt, pts, crop_w, crop_h):
            continue

        detections.append({"poly": pts, "text": txt, "score": round(sc, 3)})
    return detections


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
    roi = {"x1": 0.04, "y1": 0.62, "x2": 0.96, "y2": 0.96}
    vis_dir = os.path.join(work_dir, "bbox_vis")
    zip_path = os.path.join(work_dir, f"{JOB_ID}_bbox_vis.zip")
    duration = 60.0

    try:
        import cv2

        db_cfg = _get_dbnet_config()
        save_vis = db_cfg["save_bbox_vis"]
        if save_vis:
            os.makedirs(vis_dir, exist_ok=True)

        ocr = _create_mobile_dbnet_ocr(db_cfg)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 60.0

        step_sec = db_cfg["step_sec"]
        idle_jump_sec = db_cfg["idle_jump_sec"]
        min_score = db_cfg["min_score"]

        def _probe_at(t_sec: float):
            clamped_t = max(0.0, min(duration, float(t_sec)))
            cap.set(cv2.CAP_PROP_POS_MSEC, clamped_t * 1000.0)
            ok, frm = cap.read()
            if not ok or frm is None:
                return None, []
            fh, fw, _ = frm.shape
            crp = frm[int(fh * roi["y1"]):int(fh * roi["y2"]), int(fw * roi["x1"]):int(fw * roi["x2"])]
            if crp.size == 0:
                return frm, []
            return frm, _run_dbnet_ocr_single(ocr, crp, min_score=min_score)

        cur_t = round(min(duration * 0.01, 0.35), 2)
        consecutive_empty = 0
        steps_visited = 0
        hits = 0
        screen_texts: list[dict] = []
        vis_files: list[str] = []

        while cur_t < duration:
            steps_visited += 1
            frame, dets = _probe_at(cur_t)
            if frame is None:
                cur_t = round(cur_t + step_sec, 2)
                continue

            texts_found = [d["text"] for d in dets if str(d.get("text") or "").strip()]

            if dets or texts_found:
                hits += 1
                consecutive_empty = 0
                vis_name = f"frame_{hits - 1:03d}_{cur_t:.2f}s.jpg"
                if save_vis:
                    vis_path = os.path.join(vis_dir, vis_name)
                    _draw_bbox_visualization(frame, roi, dets, vis_path, cur_t)
                    if os.path.exists(vis_path):
                        vis_files.append(vis_path)
                screen_texts.append({
                    "time": cur_t,
                    "texts": texts_found,
                    "boxes": dets,
                    "vis_image": vis_name if save_vis else "",
                })
                cur_t = round(cur_t + step_sec, 2)
            else:
                consecutive_empty += 1
                jump = idle_jump_sec if consecutive_empty >= 2 else step_sec
                cur_t = round(cur_t + jump, 2)

        cap.release()

        if save_vis and vis_files:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for vf in vis_files:
                    zf.write(vf, arcname=os.path.basename(vf))

        print(
            f"[INFO] [RapidOCR PP-OCRv5 Mobile + DBNet | Step={step_sec}s/Idle={idle_jump_sec}s] "
            f"mốc đã nhảy qua={steps_visited} | hits={hits}"
        )
        return roi, screen_texts, zip_path if (save_vis and vis_files) else "", duration
    except Exception as e:
        print(f"[WARN] RapidOCR Mobile / DBNet bỏ qua: {e}")
        return roi, [], "", duration


def _is_same_subtitle(t1: str, t2: str) -> bool:
    s1 = re.sub(r"[\s\W_]+", "", str(t1 or ""), flags=re.UNICODE).lower()
    s2 = re.sub(r"[\s\W_]+", "", str(t2 or ""), flags=re.UNICODE).lower()
    if not s1 or not s2:
        return False
    if s1 == s2:
        return True
    if min(len(s1), len(s2)) <= 3:
        return SequenceMatcher(None, s1, s2).ratio() >= 0.75
    return SequenceMatcher(None, s1, s2).ratio() >= 0.85


def build_ocr_segments_from_jumps(screen_texts: list[dict], duration: float = 60.0, step_sec: float = 0.65) -> list[dict]:
    grouped: list[dict] = []
    max_merge_gap = max(1.15, step_sec * 1.9)

    for item in sorted(screen_texts, key=lambda x: float(x.get("time", 0.0))):
        ts = float(item.get("time", 0.0))
        texts = [str(t).strip() for t in (item.get("texts") or []) if str(t).strip()]
        joined = " ".join(texts).strip()
        if not joined:
            continue

        if grouped and _is_same_subtitle(grouped[-1]["ocr_text"], joined) and (ts - grouped[-1]["_last_ts"]) <= max_merge_gap:
            if len(joined) > len(grouped[-1]["ocr_text"]):
                grouped[-1]["ocr_text"] = joined
                grouped[-1]["best_text"] = joined
            grouped[-1]["_last_ts"] = ts
            grouped[-1]["end"] = round(min(duration, ts + step_sec * 0.5), 2)
            continue

        start_t = round(max(0.0, ts - step_sec * 0.4), 2)
        end_t = round(min(duration, max(start_t + 0.45, ts + step_sec * 0.5)), 2)
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
            "subtitle_source": "rapidocr_mobile_dbnet_jump",
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
                if new_e - new_s >= 0.25:
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
        end_t = max(start_t + 0.25, float(seg.get("end", start_t + 0.65)))
        spk = seg.get("speaker_id", "SPEAKER_00")
        src_tag = seg.get("subtitle_source", "rapidocr_mobile_dbnet_jump")
        blocks.append(
            f"{idx_out}\n"
            f"{_format_srt_time(start_t)} --> {_format_srt_time(end_t)}\n"
            f"[{spk} | {src_tag}] {txt}\n"
        )
    if not blocks:
        blocks.append("1\n00:00:00,000 --> 00:00:01,000\n[NO_OCR_TEXT_DETECTED]\n")
    with open(out_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks).strip() + "\n")
    print(f"[INFO] Đã xuất file SRT Stage 1 (RapidOCR Mobile + DBNet + VAD): {out_srt_path} ({idx_out} dòng)")
    return out_srt_path


sync = GitHubSync(GH_REPO, GH_TOKEN)


def main():
    send_ntfy(
        f"[Stage 1] {JOB_ID}",
        f"Bắt đầu nhận diện RapidOCR PP-OCRv5 Mobile + DBNet + VAD (bỏ ASR) cho job {JOB_ID}.",
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
        "ocr_engine": "rapidocr_mobile_dbnet_jump",
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

    sync.write_json(f"state/{JOB_ID}_stage1.json", stage1_payload, f"stage1 done (rapidocr mobile+dbnet): {JOB_ID}")

    updated_jobs, _ = sync.read_json("state/jobs.json")
    if JOB_ID in updated_jobs.get("jobs", {}):
        updated_jobs["jobs"][JOB_ID]["status"] = "STAGE1_DONE"
        sync.write_json("state/jobs.json", updated_jobs, f"update status STAGE1_DONE: {JOB_ID}")

    sync.trigger_dispatch("stage1_completed", {"job_id": JOB_ID})
    send_ntfy(
        f"[Stage 1 Done] {JOB_ID}",
        f"Hoàn tất RapidOCR Mobile + DBNet + VAD ({len(segments)} đoạn, {len(speaker_refs)} người nói) -> sang Stage 2.",
        tags="white_check_mark",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_ntfy(f"[Stage 1 Lỗi] {JOB_ID}", err, tags="x", priority="high")
        raise
