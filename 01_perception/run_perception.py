import functools
import os

# Vá triệt để lỗi Python 3.13 + numpy.ufunc cho pyannote/scipy/speechbrain trước mọi import
_orig_update_wrapper = functools.update_wrapper


def _safe_update_wrapper(
    wrapper,
    wrapped,
    assigned=functools.WRAPPER_ASSIGNMENTS,
    updated=functools.WRAPPER_UPDATES,
):
    for attr in assigned:
        try:
            val = getattr(wrapped, attr)
        except AttributeError:
            if attr == "__qualname__":
                val = getattr(wrapped, "__name__", "ufunc")
            elif attr == "__module__":
                val = "numpy"
            else:
                continue
        try:
            setattr(wrapper, attr, val)
        except (AttributeError, TypeError):
            pass
    for attr in updated:
        try:
            getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
        except (AttributeError, TypeError):
            pass
    try:
        wrapper.__wrapped__ = wrapped
    except (AttributeError, TypeError):
        pass
    return wrapper


try:
    _orig_update_wrapper.__code__ = _safe_update_wrapper.__code__
except Exception:
    pass
functools.update_wrapper = _safe_update_wrapper
functools.wraps = (
    lambda wrapped, assigned=functools.WRAPPER_ASSIGNMENTS, updated=functools.WRAPPER_UPDATES: functools.partial(
        _safe_update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated
    )
)


def _patch_pyannote_ufunc_compat():
    """Gắn module/qualname giả lập và vá wrapt/decorator/inspect cho numpy.ufunc trên Python 3.13."""
    import inspect

    try:
        import numpy as np
        if not hasattr(np.ufunc, "__qualname__"):
            try:
                type.__setattr__(np.ufunc, "__qualname__", "ufunc")
            except Exception:
                pass
        if not hasattr(np.ufunc, "__module__"):
            try:
                type.__setattr__(np.ufunc, "__module__", "numpy")
            except Exception:
                pass
    except Exception:
        pass

    try:
        orig_unwrap = inspect.unwrap

        def safe_unwrap(func, *, stop=None):
            if type(func).__name__ == "ufunc":
                return func
            return orig_unwrap(func, stop=stop)

        inspect.unwrap = safe_unwrap
    except Exception:
        pass

    try:
        import wrapt
        if hasattr(wrapt, "FunctionWrapper"):
            _orig_fw_init = wrapt.FunctionWrapper.__init__

            def _safe_fw_init(self, wrapped, wrapper, enabled=None):
                if type(wrapped).__name__ == "ufunc":
                    _w = lambda *a, **kw: wrapped(*a, **kw)
                    _w.__name__ = getattr(wrapped, "__name__", "ufunc")
                    _w.__qualname__ = getattr(wrapped, "__name__", "ufunc")
                    _w.__module__ = "numpy"
                    return _orig_fw_init(self, _w, wrapper, enabled=enabled)
                return _orig_fw_init(self, wrapped, wrapper, enabled=enabled)

            wrapt.FunctionWrapper.__init__ = _safe_fw_init
    except Exception:
        pass


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import base64
from difflib import SequenceMatcher
import gc
import glob
import json
import re
import shutil
import statistics
import subprocess
import sys
import sysconfig
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
HF_TOKEN = get_cfg("HF_TOKEN", "__INJECT_HF_TOKEN__")
NTFY_TOPIC = get_cfg("NTFY_TOPIC", "__INJECT_NTFY_TOPIC__")
NTFY_SERVER = (get_cfg("NTFY_SERVER", "__INJECT_NTFY_SERVER__") or "https://ntfy.sh").rstrip("/")
NTFY_TOKEN = get_cfg("NTFY_TOKEN", "__INJECT_NTFY_TOKEN__")

if HF_TOKEN and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = HF_TOKEN
if HF_TOKEN and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

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
    """Chỉ cài các gói còn thiếu một cách nhẹ nhàng để giảm tối đa thời gian khởi tạo."""
    wanted = [
        ("demucs", "demucs"),
        ("soundfile", "soundfile"),
        ("cv2", "opencv-python"),
        ("gdown", "gdown"),
    ]
    missing = [pip_name for mod_name, pip_name in wanted if not _is_installed(mod_name)]
    if missing:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", *missing],
            check=False,
        )


def _build_cuda_ld_library_path(venv_dir: str = "/tmp/ocr_venv") -> str:
    """Thu thập nhanh đường dẫn lib CUDA/cuDNN từ hệ thống & venv."""
    lib_dirs: list[str] = []
    search_roots = [
        os.path.join(venv_dir, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia"),
        os.path.join(sysconfig.get_paths().get("purelib", ""), "nvidia"),
        "/usr/local/lib/python3.13/dist-packages/nvidia",
        "/usr/local/lib/python3.12/dist-packages/nvidia",
    ]
    for root in search_roots:
        if os.path.isdir(root):
            for p in sorted(glob.glob(os.path.join(root, "*", "lib"))):
                if os.path.isdir(p) and p not in lib_dirs:
                    lib_dirs.append(p)

    for std_dir in [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ]:
        if os.path.isdir(std_dir) and std_dir not in lib_dirs:
            lib_dirs.append(std_dir)

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        for p in existing.split(":"):
            if p and p not in lib_dirs:
                lib_dirs.append(p)
    return ":".join(lib_dirs)


def ensure_ocr_venv(venv_dir: str = "/tmp/ocr_venv") -> str:
    """Khởi tạo venv cô lập nhanh qua --target, chỉ cài gói tối thiểu cần thiết cho RapidOCR GPU."""
    venv_py = os.path.join(venv_dir, "bin", "python")
    marker = os.path.join(venv_dir, ".rapidocr_gpu_v15_binsearch_ready")
    if os.path.exists(venv_py) and os.path.exists(marker):
        return venv_py

    if os.path.exists(venv_dir) and not os.path.exists(marker):
        shutil.rmtree(venv_dir, ignore_errors=True)

    subprocess.run([sys.executable, "-S", "-m", "venv", "--without-pip", venv_dir], check=True)

    site_pkgs = os.path.join(
        venv_dir, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"
    )
    os.makedirs(site_pkgs, exist_ok=True)

    pkgs = [
        "wrapt",
        "numpy>=1.26.4,<2.3.0",
        "onnxruntime-gpu==1.20.2",
        "rapidocr",
        "opencv-python-headless",
    ]

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--target", site_pkgs, *pkgs],
        check=True,
    )

    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    return venv_py


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
        "dbnet_thresh": float(os.environ.get("DBNET_THRESH", "0.34")),
        "dbnet_box_thresh": float(os.environ.get("DBNET_BOX_THRESH", "0.52")),
        "dbnet_unclip_ratio": float(os.environ.get("DBNET_UNCLIP_RATIO", "1.55")),
        "dbnet_limit_side_len": int(os.environ.get("DBNET_LIMIT_SIDE_LEN", "768")),
        "min_score": float(os.environ.get("OCR_MIN_SCORE", "0.60")),
        "step_sec": max(0.25, float(os.environ.get("OCR_STEP_SEC", "0.50"))),
        "coarse_step_sec": max(0.50, float(os.environ.get("OCR_COARSE_STEP_SEC", "1.00"))),
        "bin_tol_sec": max(0.12, float(os.environ.get("OCR_BIN_TOL_SEC", "0.20"))),
        "min_clean_chars": max(2, int(os.environ.get("OCR_MIN_CLEAN_CHARS", "2"))),
        "edge_min_density": float(os.environ.get("OCR_EDGE_MIN_DENSITY", "0.025")),
        "use_clahe": os.environ.get("OCR_USE_CLAHE", "1").strip() == "1",
        "save_bbox_vis": os.environ.get("SAVE_BBOX_VIS", "0").strip() == "1",
        "use_cuda": os.environ.get("OCR_USE_CUDA", "1").strip() == "1",
    }


_OCR_VENV_WORKER_CODE = r"""
import json
import os
import re
import sys
import time
import cv2
from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

req_path = sys.argv[1]
out_path = sys.argv[2]

with open(req_path, "r", encoding="utf-8") as f:
    req = json.load(f)

cfg = req["cfg"]
samples = req["samples"]
min_score = float(cfg.get("min_score", 0.60))
want_cuda = bool(cfg.get("use_cuda", True))

has_cuda = False
providers = []
if want_cuda:
    try:
        import onnxruntime as ort
        try:
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        except Exception as _dll_err:
            print(f"[VENV-OCR][WARN] preload_dlls cảnh báo: {_dll_err}", flush=True)
        raw_provs = list(ort.get_available_providers() or [])
        providers = [p for p in raw_provs if p != "TensorrtExecutionProvider"] or raw_provs
        has_cuda = "CUDAExecutionProvider" in providers
    except Exception as _ort_err:
        print(f"[VENV-OCR][WARN] Không kiểm tra được onnxruntime providers: {_ort_err}", flush=True)
        has_cuda = False

def build_ocr(use_gpu: bool):
    params = {
        "Global.use_cls": False,
        "Global.text_score": min_score,
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.lang_type": LangDet.CH,
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.thresh": float(cfg.get("dbnet_thresh", 0.34)),
        "Det.box_thresh": float(cfg.get("dbnet_box_thresh", 0.52)),
        "Det.unclip_ratio": float(cfg.get("dbnet_unclip_ratio", 1.55)),
        "Det.limit_side_len": int(cfg.get("dbnet_limit_side_len", 768)),
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": LangRec.CH,
        "Rec.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "EngineConfig.onnxruntime.use_cuda": use_gpu,
        "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "DEFAULT",
        "EngineConfig.onnxruntime.providers": ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"],
    }
    return RapidOCR(params=params), use_gpu

t0_init = time.time()
try:
    ocr, active_gpu = build_ocr(has_cuda)
except Exception as _gpu_init_err:
    print(f"[VENV-OCR][WARN] Khởi tạo GPU thất bại ({_gpu_init_err}), thử CPU.", flush=True)
    ocr, active_gpu = build_ocr(False)

actual_det_providers = []
for _attr_chain in (
    lambda o: o.text_det.session.session.get_providers(),
    lambda o: o.text_det.infer.session.get_providers(),
    lambda o: o.text_det.session.get_providers(),
):
    try:
        actual_det_providers = list(_attr_chain(ocr) or [])
        if actual_det_providers:
            break
    except Exception:
        continue

print(
    f"[VENV-OCR] RapidOCR PP-OCRv5 Mobile + DBNet sẵn sàng (use_cuda={active_gpu}, "
    f"det_providers={actual_det_providers or providers}).",
    flush=True,
)

def pad_to_multiple_of_32(img):
    h, w = img.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h == 0 and pad_w == 0:
        return img
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

def is_valid_det(txt: str, pts: list, crop_w: int, crop_h: int) -> bool:
    if not txt:
        return False
    if len(pts) >= 4 and crop_w > 0 and crop_h > 0:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        if bh > bw * 1.45 and min(xs) > crop_w * 0.75:
            return False
    return True

def apply_clahe_if_enabled(img, enabled: bool):
    if not enabled or img is None or img.size == 0:
        return img
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 4))
        l_eq = clahe.apply(l_ch)
        merged = cv2.merge((l_eq, a_ch, b_ch))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return img

def run_one(engine, crop):
    crop_h, crop_w = crop.shape[:2]
    enhanced = apply_clahe_if_enabled(crop, bool(cfg.get("use_clahe", True)))
    padded_crop = pad_to_multiple_of_32(enhanced)
    res = engine(padded_crop)
    boxes = getattr(res, "boxes", None)
    txts = getattr(res, "txts", None)
    scores = getattr(res, "scores", None)
    if boxes is None and isinstance(res, tuple) and len(res) >= 1 and res[0]:
        boxes = [x[0] for x in res[0]]
        txts = [x[1] for x in res[0]]
        scores = [x[2] for x in res[0]]
    if not txts:
        return []
    dets = []
    for i, txt_raw in enumerate(txts):
        txt = str(txt_raw or "").strip()
        sc = float(scores[i]) if (scores is not None and i < len(scores) and scores[i] is not None) else 1.0
        clean_chars = re.sub(r"[\s\W_]+", "", txt, flags=re.UNICODE)
        min_clean_chars = int(cfg.get("min_clean_chars", 2))
        if not clean_chars or sc < min_score:
            continue
        if len(clean_chars) < min_clean_chars:
            continue
        if len(set(clean_chars)) == 1 and len(clean_chars) >= 2:
            continue
        pts = []
        if boxes is not None and i < len(boxes) and boxes[i] is not None:
            try:
                arr = boxes[i].tolist() if hasattr(boxes[i], "tolist") else list(boxes[i])
                for p in arr:
                    if len(p) >= 2:
                        px = min(crop_w, max(0, int(round(float(p[0])))))
                        py = min(crop_h, max(0, int(round(float(p[1])))))
                        pts.append([px, py])
            except Exception:
                pts = []
        if len(pts) >= 4:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bh = max(ys) - min(ys)
            bw = max(xs) - min(xs)
            if bh < 9 or bw < 14:
                continue
            if min(ys) <= 2 or max(ys) >= crop_h - 2:
                continue
        if clean_chars.isascii() and len(clean_chars) <= 2 and sc < 0.80:
            continue
        if is_valid_det(txt, pts, crop_w, crop_h):
            dets.append({"poly": pts, "text": txt, "score": round(sc, 3)})
    return dets

gpu_fallback_done = False
results = []
for item in samples:
    ts = float(item["time"])
    cp = item["crop_path"]
    crop = cv2.imread(cp)
    if crop is None or crop.size == 0:
        results.append({"time": ts, "boxes": []})
        continue
    try:
        dets = run_one(ocr, crop)
    except Exception as e:
        if active_gpu and not gpu_fallback_done:
            print(f"[VENV-OCR] GPU lỗi ({e}), chuyển sang CPU 1 lần trong venv.", flush=True)
            gpu_fallback_done = True
            ocr, active_gpu = build_ocr(False)
            try:
                dets = run_one(ocr, crop)
            except Exception:
                dets = []
        else:
            dets = []
    results.append({"time": ts, "boxes": dets})

print(f"[VENV-OCR] Hoàn tất OCR {len(samples)} ảnh trong {time.time() - t0_init:.2f}s.", flush=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"active_gpu": active_gpu, "results": results}, f, ensure_ascii=False)
"""


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


def _has_subtitle_edge_signal(gray_band, cv2_mod, edge_thresh: float = 75.0, edge_min_density: float = 0.025) -> tuple[bool, float]:
    gx = cv2_mod.Sobel(gray_band, cv2_mod.CV_32F, 1, 0, ksize=3)
    gy = cv2_mod.Sobel(gray_band, cv2_mod.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2_mod.magnitude(gx, gy)
    density = float((grad_mag >= edge_thresh).mean())
    return (density >= edge_min_density), density


def _extract_roi_crop_at_time(cap, fps: float, duration: float, t_sec: float, roi: dict, cv2_mod):
    t_clamped = max(0.0, min(max(0.0, duration - 0.04), float(t_sec)))
    frame_no = int(round(t_clamped * fps))
    cap.set(cv2_mod.CAP_PROP_POS_FRAMES, frame_no)
    ok, frm = cap.read()
    if not ok or frm is None:
        return None
    fh, fw, _ = frm.shape
    crp = frm[int(fh * roi["y1"]) : int(fh * roi["y2"]), int(fw * roi["x1"]) : int(fw * roi["x2"])]
    return crp if (crp is not None and crp.size > 0) else None


def run_ocr_with_jump_step(video_path: str, work_dir: str) -> tuple[dict, list[dict], str, float]:
    """Định vị biên phụ đề bằng tìm kiếm nhị phân (Binary Search) trên trục thời gian."""
    roi = {"x1": 0.02, "y1": 0.58, "x2": 0.98, "y2": 0.97}
    vis_dir = os.path.join(work_dir, "bbox_vis")
    crops_dir = os.path.join(work_dir, "ocr_crops")
    zip_path = os.path.join(work_dir, f"{JOB_ID}_bbox_vis.zip")
    duration = 60.0

    try:
        import cv2

        db_cfg = _get_dbnet_config()
        save_vis = db_cfg["save_bbox_vis"]
        if save_vis:
            os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(crops_dir, exist_ok=True)

        venv_dir = "/tmp/ocr_venv"
        venv_py = ensure_ocr_venv(venv_dir)
        worker_py = os.path.join(work_dir, "_rapidocr_venv_worker.py")
        with open(worker_py, "w", encoding="utf-8") as f:
            f.write(_OCR_VENV_WORKER_CODE.lstrip())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 60.0

        step_sec = db_cfg["step_sec"]
        bin_tol_sec = float(db_cfg.get("bin_tol_sec", 0.20))
        edge_min_density = float(db_cfg.get("edge_min_density", 0.025))
        use_clahe = bool(db_cfg.get("use_clahe", True))
        clahe_obj = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 4)) if use_clahe else None

        edge_cache: dict[float, tuple[bool, float]] = {}
        crop_path_cache: dict[float, str] = {}
        idx_s = 0

        def eval_time(t_val: float) -> tuple[bool, float, str]:
            nonlocal idx_s
            t_r = round(max(0.0, min(duration, float(t_val))), 2)
            if t_r in edge_cache:
                has_sig, dens = edge_cache[t_r]
                return has_sig, dens, crop_path_cache.get(t_r, "")
            crp = _extract_roi_crop_at_time(cap, fps, duration, t_r, roi, cv2)
            if crp is None:
                edge_cache[t_r] = (False, 0.0)
                return False, 0.0, ""
            ch_h = crp.shape[0]
            sub_band = crp[int(ch_h * 0.25) :, :]
            gray = cv2.cvtColor(sub_band, cv2.COLOR_BGR2GRAY)
            if clahe_obj is not None:
                gray = clahe_obj.apply(gray)
            has_sig, dens = _has_subtitle_edge_signal(gray, cv2, edge_min_density=edge_min_density)
            edge_cache[t_r] = (has_sig, dens)
            if has_sig:
                cp = os.path.join(crops_dir, f"crop_{idx_s:04d}_{t_r:.2f}s.jpg")
                cv2.imwrite(cp, crp)
                crop_path_cache[t_r] = cp
                idx_s += 1
                return True, dens, cp
            return False, dens, ""

        def binary_search_onset(t_off: float, t_on: float) -> float:
            lo, hi = t_off, t_on
            for _ in range(8):
                if (hi - lo) <= bin_tol_sec:
                    break
                mid = round((lo + hi) * 0.5, 2)
                sig_mid, _, _ = eval_time(mid)
                if sig_mid:
                    hi = mid
                else:
                    lo = mid
            return hi

        def binary_search_offset(t_on: float, t_off: float) -> float:
            lo, hi = t_on, t_off
            for _ in range(8):
                if (hi - lo) <= bin_tol_sec:
                    break
                mid = round((lo + hi) * 0.5, 2)
                sig_mid, _, _ = eval_time(mid)
                if sig_mid:
                    lo = mid
                else:
                    hi = mid
            return lo

        # Tìm kiếm nhị phân trực tiếp trên toàn bộ khoảng [0.0, duration] (bỏ bước quét thô)
        candidate_times: list[float] = []

        def _collect_point(t_pt: float):
            pt_r = round(max(0.0, min(duration, float(t_pt))), 2)
            sig_p, _, _ = eval_time(pt_r)
            if sig_p and pt_r not in candidate_times:
                candidate_times.append(pt_r)

        def _binsearch_span(t_left: float, t_right: float, depth: int = 0):
            t_l = round(max(0.0, min(duration, float(t_left))), 2)
            t_r = round(max(0.0, min(duration, float(t_right))), 2)
            if t_r <= t_l:
                return
            sig_l, _, _ = eval_time(t_l)
            sig_r, _, _ = eval_time(t_r)
            if sig_l and t_l not in candidate_times:
                candidate_times.append(t_l)
            if sig_r and t_r not in candidate_times:
                candidate_times.append(t_r)

            if (t_r - t_l) <= bin_tol_sec or depth >= 10:
                if sig_l or sig_r:
                    _collect_point((t_l + t_r) * 0.5)
                return

            if not sig_l and sig_r:
                onset = binary_search_onset(t_l, t_r)
                _collect_point(onset)
                _collect_point((onset + t_r) * 0.5)
                return

            if sig_l and not sig_r:
                offset = binary_search_offset(t_l, t_r)
                _collect_point(offset)
                _collect_point((t_l + offset) * 0.5)
                return

            mid = round((t_l + t_r) * 0.5, 2)
            _binsearch_span(t_l, mid, depth + 1)
            _binsearch_span(mid, t_r, depth + 1)

        _binsearch_span(0.0, duration, depth=0)

        cap.release()

        sampled_entries = [
            {"time": t_k, "crop_path": crop_path_cache[t_k]}
            for t_k in sorted(candidate_times)
            if t_k in crop_path_cache and os.path.exists(crop_path_cache[t_k])
        ]

        req_json = os.path.join(work_dir, "_ocr_req.json")
        res_json = os.path.join(work_dir, "_ocr_res.json")
        with open(req_json, "w", encoding="utf-8") as f:
            json.dump({"cfg": db_cfg, "samples": sampled_entries}, f, ensure_ascii=False)

        worker_env = os.environ.copy()
        worker_env["LD_LIBRARY_PATH"] = _build_cuda_ld_library_path(venv_dir)

        proc = subprocess.run([venv_py, worker_py, req_json, res_json], env=worker_env, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout.strip())
        if proc.returncode != 0:
            raise RuntimeError(f"Worker RapidOCR trong venv lỗi (code {proc.returncode}): {proc.stderr[-500:]}")

        with open(res_json, "r", encoding="utf-8") as f:
            res_payload = json.load(f)
            active_gpu = bool(res_payload.get("active_gpu", False))
            raw_ocr_map = {float(x["time"]): (x.get("boxes") or []) for x in res_payload.get("results", [])}

        screen_texts: list[dict] = []
        vis_files: list[str] = []
        hits = 0

        cap_vis = cv2.VideoCapture(video_path) if save_vis else None
        for entry in sampled_entries:
            ts = float(entry["time"])
            dets = raw_ocr_map.get(ts, [])
            texts_found = [d["text"] for d in dets if str(d.get("text") or "").strip()]
            if dets or texts_found:
                hits += 1
                vis_name = f"frame_{hits - 1:03d}_{ts:.2f}s.jpg"
                if save_vis and cap_vis is not None:
                    cap_vis.set(cv2.CAP_PROP_POS_FRAMES, int(round(ts * fps)))
                    ok_v, frm_v = cap_vis.read()
                    if ok_v and frm_v is not None:
                        vp = os.path.join(vis_dir, vis_name)
                        _draw_bbox_visualization(frm_v, roi, dets, vp, ts)
                        if os.path.exists(vp):
                            vis_files.append(vp)
                screen_texts.append(
                    {
                        "time": ts,
                        "texts": texts_found,
                        "boxes": dets,
                        "vis_image": vis_name if save_vis else "",
                    }
                )

        if cap_vis is not None:
            cap_vis.release()

        if save_vis and vis_files:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for vf in vis_files:
                    zf.write(vf, arcname=os.path.basename(vf))

        shutil.rmtree(crops_dir, ignore_errors=True)

        print(
            f"[INFO] [RapidOCR venv (GPU={active_gpu}) | BinarySearch(tol={bin_tol_sec}s)] "
            f"ứng viên OCR={len(sampled_entries)} | hits={hits}"
        )
        return roi, screen_texts, zip_path if (save_vis and vis_files) else "", duration
    except Exception as e:
        shutil.rmtree(crops_dir, ignore_errors=True)
        print(f"[WARN] RapidOCR venv bỏ qua: {e}")
        return roi, [], "", duration


def _is_same_subtitle(t1: str, t2: str) -> bool:
    s1 = re.sub(r"[\s\W_]+", "", str(t1 or ""), flags=re.UNICODE).lower()
    s2 = re.sub(r"[\s\W_]+", "", str(t2 or ""), flags=re.UNICODE).lower()
    if not s1 or not s2:
        return False
    if s1 == s2 or s1 in s2 or s2 in s1:
        return True
    if min(len(s1), len(s2)) <= 3:
        return SequenceMatcher(None, s1, s2).ratio() >= 0.88
    return SequenceMatcher(None, s1, s2).ratio() >= 0.90


def build_ocr_segments_from_jumps(screen_texts: list[dict], duration: float = 60.0, step_sec: float = 0.50) -> list[dict]:
    grouped: list[dict] = []
    max_merge_gap = max(1.20, step_sec * 3.0)

    for item in sorted(screen_texts, key=lambda x: float(x.get("time", 0.0))):
        ts = float(item.get("time", 0.0))
        texts = [str(t).strip() for t in (item.get("texts") or []) if str(t).strip()]
        joined = " ".join(texts).strip()
        clean_joined = re.sub(r"[\s\W_]+", "", joined, flags=re.UNICODE)
        if not joined or len(clean_joined) < 2:
            continue

        merged_into_idx = -1
        for back_i in range(len(grouped) - 1, max(-1, len(grouped) - 3), -1):
            prev_g = grouped[back_i]
            if (ts - prev_g["_last_ts"]) <= max_merge_gap and _is_same_subtitle(prev_g["ocr_text"], joined):
                merged_into_idx = back_i
                break

        if merged_into_idx >= 0:
            target_g = grouped[merged_into_idx]
            if len(clean_joined) > len(re.sub(r"[\s\W_]+", "", target_g["ocr_text"], flags=re.UNICODE)):
                target_g["ocr_text"] = joined
                target_g["best_text"] = joined
            target_g["_last_ts"] = max(target_g["_last_ts"], ts)
            target_g["end"] = round(min(duration, max(target_g["end"], ts + step_sec * 0.5)), 2)
            continue

        start_t = round(max(0.0, ts - step_sec * 0.4), 2)
        end_t = round(min(duration, max(start_t + 0.35, ts + step_sec * 0.6)), 2)
        grouped.append(
            {
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
                "subtitle_source": "rapidocr_mobile_dbnet_binsearch",
                "boxes": item.get("boxes") or [],
                "vis_image": item.get("vis_image") or "",
            }
        )

    deduped: list[dict] = []
    for g in grouped:
        g.pop("_last_ts", None)
        if deduped:
            prev = deduped[-1]
            gap = float(g["start"]) - float(prev["end"])
            if gap <= 1.25 and _is_same_subtitle(prev["best_text"], g["best_text"]):
                if len(g["best_text"]) > len(prev["best_text"]):
                    prev["ocr_text"] = g["ocr_text"]
                    prev["best_text"] = g["best_text"]
                prev["end"] = round(min(duration, max(prev["end"], g["end"])), 2)
                continue
        deduped.append(g)

    for idx, g in enumerate(deduped):
        g["id"] = idx
        if idx + 1 < len(deduped) and deduped[idx]["end"] > deduped[idx + 1]["start"] - 0.03:
            deduped[idx]["end"] = round(max(deduped[idx]["start"] + 0.25, deduped[idx + 1]["start"] - 0.03), 2)

    return deduped


def _extract_voice_features(samples, sr: int) -> tuple[float | None, float]:
    import numpy as np

    if samples is None or len(samples) < max(64, sr // 10):
        return None, 0.0
    s = samples.astype(np.float32)
    s = s - float(np.mean(s))
    energy = float(np.sqrt(np.mean(s ** 2)))
    if energy < 1e-4:
        return None, 0.0

    frame_len = max(128, int(0.04 * sr))
    hop = max(64, frame_len // 2)
    min_lag = max(1, int(sr / 450.0))
    max_lag = min(frame_len - 1, int(sr / 75.0))
    frame_pitches: list[float] = []
    centroids: list[float] = []
    freqs = np.fft.rfftfreq(512, d=1.0 / sr)
    win = np.hanning(frame_len).astype(np.float32)

    for start in range(0, len(s) - frame_len + 1, hop):
        fr = s[start : start + frame_len]
        rms = float(np.sqrt(np.mean(fr ** 2)))
        if rms < energy * 0.35:
            continue
        win_fr = fr * win
        spec = np.abs(np.fft.rfft(win_fr, n=512))
        s_sum = float(np.sum(spec))
        if s_sum > 1e-6:
            centroids.append(float(np.sum(freqs * spec) / s_sum))

        corr = np.correlate(fr, fr, mode="full")[frame_len - 1 :]
        if corr[0] > 0 and max_lag > min_lag:
            sub_c = corr[min_lag:max_lag]
            b_idx = int(np.argmax(sub_c)) + min_lag
            if corr[b_idx] / corr[0] >= 0.28:
                frame_pitches.append(float(sr) / float(b_idx))

    pitch_med = round(float(statistics.median(frame_pitches)), 1) if frame_pitches else None
    cent_med = float(statistics.median(centroids)) if centroids else 0.0
    return pitch_med, cent_med


def _diarize_with_pyannote(vocals_wav: str, segments: list[dict], wav_data, sr: int, work_dir: str) -> dict | None:
    hf_tok = _clean_url_or_str(
        os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "") or HF_TOKEN
    )
    if not hf_tok:
        print("[WARN] Thiếu HF_TOKEN cho pyannote.audio, dùng VAD/pitch dự phòng.")
        return None

    model_id = (
        os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1").strip()
        or "pyannote/speaker-diarization-3.1"
    )

    try:
        _patch_pyannote_ufunc_compat()
        import numpy as np
        import soundfile as sf
        import torch
        from pyannote.audio import Pipeline

        try:
            pipeline = Pipeline.from_pretrained(model_id, token=hf_tok)
        except TypeError:
            pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_tok)

        if pipeline is None:
            print(f"[WARN] Không tải được pipeline pyannote ({model_id}).")
            return None

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        waveform_tensor = torch.from_numpy(wav_data.astype(np.float32)).unsqueeze(0)
        diarization = pipeline({"waveform": waveform_tensor, "sample_rate": int(sr)})
        annotation = getattr(diarization, "speaker_diarization", diarization)

        turns: list[tuple[float, float, str]] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            t0 = float(turn.start)
            t1 = float(turn.end)
            if t1 > t0:
                turns.append((t0, t1, str(speaker)))

        if not turns:
            print("[WARN] pyannote không trả về lượt nói nào, dùng VAD/pitch dự phòng.")
            return None

        raw_spk_order = []
        for _, _, spk_raw in sorted(turns, key=lambda x: x[0]):
            if spk_raw not in raw_spk_order:
                raw_spk_order.append(spk_raw)

        spk_pitches: dict[str, list[float]] = {k: [] for k in raw_spk_order}
        for t0, t1, spk_raw in turns:
            i0 = max(0, int(t0 * sr))
            i1 = min(len(wav_data), int(t1 * sr))
            if i1 > i0:
                p_val, _ = _extract_voice_features(wav_data[i0:i1], sr)
                if p_val is not None:
                    spk_pitches[spk_raw].append(p_val)

        raw_to_clean: dict[str, str] = {}
        male_idx = 0
        female_idx = 0
        for spk_raw in raw_spk_order:
            p_list = spk_pitches.get(spk_raw) or []
            med_p = float(statistics.median(p_list)) if p_list else None
            if med_p is not None and med_p >= 185.0:
                raw_to_clean[spk_raw] = f"SPEAKER_{female_idx:02d}_F"
                female_idx += 1
            else:
                raw_to_clean[spk_raw] = f"SPEAKER_{male_idx:02d}_M"
                male_idx += 1

        for seg in segments:
            s_start = float(seg["start"])
            s_end = float(seg["end"])
            overlap_by_spk: dict[str, float] = {}
            nearest_spk = raw_spk_order[0]
            best_dist = float("inf")

            for t0, t1, spk_raw in turns:
                ov = max(0.0, min(s_end, t1) - max(s_start, t0))
                if ov > 0.0:
                    overlap_by_spk[spk_raw] = overlap_by_spk.get(spk_raw, 0.0) + ov
                dist = (
                    0.0
                    if (t0 <= s_start <= t1 or t0 <= s_end <= t1)
                    else min(abs(t0 - s_end), abs(s_start - t1))
                )
                if dist < best_dist:
                    best_dist = dist
                    nearest_spk = spk_raw

            chosen_raw = max(overlap_by_spk.items(), key=lambda x: x[1])[0] if overlap_by_spk else nearest_spk
            clean_spk = raw_to_clean.get(chosen_raw, "SPEAKER_00_M")
            seg["speaker_id"] = clean_spk
            seg["gender_hint"] = "female" if clean_spk.endswith("_F") else "male"

            i0 = max(0, int(s_start * sr))
            i1 = min(len(wav_data), int(s_end * sr))
            p_seg, _ = _extract_voice_features(wav_data[i0:i1], sr) if i1 > i0 else (None, 0.0)
            if p_seg is not None:
                seg["pitch_hz"] = p_seg

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
            avg_pitch = round(float(statistics.mean(pitches)), 1) if pitches else None
            g_hint = "female" if spk.endswith("_F") else "male"
            speaker_refs[spk] = {
                "ref_asset": ref_name,
                "gender_hint": g_hint,
                "pitch_hz": avg_pitch,
            }

        del pipeline, diarization, waveform_tensor
        free_gpu()
        print(
            f"[INFO] pyannote.audio ({model_id}) phân tách thành công {len(speaker_refs)} người nói: "
            f"{list(speaker_refs.keys())}"
        )
        return speaker_refs
    except Exception as e:
        tb_full = traceback.format_exc()[-1500:]
        print(f"[WARN] pyannote.audio lỗi ({e}), chuyển sang VAD/pitch dự phòng.\n{tb_full}", flush=True)
        free_gpu()
        return None


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
        prev_end = float(segments[idx - 1]["end"]) + 0.03 if idx > 0 else 0.0
        next_start = float(segments[idx + 1]["start"]) - 0.03 if idx + 1 < len(segments) else total_dur

        search_s = max(prev_end, float(seg["start"]) - 0.30)
        search_e = min(next_start, float(seg["end"]) + 0.30)
        if search_e <= search_s + 0.15:
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
                new_e = round(
                    min(next_start, search_s + min((i1 - i0) / float(sr), (active[-1] + win) / float(sr))), 2
                )
                if new_e - new_s >= 0.20:
                    seg["start"] = new_s
                    seg["end"] = new_e

    pyannote_refs = _diarize_with_pyannote(vocals_wav, segments, wav_data, sr, work_dir)
    if pyannote_refs is not None:
        return pyannote_refs

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
        valid_p = [
            float(segments[k]["pitch_hz"])
            for k in idxs
            if segments[k].get("pitch_hz") and 65.0 <= float(segments[k]["pitch_hz"]) <= 185.0
        ]
        valid_c = [centroids[k] for k in idxs if centroids[k] > 0]
        p_med = float(statistics.median(valid_p)) if valid_p else 108.0
        c_med = float(statistics.median(valid_c)) if valid_c else 1100.0
        turn_scores.append((t_idx, p_med, c_med))

    turn_speaker: dict[int, str] = {}
    if len(turn_scores) >= 2:
        p_vals = np.array([x[1] for x in turn_scores], dtype=np.float32)
        c_vals = np.array([x[2] for x in turn_scores], dtype=np.float32)
        p_norm = (p_vals - float(np.mean(p_vals))) / max(1.0, float(np.std(p_vals)))
        c_norm = (c_vals - float(np.mean(c_vals))) / max(1.0, float(np.std(c_vals)))
        combo = [float(0.65 * p + 0.35 * c) for p, c in zip(p_norm, c_norm)]
        split_thr = float(statistics.median(combo))
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
        avg_pitch = round(float(statistics.mean(pitches)), 1) if pitches else None
        g_hint = "female" if spk.endswith("_F") else "male"
        speaker_refs[spk] = {
            "ref_asset": ref_name,
            "gender_hint": g_hint,
            "pitch_hz": avg_pitch,
        }

    print(
        f"[INFO] Căn chỉnh VAD & phân tách người nói (fallback) tạo {len(speaker_refs)} nhân vật: "
        f"{list(speaker_refs.keys())}"
    )
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
        src_tag = seg.get("subtitle_source", "rapidocr_mobile_dbnet_binsearch")
        blocks.append(
            f"{idx_out}\n"
            f"{_format_srt_time(start_t)} --> {_format_srt_time(end_t)}\n"
            f"[{spk} | {src_tag}] {txt}\n"
        )
    if not blocks:
        blocks.append("1\n00:00:00,000 --> 00:00:01,000\n[NO_OCR_TEXT_DETECTED]\n")
    with open(out_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks).strip() + "\n")
    print(f"[INFO] Đã xuất file SRT Stage 1 (RapidOCR venv GPU + BinarySearch): {out_srt_path} ({idx_out} dòng)")
    return out_srt_path


sync = GitHubSync(GH_REPO, GH_TOKEN)


def run_auto_health_check(work_dir: str, venv_dir: str = "/tmp/ocr_venv") -> dict:
    """Kiểm tra nhanh môi trường CUDA, ONNXRuntime GPU provider và RapidOCR venv."""
    import numpy as np

    report = {
        "python": sys.version.split()[0],
        "numpy": getattr(np, "__version__", "unknown"),
        "torch_cuda": False,
        "gpu_name": "none",
        "venv_python_exists": os.path.exists(os.path.join(venv_dir, "bin", "python")),
        "venv_ort_providers": [],
        "venv_cuda_provider": False,
        "venv_rapidocr_ok": False,
        "hf_token_present": bool(
            _clean_url_or_str(os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "") or HF_TOKEN)
        ),
        "ffmpeg_ok": shutil.which("ffmpeg") is not None,
    }

    try:
        import torch

        report["torch_cuda"] = bool(torch.cuda.is_available())
        if report["torch_cuda"]:
            report["gpu_name"] = str(torch.cuda.get_device_name(0))
    except Exception as e:
        report["torch_error"] = str(e)

    ld_path = _build_cuda_ld_library_path(venv_dir)
    venv_py = os.path.join(venv_dir, "bin", "python")
    if os.path.exists(venv_py):
        probe_code = (
            "import json\n"
            "out = {'providers': [], 'has_cuda': False, 'rapidocr_ok': False}\n"
            "try:\n"
            "    import onnxruntime as ort\n"
            "    provs = list(ort.get_available_providers() or [])\n"
            "    out['providers'] = provs\n"
            "    out['has_cuda'] = 'CUDAExecutionProvider' in provs\n"
            "    from rapidocr import RapidOCR\n"
            "    out['rapidocr_ok'] = True\n"
            "except Exception:\n"
            "    pass\n"
            "print(json.dumps(out))\n"
        )
        probe_env = os.environ.copy()
        probe_env["LD_LIBRARY_PATH"] = ld_path
        try:
            res = subprocess.run([venv_py, "-c", probe_code], env=probe_env, capture_output=True, text=True, timeout=20)
            if res.returncode == 0 and res.stdout.strip():
                parsed = json.loads(res.stdout.strip().splitlines()[-1])
                report["venv_ort_providers"] = parsed.get("providers") or []
                report["venv_cuda_provider"] = bool(parsed.get("has_cuda", False))
                report["venv_rapidocr_ok"] = bool(parsed.get("rapidocr_ok", False))
        except Exception:
            pass

    check_path = os.path.join(work_dir, f"{JOB_ID}_stage1_healthcheck.json")
    os.makedirs(work_dir, exist_ok=True)
    with open(check_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"[AUTO-CHECK] py={report['python']} numpy={report['numpy']} "
        f"torch_cuda={report['torch_cuda']}({report['gpu_name']}) | "
        f"venv_cuda={report['venv_cuda_provider']} rapidocr={report['venv_rapidocr_ok']} "
        f"hf_token={report['hf_token_present']} ffmpeg={report['ffmpeg_ok']}",
        flush=True,
    )
    return report


def main():
    send_ntfy(
        f"[Stage 1] {JOB_ID}",
        f"Bắt đầu nhận diện RapidOCR venv GPU (Binary Search) cho job {JOB_ID}.",
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
    ensure_ocr_venv("/tmp/ocr_venv")
    health_report = run_auto_health_check(work_dir, "/tmp/ocr_venv")
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
        "ocr_engine": "rapidocr_mobile_dbnet_binsearch",
        "ocr_step_sec": step_sec,
        "subtitle_roi": subtitle_roi,
        "screen_texts": screen_texts,
        "bbox_zip_asset": os.path.basename(bbox_zip_path) if bbox_zip_path else "",
        "scene_context": "",
        "health_check": health_report,
        "speaker_refs": speaker_refs,
        "segments": segments,
    }
    with open(os.path.join(work_dir, f"{JOB_ID}_stage1.json"), "w", encoding="utf-8") as f:
        json.dump(stage1_payload, f, ensure_ascii=False, indent=2)

    sync.write_json(
        f"state/{JOB_ID}_stage1.json",
        stage1_payload,
        f"stage1 done (rapidocr venv gpu+binsearch): {JOB_ID}",
    )

    updated_jobs, _ = sync.read_json("state/jobs.json")
    if JOB_ID in updated_jobs.get("jobs", {}):
        updated_jobs["jobs"][JOB_ID]["status"] = "STAGE1_DONE"
        sync.write_json("state/jobs.json", updated_jobs, f"update status STAGE1_DONE: {JOB_ID}")

    sync.trigger_dispatch("stage1_completed", {"job_id": JOB_ID})
    send_ntfy(
        f"[Stage 1 Done] {JOB_ID}",
        f"Hoàn tất RapidOCR venv GPU (Binary Search) ({len(segments)} đoạn, {len(speaker_refs)} người nói) -> sang Stage 2.",
        tags="white_check_mark",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_ntfy(f"[Stage 1 Lỗi] {JOB_ID}", err, tags="x", priority="high")
        raise
