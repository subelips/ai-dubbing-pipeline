import functools
import os

# Vá triệt để lỗi Python 3.13 + numpy.ufunc cho pyannote/scipy trước mọi import
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
    """Vá tương thích numpy.ma / scipy / pyannote trên Python 3.13 mà không làm hỏng C-type ufunc."""
    import builtins
    import inspect

    try:
        import numpy as np
        for _mod_attr in ("_core",):
            _sub = getattr(np, _mod_attr, None)
            _umath = getattr(_sub, "_multiarray_umath", None) if _sub is not None else None
            if _umath is not None and not hasattr(_umath, "_blas_supports_fpe"):
                try:
                    setattr(_umath, "_blas_supports_fpe", lambda *a, **kw: False)
                except Exception:
                    pass

        _orig_getattr = builtins.getattr
        if not getattr(_orig_getattr, "_ufunc_patched", False):
            def _safe_getattr(obj, name, *default):
                if name == "__qualname__" and isinstance(obj, np.ufunc):
                    return _orig_getattr(obj, "__name__", "ufunc")
                if name == "__module__" and isinstance(obj, np.ufunc):
                    return "numpy"
                return _orig_getattr(obj, name, *default)
            _safe_getattr._ufunc_patched = True
            builtins.getattr = _safe_getattr
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
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/root/.cache/huggingface/hub")
os.environ.setdefault("TORCH_HOME", "/root/.cache/torch")

# ROI phụ đề (dùng chung cho dò sự kiện và cắt ảnh OCR)
SUBTITLE_ROI = {"x1": 0.02, "y1": 0.55, "x2": 0.98, "y2": 0.72}
_RAPID_OCR_INSTANCE = None
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import base64
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import importlib.util
import gc
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import zipfile
import requests


def _install_numpy_py313_hook():
    """Vá tự động `numpy.ma.core` và `numpy._core.strings` trên Python 3.13 trước khi được nạp."""
    import importlib.abc
    import importlib.util

    targets = {"numpy.ma.core", "numpy._core.strings", "numpy.core.strings"}
    if targets.issubset(sys.modules.keys()):
        return

    class _NumpyPy313Patcher(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def __init__(self):
            self._origins = {}

        def find_spec(self, fullname, path, target=None):
            if fullname not in targets:
                return None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                if hasattr(finder, "find_spec"):
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None and spec.origin and spec.origin.endswith(".py"):
                        self._origins[fullname] = spec.origin
                        return importlib.util.spec_from_loader(fullname, self, origin=spec.origin)
            return None

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            origin = self._origins[module.__name__]
            with open(origin, "r", encoding="utf-8") as fh:
                source = fh.read()
            if module.__name__ == "numpy.ma.core":
                source = source.replace(
                    "ufunc.__qualname__",
                    "getattr(ufunc, '__qualname__', getattr(ufunc, '__name__', 'ufunc'))",
                )
            elif module.__name__ in ("numpy._core.strings", "numpy.core.strings"):
                source = re.sub(r"(?m)^_override___module__\(\)", "pass", source)
            code = compile(source, origin, "exec")
            exec(code, module.__dict__)

    if not any(type(f).__name__ == "_NumpyPy313Patcher" for f in sys.meta_path):
        sys.meta_path.insert(0, _NumpyPy313Patcher())


_install_numpy_py313_hook()


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
        return importlib.util.find_spec(pkg_name) is not None
    except ModuleNotFoundError:
        return False


def ensure_deps():
    """Chỉ cài gói thật sự còn thiếu; cho phép bỏ qua qua SKIP_ENSURE_DEPS=1."""
    if os.environ.get("SKIP_ENSURE_DEPS", "0").strip() == "1":
        return

    wanted = [
        ("demucs", "demucs"),
        ("sphn", "sphn"),
        ("julius", "julius"),
        ("lameenc", "lameenc"),
        ("soundfile", "soundfile"),
        ("cv2", "opencv-python"),
        ("gdown", "gdown"),
        ("pyannote.audio", "pyannote.audio"),
        ("transformers", "transformers>=4.49.0"),
        ("accelerate", "accelerate"),
        ("qwen_vl_utils", "qwen-vl-utils"),
        ("PIL", "pillow"),
    ]
    missing = [pip_spec for mod_name, pip_spec in wanted if not _is_installed(mod_name)]
    if missing:
        heavy_pkgs = {"pyannote.audio", "transformers>=4.49.0", "accelerate", "qwen-vl-utils", "demucs"}
        needs_deps = any(p in heavy_pkgs for p in missing)
        pip_cmd = [sys.executable, "-m", "pip", "install", "-q"]
        if not needs_deps:
            pip_cmd.append("--no-deps")
        pip_cmd.extend(missing)
        subprocess.run(pip_cmd, check=False)


def _get_rapidocr():
    """Khởi tạo RapidOCR (ONNX, CPU) theo kiểu lazy; cài --no-deps để không phá opencv sẵn có."""
    global _RAPID_OCR_INSTANCE
    if _RAPID_OCR_INSTANCE is not None:
        return _RAPID_OCR_INSTANCE
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "rapidocr-onnxruntime"],
            check=False,
        )
        if not _is_installed("onnxruntime"):
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=False)
        from rapidocr_onnxruntime import RapidOCR

    _RAPID_OCR_INSTANCE = RapidOCR()
    return _RAPID_OCR_INSTANCE


def prefetch_qwen_model_async(model_id: str):
    """Tải trước model Qwen về cache local song song khi Demucs tách nhạc."""
    try:
        from huggingface_hub import snapshot_download

        cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE") or None
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=False,
        )
    except Exception as e:
        print(f"[PREFETCH][WARN] Không thể tải trước {model_id}: {e}", flush=True)


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


def extract_vad_intervals_and_timestamps(vocals_wav: str, duration: float) -> tuple[list[list[float]], list[float]]:
    """Dùng VAD năng lượng thích ứng trên track vocals.wav sạch để tìm khoảng thoại và mốc lấy mẫu OCR tối ưu."""
    import numpy as np
    import soundfile as sf

    if not os.path.exists(vocals_wav):
        fallback_ts = [round(float(t), 2) for t in np.arange(0.25, max(0.3, duration), 0.50)]
        return [], fallback_ts

    wav, sr = sf.read(vocals_wav)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    total_dur = max(duration, len(wav) / float(sr))

    # Cửa sổ 30ms, bước nhảy 10ms
    win = max(1, int(0.03 * sr))
    hop = max(1, int(0.01 * sr))
    if len(wav) <= win:
        return [[0.0, round(total_dur, 2)]], [round(total_dur * 0.5, 2)]

    # Tính RMS nhanh bằng cumulative sum
    sq = np.square(wav.astype(np.float64))
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    starts = np.arange(0, len(wav) - win, hop)
    rms = np.sqrt(np.maximum(0.0, (csum[starts + win] - csum[starts]) / float(win)))

    noise_floor = float(np.percentile(rms, 20)) if len(rms) > 0 else 0.0
    peak_level = float(np.percentile(rms, 95)) if len(rms) > 0 else 0.01
    thresh = max(noise_floor * 2.5, peak_level * 0.08, 1e-4)

    active = rms >= thresh
    raw_intervals: list[list[float]] = []
    in_seg = False
    seg_s = 0.0

    for idx, is_speech in enumerate(active):
        t = (idx * hop) / float(sr)
        if is_speech and not in_seg:
            in_seg = True
            seg_s = t
        elif not is_speech and in_seg:
            in_seg = False
            if t - seg_s >= 0.16:
                raw_intervals.append([round(seg_s, 2), round(t, 2)])
    if in_seg and (total_dur - seg_s >= 0.16):
        raw_intervals.append([round(seg_s, 2), round(total_dur, 2)])

    # Gộp các khoảng ngắt hơi ngắn (< 0.25s) thuộc cùng một câu thoại
    merged: list[list[float]] = []
    for s, e in raw_intervals:
        if merged and (s - merged[-1][1]) < 0.25:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Chọn mốc thời gian OCR bên trong từng đoạn VAD
    sample_ts: list[float] = []
    for s, e in merged:
        seg_dur = e - s
        if seg_dur <= 1.65:
            # Câu ngắn/trung bình: lấy đúng điểm 45% thời lượng (tránh fade-in/out và không sót câu ngắn)
            mid_t = round(min(total_dur - 0.05, max(0.05, s + seg_dur * 0.45)), 2)
            sample_ts.append(mid_t)
        else:
            # Câu dài có thể chứa 2-3 dòng phụ đề liên tiếp: lấy mẫu mỗi 0.85s bên trong đoạn
            cur = s + 0.35
            while cur <= e - 0.20:
                sample_ts.append(round(min(total_dur - 0.05, cur), 2))
                cur += 0.85
            tail_t = round(min(total_dur - 0.05, e - 0.25), 2)
            if not sample_ts or (tail_t - sample_ts[-1]) >= 0.45:
                sample_ts.append(tail_t)

    # Loại bỏ các mốc quá sát nhau (< 0.20s)
    dedup_ts: list[float] = []
    for ts in sorted(set(sample_ts)):
        if not dedup_ts or (ts - dedup_ts[-1]) >= 0.20:
            dedup_ts.append(ts)

    # Video im lặng / VAD không bắt được thoại: fallback lấy mẫu đều để OCR vẫn chạy được
    if not dedup_ts:
        dedup_ts = [round(float(t), 2) for t in np.arange(0.25, max(0.3, total_dur), 0.50)]

    print(
        f"[VAD-GUIDED OCR] Tìm thấy {len(merged)} khoảng thoại -> chọn {len(dedup_ts)} mốc khung hình tối ưu.",
        flush=True,
    )
    return merged, dedup_ts


def _get_qwen25_vl_config() -> dict:
    return {
        "model_id": (
            os.environ.get("QWEN25_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct").strip()
            or "Qwen/Qwen2.5-VL-3B-Instruct"
        ),
        "step_sec": max(0.25, float(os.environ.get("OCR_STEP_SEC", "0.50"))),
        "batch_size": max(1, min(16, int(os.environ.get("QWEN_OCR_BATCH_SIZE", "10")))),
        "max_pixels": max(65536, int(os.environ.get("QWEN_OCR_MAX_PIXELS", str(256 * 28 * 28)))),
        "max_new_tokens": max(48, int(os.environ.get("QWEN_OCR_MAX_NEW_TOKENS", "96"))),
        "save_bbox_vis": os.environ.get("SAVE_BBOX_VIS", "0").strip() == "1",
        "use_vad_ocr": os.environ.get("USE_VAD_GUIDED_OCR", "1").strip() == "1",
    }


def _is_garbage_text(txt: str) -> bool:
    """True nếu chuỗi không có chữ/số thực sự (chỉ dấu câu, khoảng trắng...)."""
    s = str(txt or "").strip()
    if not s:
        return True
    return not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", s)


def _parse_qwen_ocr_and_speaker(raw_text: str) -> tuple[list[str], str]:
    """Tách văn bản phụ đề và mô tả ngắn nhân vật/người nói từ phản hồi Qwen2.5-VL-3B (reasoning=OFF)."""
    txt = re.sub(r"<think>.*?</think>", "", str(raw_text or ""), flags=re.DOTALL | re.IGNORECASE).strip()
    if not txt or txt.upper() in ("NONE", "NO_TEXT", "EMPTY", "N/A"):
        return [], "unknown"

    visual_spk = "unknown"
    text_lines: list[str] = []

    for ln in re.split(r"[\r\n]+", txt):
        s = ln.strip()
        if not s:
            continue
        m_spk = re.match(r"^(?:SPEAKER|SPK|PERSON|NHAN_VAT|NHÂN VẬT)\s*[:=]\s*(.+)$", s, flags=re.IGNORECASE)
        if m_spk:
            val = m_spk.group(1).strip().strip('"').strip("'").lower()
            if val and val not in ("none", "unknown", "n/a", "không rõ"):
                visual_spk = val
            continue
        m_txt = re.match(r"^(?:TEXT|SUBTITLE|OCR|PHU_DE|PHỤ ĐỀ)\s*[:=]\s*(.+)$", s, flags=re.IGNORECASE)
        if m_txt:
            s = m_txt.group(1).strip()

        cleaned = re.sub(r"^\s*[-*•\d.)]+\s*", "", s).strip().strip('"').strip("'")
        if not cleaned or cleaned.upper() in ("NONE", "NO_TEXT", "EMPTY", "N/A"):
            continue
        text_lines.append(cleaned)

    return text_lines, visual_spk


def _run_qwen25_vl_ocr(sampled_entries: list[dict], cfg: dict, cv2_mod) -> tuple[dict[float, dict], bool]:
    """Chạy Qwen/Qwen2.5-VL-3B-Instruct theo batch trên GPU (reasoning=OFF) cho cả OCR và gợi ý nhân vật."""
    if not sampled_entries:
        return {}, False

    _install_numpy_py313_hook()
    _patch_pyannote_ufunc_compat()
    import logging
    import warnings
    import torch
    from PIL import Image

    warnings.filterwarnings("ignore", message=".*Kwargs passed to `processor.__call__`.*")
    logging.getLogger("transformers").setLevel(logging.ERROR)

    model_id = cfg["model_id"]
    batch_size = max(1, int(cfg.get("batch_size", 8)))
    max_pixels = int(cfg.get("max_pixels", 384 * 28 * 28))
    max_new_tokens = int(cfg.get("max_new_tokens", 160))

    try:
        from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor as VLProcessorClass
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as VLModelClass
        processor = VLProcessorClass.from_pretrained(model_id, trust_remote_code=True)
    except Exception as _sub_err:
        print(f"[QWEN2.5-VL-3B][INFO] Submodule import fallback ({_sub_err}), dùng AutoProcessor...", flush=True)
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration as VLModelClass
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    img_proc = None
    try:
        img_proc = getattr(processor, "image_processor", None)
        if img_proc is not None:
            if hasattr(img_proc, "max_pixels"):
                img_proc.max_pixels = max_pixels
            if hasattr(img_proc, "min_pixels"):
                img_proc.min_pixels = min(getattr(img_proc, "min_pixels", 3136), 64 * 28 * 28)
    except Exception:
        pass

    try:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"
    except Exception:
        pass

    has_cuda = bool(torch.cuda.is_available())
    dtype = torch.float16 if has_cuda else torch.float32

    t0 = time.time()
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if has_cuda:
        model_kwargs["device_map"] = "auto"

    model = VLModelClass.from_pretrained(model_id, **model_kwargs)
    if not has_cuda:
        model.to("cpu")
    model.eval()
    print(
        f"[QWEN2.5-VL-3B] Đã tải {model_id} (dtype={dtype}, cuda={has_cuda}, "
        f"batch_size={batch_size}, max_pixels={max_pixels}, reasoning=OFF) trong {time.time() - t0:.2f}s",
        flush=True,
    )

    prompt_text = (
        "Đọc phụ đề và nhận diện người nói trong ảnh. Trả lời ngắn gọn đúng định dạng sau (không suy luận, không giải thích):\n"
        "SPEAKER: <mô tả ngắn gọn giới tính/độ tuổi/trang phục người nói hoặc off_screen/unknown>\n"
        "TEXT: <nội dung chữ phụ đề gốc, hoặc NONE nếu không có>"
    )

    raw_ocr_map: dict[float, dict] = {}
    t_inf = time.time()

    def _build_chat_str(img_obj) -> str:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_obj},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        try:
            return processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )

    try:
        target_dev = next(model.parameters()).device

        # Đọc ảnh theo từng batch để tiết kiệm RAM CPU trên video dài
        for b_start in range(0, len(sampled_entries), batch_size):
            raw_chunk = sampled_entries[b_start : b_start + batch_size]
            chunk = []
            for entry in raw_chunk:
                ts = float(entry["time"])
                cp = entry["crop_path"]
                bgr = cv2_mod.imread(cp)
                if bgr is None or bgr.size == 0:
                    raw_ocr_map[ts] = {"boxes": [], "visual_speaker": "unknown"}
                    continue
                rgb = cv2_mod.cvtColor(bgr, cv2_mod.COLOR_BGR2RGB)
                chunk.append((ts, Image.fromarray(rgb)))

            if not chunk:
                continue

            batch_ts = [x[0] for x in chunk]
            batch_imgs = [x[1] for x in chunk]
            batch_chats = [_build_chat_str(im) for im in batch_imgs]

            try:
                inputs = processor(
                    text=batch_chats,
                    images=batch_imgs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: (v.to(target_dev) if hasattr(v, "to") else v) for k, v in inputs.items()}

                with torch.inference_mode():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )

                in_len = inputs["input_ids"].shape[1]
                trimmed = [out_ids[in_len:] for out_ids in gen_ids]
                decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                for idx_c, ts_c in enumerate(batch_ts):
                    out_str = decoded[idx_c] if idx_c < len(decoded) else ""
                    lines, v_spk = _parse_qwen_ocr_and_speaker(out_str)
                    raw_ocr_map[ts_c] = {
                        "boxes": [{"poly": [], "text": ln, "score": 1.0} for ln in lines],
                        "visual_speaker": v_spk,
                    }
            except Exception as e_batch:
                print(f"[QWEN2.5-VL-3B][WARN] Batch lỗi ({e_batch}), chạy dự phòng từng ảnh trong batch...", flush=True)
                for ts_c, im_c in chunk:
                    try:
                        c_txt = _build_chat_str(im_c)
                        inp1 = processor(text=[c_txt], images=[im_c], padding=True, return_tensors="pt")
                        inp1 = {k: (v.to(target_dev) if hasattr(v, "to") else v) for k, v in inp1.items()}
                        with torch.inference_mode():
                            g1 = model.generate(**inp1, max_new_tokens=max_new_tokens, do_sample=False)
                        l1 = inp1["input_ids"].shape[1]
                        dec1 = processor.batch_decode([g1[0][l1:]], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        lines1, v_spk1 = _parse_qwen_ocr_and_speaker(dec1[0] if dec1 else "")
                        raw_ocr_map[ts_c] = {
                            "boxes": [{"poly": [], "text": ln, "score": 1.0} for ln in lines1],
                            "visual_speaker": v_spk1,
                        }
                    except Exception:
                        raw_ocr_map[ts_c] = {"boxes": [], "visual_speaker": "unknown"}

        # Fallback các khung OCR ra rác (chỉ dấu câu) bằng RapidOCR (ONNX, CPU) — ổn định hơn Qwen
        retry_items = []
        for entry in sampled_entries:
            ts_r = float(entry["time"])
            res_r = raw_ocr_map.get(ts_r) or {}
            txt_r = " ".join(str(b.get("text") or "") for b in (res_r.get("boxes") or []))
            if entry.get("event") and _is_garbage_text(txt_r):
                retry_items.append(entry)

        if retry_items:
            print(f"[QWEN2.5-VL-3B] {len(retry_items)} khung OCR rác -> retry zoom + RapidOCR...", flush=True)
            still_bad = []
            for entry in retry_items[:12]:
                zp = entry.get("zoom_path")
                if not zp:
                    still_bad.append(entry)
                    continue
                try:
                    bgr_z = cv2_mod.imread(zp)
                    if bgr_z is None or bgr_z.size == 0:
                        still_bad.append(entry)
                        continue
                    im_z = Image.fromarray(cv2_mod.cvtColor(bgr_z, cv2_mod.COLOR_BGR2RGB))
                    chat_z = _build_chat_str(im_z)
                    inp_z = processor(text=[chat_z], images=[im_z], padding=True, return_tensors="pt")
                    inp_z = {k: (v.to(target_dev) if hasattr(v, "to") else v) for k, v in inp_z.items()}
                    with torch.inference_mode():
                        g_z = model.generate(**inp_z, max_new_tokens=max_new_tokens, do_sample=False)
                    l_z = inp_z["input_ids"].shape[1]
                    dec_z = processor.batch_decode(
                        [g_z[0][l_z:]], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    lines_z, spk_z = _parse_qwen_ocr_and_speaker(dec_z[0] if dec_z else "")
                    if lines_z and not _is_garbage_text(" ".join(lines_z)):
                        raw_ocr_map[float(entry["time"])] = {
                            "boxes": [{"poly": [], "text": x, "score": 1.0} for x in lines_z],
                            "visual_speaker": spk_z,
                        }
                    else:
                        still_bad.append(entry)
                except Exception as e_z:
                    print(f"[ZOOM][WARN] {e_z}", flush=True)
                    still_bad.append(entry)
            retry_items = still_bad

            try:
                fb_ocr = _get_rapidocr()
                for entry in retry_items[:12]:
                    for cand_path in (
                        entry.get("crop_path"),
                        entry.get("alt2_path"),
                        entry.get("alt3_path"),
                        entry.get("alt_path"),
                    ):
                        if not cand_path:
                            continue
                        try:
                            bgr_r = cv2_mod.imread(cand_path)
                            if bgr_r is None or bgr_r.size == 0:
                                continue
                            fb_res, _ = fb_ocr(bgr_r)
                            texts_fb = [str(r[1]).strip() for r in (fb_res or []) if str(r[1]).strip()]
                            joined_fb = " ".join(texts_fb).strip()
                            if joined_fb and not _is_garbage_text(joined_fb):
                                raw_ocr_map[float(entry["time"])] = {
                                    "boxes": [{"poly": [], "text": t, "score": 1.0} for t in texts_fb],
                                    "visual_speaker": "unknown",
                                }
                                break
                        except Exception as e_fb:
                            print(f"[RAPIDOCR][WARN] {e_fb}", flush=True)
            except Exception as e_fb0:
                print(f"[RAPIDOCR][WARN] Không khởi tạo được fallback: {e_fb0}", flush=True)
    finally:
        del model, processor
        free_gpu()

    print(f"[QWEN2.5-VL-3B] Hoàn tất OCR + gợi ý nhân vật {len(sampled_entries)} ảnh trong {time.time() - t_inf:.2f}s.", flush=True)
    return raw_ocr_map, has_cuda


def _draw_roi_visualization(full_frame, roi: dict, out_vis_path: str, ts_sec: float):
    import cv2

    vis = full_frame.copy()
    h, w, _ = vis.shape
    rx1 = int(w * roi["x1"])
    ry1 = int(h * roi["y1"])
    rx2 = int(w * roi["x2"])
    ry2 = int(h * roi["y2"])

    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 220, 255), 2)
    cv2.putText(
        vis,
        f"t={ts_sec:.2f}s | Qwen2.5-VL-3B (VAD)",
        (max(8, rx1), max(22, ry1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        2,
    )
    cv2.imwrite(out_vis_path, vis)


def detect_subtitle_events(video_path: str, roi: dict, sample_fps: float = 10.0) -> list[dict]:
    """Dò nhanh các 'sự kiện' phụ đề (xuất hiện / đổi nội dung) bằng CV, không cần model OCR.

    Ý tưởng: phụ đề là chữ trắng có viền tối. Lấy mask = pixel sáng nằm cạnh pixel tối,
    chạy quét ROI ở ~10 fps; mask đổi mạnh (IoU thấp) => phụ đề mới. Cách này bắt được cả
    phụ đề xuất hiện lúc không có thoại (VAD bỏ qua) và rẻ hơn nhiều so với OCR mọi khung.
    """
    import cv2
    import numpy as np

    events: list[dict] = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        cap.release()
        return events

    step = max(1, int(round(fps / max(2.0, sample_fps))))
    kernel = np.ones((5, 5), np.uint8)
    close_k = np.ones((3, 3), np.uint8)
    cur = None
    idx = 0

    try:
        while True:
            # grab() rẻ hơn read() vì bỏ qua decode các khung không lấy mẫu
            if not cap.grab():
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    idx += 1
                    continue
                h, w, _ = frame.shape
                bx0, by0 = int(w * roi["x1"]), int(h * roi["y1"])
                band = frame[by0 : int(h * roi["y2"]), bx0 : int(w * roi["x2"])]
                if band.size == 0:
                    idx += 1
                    continue
                # Thu nhỏ để threshold nhanh hơn ~4 lần, không ảnh hưởng độ chính xác mask
                bh, bw = band.shape[:2]
                if bw > 720:
                    nw, nh = 720, max(1, int(bh * 720 / bw))
                    band = cv2.resize(band, (nw, nh))
                else:
                    nw, nh = bw, bh
                scale_x, scale_y = bw / float(nw), bh / float(nh)
                hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
                v = hsv[:, :, 2]
                s = hsv[:, :, 1]
                white = (v > 195) & (s < 70)
                dark = cv2.dilate((v < 80).astype(np.uint8), kernel)
                mask = cv2.morphologyEx((white & (dark > 0)).astype(np.uint8), cv2.MORPH_CLOSE, close_k)
                present = int(mask.sum()) >= 170 and int((mask.sum(axis=0) > 2).sum()) >= 28
                t = round(idx / fps, 2)

                # Bounding box của chữ trong khung hình (dùng để crop zoom khi ký tự quá ngắn)
                bbox = None
                if present:
                    # Bbox lấy trong dải dòng phụ đề (0.60-0.71) để tránh vật sáng ở nền
                    sub_y1, sub_y2 = int(h * 0.60), int(h * 0.71)
                    sub = frame[sub_y1:sub_y2, bx0 : int(w * roi["x2"])]
                    if sub.size:
                        hsv_s = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
                        wmask = (hsv_s[:, :, 2] > 195) & (hsv_s[:, :, 1] < 70)
                        ys, xs = np.where(wmask)
                        if len(xs):
                            bbox = (
                                int(bx0 + xs.min()),
                                int(sub_y1 + ys.min()),
                                int(bx0 + xs.max()),
                                int(sub_y1 + ys.max()),
                            )

                if present:
                    if cur is None:
                        cur = {"start": t, "end": t, "mask": mask, "bbox": bbox, "bbox_area": 0}
                    else:
                        inter = float((mask & cur["mask"]).sum())
                        union = float((mask | cur["mask"]).sum()) or 1.0
                        if inter / union < 0.45:
                            events.append({"start": cur["start"], "end": cur["end"], "bbox": cur.get("bbox")})
                            cur = {"start": t, "end": t, "mask": mask, "bbox": bbox, "bbox_area": 0}
                        else:
                            cur["end"] = t
                            cur["mask"] = mask
                    if bbox:
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area > cur.get("bbox_area", 0):
                            cur["bbox"] = bbox
                            cur["bbox_area"] = area
                elif cur is not None:
                    events.append({"start": cur["start"], "end": cur["end"], "bbox": cur.get("bbox")})
                    cur = None
            idx += 1
    finally:
        cap.release()

    if cur is not None:
        events.append({"start": cur["start"], "end": cur["end"], "bbox": cur.get("bbox")})

    cleaned = []
    for ev in events:
        if ev["end"] - ev["start"] < 0.08:
            continue
        cleaned.append(
            {
                "start": round(max(0.0, ev["start"] - 0.05), 2),
                "end": round(ev["end"] + 0.10, 2),
                "bbox": ev.get("bbox"),
            }
        )
    print(f"[SUBTITLE-DETECT] Tìm thấy {len(cleaned)} sự kiện phụ đề.", flush=True)
    return cleaned


def run_ocr_with_qwen25_vl(
    video_path: str,
    work_dir: str,
    vocals_wav: str = "",
    subtitle_events_in: list[dict] | None = None,
) -> tuple[dict, list[dict], str, float, list[list[float]], list[dict]]:
    """Lấy mẫu khung hình theo sự kiện phụ đề + mốc VAD và trích xuất phụ đề + gợi ý nhân vật bằng Qwen2.5-VL-3B."""
    roi = dict(SUBTITLE_ROI)
    vis_dir = os.path.join(work_dir, "bbox_vis")
    crops_dir = os.path.join(work_dir, "ocr_crops")
    zip_path = os.path.join(work_dir, f"{JOB_ID}_bbox_vis.zip")
    duration = 60.0
    vad_intervals: list[list[float]] = []
    subtitle_events: list[dict] = list(subtitle_events_in or [])

    try:
        import cv2

        cfg = _get_qwen25_vl_config()
        save_vis = cfg["save_bbox_vis"]
        if save_vis:
            os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(crops_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = round(total_frames / float(fps), 2) if fps > 0 else 60.0
        cap.release()

        # Dò sự kiện phụ đề bằng CV (rẻ, không cần model) để lấy mẫu đúng chỗ và đủ câu
        if not subtitle_events:
            subtitle_events = detect_subtitle_events(video_path, roi)

        if cfg["use_vad_ocr"] and vocals_wav and os.path.exists(vocals_wav):
            vad_intervals, vad_ts = extract_vad_intervals_and_timestamps(vocals_wav, duration)
        else:
            vad_ts = []
            _t = 0.25
            while _t < max(0.3, duration):
                vad_ts.append(round(float(_t), 2))
                _t += cfg["step_sec"]

        # Mỗi sự kiện phụ đề: 1 khung giữa + 1 khung dự phòng (dùng khi OCR ra rác)
        specs: list[dict] = []
        for ev in subtitle_events:
            span = max(0.12, ev["end"] - ev["start"])
            mid_t = round(
                min(max(ev["start"] + 0.18, ev["start"] + span * 0.45), max(ev["start"] + 0.10, ev["end"] - 0.06)),
                2,
            )
            alt_t = round(
                min(max(ev["start"] + 0.06, ev["start"] + span * 0.80), max(ev["start"] + 0.05, ev["end"] - 0.04)),
                2,
            )
            specs.append(
                {
                    "time": mid_t,
                    "alt_time": alt_t,
                    "start_time": round(ev["start"] + 0.10, 2),
                    "mid2_time": round(ev["start"] + span * 0.35, 2),
                    "event": ev,
                }
            )

        # Lưới an toàn: mốc VAD không nằm trong sự kiện nào (phụ đề lạ màu/vị trí, detector bỏ sót)
        def _covered(t: float, pad: float = 0.9) -> bool:
            return any((ev["start"] - pad) <= t <= (ev["end"] + pad) for ev in subtitle_events)

        for _t in vad_ts:
            _t = round(float(_t), 2)
            if specs and _covered(_t):
                continue
            specs.append({"time": _t, "alt_time": round(min(duration - 0.05, _t + 0.28), 2), "event": None})

        specs.sort(key=lambda x: x["time"])
        dedup_specs: list[dict] = []
        for sp in specs:
            if not dedup_specs or sp["time"] - dedup_specs[-1]["time"] >= 0.20:
                dedup_specs.append(sp)
        specs = dedup_specs

        cap = cv2.VideoCapture(video_path)
        sampled_entries = []
        for idx_s, sp in enumerate(specs):
            cur_t = sp["time"]
            cur_entry = None
            frames = [(cur_t, "")]
            alt_t = sp.get("alt_time")
            if alt_t is not None and abs(float(alt_t) - cur_t) >= 0.08:
                frames.append((round(float(alt_t), 2), "_alt"))
            start_t = sp.get("start_time")
            if start_t is not None and abs(float(start_t) - cur_t) >= 0.08 and (
                alt_t is None or abs(float(start_t) - float(alt_t)) >= 0.08
            ):
                frames.append((round(float(start_t), 2), "_alt2"))
            mid2_t = sp.get("mid2_time")
            used_ts = [cur_t] + [float(x[0]) for x in frames[1:]]
            if mid2_t is not None and all(abs(float(mid2_t) - u) >= 0.08 for u in used_ts):
                frames.append((round(float(mid2_t), 2), "_alt3"))
            for t_c, suffix in frames:
                target_f = min(max(0, total_frames - 1), int(round(t_c * fps))) if total_frames > 0 else int(round(t_c * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
                ok, frm = cap.read()
                if not ok or frm is None:
                    continue
                fh, fw, _ = frm.shape
                crp = frm[int(fh * roi["y1"]) : int(fh * roi["y2"]), int(fw * roi["x1"]) : int(fw * roi["x2"])]
                if crp is None or crp.size == 0:
                    continue
                cp = os.path.join(crops_dir, f"crop_{idx_s:04d}_{t_c:.2f}s{suffix}.jpg")
                cv2.imwrite(cp, crp)
                if suffix:
                    if cur_entry is not None:
                        if suffix == "_alt2":
                            cur_entry["alt2_path"] = cp
                        elif suffix == "_alt3":
                            cur_entry["alt3_path"] = cp
                        else:
                            cur_entry["alt_path"] = cp
                            cur_entry["alt_time"] = t_c
                else:
                    cur_entry = {"time": t_c, "crop_path": cp, "event": sp.get("event")}
                    sampled_entries.append(cur_entry)
                    # Crop zoom quanh bbox chữ (dòng ngắn 1-2 ký tự) để model đọc được
                    bb = (sp.get("event") or {}).get("bbox")
                    if bb:
                        zx1, zy1, zx2, zy2 = bb
                        bh0 = max(1, zy2 - zy1)
                        pad_x = max(14, int((zx2 - zx1) * 0.45))
                        pad_y = max(12, int((zy2 - zy1) * 0.55))
                        zx1, zy1 = max(0, zx1 - pad_x), max(0, zy1 - pad_y)
                        zx2, zy2 = min(frm.shape[1], zx2 + pad_x), min(frm.shape[0], zy2 + pad_y)
                        zc = frm[zy1:zy2, zx1:zx2]
                        if zc is not None and zc.size > 0:
                            sc = max(2.0, min(8.0, 200.0 / bh0))
                            zc = cv2.resize(zc, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
                            zp = os.path.join(crops_dir, f"crop_{idx_s:04d}_{t_c:.2f}s_zoom.jpg")
                            cv2.imwrite(zp, zc)
                            cur_entry["zoom_path"] = zp

        cap.release()
        print(
            f"[SUBTITLE-DETECT] {len(subtitle_events)} sự kiện phụ đề -> {len(sampled_entries)} khung OCR "
            f"(VAD {len(vad_intervals)} khoảng thoại).",
            flush=True,
        )

        raw_ocr_map, active_gpu = _run_qwen25_vl_ocr(sampled_entries, cfg, cv2)

        screen_texts: list[dict] = []
        vis_files: list[str] = []
        hits = 0

        cap_vis = cv2.VideoCapture(video_path) if save_vis else None
        for entry in sampled_entries:
            ts = float(entry["time"])
            item_res = raw_ocr_map.get(ts) or {"boxes": [], "visual_speaker": "unknown"}
            dets = item_res.get("boxes") or []
            v_spk = str(item_res.get("visual_speaker") or "unknown")
            texts_found = [d["text"] for d in dets if str(d.get("text") or "").strip()]
            if texts_found:
                hits += 1
                vis_name = f"frame_{hits - 1:03d}_{ts:.2f}s.jpg"
                if save_vis and cap_vis is not None:
                    cap_vis.set(cv2.CAP_PROP_POS_FRAMES, int(round(ts * fps)))
                    ok_v, frm_v = cap_vis.read()
                    if ok_v and frm_v is not None:
                        vp = os.path.join(vis_dir, vis_name)
                        _draw_roi_visualization(frm_v, roi, vp, ts)
                        if os.path.exists(vp):
                            vis_files.append(vp)
                screen_texts.append(
                    {
                        "time": ts,
                        "texts": texts_found,
                        "visual_speaker": v_spk,
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

        mode_str = "VAD-Guided" if (cfg["use_vad_ocr"] and vad_intervals) else f"Step={cfg['step_sec']}s"
        print(
            f"[INFO] [Qwen2.5-VL-3B (GPU={active_gpu}) | {mode_str}] "
            f"tổng frame OCR={len(sampled_entries)} | hits={hits}"
        )
        return roi, screen_texts, zip_path if (save_vis and vis_files) else "", duration, vad_intervals, subtitle_events
    except Exception as e:
        shutil.rmtree(crops_dir, ignore_errors=True)
        print(f"[WARN] Qwen2.5-VL-3B OCR lỗi: {e}\n{traceback.format_exc()[-1500:]}")
        return roi, [], "", duration, vad_intervals, subtitle_events


def _is_same_subtitle(t1: str, t2: str, strict: bool = False) -> bool:
    s1 = re.sub(r"[\s\W_]+", "", str(t1 or ""), flags=re.UNICODE).lower()
    s2 = re.sub(r"[\s\W_]+", "", str(t2 or ""), flags=re.UNICODE).lower()
    if not s1 or not s2:
        return False
    if s1 == s2:
        return True
    min_len = min(len(s1), len(s2))
    max_len = max(len(s1), len(s2))
    # "吃" nằm trong "该你吃了" nhưng là câu thoại khác -> chỉ coi là cùng câu khi độ dài tương đương
    if (s1 in s2 or s2 in s1) and (min_len / float(max(1, max_len))) >= 0.6:
        return True
    if min_len <= 3:
        return SequenceMatcher(None, s1, s2).ratio() >= (0.88 if strict else 0.85)
    thr = 0.85 if strict else 0.78
    return SequenceMatcher(None, s1, s2).ratio() >= thr


def _find_enclosing_vad(ts: float, vad_intervals: list[list[float]]) -> tuple[float, float] | None:
    """Tìm khoảng VAD chứa (hoặc gần nhất trong 0.35s) mốc thời gian ts."""
    best = None
    best_dist = 1e9
    for s, e in vad_intervals:
        if (s - 0.10) <= ts <= (e + 0.15):
            return (float(s), float(e))
        dist = min(abs(ts - s), abs(ts - e))
        if dist < best_dist and dist <= 0.35:
            best_dist = dist
            best = (float(s), float(e))
    return best


def _find_subtitle_event(ts: float, events: list[dict]) -> dict | None:
    """Tìm sự kiện phụ đề chứa mốc ts (hoặc gần nhất trong 0.45s)."""
    best = None
    best_dist = 1e9
    for ev in events or []:
        if (ev["start"] - 0.25) <= ts <= (ev["end"] + 0.25):
            return ev
        dist = min(abs(ts - ev["start"]), abs(ts - ev["end"]))
        if dist < best_dist and dist <= 0.45:
            best_dist = dist
            best = ev
    return best


def build_ocr_segments(
    screen_texts: list[dict],
    duration: float = 60.0,
    step_sec: float = 0.50,
    vad_intervals: list[list[float]] | None = None,
    subtitle_events: list[dict] | None = None,
) -> list[dict]:
    grouped: list[dict] = []
    max_merge_gap = max(1.35, step_sec * 3.0)
    vad_list = vad_intervals or []
    ev_list = subtitle_events or []

    for item in sorted(screen_texts, key=lambda x: float(x.get("time", 0.0))):
        ts = float(item.get("time", 0.0))
        texts = [str(t).strip() for t in (item.get("texts") or []) if str(t).strip()]
        v_spk = str(item.get("visual_speaker") or "unknown").strip()
        joined = " ".join(texts).strip()
        if not joined:
            continue

        ev = _find_subtitle_event(ts, ev_list)
        enc = _find_enclosing_vad(ts, vad_list) if vad_list else None

        if (
            grouped
            and _is_same_subtitle(grouped[-1]["ocr_text"], joined, strict=False)
            and (ts - grouped[-1]["_last_ts"]) <= max_merge_gap
        ):
            if len(joined) > len(grouped[-1]["ocr_text"]):
                grouped[-1]["ocr_text"] = joined
                grouped[-1]["best_text"] = joined
            if grouped[-1].get("visual_speaker", "unknown") == "unknown" and v_spk != "unknown":
                grouped[-1]["visual_speaker"] = v_spk
            grouped[-1]["_last_ts"] = ts
            if ev:
                ext_end = ev["end"]
            elif enc:
                ext_end = enc[1]
            else:
                ext_end = ts + step_sec * 0.5
            grouped[-1]["end"] = round(min(duration, max(grouped[-1]["end"], ext_end)), 2)
            continue

        if ev:
            # Dùng đúng biên hiển thị thật của phụ đề (chính xác hơn VAD)
            start_t = round(max(0.0, ev["start"]), 2)
            end_t = round(min(duration, ev["end"]), 2)
            if grouped and grouped[-1]["end"] > start_t:
                start_t = round(max(ev["start"], grouped[-1]["end"] + 0.03), 2)
                end_t = round(max(end_t, start_t + 0.12), 2)
        elif enc:
            # Nếu trong cùng 1 khoảng VAD dài có nhiều câu phụ đề khác nhau nối tiếp, giới hạn biên theo ts
            start_t = round(max(0.0, enc[0]), 2)
            if grouped and grouped[-1]["end"] > start_t:
                start_t = round(max(grouped[-1]["end"] + 0.03, ts - 0.35), 2)
            end_t = round(min(duration, max(start_t + 0.30, enc[1])), 2)
        else:
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
                "visual_speaker": v_spk,
                "pitch_hz": None,
                "asr_raw": "",
                "asr_text": "",
                "ocr_text": joined,
                "best_text": joined,
                "subtitle_source": "qwen2_5_vl_3b_instruct",
                "boxes": item.get("boxes") or [],
                "vis_image": item.get("vis_image") or "",
            }
        )

    for idx, g in enumerate(grouped):
        g["id"] = idx
        g.pop("_last_ts", None)
        if idx + 1 < len(grouped) and grouped[idx]["end"] > grouped[idx + 1]["start"] - 0.03:
            grouped[idx]["end"] = round(max(grouped[idx]["start"] + 0.25, grouped[idx + 1]["start"] - 0.03), 2)

    return grouped


def deduplicate_adjacent_segments(segments: list[dict]) -> list[dict]:
    """Khử trùng lặp hậu kỳ: gộp các đoạn liền kề có nội dung giống nhau bị tách nhầm do nhiễu biên."""
    if len(segments) <= 1:
        return segments

    deduped: list[dict] = [segments[0]]
    for seg in segments[1:]:
        prev = deduped[-1]
        gap = float(seg["start"]) - float(prev["end"])
        if gap <= 0.30 and _is_same_subtitle(prev.get("best_text", ""), seg.get("best_text", ""), strict=True):
            prev["end"] = max(float(prev["end"]), float(seg["end"]))
            if len(str(seg.get("best_text", ""))) > len(str(prev.get("best_text", ""))):
                prev["best_text"] = seg["best_text"]
                prev["ocr_text"] = seg["ocr_text"]
            if prev.get("visual_speaker", "unknown") == "unknown" and seg.get("visual_speaker", "unknown") != "unknown":
                prev["visual_speaker"] = seg["visual_speaker"]
        else:
            deduped.append(seg)

    for idx, g in enumerate(deduped):
        g["id"] = idx
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


def _extract_acoustic_fingerprint(samples, sr: int):
    """Trích xuất vector đặc trưng âm sắc (MFCC/spectral envelope + pitch) ổn định bằng numpy/torch."""
    import numpy as np

    if samples is None or len(samples) < max(160, sr // 5):
        return None
    s = samples.astype(np.float32)
    s = s - float(np.mean(s))
    rms = float(np.sqrt(np.mean(s ** 2)))
    if rms < 1e-4:
        return None
    s = s / rms

    n_fft = 512
    hop = 160
    if len(s) < n_fft:
        s = np.pad(s, (0, n_fft - len(s)))

    win = np.hanning(n_fft).astype(np.float32)
    frames = []
    for st in range(0, len(s) - n_fft + 1, hop):
        fr = s[st : st + n_fft]
        if float(np.sqrt(np.mean(fr ** 2))) >= 0.15:
            frames.append(fr * win)
    if not frames:
        frames.append(s[:n_fft] * win)

    mat = np.stack(frames, axis=0)
    mag = np.abs(np.fft.rfft(mat, n=n_fft, axis=-1)) + 1e-7
    log_mag = np.log(mag)

    n_bins = log_mag.shape[1]
    edges = np.linspace(0, n_bins, 33, dtype=int)
    bands = np.stack(
        [np.mean(log_mag[:, edges[i] : max(edges[i] + 1, edges[i + 1])], axis=-1) for i in range(32)],
        axis=-1,
    )
    mean_vec = np.mean(bands, axis=0)
    std_vec = np.std(bands, axis=0)
    feat = np.concatenate([mean_vec, std_vec]).astype(np.float32)
    nrm = float(np.linalg.norm(feat))
    if nrm <= 1e-8:
        return None
    return feat / nrm


def _infer_cluster_gender(pitches: list[float], visual_hints: list[str]) -> str:
    """Xác định giới tính ở cấp cụm speaker (ổn định hơn so với xét từng câu ngắn)."""
    female_votes = 0
    male_votes = 0
    for v in visual_hints:
        vl = (v or "").lower()
        f_kw = any(k in vl for k in ("nữ", "nu", "female", "woman", "girl", "cô", "chị", "bà"))
        m_kw = any(k in vl for k in ("nam", "male", "man", "boy", "anh", "chú", "ông", "thầy"))
        if f_kw and not m_kw:
            female_votes += 1
        elif m_kw and not f_kw:
            male_votes += 1

    if female_votes > male_votes:
        return "female"
    if male_votes > female_votes:
        return "male"

    valid_p = [float(p) for p in pitches if p is not None and 65.0 <= float(p) <= 380.0]
    if valid_p:
        return "female" if float(statistics.median(valid_p)) >= 185.0 else "male"
    return "male"


def _diarize_with_pyannote(vocals_wav: str, segments: list[dict], wav_data, sr: int, work_dir: str) -> dict | None:
    """Chạy pyannote.audio làm trục phân tách người nói chính, tránh gắn nhầm do đặc trưng câu ngắn."""
    hf_tok = _clean_url_or_str(
        os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "") or HF_TOKEN
    )
    model_id = (
        os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1").strip()
        or "pyannote/speaker-diarization-3.1"
    )

    import numpy as np
    import soundfile as sf

    if not hf_tok:
        return None

    pyannote_turns: list[tuple[float, float, str]] = []
    t_dia = time.time()
    try:
        _patch_pyannote_ufunc_compat()
        import torch
        from pyannote.audio import Pipeline

        cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE") or None
        try:
            pipeline = Pipeline.from_pretrained(model_id, token=hf_tok, cache_dir=cache_dir)
        except TypeError:
            try:
                pipeline = Pipeline.from_pretrained(model_id, token=hf_tok)
            except TypeError:
                pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_tok)

        if pipeline is None:
            return None

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        dia_kwargs = {}
        num_spk_env = os.environ.get("PYANNOTE_NUM_SPEAKERS", "").strip()
        min_spk_env = os.environ.get("PYANNOTE_MIN_SPEAKERS", "").strip()
        max_spk_env = os.environ.get("PYANNOTE_MAX_SPEAKERS", "").strip()
        if num_spk_env.isdigit() and int(num_spk_env) > 0:
            dia_kwargs["num_speakers"] = int(num_spk_env)
        else:
            if min_spk_env.isdigit() and int(min_spk_env) > 0:
                dia_kwargs["min_speakers"] = int(min_spk_env)
            if max_spk_env.isdigit() and int(max_spk_env) > 0:
                dia_kwargs["max_speakers"] = int(max_spk_env)

        waveform_tensor = torch.from_numpy(wav_data.astype(np.float32)).unsqueeze(0)
        diarization = pipeline({"waveform": waveform_tensor, "sample_rate": int(sr)}, **dia_kwargs)
        annotation = getattr(diarization, "speaker_diarization", diarization)
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            if float(turn.end) > float(turn.start):
                pyannote_turns.append((float(turn.start), float(turn.end), str(speaker)))

        del pipeline, diarization, waveform_tensor
        free_gpu()
        print(
            f"[PYANNOTE] Xong {len(pyannote_turns)} lượt nói trong {time.time() - t_dia:.2f}s.",
            flush=True,
        )
    except Exception as e:
        tb_full = traceback.format_exc()[-1200:]
        print(f"[WARN] pyannote.audio lỗi ({e}), chuyển sang fallback.\n{tb_full}", flush=True)
        free_gpu()
        return None

    if not pyannote_turns:
        return None

    raw_spk_per_seg: list[str] = []
    for seg in segments:
        s0 = float(seg["start"])
        s1 = float(seg["end"])
        i0 = max(0, int(s0 * sr))
        i1 = min(len(wav_data), int(s1 * sr))
        chunk = wav_data[i0:i1] if i1 > i0 else np.zeros(0, dtype=np.float32)

        pitch_val, _ = _extract_voice_features(chunk, sr) if len(chunk) >= max(64, sr // 10) else (None, 0.0)
        seg["pitch_hz"] = pitch_val

        ov_map: dict[str, float] = {}
        for pt0, pt1, p_spk in pyannote_turns:
            ov = max(0.0, min(s1, pt1) - max(s0, pt0))
            if ov > 0.0:
                ov_map[p_spk] = ov_map.get(p_spk, 0.0) + ov

        if ov_map:
            raw_spk_per_seg.append(max(ov_map.items(), key=lambda x: x[1])[0])
        else:
            raw_spk_per_seg.append("")

    for idx_s, spk_lbl in enumerate(raw_spk_per_seg):
        if spk_lbl:
            continue
        cur_s = float(segments[idx_s]["start"])
        cur_e = float(segments[idx_s]["end"])
        best_lbl = ""
        best_gap = 0.40
        for j_s, other_lbl in enumerate(raw_spk_per_seg):
            if not other_lbl or j_s == idx_s:
                continue
            o_s = float(segments[j_s]["start"])
            o_e = float(segments[j_s]["end"])
            gap = max(0.0, max(cur_s - o_e, o_s - cur_e))
            if gap <= best_gap:
                best_gap = gap
                best_lbl = other_lbl
        raw_spk_per_seg[idx_s] = best_lbl or "PY_SPK_00"

    raw_groups: dict[str, list[int]] = {}
    for idx_s, lbl in enumerate(raw_spk_per_seg):
        raw_groups.setdefault(lbl, []).append(idx_s)

    ordered_raw = sorted(raw_groups.items(), key=lambda kv: min(kv[1]))
    male_cnt = 0
    female_cnt = 0
    speaker_refs = {}

    for raw_lbl, mem_idxs in ordered_raw:
        pitches = [float(segments[k]["pitch_hz"]) for k in mem_idxs if segments[k].get("pitch_hz") is not None]
        v_hints = [str(segments[k].get("visual_speaker") or "") for k in mem_idxs]
        gender = _infer_cluster_gender(pitches, v_hints)

        if gender == "female":
            final_label = f"SPEAKER_{female_cnt:02d}_F"
            female_cnt += 1
        else:
            final_label = f"SPEAKER_{male_cnt:02d}_M"
            male_cnt += 1

        for k in mem_idxs:
            segments[k]["speaker_id"] = final_label
            segments[k]["gender_hint"] = gender

        ranked_m = sorted(mem_idxs, key=lambda k: float(segments[k]["end"]) - float(segments[k]["start"]), reverse=True)
        chunks = []
        total_samples = 0
        for k in ranked_m:
            t0 = float(segments[k]["start"])
            t1 = float(segments[k]["end"])
            a0 = max(0, int(t0 * sr))
            a1 = min(len(wav_data), int(t1 * sr))
            if a1 > a0:
                chunks.append(wav_data[a0:a1])
                total_samples += (a1 - a0)
            if total_samples >= sr * 5:
                break

        ref_name = f"ref_speaker_{final_label}.wav"
        ref_path = os.path.join(work_dir, ref_name)
        merged_wav = np.concatenate(chunks)[: sr * 6] if chunks else np.zeros(sr, dtype=np.float32)
        sf.write(ref_path, merged_wav, sr)

        avg_pitch = round(float(statistics.mean(pitches)), 1) if pitches else None
        speaker_refs[final_label] = {
            "ref_asset": ref_name,
            "gender_hint": gender,
            "pitch_hz": avg_pitch,
            "segments_count": len(mem_idxs),
            "pyannote_label": raw_lbl,
        }

    print(
        f"[INFO] Pyannote Speaker Diarization phân tách {len(speaker_refs)} người nói: "
        f"{list(speaker_refs.keys())}"
    )
    return speaker_refs

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
        src_tag = seg.get("subtitle_source", "qwen2_5_vl_3b_instruct")
        blocks.append(
            f"{idx_out}\n"
            f"{_format_srt_time(start_t)} --> {_format_srt_time(end_t)}\n"
            f"[{spk} | {src_tag}] {txt}\n"
        )
    if not blocks:
        blocks.append("1\n00:00:00,000 --> 00:00:01,000\n[NO_OCR_TEXT_DETECTED]\n")
    with open(out_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks).strip() + "\n")
    print(f"[INFO] Đã xuất file SRT Stage 1 (Qwen2.5-VL-3B): {out_srt_path} ({idx_out} dòng)")
    return out_srt_path


sync = GitHubSync(GH_REPO, GH_TOKEN)


def run_auto_health_check(work_dir: str) -> dict:
    """Kiểm tra nhanh môi trường CUDA, token HF và ffmpeg."""
    import numpy as np

    report = {
        "python": sys.version.split()[0],
        "numpy": getattr(np, "__version__", "unknown"),
        "torch_cuda": False,
        "gpu_name": "none",
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

    check_path = os.path.join(work_dir, f"{JOB_ID}_stage1_healthcheck.json")
    os.makedirs(work_dir, exist_ok=True)
    with open(check_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"[AUTO-CHECK] py={report['python']} numpy={report['numpy']} "
        f"torch_cuda={report['torch_cuda']}({report['gpu_name']}) | "
        f"hf_token={report['hf_token_present']} ffmpeg={report['ffmpeg_ok']}",
        flush=True,
    )
    return report


def main():
    send_ntfy(
        f"[Stage 1] {JOB_ID}",
        f"Bắt đầu nhận diện VAD-Guided OCR bằng Qwen2.5-VL-3B cho job {JOB_ID}.",
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
    health_report = run_auto_health_check(work_dir)
    download_video_source(sync, JOB_ID, video_asset, video_path)
    extract_audio_16k(video_path, audio_16k)

    qwen_model_id = _get_qwen25_vl_config()["model_id"]
    t_demucs = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_qwen = pool.submit(prefetch_qwen_model_async, qwen_model_id)
        vocals_wav, bgm_wav = separate_vocals_and_bgm(audio_16k, work_dir)
        fut_qwen.result()
    print(f"[TIMING] Tách vocals/BGM (demucs, song song prefetch Qwen): {time.time() - t_demucs:.2f}s", flush=True)

    t_ocr = time.time()
    subtitle_roi, screen_texts, bbox_zip_path, duration, vad_intervals, subtitle_events = run_ocr_with_qwen25_vl(
        video_path, work_dir, vocals_wav=vocals_wav
    )
    print(f"[TIMING] Dò phụ đề + OCR Qwen: {time.time() - t_ocr:.2f}s", flush=True)

    step_sec = _get_qwen25_vl_config()["step_sec"]
    segments = build_ocr_segments(
        screen_texts,
        duration=duration,
        step_sec=step_sec,
        vad_intervals=vad_intervals,
        subtitle_events=subtitle_events,
    )
    n_before = len(segments)
    segments = [s for s in segments if not _is_garbage_text(s.get("best_text") or s.get("ocr_text") or "")]
    if len(segments) != n_before:
        print(f"[INFO] Bỏ {n_before - len(segments)} đoạn rác (chỉ dấu câu).", flush=True)

    t_diar = time.time()
    speaker_refs = align_vad_and_diarize_from_vocals(vocals_wav, segments, work_dir)
    segments = deduplicate_adjacent_segments(segments)
    print(f"[TIMING] Diarization + gán speaker: {time.time() - t_diar:.2f}s", flush=True)

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
        "ocr_engine": "qwen2_5_vl_3b_instruct",
        "vad_guided_ocr": True,
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
        f"stage1 done (qwen2.5-vl-3b vad-guided): {JOB_ID}",
    )

    updated_jobs, _ = sync.read_json("state/jobs.json")
    if JOB_ID in updated_jobs.get("jobs", {}):
        updated_jobs["jobs"][JOB_ID]["status"] = "STAGE1_DONE"
        sync.write_json("state/jobs.json", updated_jobs, f"update status STAGE1_DONE: {JOB_ID}")

    sync.trigger_dispatch("stage1_completed", {"job_id": JOB_ID})
    send_ntfy(
        f"[Stage 1 Done] {JOB_ID}",
        f"Hoàn tất Qwen2.5-VL-3B (VAD-Guided) + AV Diarization ({len(segments)} đoạn, {len(speaker_refs)} người nói) -> sang Stage 2.",
        tags="white_check_mark",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_ntfy(f"[Stage 1 Lỗi] {JOB_ID}", err, tags="x", priority="high")
        raise
