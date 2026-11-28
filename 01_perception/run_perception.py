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

        # Trên Python 3.13, numpy.ma.core truy cập trực tiếp `ufunc.__qualname__` (C slot không có sẵn).
        # Thay vì đụng vào C-type slot, chỉ chặn nếu có code đọc qua getattr hoặc vá numpy.ma.core trước khi import.
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
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import base64
from difflib import SequenceMatcher
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
                source = source.replace(
                    'ufunc.__module__ = "numpy.strings"',
                    'try:\n            ufunc.__module__ = "numpy.strings"\n        except (AttributeError, TypeError):\n            pass',
                )
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
        __import__(pkg_name)
        return True
    except Exception:
        return False


def ensure_deps():
    """Cài các gói thiếu cho tách giọng, diarization và Qwen2.5-VL-3B."""
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
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "numpy>=1.26.4,<2.3.0", *missing],
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


def _get_qwen25_vl_config() -> dict:
    return {
        "model_id": (
            os.environ.get("QWEN25_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct").strip()
            or "Qwen/Qwen2.5-VL-3B-Instruct"
        ),
        "step_sec": max(0.25, float(os.environ.get("OCR_STEP_SEC", "0.50"))),
        "max_new_tokens": max(32, int(os.environ.get("QWEN_OCR_MAX_NEW_TOKENS", "128"))),
        "save_bbox_vis": os.environ.get("SAVE_BBOX_VIS", "0").strip() == "1",
    }


def _parse_qwen_ocr_response(raw_text: str) -> list[str]:
    # Loại bỏ mọi khối <think>...</think> nếu có và đọc trực tiếp văn bản phụ đề
    txt = re.sub(r"<think>.*?</think>", "", str(raw_text or ""), flags=re.DOTALL | re.IGNORECASE).strip()
    if not txt or txt.upper() in ("NONE", "NO_TEXT", "EMPTY", "N/A"):
        return []
    lines = []
    for ln in re.split(r"[\r\n]+", txt):
        cleaned = re.sub(r"^\s*[-*•\d.)]+\s*", "", ln).strip().strip('"').strip("'")
        if not cleaned or cleaned.upper() in ("NONE", "NO_TEXT", "EMPTY", "N/A"):
            continue
        lines.append(cleaned)
    return lines


def _run_qwen25_vl_ocr(sampled_entries: list[dict], cfg: dict, cv2_mod) -> tuple[dict[float, list[dict]], bool]:
    """Chạy Qwen/Qwen2.5-VL-3B-Instruct trực tiếp trên ảnh RGB gốc, vô hiệu hóa hoàn toàn reasoning."""
    if not sampled_entries:
        return {}, False

    _install_numpy_py313_hook()
    _patch_pyannote_ufunc_compat()
    import torch
    from PIL import Image

    model_id = cfg["model_id"]
    max_new_tokens = int(cfg.get("max_new_tokens", 128))

    try:
        from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor as VLProcessorClass
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as VLModelClass
        processor = VLProcessorClass.from_pretrained(model_id, trust_remote_code=True)
    except Exception as _sub_err:
        print(f"[QWEN2.5-VL-3B][INFO] Submodule import fallback ({_sub_err}), dùng AutoProcessor...", flush=True)
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration as VLModelClass
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
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
    print(f"[QWEN2.5-VL-3B] Đã tải {model_id} (dtype={dtype}, cuda={has_cuda}, reasoning=OFF) trong {time.time() - t0:.2f}s", flush=True)

    prompt_text = (
        "Trích xuất chính xác toàn bộ chữ phụ đề xuất hiện trong ảnh. "
        "Chỉ xuất trực tiếp văn bản gốc, không suy luận, không giải thích. "
        "Nếu không có chữ phụ đề, trả lời duy nhất: NONE"
    )

    raw_ocr_map: dict[float, list[dict]] = {}
    t_inf = time.time()

    try:
        for entry in sampled_entries:
            ts = float(entry["time"])
            cp = entry["crop_path"]
            bgr = cv2_mod.imread(cp)
            if bgr is None or bgr.size == 0:
                raw_ocr_map[ts] = []
                continue

            rgb = cv2_mod.cvtColor(bgr, cv2_mod.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            try:
                # Vô hiệu hóa reasoning / thinking mode ở mức chat template
                try:
                    chat_text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    chat_text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                inputs = processor(
                    text=[chat_text],
                    images=[pil_img],
                    padding=True,
                    return_tensors="pt",
                )
                target_dev = next(model.parameters()).device
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
                out_str = decoded[0] if decoded else ""
                lines = _parse_qwen_ocr_response(out_str)
                raw_ocr_map[ts] = [{"poly": [], "text": ln, "score": 1.0} for ln in lines]
            except Exception as e_item:
                print(f"[QWEN2.5-VL-3B][WARN] Lỗi OCR tại t={ts:.2f}s: {e_item}", flush=True)
                raw_ocr_map[ts] = []
    finally:
        del model, processor
        free_gpu()

    print(f"[QWEN2.5-VL-3B] Hoàn tất OCR {len(sampled_entries)} ảnh trong {time.time() - t_inf:.2f}s.", flush=True)
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
        f"t={ts_sec:.2f}s | Qwen2.5-VL-3B",
        (max(8, rx1), max(22, ry1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        2,
    )
    cv2.imwrite(out_vis_path, vis)


def run_ocr_with_qwen25_vl(video_path: str, work_dir: str) -> tuple[dict, list[dict], str, float]:
    """Lấy mẫu khung hình tuần tự theo step_sec và trích xuất phụ đề trực tiếp bằng Qwen2.5-VL-3B."""
    roi = {"x1": 0.02, "y1": 0.58, "x2": 0.98, "y2": 0.97}
    vis_dir = os.path.join(work_dir, "bbox_vis")
    crops_dir = os.path.join(work_dir, "ocr_crops")
    zip_path = os.path.join(work_dir, f"{JOB_ID}_bbox_vis.zip")
    duration = 60.0

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
        duration = total_frames / fps if fps > 0 else 60.0

        step_sec = cfg["step_sec"]
        step_frames = max(1, int(round(step_sec * fps)))

        sampled_entries = []
        frame_idx = 0
        idx_s = 0
        while True:
            ok, frm = cap.read()
            if not ok or frm is None:
                break
            if frame_idx % step_frames == 0:
                cur_t = round(frame_idx / float(fps), 2)
                fh, fw, _ = frm.shape
                crp = frm[int(fh * roi["y1"]) : int(fh * roi["y2"]), int(fw * roi["x1"]) : int(fw * roi["x2"])]
                if crp is not None and crp.size > 0:
                    cp = os.path.join(crops_dir, f"crop_{idx_s:04d}_{cur_t:.2f}s.jpg")
                    cv2.imwrite(cp, crp)
                    sampled_entries.append({"time": cur_t, "crop_path": cp})
                    idx_s += 1
            frame_idx += 1

        if frame_idx > 0:
            duration = round(frame_idx / float(fps), 2)
        cap.release()

        raw_ocr_map, active_gpu = _run_qwen25_vl_ocr(sampled_entries, cfg, cv2)

        screen_texts: list[dict] = []
        vis_files: list[str] = []
        hits = 0

        cap_vis = cv2.VideoCapture(video_path) if save_vis else None
        for entry in sampled_entries:
            ts = float(entry["time"])
            dets = raw_ocr_map.get(ts, [])
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
            f"[INFO] [Qwen2.5-VL-3B (GPU={active_gpu}) | Step={step_sec}s] "
            f"tổng frame OCR={len(sampled_entries)} | hits={hits}"
        )
        return roi, screen_texts, zip_path if (save_vis and vis_files) else "", duration
    except Exception as e:
        shutil.rmtree(crops_dir, ignore_errors=True)
        print(f"[WARN] Qwen2.5-VL-3B OCR lỗi: {e}\n{traceback.format_exc()[-1500:]}")
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


def build_ocr_segments(screen_texts: list[dict], duration: float = 60.0, step_sec: float = 0.50) -> list[dict]:
    grouped: list[dict] = []
    max_merge_gap = max(1.20, step_sec * 3.0)

    for item in sorted(screen_texts, key=lambda x: float(x.get("time", 0.0))):
        ts = float(item.get("time", 0.0))
        texts = [str(t).strip() for t in (item.get("texts") or []) if str(t).strip()]
        joined = " ".join(texts).strip()
        if not joined:
            continue

        if (
            grouped
            and _is_same_subtitle(grouped[-1]["ocr_text"], joined)
            and (ts - grouped[-1]["_last_ts"]) <= max_merge_gap
        ):
            if len(joined) > len(grouped[-1]["ocr_text"]):
                grouped[-1]["ocr_text"] = joined
                grouped[-1]["best_text"] = joined
            grouped[-1]["_last_ts"] = ts
            grouped[-1]["end"] = round(min(duration, max(grouped[-1]["end"], ts + step_sec * 0.5)), 2)
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
        f"Bắt đầu nhận diện OCR bằng Qwen2.5-VL-3B cho job {JOB_ID}.",
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

    vocals_wav, bgm_wav = separate_vocals_and_bgm(audio_16k, work_dir)
    subtitle_roi, screen_texts, bbox_zip_path, duration = run_ocr_with_qwen25_vl(video_path, work_dir)
    step_sec = _get_qwen25_vl_config()["step_sec"]
    segments = build_ocr_segments(screen_texts, duration=duration, step_sec=step_sec)
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
        "ocr_engine": "qwen2_5_vl_3b_instruct",
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
        f"stage1 done (qwen2.5-vl-3b): {JOB_ID}",
    )

    updated_jobs, _ = sync.read_json("state/jobs.json")
    if JOB_ID in updated_jobs.get("jobs", {}):
        updated_jobs["jobs"][JOB_ID]["status"] = "STAGE1_DONE"
        sync.write_json("state/jobs.json", updated_jobs, f"update status STAGE1_DONE: {JOB_ID}")

    sync.trigger_dispatch("stage1_completed", {"job_id": JOB_ID})
    send_ntfy(
        f"[Stage 1 Done] {JOB_ID}",
        f"Hoàn tất Qwen2.5-VL-3B ({len(segments)} đoạn, {len(speaker_refs)} người nói) -> sang Stage 2.",
        tags="white_check_mark",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_ntfy(f"[Stage 1 Lỗi] {JOB_ID}", err, tags="x", priority="high")
        raise
