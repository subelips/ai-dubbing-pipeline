import base64
import gc
import json
import os
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
        upload_url = release["upload_url"].split("{")[0]
        filename = os.path.basename(file_path)
        for asset in release.get("assets", []):
            if asset["name"] == filename:
                requests.delete(f"{self.api_base}/releases/assets/{asset['id']}", headers=self.headers, timeout=30)
        up_headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/octet-stream"}
        with open(file_path, "rb") as f:
            r = requests.post(f"{upload_url}?name={filename}", headers=up_headers, data=f.read(), timeout=300)
        r.raise_for_status()
        return r.json().get("browser_download_url")

    def download_asset(self, tag: str, filename: str, dest_path: str):
        release = self.ensure_release(tag)
        for asset in release.get("assets", []):
            if asset["name"] == filename:
                dl_headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/octet-stream"}
                with requests.get(asset["url"], headers=dl_headers, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(dest_path, "wb") as out:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                out.write(chunk)
                return dest_path
        raise FileNotFoundError(f"Không tìm thấy asset {filename} trong release {tag}")


def send_telegram(bot_token: str, chat_id: str, text: str):
    if not bot_token or not chat_id:
        return
    clean_token = str(bot_token).strip()
    clean_chat = str(chat_id).strip()
    url = f"https://api.telegram.org/bot{clean_token}/sendMessage"
    try:
        requests.post(
            url,
            json={
                "chat_id": clean_chat,
                "text": text[:4000],
                "parse_mode": "HTML",
                "disable_web_page_preview": False,
            },
            timeout=20,
        )
    except Exception as e:
        print(f"[WARN] Gửi Telegram thất bại: {e}")


def download_video_source(sync: GitHubSync, tag: str, source_ref: str, dest_path: str):
    ref = (source_ref or "source_input.mp4").strip()
    if ref.startswith(("http://", "https://")):
        if "drive.google.com" in ref:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
            subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", ref, "-O", dest_path], check=True)
        else:
            subprocess.run(["curl", "-fSL", ref, "-o", dest_path], check=True)
    else:
        sync.download_asset(tag, ref, dest_path)


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def try_load_vieneu_tts():
    repo_url = os.environ.get("VIENEU_TTS_REPO", "").strip()
    local_dir = "/kaggle/working/VieNeu-TTS"
    if repo_url and not os.path.isdir(local_dir):
        subprocess.run(["git", "clone", "--depth", "1", repo_url, local_dir], check=False)

    for candidate_dir in [local_dir, "/kaggle/working/vieneu-tts", "."]:
        if os.path.isdir(candidate_dir) and candidate_dir not in sys.path:
            sys.path.insert(0, candidate_dir)

    try:
        import vieneu_tts  # type: ignore
        return ("module", vieneu_tts)
    except Exception:
        pass

    for entry in [os.path.join(local_dir, "infer.py"), os.path.join(local_dir, "tts.py")]:
        if os.path.isfile(entry):
            return ("script", entry)

    return None


async def _edge_tts_save(text: str, out_wav: str, voice: str = "vi-VN-HoaiMyNeural"):
    import edge_tts
    tmp_mp3 = out_wav + ".mp3"
    comm = edge_tts.Communicate(text, voice)
    await comm.save(tmp_mp3)
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_mp3, "-ac", "2", "-ar", "44100", out_wav],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if os.path.exists(tmp_mp3):
        os.remove(tmp_mp3)


def synth_line_audio(text_vi: str, ref_wav: str | None, out_wav: str, vieneu_handle):
    import asyncio

    if not text_vi.strip():
        return False

    if vieneu_handle is not None:
        kind, obj = vieneu_handle
        try:
            if kind == "module" and hasattr(obj, "synthesize"):
                obj.synthesize(text=text_vi, ref_audio=ref_wav, output_path=out_wav)
                if os.path.exists(out_wav) and os.path.getsize(out_wav) > 512:
                    return True
            elif kind == "script":
                cmd = [sys.executable, obj, "--text", text_vi, "--out", out_wav]
                if ref_wav and os.path.exists(ref_wav):
                    cmd.extend(["--ref", ref_wav])
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode == 0 and os.path.exists(out_wav) and os.path.getsize(out_wav) > 512:
                    return True
        except Exception:
            pass

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "edge-tts"], check=True)
        asyncio.run(_edge_tts_save(text_vi, out_wav))
        return os.path.exists(out_wav) and os.path.getsize(out_wav) > 512
    except Exception:
        return False


def mix_dub_and_mux(video_in: str, bgm_in: str, lines: list[dict], work_dir: str, final_mp4: str):
    import numpy as np
    import soundfile as sf

    bgm_data, sr = sf.read(bgm_in)
    if bgm_data.ndim == 1:
        bgm_data = np.stack([bgm_data, bgm_data], axis=-1)
    total_samples = len(bgm_data)
    voice_track = np.zeros_like(bgm_data, dtype=np.float32)

    vieneu_handle = try_load_vieneu_tts()
    default_ref = os.path.join(work_dir, "ref_speaker_SPEAKER_00.wav")
    if not os.path.exists(default_ref):
        default_ref = None

    for idx, line in enumerate(lines):
        text_vi = (line.get("text_vi") or "").strip()
        if not text_vi:
            continue
        start_t = max(0.0, float(line.get("start", 0.0)))
        line_wav = os.path.join(work_dir, f"line_{idx:04d}.wav")
        ok = synth_line_audio(text_vi, default_ref, line_wav, vieneu_handle)
        if not ok or not os.path.exists(line_wav):
            continue

        seg_audio, seg_sr = sf.read(line_wav)
        if seg_audio.ndim == 1:
            seg_audio = np.stack([seg_audio, seg_audio], axis=-1)
        if seg_sr != sr:
            resampled_wav = os.path.join(work_dir, f"line_{idx:04d}_rs.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", line_wav, "-ac", "2", "-ar", str(sr), resampled_wav],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            seg_audio, _ = sf.read(resampled_wav)
            if seg_audio.ndim == 1:
                seg_audio = np.stack([seg_audio, seg_audio], axis=-1)

        start_sample = int(start_t * sr)
        if start_sample >= total_samples:
            continue
        end_sample = min(total_samples, start_sample + len(seg_audio))
        copy_len = end_sample - start_sample
        if copy_len > 0:
            voice_track[start_sample:end_sample] += seg_audio[:copy_len].astype(np.float32)

    has_voice = np.abs(voice_track).max(axis=-1) > 1e-3
    ducked_bgm = bgm_data.astype(np.float32).copy()
    ducked_bgm[has_voice] *= 0.35

    mixed_audio = np.clip(ducked_bgm + voice_track, -1.0, 1.0)
    mixed_wav = os.path.join(work_dir, "final_dub_mix.wav")
    sf.write(mixed_wav, mixed_audio, sr)
    free_gpu()

    ff_res = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_in,
            "-i", mixed_wav,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            final_mp4,
        ],
        capture_output=True,
        text=True,
    )
    if ff_res.returncode != 0:
        raise RuntimeError(f"FFmpeg mux lỗi (code {ff_res.returncode}):\n{ff_res.stderr[-1500:]}")


sync = GitHubSync(GH_REPO, GH_TOKEN)


def main():
    send_telegram(TG_TOKEN, TG_CHAT, f"🎙️ <b>[Stage 3 - Synthesis]</b> Bắt đầu job <code>{JOB_ID}</code>")
    stage2_data, _ = sync.read_json(f"state/{JOB_ID}_stage2.json")

    out_dir = "/kaggle/working/out"
    os.makedirs(out_dir, exist_ok=True)
    video_in = os.path.join(out_dir, "source.mp4")
    bgm_in = os.path.join(out_dir, "no_vocals.wav")
    final_mp4 = os.path.join(out_dir, f"final_dubbed_{JOB_ID}.mp4")

    video_asset = (stage2_data.get("video_asset") or stage2_data.get("video_url") or "source_input.mp4").strip()
    download_video_source(sync, JOB_ID, video_asset, video_in)
    if not os.path.exists(video_in) or os.path.getsize(video_in) < 1024:
        raise RuntimeError(f"File video '{video_asset}' tải về không hợp lệ hoặc rỗng.")

    sync.download_asset(JOB_ID, "no_vocals.wav", bgm_in)

    release = sync.ensure_release(JOB_ID)
    for asset in release.get("assets", []):
        if asset["name"].startswith("ref_speaker_") and asset["name"].endswith(".wav"):
            sync.download_asset(JOB_ID, asset["name"], os.path.join(out_dir, asset["name"]))

    lines = stage2_data.get("lines", [])
    mix_dub_and_mux(video_in, bgm_in, lines, out_dir, final_mp4)

    download_url = sync.upload_asset(JOB_ID, final_mp4)

    updated_jobs, _ = sync.read_json("state/jobs.json")
    if JOB_ID in updated_jobs.get("jobs", {}):
        updated_jobs["jobs"][JOB_ID]["status"] = "COMPLETED"
        updated_jobs["jobs"][JOB_ID]["output_url"] = download_url
        sync.write_json("state/jobs.json", updated_jobs, f"completed job: {JOB_ID}")

    send_telegram(
        TG_TOKEN,
        TG_CHAT,
        f"🎉 <b>Hoàn tất lồng tiếng!</b> Job: <code>{JOB_ID}</code>\n🔗 Link tải MP4: {download_url}",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()[-1200:]
        send_telegram(TG_TOKEN, TG_CHAT, f"❌ <b>[Stage 3 Lỗi]</b> <code>{JOB_ID}</code>:\n<pre>{err}</pre>")
        raise
