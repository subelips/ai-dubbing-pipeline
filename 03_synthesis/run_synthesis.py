import base64
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
import re
import subprocess
import sys
import threading
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
GH_TOKEN = get_cfg("GH_PAT_TOKEN", "__INJECT_GH_PAT_TOKEN__")
TG_TOKEN = get_cfg("TELEGRAM_BOT_TOKEN", "__INJECT_TELEGRAM_BOT_TOKEN__")
TG_CHAT = get_cfg("TELEGRAM_CHAT_ID", "__INJECT_TELEGRAM_CHAT_ID__")

if not JOB_ID:
    raise RuntimeError("Thiếu JOB_ID khi khởi chạy kernel.")


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
        return r.json().get("browser_download_url")

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


def send_telegram(bot_token: str, chat_id: str, text: str):
    clean_token = _clean_url_or_str(bot_token)
    clean_chat = _clean_url_or_str(chat_id)
    if not clean_token or not clean_chat:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{clean_token}/sendMessage",
            json={"chat_id": clean_chat, "text": text[:4000], "parse_mode": "HTML", "disable_web_page_preview": False},
            timeout=20,
        )
    except Exception as e:
        print(f"[WARN] Gửi Telegram thất bại: {e}")


def ensure_deps():
    subprocess.run(["apt-get", "-qq", "update"], check=False)
    subprocess.run(["apt-get", "-qq", "install", "-y", "espeak-ng"], check=False)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "vieneu", "numpy", "soundfile", "gdown"],
        check=True,
    )


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
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
            direct_url = _normalize_gdrive_url(ref)
            subprocess.run([sys.executable, "-m", "gdown", direct_url, "-O", dest_path], check=True)
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


def clean_bgm_residual_vocals(bgm_in: str, work_dir: str) -> str:
    if os.environ.get("BGM_VOCAL_BLEED_REDUCE", "1").strip() == "0":
        return bgm_in

    cleaned_bgm = os.path.join(work_dir, "no_vocals_cleaned.wav")
    af_chain = "highpass=f=80,lowpass=f=14000,stereotools=mlev=0.45:slev=1.05"
    res = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", bgm_in,
            "-af", af_chain,
            "-ac", "2",
            "-ar", "44100",
            cleaned_bgm,
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode == 0 and os.path.exists(cleaned_bgm) and os.path.getsize(cleaned_bgm) > 512:
        print("[INFO] Đã áp dụng bộ lọc giảm vocal dư cho nhạc nền.")
        return cleaned_bgm
    return bgm_in


def init_vieneu_engine():
    from vieneu import Vieneu  # type: ignore

    mode = os.environ.get("VIENEU_MODE", "").strip()
    if mode:
        try:
            return Vieneu(mode=mode)
        except Exception:
            pass
    return Vieneu()


def pick_voice_for_gender(tts_engine, gender: str) -> str | None:
    male_default = os.environ.get("VIENEU_MALE_VOICE", "Hải Đăng")
    female_default = os.environ.get("VIENEU_FEMALE_VOICE", "Ngọc Lan")
    target_gender = "male" if str(gender).lower() in ("male", "nam") else "female"

    try:
        voices = tts_engine.list_preset_voices() or []
        names = [v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else str(v) for v in voices]
        if target_gender == "male":
            if male_default in names:
                return male_default
            for n in names:
                if any(k in n.lower() for k in ["nam", "minh", "hải", "đăng", "quân", "adam", "vĩnh"]):
                    return n
        else:
            if female_default in names:
                return female_default
            for n in names:
                if any(k in n.lower() for k in ["nữ", "nu", "lan", "mai", "anh", "my", "huyền"]):
                    return n
        return names[0] if names else None
    except Exception:
        return male_default if target_gender == "male" else female_default


def _smooth_envelope(mask: "np.ndarray", win_samples: int) -> "np.ndarray":
    import numpy as np

    if win_samples <= 1:
        return mask.astype(np.float32)
    kernel = np.ones(win_samples, dtype=np.float32) / float(win_samples)
    padded = np.pad(mask.astype(np.float32), (win_samples // 2, win_samples // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(mask)]


def _build_atempo_chain(ratio: float) -> str:
    r = max(0.5, min(2.5, float(ratio)))
    parts = []
    while r > 2.0:
        parts.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        parts.append("atempo=0.5")
        r /= 0.5
    parts.append(f"atempo={r:.4f}")
    return ",".join(parts)


def _prepare_synth_units(stage2_data: dict) -> list[dict]:
    utterances = stage2_data.get("utterances") or []
    lines = stage2_data.get("lines") or []
    if utterances:
        units = []
        for idx, u in enumerate(utterances):
            txt = (u.get("text_vi") or u.get("src_text") or "").strip()
            if not txt:
                continue
            units.append({
                "id": u.get("id", idx),
                "start": float(u.get("start", 0.0)),
                "end": float(u.get("end", float(u.get("start", 0.0)) + 1.0)),
                "speaker_id": u.get("speaker_id", "SPEAKER_00"),
                "gender": u.get("gender", "female"),
                "text_vi": txt,
                "line_count": len(u.get("segment_ids") or [1]),
            })
        if units:
            return units

    return [
        {
            "id": l.get("id", idx),
            "start": float(l.get("start", 0.0)),
            "end": float(l.get("end", float(l.get("start", 0.0)) + 1.0)),
            "speaker_id": l.get("speaker_id", "SPEAKER_00"),
            "gender": l.get("gender", "female"),
            "text_vi": (l.get("text_vi") or l.get("src_text") or "").strip(),
            "line_count": 1,
        }
        for idx, l in enumerate(lines)
        if (l.get("text_vi") or l.get("src_text") or "").strip()
    ]


def synth_single_unit(
    tts_engine,
    lock: threading.Lock,
    unit: dict,
    idx: int,
    work_dir: str,
    target_sr: int,
    speaker_ref_map: dict,
    next_start_t: float | None = None,
) -> tuple[int, float, str | None]:
    import numpy as np
    import soundfile as sf

    text_vi = (unit.get("text_vi") or "").strip()
    start_t = max(0.0, float(unit.get("start", 0.0)))
    end_t = max(start_t + 0.5, float(unit.get("end", start_t + 1.2)))
    slot_dur = max(0.5, end_t - start_t)

    if next_start_t is not None and next_start_t > start_t + slot_dur:
        safe_gap_end = max(start_t + slot_dur, next_start_t - 0.06)
        slot_dur = min(safe_gap_end - start_t, slot_dur * 1.45)

    spk_id = unit.get("speaker_id", "SPEAKER_00")
    gender = (unit.get("gender") or speaker_ref_map.get(spk_id, {}).get("gender") or "female").lower()
    ref_wav = speaker_ref_map.get(spk_id, {}).get("ref_path")
    preset_voice = pick_voice_for_gender(tts_engine, gender)

    raw_wav = os.path.join(work_dir, f"raw_{idx:04d}.wav")
    aligned_wav = os.path.join(work_dir, f"aligned_{idx:04d}.wav")

    def _run_infer(use_ref: bool):
        if use_ref and ref_wav and os.path.exists(ref_wav) and os.path.getsize(ref_wav) > 1024:
            try:
                return tts_engine.infer(text_vi, ref_audio=ref_wav, denoise=True)
            except TypeError:
                return tts_engine.infer(text_vi, ref_audio=ref_wav)
        return tts_engine.infer(text_vi, voice=preset_voice)

    with lock:
        audio_arr = None
        try:
            audio_arr = _run_infer(use_ref=True)
        except Exception as ref_err:
            print(f"[WARN] Cụm {idx} clone ref_audio lỗi ({ref_err}), thử giọng preset {preset_voice}.")
            try:
                audio_arr = _run_infer(use_ref=False)
            except Exception as p_err:
                print(f"[WARN] Cụm {idx} TTS lỗi hoàn toàn: {p_err}")
                return idx, start_t, None

        if audio_arr is None:
            return idx, start_t, None

        if hasattr(tts_engine, "save"):
            tts_engine.save(audio_arr, raw_wav)
        else:
            arr_np = np.asarray(audio_arr, dtype=np.float32)
            sf.write(raw_wav, arr_np, 48000)

    if not os.path.exists(raw_wav) or os.path.getsize(raw_wav) < 512:
        return idx, start_t, None

    probe_data, src_sr = sf.read(raw_wav)
    if probe_data.ndim > 1:
        probe_data = probe_data.mean(axis=-1)
    actual_dur = max(0.1, len(probe_data) / float(src_sr or 48000))

    tempo = 1.0
    if actual_dur > slot_dur * 1.05:
        tempo = min(1.25, actual_dur / slot_dur)
    elif actual_dur < slot_dur * 0.82:
        tempo = max(0.90, actual_dur / slot_dur)

    af_parts = []
    if abs(tempo - 1.0) > 0.02:
        af_parts.append(_build_atempo_chain(tempo))

    est_dur_after_tempo = actual_dur / tempo
    max_allowed_dur = round(slot_dur * 1.15, 3)
    if est_dur_after_tempo > max_allowed_dur:
        fade_dur = min(0.06, max(0.02, max_allowed_dur * 0.1))
        fade_start = max(0.0, round(max_allowed_dur - fade_dur, 3))
        af_parts.append(f"atrim=0:{max_allowed_dur},afade=t=out:st={fade_start}:d={fade_dur},asetpts=PTS-STARTPTS")

    ff_cmd = ["ffmpeg", "-y", "-i", raw_wav]
    if af_parts:
        ff_cmd.extend(["-af", ",".join(af_parts)])
    ff_cmd.extend(["-ac", "2", "-ar", str(target_sr), aligned_wav])

    res = subprocess.run(ff_cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(aligned_wav):
        return idx, start_t, raw_wav
    return idx, start_t, aligned_wav


def mix_dub_and_mux(video_in: str, bgm_in: str, stage2_data: dict, speaker_ref_map: dict, work_dir: str, final_mp4: str):
    import numpy as np
    import soundfile as sf

    effective_bgm = clean_bgm_residual_vocals(bgm_in, work_dir)
    bgm_data, sr = sf.read(effective_bgm)
    if bgm_data.ndim == 1:
        bgm_data = np.stack([bgm_data, bgm_data], axis=-1)
    total_samples = len(bgm_data)
    voice_track = np.zeros_like(bgm_data, dtype=np.float32)

    units = _prepare_synth_units(stage2_data)
    total_lines_expected = len(stage2_data.get("lines") or units)

    tts_engine = init_vieneu_engine()
    lock = threading.Lock()
    workers = max(1, min(2, int(os.environ.get("TTS_WORKERS", "1"))))

    results = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    synth_single_unit,
                    tts_engine,
                    lock,
                    unit,
                    i,
                    work_dir,
                    sr,
                    speaker_ref_map,
                    float(units[i + 1].get("start", 0.0)) if i + 1 < len(units) else None,
                )
                for i, unit in enumerate(units)
            ]
            for fut in futures:
                results.append(fut.result())
    finally:
        try:
            if hasattr(tts_engine, "close"):
                tts_engine.close()
        except Exception:
            pass
        del tts_engine
        free_gpu()

    results.sort(key=lambda x: x[0])
    ok_units = 0
    covered_lines = 0

    target_rms = 0.11
    for u_idx, (_, start_t, wav_path) in enumerate(results):
        if not wav_path or not os.path.exists(wav_path):
            continue
        seg_audio, seg_sr = sf.read(wav_path)
        if seg_audio.ndim == 1:
            seg_audio = np.stack([seg_audio, seg_audio], axis=-1)
        if seg_sr != sr:
            rs_path = wav_path + "_rs.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-ac", "2", "-ar", str(sr), rs_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            seg_audio, _ = sf.read(rs_path)
            if seg_audio.ndim == 1:
                seg_audio = np.stack([seg_audio, seg_audio], axis=-1)

        seg_f32 = seg_audio.astype(np.float32)
        rms = float(np.sqrt(np.mean(seg_f32 ** 2)))
        if rms > 1e-4:
            gain = min(2.5, target_rms / rms)
            seg_f32 = np.clip(seg_f32 * gain, -0.95, 0.95)

        start_sample = int(round(start_t * sr))
        if start_sample >= total_samples:
            continue
        end_sample = min(total_samples, start_sample + len(seg_f32))
        copy_len = end_sample - start_sample
        if copy_len > 0:
            voice_track[start_sample:end_sample] += seg_f32[:copy_len]
            ok_units += 1
            covered_lines += int(units[u_idx].get("line_count", 1))

    print(
        f"[INFO] VieNeu-TTS hoàn tất {ok_units}/{len(units)} cụm ý "
        f"(bao phủ ~{min(total_lines_expected, covered_lines)}/{total_lines_expected} câu thoại)."
    )

    raw_voice_active = (np.abs(voice_track).max(axis=-1) > 1e-3).astype(np.float32)
    smooth_active = _smooth_envelope(raw_voice_active, max(1, int(sr * 0.18)))
    duck_factor = 1.0 - (0.65 * smooth_active)
    ducked_bgm = bgm_data.astype(np.float32) * duck_factor[:, None]

    mixed_audio = np.clip(ducked_bgm + voice_track, -1.0, 1.0)
    mixed_wav = os.path.join(work_dir, "final_dub_mix.wav")
    sf.write(mixed_wav, mixed_audio, sr)

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
    send_telegram(TG_TOKEN, TG_CHAT, f"🎙️ <b>[Stage 3 - VieNeu-TTS]</b> Bắt đầu job <code>{JOB_ID}</code>")
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

    speaker_ref_map = {}
    speakers_meta = stage2_data.get("speakers") or {}
    for spk_id, meta in speakers_meta.items():
        if isinstance(meta, dict):
            speaker_ref_map[spk_id] = dict(meta)

    release = sync.ensure_release(JOB_ID)
    for asset in release.get("assets", []):
        name = asset["name"]
        if name.startswith("ref_speaker_") and name.endswith(".wav"):
            local_ref = os.path.join(out_dir, name)
            sync.download_asset(JOB_ID, name, local_ref)
            spk_id = name[len("ref_speaker_"):-len(".wav")]
            speaker_ref_map.setdefault(spk_id, {})["ref_path"] = local_ref

    ensure_deps()
    mix_dub_and_mux(video_in, bgm_in, stage2_data, speaker_ref_map, out_dir, final_mp4)

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
