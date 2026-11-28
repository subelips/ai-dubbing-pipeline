import base64
import json
import os
import time
import requests


class GitHubSync:
    def __init__(self, repo: str, token: str, state_branch: str = "pipeline-state"):
        self.repo = repo
        self.token = token
        self.state_branch = state_branch
        self.api_base = f"https://api.github.com/repos/{repo}"
        self.upload_base = f"https://uploads.github.com/repos/{repo}"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def read_json(self, path: str) -> tuple[dict, str | None]:
        url = f"{self.api_base}/contents/{str(path).lstrip('/')}"
        r = requests.get(url, headers=self.headers, params={"ref": self.state_branch}, timeout=30)
        if r.status_code == 404:
            return {}, None
        r.raise_for_status()
        data = r.json()
        content = base64.b64decode(data["content"]).decode("utf-8").strip()
        if not content:
            return {}, data.get("sha")
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
        raise RuntimeError(f"Không thể ghi {path} sau {max_retries} lần thử do xung đột SHA.")

    def ensure_release(self, tag: str, name: str | None = None) -> dict:
        url = f"{self.api_base}/releases/tags/{tag}"
        r = requests.get(url, headers=self.headers, timeout=30)
        if r.status_code == 200:
            return r.json()
        create_url = f"{self.api_base}/releases"
        cr = requests.post(
            create_url,
            headers=self.headers,
            json={
                "tag_name": tag,
                "name": name or f"Artifacts - {tag}",
                "draft": False,
                "prerelease": False,
            },
            timeout=30,
        )
        cr.raise_for_status()
        return cr.json()

    def upload_asset(self, tag: str, file_path: str):
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"[WARN] upload_asset: thiếu file hoặc file 0 byte '{file_path}'")
            return None
        release = self.ensure_release(tag)
        upload_url = release["upload_url"].split("{")[0]
        filename = os.path.basename(file_path)

        # Xóa asset trùng tên nếu đã tồn tại
        for asset in release.get("assets", []):
            if asset["name"] == filename:
                requests.delete(f"{self.api_base}/releases/assets/{asset['id']}", headers=self.headers, timeout=30)

        up_headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/octet-stream",
        }
        with open(file_path, "rb") as f:
            r = requests.post(
                f"{upload_url}?name={filename}",
                headers=up_headers,
                data=f.read(),
                timeout=300,
            )
        r.raise_for_status()
        return r.json().get("browser_download_url")

    def download_asset(self, tag: str, filename: str, dest_path: str):
        release = self.ensure_release(tag)
        for asset in release.get("assets", []):
            if asset["name"] == filename:
                dl_headers = {
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/octet-stream",
                }
                with requests.get(asset["url"], headers=dl_headers, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(dest_path, "wb") as out:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                out.write(chunk)
                return dest_path
        raise FileNotFoundError(f"Không tìm thấy asset {filename} trong release {tag}")

    def trigger_dispatch(self, event_type: str, client_payload: dict):
        url = f"{self.api_base}/dispatches"
        r = requests.post(
            url,
            headers=self.headers,
            json={"event_type": event_type, "client_payload": client_payload},
            timeout=30,
        )
        r.raise_for_status()
