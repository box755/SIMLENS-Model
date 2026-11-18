"""Download official release checkpoints and verify archive and member hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Redirect handler that removes the `Authorization` header from redirected requests."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        # Release assets redirect to a storage URL, which does not receive the token
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


def download_asset(manifest, entry, destination, token=None):
    """Download one checkpoint archive of the release named in the manifest to `destination`."""
    if manifest["repository"] != "box755/SIMLENS-Model":
        raise ValueError("Unexpected repository in checkpoint manifest")
    opener = urllib.request.build_opener(SafeRedirect())
    # With a token, look the asset up through the GitHub API; otherwise use the public download URL
    if token:
        api = f"https://api.github.com/repos/{manifest['repository']}/releases/tags/{manifest['tag']}"
        headers = {"User-Agent": "SIMLENS-checkpoint-downloader", "Authorization": "Bearer " + token,
                   "Accept": "application/vnd.github+json"}
        with opener.open(urllib.request.Request(api, headers=headers), timeout=60) as response:
            release = json.load(response)
        assets = [a for a in release["assets"] if a["name"] == entry["asset"]]
        if len(assets) != 1:
            raise ValueError("Expected checkpoint asset is not available in this release")
        if not assets[0]["url"].startswith("https://api.github.com/repos/box755/SIMLENS-Model/releases/assets/"):
            raise ValueError("Unexpected asset URL")
        headers["Accept"] = "application/octet-stream"
        req = urllib.request.Request(assets[0]["url"], headers=headers)
    else:
        url = f"https://github.com/{manifest['repository']}/releases/download/{manifest['tag']}/{entry['asset']}"
        req = urllib.request.Request(url, headers={"User-Agent": "SIMLENS-checkpoint-downloader"})
    with opener.open(req, timeout=60) as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def digest(path):
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def install_archive(archive_path, entry, seed, destination):
    """Verify the archive and each file listed in the manifest, then extract the files to `destination`."""
    if digest(archive_path) != entry["sha256"]:
        raise ValueError("Checkpoint ZIP SHA-256 mismatch")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="simlens-", dir=destination.parent) as tmp:
        stage = Path(tmp)
        with zipfile.ZipFile(archive_path) as archive:
            for name, expected in entry["files"].items():
                # Only plain file names listed in the manifest are extracted
                if Path(name).name != name or "\\" in name:
                    raise ValueError("Invalid member name in manifest")
                target = stage / name
                with archive.open(f"seed{seed}/{name}") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if digest(target) != expected:
                    raise ValueError(f"SHA-256 mismatch: {name}")
        # Move the verified files into place
        stage.rename(destination)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", choices=["42", "7", "123", "all"], default="42")
    ap.add_argument("--output", type=Path, default=ROOT / "checkpoints")
    args = ap.parse_args()
    manifest = json.loads((ROOT / "checkpoints.json").read_text(encoding="utf-8"))
    seeds = ("42", "7", "123") if args.seed == "all" else (args.seed,)
    for seed in seeds:
        entry = manifest["checkpoints"][seed]
        dest = args.output / f"seed{seed}"
        if dest.exists():
            raise FileExistsError(f"Checkpoint directory already exists: {dest}")
        with tempfile.TemporaryDirectory(prefix="simlens-download-") as tmp:
            archive = Path(tmp) / "checkpoint.zip"
            download_asset(manifest, entry, archive, os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))
            install_archive(archive, entry, seed, dest)
        print(f"Verified seed {seed}: {dest}")


if __name__ == "__main__":
    main()
