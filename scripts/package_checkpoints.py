"""Build the release archives, checkpoints.json and SHA256SUMS.txt from the checkpoints in checkpoints/."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Files that make up one checkpoint
FILES = ("adapter_config.json", "adapter_model.safetensors", "head.pt", "meta.json")


def sha256(path):
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="v0.1.0")
    args = ap.parse_args()
    out = ROOT / "release-assets"
    out.mkdir(exist_ok=True)
    manifest = {"repository": "box755/SIMLENS-Model", "tag": args.tag, "checkpoints": {}}
    for seed in (42, 7, 123):
        folder = ROOT / "checkpoints" / f"seed{seed}"
        sources = {name: folder / name for name in FILES}
        # Store the configuration of this seed next to its weights
        cfg = yaml.safe_load((ROOT / "simlens/configs/main.yaml").read_text(encoding="utf-8"))
        cfg["seed"] = seed
        cfg["run_name"] = f"simlens_s{seed}"
        (folder / "model_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        sources.update({"model_config.yaml": folder / "model_config.yaml",
                        "MODEL_CARD.md": ROOT / "docs/model_card.md", "LICENSE": ROOT / "LICENSE"})
        for path in sources.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        name = f"simlens-seed{seed}-{args.tag}.zip"
        target = out / name
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member, source in sources.items():
                archive.write(source, f"seed{seed}/{member}")
        manifest["checkpoints"][str(seed)] = {
            "asset": name, "bytes": target.stat().st_size, "sha256": sha256(target),
            "files": {member: sha256(path) for member, path in sources.items()},
        }
    (ROOT / "checkpoints.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "SHA256SUMS.txt").write_text("".join(
        f"{row['sha256']}  {row['asset']}\n" for row in manifest["checkpoints"].values()), encoding="utf-8")
    print(json.dumps({k: {"asset": v["asset"], "bytes": v["bytes"]}
                      for k, v in manifest["checkpoints"].items()}, indent=2))


if __name__ == "__main__":
    main()
