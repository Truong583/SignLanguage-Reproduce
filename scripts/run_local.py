"""Container pipeline for one configured experiment, never silently invents missing data."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MBART_REVISION = "f417e5563320b2cc8aabe4329d986b238809067f"


def call(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


@contextmanager
def workspace_lock():
    # Linux container lock survives file existence, releases automatically on crash.
    import fcntl
    with (ROOT / "runs/.local-run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another local pipeline is using this project. Wait for it to finish.")
        yield


def code_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "repro").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(cfg):
    from scripts.fetch_phoenix import ensure_phoenix, valid_layout, check_prepare_space
    data_root = Path(cfg["data_root"])
    if data_root.as_posix() == "data/phoenix14t":
        if not valid_layout(data_root):
            check_prepare_space(data_root,ROOT/'data/phoenix-2014-T.v3.tar.gz')
        ensure_phoenix(data_root, ROOT / "data/phoenix-2014-T.v3.tar.gz")
        manifests = [Path(cfg[s]) for s in ("train", "dev", "test")]
        if not all(p.is_file() for p in manifests):
            if len({p.parent for p in manifests}) != 1:
                raise RuntimeError("Manifest paths must share one directory for automatic preparation.")
            call("scripts/prepare_data.py", "--dataset", "phoenix14t", "--annotations",
                 data_root / "annotations/manual", "--data-root", data_root,
                 "--output", manifests[0].parent, "--overwrite")
    elif not all(Path(cfg[s]).is_file() for s in ("train", "dev", "test")):
        raise RuntimeError("This dataset requires its authorized data/manifests first; see docs/AUDIT.md. Automatic download supports PHOENIX14T only.")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    if cfg.get('backbone') in ('swin_t','pvig_tiny'):
        call('scripts/fetch_backbones.py','--backbone',cfg['backbone'])
    if cfg.get("input_kind", "rgb") == "rgb":
        if cfg.get("resnet_weights") != "assets/resnet18-f37072fd.pth":
            if not Path(cfg["resnet_weights"]).is_file():
                raise FileNotFoundError(cfg["resnet_weights"])
        else:
            call("scripts/fetch_assets.py", "--resnet")
    if cfg.get("task") == "slt" and not Path(cfg["mbart_path"]).is_dir():
        if cfg["mbart_path"] != "assets/mbart-large-cc25":
            raise FileNotFoundError(cfg["mbart_path"])
        call("scripts/fetch_assets.py", "--mbart", "--mbart-revision", MBART_REVISION)


def train(cfg, config_path):
    import yaml
    output = Path(cfg["output"])
    if not output.resolve().is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Experiment output must be inside runs/.")
    output.mkdir(parents=True, exist_ok=True)
    cfg["code_sha256"] = code_hash()
    saved_config = output / "local_config.yaml"
    if saved_config.is_file():
        previous = yaml.safe_load(saved_config.read_text(encoding="utf-8"))
        if previous != cfg:
            raise ValueError("This run already has a different config/code hash. Restore the original or choose a new output directory.")
    else:
        saved_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    call("scripts/doctor.py", "--config", saved_config)
    last = output / "last.pt"
    extra = []
    if last.is_file():
        extra = ["--resume", last]
    elif cfg.get("task") == "slt":
        experiment = config_path.stem
        phase = "tctc" if cfg.get("target_field") == "pseudo_gloss" else "cslr"
        dataset = experiment.rsplit("_", 1)[0]
        source = ROOT / f"runs/{dataset}_{phase}_seed{cfg.get('seed', 0)}/best.pt"
        if not source.is_file():
            raise FileNotFoundError(f"Pretraining checkpoint required before SLT: {source}")
        extra = ["--initialize-from", source]
    call("scripts/launch.py", "--config", saved_config, *extra)
    best = output / "best.pt"
    if not best.is_file():
        raise FileNotFoundError("No best checkpoint; training has not completed successfully.")
    # Both splits use the same checkpoint chosen on dev, never selected on test.
    for split in ("dev", "test"):
        call("scripts/launch.py", "--config", saved_config, "--mode", "eval",
             "--split", split, "--checkpoint", best)
    experiment = config_path.stem
    targets = json.loads((ROOT / "docs/paper_targets.json").read_text())
    if experiment in targets:
        result = subprocess.run([sys.executable, "scripts/compare_results.py", "--experiment",
                  experiment, "--dev", str(output / "dev_metrics.json"), "--test",
                  str(output / "test_metrics.json"), "--head", cfg.get("eval_head", "sequence")],
                  cwd=ROOT, check=True, capture_output=True, text=True)
        (output / "paper_comparison.json").write_text(result.stdout, encoding="utf-8")
        print(result.stdout, flush=True)
    else:
        print("No paper target registered for this configuration; metrics saved, comparison unavailable.")
    shutil.copy2(ROOT / "docs/AUDIT.md", output / "REPRODUCTION_LIMITATIONS.md")
    (output / "COMPLETED.json").write_text(json.dumps({"status": "measured reconstruction",
        "config": str(config_path), "best_checkpoint": str(best),
        "code_sha256": cfg["code_sha256"]}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["probe", "prepare", "train"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument('--campaign',default='phoenix14t_cslr_suite_seed0')
    args = parser.parse_args()
    os.chdir(ROOT)
    import yaml
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.stage == "probe":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Docker cannot access CUDA. Set up NVIDIA Container Toolkit/WSL GPU support first.")
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                a = torch.ones((16, 16), device=f"cuda:{index}")
                (a @ a).sum().item()
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}", flush=True)
        from repro.runtime import batch_plan
        print(json.dumps(batch_plan(torch.cuda.device_count(), cfg.get("global_batch", 6), cfg.get("micro_batch", 1))))
        return
    with workspace_lock():
        if args.stage == "prepare":
            prepare(cfg)
            if args.suite:
                for kind in ("swin_t", "pvig_tiny"):
                    call("scripts/fetch_backbones.py", "--backbone", kind)
        else:
            if args.suite:
                if args.config.as_posix() != "configs_repro/phoenix14t_cslr.yaml":
                    raise ValueError("Full suite uses PHOENIX14T CSLR. Use --single for a custom config.")
                from scripts.deployment import valid_campaign
                call("scripts/run_suite.py",'--output',ROOT/'runs'/valid_campaign(args.campaign))
            else:
                train(cfg, args.config)


if __name__ == "__main__":
    main()
