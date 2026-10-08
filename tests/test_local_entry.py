import importlib.util
from pathlib import Path
import io
import tarfile
import contextlib

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docker_isolation_and_offline_training():
    entry = load("local_entry", ROOT / "run.py")
    prepare = entry.container_command("image", "prepare", "config.yaml", 16, 4, "test")
    train = entry.container_command("image", "train", "config.yaml", 16, 4, "test")
    assert train[train.index("--network") + 1] == "none"
    assert prepare[prepare.index("--network") + 1] == "bridge"
    assert "--privileged" not in train and "--read-only" in train
    assert "--gpus" in train and "--gpus" not in prepare
    mounts = [train[i + 1] for i, value in enumerate(train) if value == "--mount"]
    assert len(mounts) == 4
    assert all("readonly" in mount for mount in mounts[:3])
    assert "readonly" not in mounts[3] and "dst=/workspace/runs" in mounts[3]
    assert all("docker.sock" not in mount for mount in mounts)


def test_single_new_campaign_is_forwarded_and_uses_separate_output(tmp_path,monkeypatch):
    entry=load('single_campaign_entry',ROOT/'run.py')
    command=entry.container_command('image','train','configs_repro/phoenix14t_cslr.yaml',16,4,'test',False,'reviewed-fix')
    assert command[command.index('--campaign')+1]=='reviewed-fix' and '--suite' not in command
    pipeline=load('single_campaign_pipeline',ROOT/'scripts/run_local.py')
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    cfg=tmp_path/'phoenix14t_cslr.yaml'; cfg.write_text('task: cslr\noutput: runs/old-run\n')
    monkeypatch.setattr(pipeline.sys,'argv',['run_local.py','--stage','train','--config',str(cfg),'--campaign','reviewed-fix'])
    monkeypatch.setattr(pipeline.os,'chdir',lambda *args:None)
    monkeypatch.setattr(pipeline,'workspace_lock',contextlib.nullcontext)
    seen=[]
    monkeypatch.setattr(pipeline,'train',lambda config,*args:seen.append(config) or True)
    pipeline.main()
    assert seen[0]['output']==str(tmp_path/'runs/reviewed-fix/phoenix14t_cslr')


def test_resume_and_evaluate_same_dev_selected_checkpoint(tmp_path, monkeypatch):
    pipeline = load("local_pipeline", ROOT / "scripts/run_local.py")
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "code_hash", lambda: "fixed-code")
    monkeypatch.setattr(pipeline, "training_finished",lambda *args:True)
    output = tmp_path / "runs/experiment"
    output.mkdir(parents=True)
    (output / "last.pt").write_bytes(b"saved")
    (output / "best.pt").write_bytes(b"best")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "paper_targets.json").write_text('{}')
    (docs / "AUDIT.md").write_text("Reconstruction limitations")
    calls = []
    monkeypatch.setattr(pipeline, "call", lambda *args: calls.append(list(map(str, args))))
    cfg = {"output": str(output), "task": "cslr"}
    pipeline.train(cfg, Path("configs_repro/experiment.yaml"))
    assert "--resume" in calls[1] and str(output / "last.pt") in calls[1]
    assert len(calls) == 4
    for call, split in zip(calls[2:], ("dev", "test")):
        assert call[call.index("--split") + 1] == split
        assert call[call.index("--checkpoint") + 1] == str(output / "best.pt")
    assert (output / "COMPLETED.json").is_file()
    changed = {"output": str(output), "task": "cslr", "seed": 7}
    with pytest.raises(ValueError, match="different config"):
        pipeline.train(changed, Path("configs_repro/experiment.yaml"))


def make_archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, content, link in entries:
            info = tarfile.TarInfo(name)
            if link is not None:
                info.type = tarfile.LNKTYPE
                info.linkname = link
                archive.addfile(info)
            else:
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))


def test_official_layout_and_internal_hardlinks(tmp_path):
    fetch = load("phoenix_fetch", ROOT / "scripts/fetch_phoenix.py")
    base = "PHOENIX-2014-T-release-v3/PHOENIX-2014-T/"
    entries = []
    for split in ("train", "dev", "test"):
        entries.append((base + f"annotations/manual/PHOENIX-2014-T.{split}.corpus.csv", b"name|orth", None))
        frame = base + f"features/fullFrame-210x260px/{split}/sample/images0014.png"
        entries.extend([(frame, b"image-payload", None), (frame + ".copy", b"", frame)])
    archive = tmp_path / "release.tar.gz"
    make_archive(archive, entries)
    data = tmp_path / "phoenix14t"
    fetch.ensure_phoenix(data, archive)
    assert fetch.valid_layout(data)
    assert (data / "features/fullFrame-210x260px/train/sample/images0014.png.copy").read_bytes() == b"image-payload"


@pytest.mark.parametrize("entry", [("../outside", b"payload", None),
                                   ("copy.png", b"", "../../outside")])
def test_extraction_rejects_escape_paths_and_links(tmp_path, entry):
    fetch = load("phoenix_fetch_security", ROOT / "scripts/fetch_phoenix.py")
    archive = tmp_path / "unsafe.tar.gz"
    make_archive(archive, [entry])
    with tarfile.open(archive, "r:gz") as source:
        with pytest.raises((ValueError, tarfile.FilterError)):
            fetch.safe_extract(source, tmp_path / "output")
