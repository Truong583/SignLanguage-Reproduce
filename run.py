"""One-command Docker entry point; only the Python standard library is needed on host."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from scripts.deployment import (OperationLock,active_release,settings,valid_campaign,
    image_name,ensure_image,workspace_label,DEFAULT_CAMPAIGN)
from scripts.diagnostics import Recorder

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get('SIGNLANGUAGE_WORKSPACE',ROOT)).resolve()


def container_command(image, stage, config, memory, cpus, name, suite=False,campaign=DEFAULT_CAMPAIGN):
    uid = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "1000:1000"
    command = ["docker", "run", "--rm", "--init", "--name", name,
               '--label','signlanguage.workspace='+workspace_label(WORKSPACE),
               "--user", uid, "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--pids-limit", "2048",
               "--memory", f"{memory}g", "--memory-swap", f"{memory}g",
               "--cpus", str(cpus), "--shm-size", "2g",
               "--tmpfs", "/tmp:rw,nosuid,nodev,size=4g",
               "--network", "bridge" if stage == "prepare" else "none",
               "--mount", f"type=bind,src={ROOT},dst=/workspace,readonly"]
    for folder in ("data", "assets", "runs"):
        writable = stage == "prepare" or folder == "runs"
        command += ["--mount", f"type=bind,src={WORKSPACE / folder},dst=/workspace/{folder}" +
                    ("" if writable else ",readonly")]
    if stage != "prepare":
        command += ["--gpus", "all"]
    command += ["--workdir", "/workspace", image, "python", "scripts/run_local.py",
                "--stage", stage, "--config", config]
    if suite: command += ["--suite"]
    if suite and campaign!=DEFAULT_CAMPAIGN: command += ['--campaign',campaign]
    return command


def main(recorder=None):
    parser = argparse.ArgumentParser(description="Prepare, train, resume and evaluate in Docker.")
    parser.add_argument("--config", default="configs_repro/phoenix14t_cslr.yaml")
    parser.add_argument("--memory-gb", type=int)
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--base-image", default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    parser.add_argument("--plan", action="store_true", help="Show workflow without downloading/training")
    parser.add_argument("--single", action="store_true", help="Run just --config instead of the PHOENIX14T CSLR suite")
    parser.add_argument('--once',action='store_true',help='Run once without the GitHub/W&B supervisor')
    parser.add_argument('--campaign',default=settings(WORKSPACE).get('campaign',DEFAULT_CAMPAIGN))
    args = parser.parse_args()
    valid_campaign(args.campaign)
    path = (ROOT / args.config).resolve()
    if not path.is_relative_to(ROOT / "configs_repro") or not path.is_file():
        parser.error("--config must be an existing YAML inside configs_repro")
    config = path.relative_to(ROOT).as_posix()
    if not args.single and config != 'configs_repro/phoenix14t_cslr.yaml':
        parser.error('Full suite uses PHOENIX14T CSLR. Add --single for a custom config.')
    if args.plan:
        print(json.dumps({"config": config, "stages": ["GPU check", "prepare data/assets",
              "validate splits", "resume or train", "evaluate best checkpoint on dev/test",
              "compare paper metrics"], "scope": "one configured experiment" if args.single else "68 PHOENIX14T CSLR reconstruction experiments (67 quantitative + 1 qualitative reference)",
              "output": "runs/<campaign>/<experiment>/", "prerequisites": "Python 3.12+, Docker with NVIDIA GPU support",
              'supervisor':'setup_machine.py once; run.py supervises GitHub updates and private W&B; --once runs without supervisor',
              'ci':False}, indent=2))
        return
    if shutil.which("docker") is None:
        raise RuntimeError("Install/start Docker with NVIDIA GPU support first; see CHAY_MOT_LENH.md.")
    running=subprocess.run(['docker','ps','--filter','label=signlanguage.workspace='+workspace_label(WORKSPACE),
        '--format','{{.ID}}'],capture_output=True,text=True,check=True)
    if running.stdout.strip(): raise RuntimeError('A training container for this workspace is already running. Stop/review it before starting another run.')
    info = subprocess.run(["docker", "info", "--format", "{{json .}}"],
                          capture_output=True, text=True, check=True)
    server = json.loads(info.stdout)
    if server.get("OSType") != "linux":
        raise RuntimeError("Docker must use Linux containers with NVIDIA GPU support.")
    available_memory = server["MemTotal"] / 1024**3
    memory = args.memory_gb or int(available_memory * .7)
    cpus = args.cpus or max(1, min(8, server["NCPU"] - 2))
    if memory < 8 or memory > available_memory * .9:
        raise RuntimeError("Docker needs enough RAM: limit >=8 GiB and <=90% available RAM.")
    if cpus < 1 or cpus > server["NCPU"]:
        raise RuntimeError("CPU limit exceeds Docker resources.")
    for folder in ("data", "assets", "runs"):
        (WORKSPACE / folder).mkdir(exist_ok=True)
    call=recorder.call if recorder else subprocess.run
    # Host does not install packages or modify Docker/driver configuration.
    if recorder: recorder.stage='verify_source'
    call([sys.executable, "scripts/verify_bundle.py"], cwd=ROOT, **({} if recorder else {'check':True}))
    if recorder: recorder.stage='docker_build'
    image=ensure_image(ROOT,args.base_image,recorder.call if recorder else None)
    print(f"Docker limits: {memory} GiB RAM, {cpus} CPU(s). Effective batch remains in config.", flush=True)
    for stage in ("probe", "prepare", "train"):
        name = "signlanguage-" + uuid.uuid4().hex[:12]
        try:
            if recorder: recorder.stage=stage
            call(container_command(image, stage, config, memory, cpus, name, not args.single,args.campaign), **({} if recorder else {'check':True}))
        except KeyboardInterrupt:
            subprocess.run(["docker", "stop", "--time", "120", name], check=False)
            raise
    print("Finished. Measured metrics and paper comparison are saved under runs/.")


def monitored_run():
    # Parent holds the lock while a pinned source snapshot runs in a child.
    if not os.environ.get('SIGNLANGUAGE_LOCK_HELD'):
        with OperationLock(WORKSPACE):
            env=os.environ.copy(); env['SIGNLANGUAGE_LOCK_HELD']='1'; env['SIGNLANGUAGE_WORKSPACE']=str(WORKSPACE)
            source=active_release(WORKSPACE)
            if source is None:
                from update import bootstrap_snapshot,activate
                from scripts.deployment import release_path
                release=bootstrap_snapshot(WORKSPACE)
                activate(WORKSPACE,release,release,settings(WORKSPACE))
                source=release_path(WORKSPACE,release['release_id'])
            process=subprocess.Popen([sys.executable,str(source/'run.py'),*sys.argv[1:]],env=env)
            try: return process.wait()
            except KeyboardInterrupt:
                # Console SIGINT reaches the child too; keep the update lock until it stops.
                try: process.wait(timeout=120)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                return 130
    if '--plan' in sys.argv:
        main(); return 0
    recorder=Recorder(WORKSPACE,ROOT)
    try:
        main(recorder); recorder.finish(0); return 0
    except KeyboardInterrupt:
        print("Stopped. Run the same command again to resume the saved checkpoint.", file=sys.stderr)
        recorder.finish(130); return 130
    except Exception as error:
        code=getattr(error,'returncode',1)
        recorder.finish(code,error); return 75 if code==75 else 1


if __name__ == "__main__":
    try:
        if '--plan' in sys.argv:
            main(); raise SystemExit(0)
        if '--once' not in sys.argv and not os.environ.get('SIGNLANGUAGE_LOCK_HELD'):
            from scripts.supervisor import supervise
            raise SystemExit(supervise(WORKSPACE,ROOT,sys.argv[1:]))
        raise SystemExit(monitored_run())
    except (RuntimeError,ValueError) as error:
        print(f'STOPPED: {error}',file=sys.stderr); raise SystemExit(1)
