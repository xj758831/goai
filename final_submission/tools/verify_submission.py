#!/usr/bin/env python3
"""Read-only structural and optional runtime verification for this package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TRACK_XML = Path(
    "src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10_track.xml"
)
POLICY = Path("src/S10_sdk_deploy/policy/policy.onnx")
ENTRY = Path("scripts/mujoco/run_plan2_full_track_closed_loop_v57.py")
ROS_ENTRY = Path(
    "src/S10_sdk_deploy/scripts/plan2_expert_tail_navigator_sim_sync.py"
)
SUCCESS_SUMMARY = Path(
    "logs/mujoco/plan2_v57_full_track_final_20260818_run4/summary.json"
)

LOCKED_HASHES = {
    Path("src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10.xml"):
        "74582d1c27f43eed6c86befb0d11c039d34ae4d1cee7aa2cad3e48b7c005bcf4",
    Path("src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/scene.xml"):
        "832d326e8fb14f7976b355ee8636d125c80d74c07894adb9ab0f2232c3e9d564",
    TRACK_XML:
        "f1d373ab39ec38344aa6ada5c2ca18594c348448e6c4bb382409bebce15eee6d",
    POLICY:
        "0ac99f3093d4a984d7587b88d57300cbf7ec2f788401dfa1570d1e4800568f6b",
    Path("logs/mujoco/plan2_hooked_tuck_closed_loop_exact_official_track_peak_20260812_v4/summary.json"):
        "683e680e8b7e6ea5268e4ce0f6857b4c05278514c62580a94b57f83e25540ed8",
    Path("logs/mujoco/plan2_hooked_tuck_closed_loop_exact_official_track_peak_20260812_v4/trace.npz"):
        "7bcb30f395eb8cd557671acb245bc75104ee1e66fefb2646518eed7ad1f9b516",
    Path("logs/mujoco/s10_m20_lidar72_capability_depth021_complementary_distill_base10_alt1_lr1e6_epoch100_20260809_v1/checkpoint_best_distilled.pt"):
        "a04308a3ca40243cd9736b870471dc3ac08b481ed24d3dd71320b4789d535af3",
    Path("logs/mujoco/s10_m20_phase_estimator_19cases_20260810_v1/phase_estimator.pt"):
        "6d497ed2b21765773ab286cebedf459f46a9f7b70ace0d649f63250658e56a66",
    SUCCESS_SUMMARY:
        "26a0234734e3c87741914e6e74b6a8dd236c9536424c3175fa0e77d05436101f",
    Path("evidence/LOCKED_499S_MANIFEST.md"):
        "84bf4146ede42cd3ab43a05080e7d516635f0338c0294e191eda7648a812bef5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_structure() -> None:
    forbidden_dirs = {".git", ".codex_backups", "__pycache__"}
    forbidden_suffixes = {".gif", ".mp4", ".pyc", ".jsonl"}
    problems = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.is_dir() and path.name in forbidden_dirs:
            problems.append(f"forbidden directory: {relative}")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes:
            problems.append(f"forbidden generated file: {relative}")
    if problems:
        raise RuntimeError("\n".join(problems))

    for relative, expected in LOCKED_HASHES.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch: {relative}: {actual}")

    summary = json.loads((ROOT / SUCCESS_SUMMARY).read_text(encoding="utf-8"))
    expected_result = (
        summary.get("route_complete") is True
        and summary.get("reached_score_count") == 33
        and summary.get("last_reached_score") == 32
        and summary.get("pit_expert", {}).get("success") is True
    )
    if not expected_result:
        raise RuntimeError("locked summary does not report 33/33 and pit success")

    for path in ROOT.rglob("*.py"):
        compile(path.read_bytes(), str(path), "exec")


def verify_runtime() -> None:
    import mujoco
    import onnxruntime
    import yaml

    mujoco.MjModel.from_xml_path(str(ROOT / TRACK_XML))
    onnxruntime.InferenceSession(
        str(ROOT / POLICY), providers=["CPUExecutionProvider"]
    )
    for path in (ROOT / "config").glob("*.yaml"):
        yaml.safe_load(path.read_text(encoding="utf-8"))

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="goai_follow_submission_verify_") as output_dir:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / ENTRY),
                "--output-dir",
                output_dir,
                "--help",
            ],
            cwd=ROOT,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    subprocess.run(
        [
            "/usr/bin/python3",
            str(ROOT / ROS_ENTRY),
            "--sync-ack-topic",
            "/goai_follow_submission_verify_unused",
            "--help",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="also load MuJoCo/ONNX and import the Python and ROS entry points",
    )
    args = parser.parse_args()
    verify_structure()
    print("structure: OK")
    if args.runtime:
        verify_runtime()
        print("runtime: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
