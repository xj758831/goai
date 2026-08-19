# Plan2 v57 Locked Success

This manifest protects the visible full-track v57 run completed on 2026-08-18.

## Verified result

- Output: `logs/mujoco/plan2_v57_full_track_final_20260818_run4`
- Route complete: true
- Official scores: 33/33 (`0..32`)
- Final simulation time: `499.9599999950894 s`
- Pit depth: `0.3769105421 m`
- Point-cloud samples/errors: `4946/0`
- ROS sync requests/timeouts: `6225/0`
- Single persistent MuJoCo viewer: true
- State reset or teleport: false

## v57 entry hashes

- `scripts/mujoco/run_plan2_full_track_closed_loop_v57.py`: `61d636d2f26f60427bd8db4c2b6ea342d2fda2732e2b9b730c6a64e235269a7a`
- `scripts/mujoco/plan2_wp30_closed_loop_v57.py`: `504bdca638254ed4688806dd105571dff40055cf41580de0becb4c238efcc524`
- `scripts/mujoco/run_plan2_full_track_pointcloud_recovery_candidate.py`: `4fc83d8abcaf4e5e71a453e6ebde4d44582804af51c8aa7c1418784e500655ad`
- `src/S10_sdk_deploy/scripts/plan2_expert_tail_navigator_sim_sync.py`: `4701a376eb6118be9f354b02928693778fd5f3cb8e2a5cd7bc5a5ee079b3bdbc`
- `config/path_plan2_wp24_entry_rightshift_20260815.yaml`: `381ffbc0082d207fda1ed1af887a2821227b63fe39aef133d726751c78139add`
- `config/plan2_wp25_wp27_behavior_straight_corridor.yaml`: `0522ae3b079c178a837ab5dc7d3245193ec98da2f1ae924ad502cac2fd5c9113`

## Protected asset hashes

- `src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10.xml`: `74582d1c27f43eed6c86befb0d11c039d34ae4d1cee7aa2cad3e48b7c005bcf4`
- `src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/scene.xml`: `832d326e8fb14f7976b355ee8636d125c80d74c07894adb9ab0f2232c3e9d564`
- `src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10_track.xml`: `f1d373ab39ec38344aa6ada5c2ca18594c348448e6c4bb382409bebce15eee6d`
- `src/S10_sdk_deploy/policy/policy.onnx`: `0ac99f3093d4a984d7587b88d57300cbf7ec2f788401dfa1570d1e4800568f6b`

## Reproduction command

```bash
cd /home/xj/Downloads/goai_follow
DISPLAY=:1 MUJOCO_GL=glfw \
  /home/xj/miniconda3/envs/isaaclab-v232/bin/python \
  scripts/mujoco/run_plan2_full_track_closed_loop_v57.py \
  --output-dir logs/mujoco/plan2_v57_reproduction \
  --max-sim-seconds 600 \
  --viewer-speed 8 \
  --expert-viewer-speed 1 \
  --no-video \
  --ros-tail \
  --ros-domain 232 \
  --tail-nav-script src/S10_sdk_deploy/scripts/plan2_expert_tail_navigator_sim_sync.py \
  --sim-sync \
  --sync-timeout-ms 200 \
  --clean-hard-exit
```

The v57 takeover is a marker/height-gated experimental recovery, not a proven
point-cloud-driven navigation controller. The successful result must not be
renamed as a true radar closed loop.
