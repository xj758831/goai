#!/usr/bin/env bash
# ROS setup files may read optional variables that are unset. Load ROS before
# enabling nounset so the launcher works in a fresh terminal.
set -eo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ ! -f "$ROS_SETUP" ]]; then
  printf 'ROS setup file not found: %s\n' "$ROS_SETUP" >&2
  printf 'Set ROS_SETUP to the installed ROS 2 setup.bash path.\n' >&2
  exit 2
fi
source "$ROS_SETUP"
set -u

PLAN2_PYTHON="${PLAN2_PYTHON:-/home/xj/miniconda3/envs/isaaclab-v232/bin/python}"
DISPLAY_NUMBER="${DISPLAY:-:1}"
ROS_DOMAIN="${ROS_DOMAIN:-231}"
RUN_DIR="${RUN_DIR:-logs/mujoco/mp4_submission_1x_$(date +%Y%m%d_%H%M%S)}"

if ! [[ "$ROS_DOMAIN" =~ ^[0-9]+$ ]] || (( ROS_DOMAIN > 232 )); then
  printf 'ROS domain must be an integer from 0 to 232; got: %s\n' "$ROS_DOMAIN" >&2
  exit 2
fi

if [[ -e "$PROJECT_ROOT/$RUN_DIR" ]]; then
  printf 'Output directory already exists: %s\n' "$PROJECT_ROOT/$RUN_DIR" >&2
  exit 2
fi

"$PLAN2_PYTHON" -c 'import mujoco; print("MuJoCo", mujoco.__version__)'
printf 'Output: %s\n' "$PROJECT_ROOT/$RUN_DIR"
printf 'Display: %s, ROS domain: %s, recording: MP4 only at 1x\n' "$DISPLAY_NUMBER" "$ROS_DOMAIN"

exec env DISPLAY="$DISPLAY_NUMBER" MUJOCO_GL=glfw \
  "$PLAN2_PYTHON" scripts/mujoco/run_plan2_full_track_closed_loop_v57.py \
  --output-dir "$RUN_DIR" \
  --max-sim-seconds 600 \
  --viewer-speed 1 \
  --expert-viewer-speed 1 \
  --frame-period 0.08 \
  --pit-frame-period 0.08 \
  --video \
  --mp4 \
  --ros-tail \
  --ros-domain "$ROS_DOMAIN" \
  --tail-nav-script "$PROJECT_ROOT/src/S10_sdk_deploy/scripts/plan2_expert_tail_navigator_sim_sync.py" \
  --tail-behavior-config "$PROJECT_ROOT/config/plan2_wp25_wp27_behavior_straight_corridor.yaml" \
  --tail-waypoints "$PROJECT_ROOT/config/path_plan2_wp24_entry_rightshift_20260815.yaml" \
  --sim-sync \
  --sync-timeout-ms 200 \
  --clean-hard-exit
