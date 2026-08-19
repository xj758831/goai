## Environme

The verified local environment is Ubuntu 22.04, Python 3.11, MuJoCo 3.10.0,
NumPy 1.26.0, ONNX Runtime 1.20.1, PyTorch 2.7.0, and ROS 2 Humble. ROS 2
Jazzy is not required for this baseline; another ROS 2 distribution may be
used if the listed message packages are available.

Install ROS 2 Humble message dependencies, then Python dependencies:

```bash
ROS_SETUP=/opt/ros/humble/setup.bash
source "$ROS_SETUP"
sudo apt install \
  ros-humble-rclpy \
  ros-humble-geometry-msgs \
  ros-humble-nav-msgs \
  ros-humble-rosgraph-msgs \
  ros-humble-std-msgs

PLAN2_PYTHON=/home/xj/miniconda3/envs/isaaclab-v232/bin/python
"$PLAN2_PYTHON" -m pip install -r requirements.txt
```

Change `PLAN2_PYTHON` and `ROS_SETUP` to the paths used on another machine.
The visible launcher also requires an X11/GLFW display and uses `$DISPLAY`.

## Run

```bash
cd /home/xj/Downloads/final_submission
export ROS_SETUP=/opt/ros/humble/setup.bash
export PLAN2_PYTHON=/home/xj/miniconda3/envs/isaaclab-v232/bin/python
bash run_submission_mp4_1x.sh
```

For a non-recording visible run, use the command in `README_CN.md` and add
`--no-video`. The output directory must be new. A valid complete run reports:

```text
route_complete=true
reached_score_count=33
last_reached_score=32
```

## What This Baseline Uses

The run uses the shipped official `policy.onnx`, known official waypoints, a
MuJoCo ground-truth route controller, and the verified Plan2 expert/oracle
controller for the exact `0.3769105421 m` pit. The point-cloud module is a
read-only shadow observer: it does not publish `cmd_vel` and is not wired into
the ONNX action output. Therefore this package should be described as an
engineering baseline rather than a completed perception-control submission.

See `README_CN.md` for the full Chinese instructions and `SUBMISSION_CONTENTS.md`
for the inclusion/exclusion record.
