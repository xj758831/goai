## 一、当前已经验证的结果

在官方 MuJoCo 赛道模型上，这个版本完成了全部得分点：

- 官方得分点：`33/33`，即 0 到 32 全部到达；
- 原始仿真时间：`499.9599999950894 s`，约 500 秒；
- ROS 同步请求/超时：`6225/0`；
- 运行过程中没有瞬移或重置机器人状态。

验证摘要位于：

```text
logs/mujoco/plan2_v57_full_track_final_20260818_run4/summary.json
```

更完整的锁定信息位于：

```text
evidence/LOCKED_499S_MANIFEST.md
```

## 二、比赛计时如何换算
因此，按照当前仓库 README 的 Navigation 口径：

```text
原始时间：499.9599999950894 秒
计分时间：499.9599999950894 / 1.2
       = 416.6333333292412 秒
```

## 三、这份基线实际使用了什么

这份 499 秒结果是工程验证基线，具体使用了：

1. 官方提供的 `policy.onnx`，负责 S10 的主要低层运动；
2. MuJoCo 的 ground-truth 位姿；
3. 已知官方 waypoint 和路线导航器；
4. Plan2 专家/Oracle 控制器，用于通过精确的 `0.377 m` 深坑；
5. 通过深坑后使用 ROS 尾段导航器继续走完剩余得分点；
6. 点云模块主要用于观察和记录，目前不是完整的自训练感知控制策略。


## 四、文件夹结构

- `scripts/mujoco/`：MuJoCo 运行脚本、v57 入口及相关依赖；
- `src/S10_sdk_deploy/S10_description/s10_mjcf/`：S10 模型、官方赛道、场景和全部 mesh；
- `src/S10_sdk_deploy/policy/policy.onnx`：官方低层运动策略；
- `src/S10_sdk_deploy/scripts/`：ROS 2 尾段导航和点云相关脚本；
- `config/`：官方路线、尾段行为配置和点云配置；
- `logs/mujoco/`：运行所需的小型 checkpoint 和验证摘要；
- `BASELINE_MANIFEST.json`：机器可读的结果、哈希和依赖清单；
- `SUBMISSION_CONTENTS.md`：本次精简保留/排除内容的依据；
- `tools/verify_submission.py`：只读完整性和运行依赖自检。

## 五、环境与依赖

本机验证使用的主要环境：

- Ubuntu 22.04（本机验证环境；不要求必须是 Ubuntu 24.04）；
- Python 3.11；
- MuJoCo `3.10.0`；
- NumPy `1.26.0`；
- ONNX Runtime `1.20.1`；
- PyTorch `2.7.0`；
- ROS 2 Humble，包含 `rclpy`、`geometry_msgs`、`nav_msgs`、
  `rosgraph_msgs`、`std_msgs`。

本机使用 ROS 2 Humble。

先加载 ROS，并安装 ROS 消息包（Humble 示例）：

```bash
ROS_SETUP=/opt/ros/humble/setup.bash
source "$ROS_SETUP"
sudo apt install \
  ros-humble-rclpy \
  ros-humble-geometry-msgs \
  ros-humble-nav-msgs \
  ros-humble-rosgraph-msgs \
  ros-humble-std-msgs
```

再在 Python 3.11 环境中安装本包列出的 Python 依赖：

```bash
PLAN2_PYTHON=/home/xj/miniconda3/envs/isaaclab-v232/bin/python
"$PLAN2_PYTHON" -m pip install -r requirements.txt
"$PLAN2_PYTHON" -c "import mujoco; print('MuJoCo', mujoco.__version__)"
```

`PLAN2_PYTHON` 只是本机示例路径。另一台电脑应改成已经安装 Python 3.11、
MuJoCo 和 ONNX Runtime 的解释器。运行可视化需要可用的 X11/GLFW 显示，
并把 `DISPLAY` 改成实际显示编号。

## 六、运行完整 MuJoCo 基线

下面命令用于重新运行 v57。`--output-dir` 必须使用一个新的、尚不存在的目录。
Fast DDS 的 ROS domain 必须在 `0..232` 范围内。

```bash
cd /home/xj/Downloads/final_submission
ROS_SETUP=/opt/ros/humble/setup.bash
source "$ROS_SETUP"
PLAN2_PYTHON=/home/xj/miniconda3/envs/isaaclab-v232/bin/python
"$PLAN2_PYTHON" -c "import mujoco; print('MuJoCo', mujoco.__version__)"
RUN_DIR="logs/mujoco/v57_reproduction_$(date +%Y%m%d_%H%M%S)"

env DISPLAY=:1 MUJOCO_GL=glfw \
  "$PLAN2_PYTHON" scripts/mujoco/run_plan2_full_track_closed_loop_v57.py \
  --output-dir "$RUN_DIR" \
  --max-sim-seconds 600 \
  --viewer-speed 8 \
  --expert-viewer-speed 8 \
  --no-video \
  --ros-tail \
  --ros-domain 231 \
  --tail-nav-script "$PWD/src/S10_sdk_deploy/scripts/plan2_expert_tail_navigator_sim_sync.py" \
  --tail-behavior-config "$PWD/config/plan2_wp25_wp27_behavior_straight_corridor.yaml" \
  --tail-waypoints "$PWD/config/path_plan2_wp24_entry_rightshift_20260815.yaml" \
  --sim-sync \
  --sync-timeout-ms 200 \
  --clean-hard-exit
```

说明：

- `DISPLAY=:1` 表示使用当前桌面的显示编号。如果你的电脑不是这个编号，需要改成实际的 DISPLAY；
- `--no-video` 表示不保存视频，但仍然可以打开 MuJoCo 窗口观察；
- 如果想保存视频，删除 `--no-video`，并使用新的输出目录；
- `--viewer-speed 8` 和 `--expert-viewer-speed 8` 让所有阶段使用同一显示倍率，避免坑底降速；显示倍率不改变成绩；
- 需要使用没有被其他 ROS 程序占用的 `--ros-domain`。

为避免多行命令复制时漏掉参数，提交录制可以直接运行附带的启动脚本：

```bash
cd /home/xj/Downloads/final_submission
export ROS_SETUP=/opt/ros/humble/setup.bash
bash run_submission_mp4_1x.sh
```

脚本会先打印输出目录，然后打开 MuJoCo。完成后，在该目录中查看
`full_track_viewer_rollout.mp4` 和 `summary.json`。

提交前可在同一 Python/ROS 环境中执行只读自检：

```bash
ROS_SETUP=/opt/ros/humble/setup.bash
source "$ROS_SETUP"
PLAN2_PYTHON=/home/xj/miniconda3/envs/isaaclab-v232/bin/python
"$PLAN2_PYTHON" tools/verify_submission.py --runtime
sha256sum -c SUBMISSION_MANIFEST.sha256
```

运行结束后，只有当摘要中出现以下内容，才算一次完整成功：

```text
route_complete=true
reached_score_count=33
last_reached_score=32
```

## 八、限制说明

当前结果使用官方 `policy.onnx`、已知路线和经过验证的 Plan2 过坑专家控制器。
点云代码是只读 shadow observer，不发布 `cmd_vel`，也没有接入当前 ONNX policy
的动作输出。
