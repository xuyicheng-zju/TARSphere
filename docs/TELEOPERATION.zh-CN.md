# 机器人与 NPC 遥操

[English](TELEOPERATION.md) · [安装说明](INSTALLATION.md)

在 TARSphere 场景中，由用户通过键盘或终端交互控制机器人和 NPC。机器人使用物理运动控制器；NPC 使用原生 animation 控制器和 NavMesh 行走。走位、姿态控制和 NPC 动作不调用语言模型。截图与视频录制按需开启。

## 启动

按照[安装说明](INSTALLATION.md)准备 Linux NVIDIA 环境、Isaac Sim 5.1、项目 Python 环境和外部资产，并配置仓库根目录的 `host_env.sh`。NPC 需要完整的带骨骼角色和原生动画资源，见[资产说明](../assets/README.md)。

在仓库根目录运行：

```bash
scripts/teleop.sh --episode examples/episodes/navigation.yaml
```

Episode 决定场景、机器人初始位置和 NPC。用 `--episode /path/to/episode.yaml` 选择其他场景配置，用 `--human-id Nora` 选择初始 NPC。首次加载可能准备材质、烘焙 NavMesh；出现 `[TELEOP] Ready` 后开始操作。

通过 SSH 使用终端控制：

```bash
scripts/teleop.sh --headless --episode examples/episodes/navigation.yaml
```

Headless 模式支持下述终端命令。窗口键盘和 GUI 视角切换需要可见的 Isaac 窗口；`--stream` 开启 Isaac Sim 原生串流。普通机器人和 NPC 控制不需要 agent API 密钥，也不需要启动感知或导航模型服务。

## 切换机器人和 NPC

启动时控制机器人。操作运动键时聚焦 Isaac 窗口；输入命令时切回启动它的终端。

| 输入 | 功能 |
| --- | --- |
| `list` | 列出场景中的 NPC ID。 |
| `human <id>` | 把键盘控制交给指定 NPC。 |
| `human` | 选择本会话的初始 NPC。 |
| `robot` | 把键盘控制交回机器人。 |
| Tab | 在机器人和选中的 NPC 之间切换。 |
| `status` | 查看当前控制对象、机器人姿态和 NPC 的位置、动作状态。 |

导航样例中可以依次输入：

```text
list
human Nora
look Nora 2
robot
move 0 0 15
status
```

执行 `human Nora` 后，在窗口中用运动键控制 Nora 行走；用 Tab 或 `robot` 切回机器人。`move` 等显式机器人命令始终操作机器人，NPC 动作命令则始终带 NPC ID。

## 窗口键盘

| 按键 | 机器人 | 动画 NPC |
| --- | --- | --- |
| I / K | 前进 / 后退 | 相对控制朝向向前 / 向后行走。 |
| J / L | 向左 / 向右移动 | 相对控制朝向向左 / 向右行走。 |
| U / O | 左转 / 右转 | 调整下一次行走的方向。 |
| F8 | 保存机器人第一视角静帧。 | 保存机器人第一视角静帧。 |

运动键可以组合使用；松开一个键不会清除其他键。短按 U/O 至少保留 0.15 秒仿真时间。切换控制对象或输入终端命令会清除按键状态，继续走动时需要重新按键。

每个 NPC 独立维护控制朝向。U/O 修改方向后，行走动画会转向目标并移动；站立时只按 U/O 不会让 NPC 原地旋转。NPC 键盘输入生成原生行走目标，经 NavMesh 和动画控制器执行。目标被拒绝时终端会显示原因。

## 机器人终端命令

| 命令 | 功能 |
| --- | --- |
| `move <前向米数> <左向米数> <旋转度数>` | 机器人自身坐标系中的一次运动，正角度向左转。 |
| `crouch [高度米数]` | 降低骨盆，默认高度 0.55 m。 |
| `stand` | 恢复行走姿态。 |
| `tilt <角度>` | 在支持范围内调整机器人相机俯仰。 |
| `wait <秒数>` | 推进仿真，让人物动画和动作继续执行。 |
| `ego on` / `ego off` | 在 GUI 中切换机器人第一视角与第三视角。 |

蹲姿下的行走、转向会被拒绝，先输入 `stand` 再运动。搬运时移动会保持已有的携物姿态。

如需物理抓取，准备 SAM3、GraspGenX 和任务物体资产，先启动配置好的工具服务：

```bash
source host_env.sh
source scripts/activate.sh
tarsphere/infra/services/start.sh
scripts/teleop.sh --episode episodes/case/fetch_and_deliver/base.yaml --with-grasp
```

输入 `grasp <物体外观描述>`，使用当前 ego RGB-D 定位物体并在同一次调用中执行默认物理抓取；反馈反映实际执行状态。对应 delivery episode 需包含匹配的 physical task 和物体资产。

## NPC 动画命令

| 命令 | 功能 |
| --- | --- |
| `goto <id> <x> <y> [z] [end_rot]` | 在世界坐标中指定目的地，通过 NavMesh 和原生动画行走。省略 z 时使用当前人物高度；可选 end_rot 为原生角色 yaw，单位为度。 |
| `look <id> [秒数]` | 执行原生观望动作。 |
| `idle <id>` | 请求原生 idle 动作。 |
| `sit <id> <座椅prim路径> [秒数]` | 使用场景与原生动画控制器支持的座位，接近并入座。 |

用 `list` 获取真实 NPC ID。`goto` 请选择可行走表面上的可达位置；`sit` 使用座椅实际的 USD prim 路径。`executed=True` 表示控制器接受请求，此时行走或入座可能还在进行。让仿真继续运行，并通过 `status` 检查完成或失败状态。

NPC 始终使用 animation 后端。键盘操作转为原生行走目标，人物不会通过传送或直接写位姿来移动。

## 可选截图与视频

输入 `capture [标签]` 或按 F8，保存原生 2560×1920 无损机器人第一视角 PNG。标签只使用字母、数字、`_`、`-`。用 `--capture-width`、`--capture-height` 修改分辨率；`--capture-layers` 额外保存原生实例分割得到的可见角色透明 PNG。

截图使用临时独立相机，并核验拍摄前后的物理时间、机器人与 NPC 位姿和原相机状态一致；图片与 metadata 一起发布后释放相机与渲染缓冲。

`--record-video` 开启连续机器人第一视角视频，需要 `ffmpeg`；`--record-ego` 保留抓取使用的 ego 观测。遥操本身不依赖这些选项。

会话日志和可选媒体默认写在源码外的 `~/.local/share/tarsphere/demos/teleop/<timestamp>/`。用 `--output /path/to/new/session.jsonl` 指定其他位置，已有会话日志不会被覆盖。

## 退出与常见问题

输入 `help` 查看命令，输入 `quit` 结束。Headless 模式关闭标准输入也会退出。`--max-loops` 可限制开发会话长度，默认不限制。

| 问题 | 检查方式 |
| --- | --- |
| 按键没有响应 | 聚焦 Isaac 窗口，用 `status` 确认控制对象；终端命令后重新按下运动键。 |
| NPC 站着不转身 | U/O 调整下一次行走方向，随后按行走键执行动画。 |
| NPC 目标被拒绝 | 查看终端反馈，选择 NavMesh 可达点，并检查是否仍有原生动作进行中。 |
| 缺少 NPC 或动画 | 检查 episode 的 NPC ID、带骨骼角色和原生动画资产。 |
| 蹲姿不能走动 | 输入 `stand` 再重试。 |
| 不能抓取 | 确认工具服务、物理任务配置，以及启动时的 `--with-grasp`。 |

用 `scripts/teleop.sh --help` 查看运动数值、时间、相机和输出参数。
