# Pipeline Tool

算子模型开发 Pipeline 工具 — 从 RGB-D 数据采集到 FoundationPose 6-DoF 位姿估计的自动化数据准备流水线。

## 流水线概览

```
采集(RGB-D + 多视角照片) → SAM2 蒙版 → Hunyuan 3D 生成 → 缩放校正 → 数据集打包 → FoundationPose 推理
```

| 阶段 | 功能 | 依赖 |
|------|------|------|
| `masks` | SAM2 分割，从第一帧 RGB 生成目标蒙版 | Docker (sam2-backend-1) |
| `hunyuangen` | 腾讯混元 3D API，多视角照片 → OBJ 模型 | 腾讯云 API Key |
| `scale` | OBJ 等比缩放至真实物理尺寸 | 无 |
| `package` | 组装 FoundationPose 标准数据集 | 无 |
| `foundationpose` | FoundationPose 推理，逐帧位姿估计 | Docker (foundationpose) |

## 安装

```bash
# 基础依赖
pip install -r requirements.txt

# 混元 3D 阶段（可选）
pip install -r requirements-hunyuan.txt

# 开发/测试
pip install -r requirements-dev.txt
```

需要 Python >= 3.10，推荐使用 `obe` conda 环境。

## 快速开始

```bash
# 1. 在 Web 上上传/选择 Task 后进入配置页
WEB_PASSWORD=your-password ./run.sh web
# 打开 /configs，只配置 Pipeline、目标类别、步骤 Enable/Skip、点位和真实尺寸

# 2. 或者直接使用 Task 级配置运行
./run.sh run --config tasks/mouse002/task.yaml

# 3. 查看进度
./run.sh status --config tasks/mouse002/task.yaml

# 4. 单独重跑某个阶段
./run.sh stage package --config tasks/mouse002/task.yaml --force
```

## CLI 命令

```
pipeline run [preset] --config <path> [--force]   # 运行预设流水线
pipeline stage <name> --config <path> [--force]    # 运行单个阶段
pipeline status --config <path>                    # 查看任务状态
```

## 配置文件

根级 `configs/foundationpose.yaml` 已退役。当前配置分为两层：

- `tasks/<task>/task.yaml`：业务配置，记录 Task、Pipeline、类别、输入目录、SAM2 点位、真实尺寸。
- `configs/pipelines/*.yaml`、`configs/algorithms/*.yaml`、`configs/runtime/*.yaml`：平台配置，记录阶段顺序、算法默认参数和服务器运行环境。普通用户不需要在 Web 上编辑这些字段。

Task 配置示例：

```yaml
task_id: mouse002
pipeline: pose6d
runtime: server
class_id: 0

input:
  rgbd_dir: ./tasks/mouse002/
  multi_views_dir: ./tasks/mouse002/views/
  first_frame: 0

sam2:
  points:
    - [326, 258]
  labels: [1]

real_size:
  longest_edge: 0.126  # 米，物体的最长边

output_dir: output/
```

验证新配置结构：

```bash
PYTHONPATH=. python -c 'from pipeline.config import load_config; c=load_config("tasks/mouse002/task.yaml", project_root="."); print(c.task, c.preset, c.sam2.points)'
```

## 架构

```
pipeline/
├── cli.py              # argparse CLI 入口
├── config.py           # 配置加载、数据类、校验
├── manifest.py         # JSON manifest 阶段追踪、断点续跑
├── pipeline.py         # PipelineOrchestrator 编排器
├── presets.yaml        # 预设流水线定义
└── stages/
    ├── base.py         # BaseStage ABC
    ├── masks.py        # SAM2 蒙版
    ├── hunyuangen.py   # 混元 3D
    ├── scale.py        # OBJ 缩放
    ├── package.py      # FP 数据集打包
    └── foundationpose.py  # 位姿推理
```

核心设计：
- **Stage 注册表模式** — `@register_stage("name")` 装饰器，加新阶段只需继承 `BaseStage` 并注册
- **Manifest 断点续跑** — 每个阶段完成后记录到 JSON manifest，失败/中断后跳过已完成阶段
- **Pipeline 编排** — `configs/pipelines/*.yaml` 定义阶段序列，一个命令跑完整流程
- **配置分层** — Task 级业务配置和平台级运行配置分离，Web 配置页只暴露业务决策

## 输出目录结构

```
output/<task>/
└── runs/<run_id>/
    ├── resolved_config.yaml
    ├── manifest.json
    └── stages/
        ├── masks/
        ├── hunyuangen/
        ├── scale/
        ├── package/
│   ├── masks/00000.png
│   ├── mesh/textured_simple.obj
│   ├── cam_K.txt
│   └── camera_params.json
└── foundationpose/
    └── ob_in_cam/*.txt
```

## 开发

```bash
# 运行测试
pytest tests/ -v

# 端到端测试（需要 bottle 参考数据集）
pytest tests/test_e2e_bottle.py -v
```

## 服务器部署

流水线涉及 Docker 容器（SAM2、FoundationPose）和腾讯云 API。通过 `upload.sh`（位于 `~/code/obe/`）将本地采集数据 rsync 到部署服务器 `180.127.11.166:21240`，然后在服务器上执行流水线。


数据采集阶段（本机 code/obe/）

  入口命令：
    python rgbd_resize_copy.py --name mouse008 --views --pick --len 0.126
  流程：
    1. 多视角拍摄--views — 相机固定，用户手动旋转物体。Orbbec Gemini 335L RGB 流预热 30 帧后，
       交互式拍摄 front/left/back/right 四个方向，Space 拍摄、R 重拍、Q 退出，
       保存为 views/{front,left,back,right}.jpg。

    2. RGB-D 序列采集 — 彩色+深度流帧同步，深度对齐到彩色视点，预热 30 帧后连续采集
       80 帧（可配）。RGB 存为 rgb/00000.png...00079.png（8-bit），深度存为
       depth/00000.png...00079.png（16-bit，毫米）。同时从相机 SDK 读取内参，
       写入 cam_K.txt（3x3 矩阵）和 camera_params.json（fx/fy/ppx/ppy/depth_scale）。
       采集过程有 RGB + 伪彩色深度可视化预览，按 Q 提前结束。

    3. 交互式选点--pick — 打开首帧 RGB，WASD/方向键移动十字光标，Space 标记前景点（绿色），
       B 标记背景点（红色），Enter 确认。坐标和标签写入 dataset_info.json，
       作为 SAM2 分割的 prompt。

    4. 写入元数据 — 汇总时间戳、分辨率、帧数、相机内参、选点坐标、real_size.longest_edge
       （--len 参数，米，用于scale_stage）到 dataset_info.json。

  产出目录：
    cam_data/mouse008/
    ├── rgb/00000.png ... 00079.png
    ├── depth/00000.png ... 00079.png
    ├── views/{front,left,back,right}.jpg
    ├── camera_params.json
    ├── cam_K.txt
    └── dataset_info.json


上传阶段（本机 code/obe/upload.sh）

  入口命令：
    ./upload.sh cam_data/mouse008/                # 交互式，每步 y/n
    ./upload.sh cam_data/mouse008/ -y             # 一键模式，跳过所有询问
    ./upload.sh cam_data/mouse008/ --from scale   # 从指定阶段开始

  流程：
    1. SSH 免密配置 — 首次使用 ./upload.sh --setup，自动生成 ed25519 密钥对，
       ssh-copy-id 到服务器，验证免密登录。

    2. rsync 上传 — 将本地 cam_data/<task>/ 增量同步到服务器
       /home/vipuser/pipeline-tool/tasks/<task>/，显示文件数和总大小。

    3. 远程 setup — 在服务器上执行 pipeline setup --task <task>，生成或更新
       tasks/<task>/task.yaml，并从 dataset_info.json 填入 sam2 选点和 real_size。

    4. 依次触发 pipeline 阶段 — 通过 SSH 远程执行 run.sh stage，按顺序跑
       ① masks → ② hunyuangen → ③ scale → ④ package → ⑤ foundationpose，
       每步交互确认（-y 跳过），失败可继续或中止。

    5. 查看状态 — 最后执行 pipeline status，展示各阶段完成情况。

  关键参数：
    --from <stage>    从指定阶段开始（upload | masks | hunyuangen | scale | package | foundationpose）
    --force           全部步骤强制重跑
    -y                跳过所有交互确认
    --debug           显示完整远程命令和实时输出
    --log <file>      将所有输出写入本地日志文件


pipeline-tool流程(服务器端)
  1. masks — 取 RGB-D 的第一帧，docker cp 进 SAM2 容器，用采集时记录的点击坐标（前景/背景点）做分割推理，把生成的蒙版docker cp 出来。优先读取 dataset_info.json 里的点坐标，config 里的作为 fallback。
  2.  hunyuangen — 将 front/left/right/back 四张多视角照片 base64 编码，通过腾讯云 AI3D API 提交混元 3D 生成任务，每 10秒轮询直到完成（超时 20 分钟），下载 OBJ，解压，输出 raw.obj。
  3. scale — 解析 OBJ 的所有顶点，算出包围盒最长边，与 config 中real_size.longest_edge（真实物理尺寸，米）做等比缩放，重新写回顶点坐标，输出 scaled.obj。
  4. package — 将前面所有产物组装成 FoundationPose 标准数据集布局：RGB/深度帧重命名为 00000.png 序列，蒙版放入masks/，缩放后的 OBJ 放入 mesh/scaled.obj，保留 cam_K.txt 和 camera_params.json。
  5. foundationpose — 从 manifest 读取 package 输出路径，检查 FoundationPose 容器是否运行，未运行则自动启动，docker exec 执行 run_demo.py 传入 mesh 和场景目录，验证输出 ob_in_cam/*.txt 数量与 RGB 帧数一致，同时包含目标位姿的可视化输出。
