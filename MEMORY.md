# Pipeline 调试进展

> 2026-05-29

## 整体状态

| Stage | 状态 | 位置 |
|---|---|---|
| RGBD 采集 | 代码就绪 | 本机 `code/obe/rgbd_resize_copy.py` |
| 视图采集 | 代码就绪 | 本机 `code/obe/take_views.py` |
| masks (sam2mask) | 调试完成 | 本机 pipeline，SAM2 Docker 在 10.10.16.58 |
| hunyuangen | 调试完成 | 本机，调腾讯云 API |
| scale | 调试完成 | 本机 |
| package | 调试完成 | 本机 + 服务器实测通过 |
| **foundationpose** | **已实现，待实测** | 本机 + 服务器已部署 |

## 本机 vs 服务器差异（已对齐）

本机 `/home/ubt2204/code/pipeline-tool/`，服务器 `/home/vipuser/pipeline-tool/`。

已对齐的差异：
- `package.py` manifest key：本机 `"sam2mask"` → 修正为 `"masks"`（对齐远程）
- `hunyuangen.py` 超时：600 → 1200s，face_count：500000 → 50000
- config `task: mouse_001` → `mouse002`

本机独有文件（未上传服务器）：`sam2_client.py`、`README.md`

## 本次对话完成（2026-05-29）

### 1. FoundationPose stage 实现

**新建 `pipeline/stages/foundationpose.py`**：
- 从 manifest 读取 package 输出路径
- 自动管理容器生命周期：`docker ps` 检查 → 未运行则执行 `{workdir}/docker/run_container.sh` → 等 10s → 二次确认
- `docker exec` 执行 `run_demo.py`，stdout/stderr 实时透传
- 路径用 `.resolve()` 转绝对路径（容器内 `/home:/home` 挂载一致）
- 验证输出：`ob_in_cam/*.txt` 数量 == `rgb/*.png` 数量

**修改 `pipeline/config.py`**：新增 `FoundationPoseConfig` dataclass（container/workdir/debug，均有默认值），集成到 `PipelineConfig`

**修改 `pipeline/stages/__init__.py`**：注册 foundationpose stage

**修改 `pipeline/presets.yaml`**：foundationpose preset 加入 foundationpose 作为第 5 步

### 2. upload.sh 重构

- 从 4 步变为 7 步：upload / masks / hunyuangen / scale / package / foundationpose / status
- 每个 stage 独立 y/n 交互，不再有 all/single 选择
- 新增 `-y` 一键模式：`./upload.sh data/mouse001 -y` 跳过全部询问
- 新增 `--from <stage>` 从指定步骤开始
- 提取 `run_stage_step` 函数消除重复代码

### 3. cam_K.txt 修复

- `package` stage 改为直接 `shutil.copy2` 源 `tasks/{task}/cam_K.txt`，不再重新生成
- 避免格式转换（`rgbd_resize_copy.py` 用 `:.6f` 或 `:.18e` 到 package 的 `:.6f` 精度丢失风险）
- fallback：如果源文件不存在，才从 `camera_params.json` 生成

### 4. foundationpose.py 路径 bug 修复

- 问题：传给 `docker exec` 的是相对路径，容器内 `cd /home/vipuser/FoundationPose` 后解析不到
- 修复：`mesh_file.resolve()` / `package_path.resolve()` / `output_dir.resolve()` 转绝对路径

## 服务器信息

- **180.127.11.166:21240** (root) — 电信，pipeline 部署服务器
- **10.10.16.58** (try:22) — SAM2 Docker 容器 `sam2-backend-1`，FoundationPose Docker
- **js4.blockelite.cn** (root:24612)
- **223.109.239.36** (root:24612)

## 关键路径

| 项目 | 路径 |
|---|---|
| Pipeline 根目录（本机） | `/home/ubt2204/code/pipeline-tool/` |
| Pipeline 根目录（服务器） | `/home/vipuser/pipeline-tool/` |
| upload.sh（本机） | `/home/ubt2204/code/obe/upload.sh` |
| upload.sh（服务器） | `/home/vipuser/code/obe/upload.sh` |
| RGBD 采集脚本 | `/home/ubt2204/code/obe/rgbd_resize_copy.py` |
| 视图采集脚本 | `/home/ubt2204/code/obe/take_views.py` |
| FoundationPose（服务器容器内） | `/home/vipuser/FoundationPose` |
| SAM2 容器内 CLI | `/opt/sam2_cli.py` |
| SAM2 checkpoint | `/opt/sam2/checkpoints/sam2.1_hiera_base_plus.pt` |
| Python 环境（服务器） | `/home/vipuser/miniconda3/envs/fpp/bin` |

## 服务器目录结构

```
tasks/{task_name}/
├── rgb/                  ← RGB 图像（00000.png ...）
├── depth/                ← 深度图
├── views/                ← 多视角照片（front/left/back/right.jpg）
├── camera_params.json    ← 相机内参
├── cam_K.txt             ← 3x3 内参矩阵（rgbd_resize_copy.py 生成）
└── dataset_info.json     ← sam2_points、real_size

output/{task_name}/
├── manifest.json
├── masks/
├── hunyuangen/
├── scale/
├── package/
│   ├── rgb/
│   ├── depth/
│   ├── masks/
│   ├── mesh/textured_simple.obj
│   ├── cam_K.txt
│   └── camera_params.json
└── foundationpose/
    └── ob_in_cam/        ← 位姿 txt 文件
```

## 常用命令

```bash
# 一键上传+跑全流程
./upload.sh data/mouse001 -y

# 单步重新跑
bash run.sh stage foundationpose --config tasks/mouse001/task.yaml --force

# 查看状态
bash run.sh status --config tasks/mouse001/task.yaml
```
