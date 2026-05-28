# Pipeline 调试进展

> 2026-05-26

## 整体状态

| Stage | 状态 | 位置 |
|---|---|---|
| RGBD 采集 | 代码就绪 | 本机 `code/obe/rgbd_resize copy.py` |
| sam2mask | 代码就绪，待服务器实测 | 本机 pipeline，SAM2 Docker 在 10.10.16.58 |
| hunyuangen | 调试完成 | 本机，调腾讯云 API |
| scale | 调试完成 | 本机 |
| package | 代码就绪，待 RGBD+mask 凑齐后实测 | 本机 |

## 本次对话完成的改动

### 1. RGBD 采集脚本适配 (`code/obe/rgbd_resize copy.py`)
- camera_params.json 从嵌套 `{"color": {...}}` 改为平铺格式，对齐 bottle 参考
- 图像命名从时间戳改为 `00000.png` 序号格式（对齐 package 阶段预期）

### 2. package.py 修复 (`pipeline/stages/package.py`)
- `_generate_cam_k_txt` 矩阵布局修复：ppx/ppy 从第三行移到正确位置（第一行/第二行第三列）

### 3. sam2mask.py 重写 (`pipeline/stages/sam2mask.py`)
- 从 `docker run --rm` 改为 `docker cp → docker exec → docker cp` 容器交互模式
- 输入从 bbox 改为 points/labels
- 调用容器内 `/opt/sam2_cli.py`

### 4. Sam2Config 重构 (`pipeline/config.py`)
- 字段从 `image + bbox` 改为 `container + checkpoint + config_file + points + labels`
- 必填校验同步更新

### 5. 配置解耦 (`pipeline/config.py`)
- `_resolve_env_vars` 不再对缺失 env var 抛异常，保留 `${VAR}` 原样
- `hunyuan` / `real_size` section 改为可选，缺失时用空默认值
- 跑 sam2mask 不需要设置 TENCENT 环境变量

### 6. SSH MCP 配置
- `~/.claude/ssh-config.json` — 3 台主机：js4.blockelite.cn / 10.10.16.58 / 223.109.239.36
- `~/.claude.json` — 已合并 mcpServers 条目，命令白名单已配置

## 服务器信息

- **10.10.16.58** (用户 try:22) — SAM2 Docker 容器 `sam2-backend-1`，FoundationPose Docker
- **js4.blockelite.cn** (root:24612)
- **223.109.239.36** (root:24612)

## 关键路径

| 项目 | 路径 |
|---|---|
| Pipeline 根目录 | `/home/ubt2204/code/pipeline-tool/` |
| RGBD 采集脚本 | `/home/ubt2204/code/obe/rgbd_resize copy.py` |
| bottle 参考数据集 | `/home/ubt2204/Downloads/data/bottle/` |
| SAM2 容器内 CLI | `/opt/sam2_cli.py` |
| SAM2 checkpoint | `/opt/sam2/checkpoints/sam2.1_hiera_base_plus.pt` |
| Python 环境 | `conda activate obe` |

## 下一步

1. 服务器上确认 SAM2 容器名和 CLI 接口 → 跑 `pipeline stage sam2mask`
2. 本机采集 RGBD → `python rgbd_resize_copy.py --output ./data/mouse_001/rgbd/`
3. 三条输入凑齐后跑 `pipeline stage package`
