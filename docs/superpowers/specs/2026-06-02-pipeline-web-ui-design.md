# Pipeline Web UI — Design Spec

2026-06-02 | status: draft

## Overview

为 pipeline-tool 添加 Web UI，部署在服务器 `180.127.11.166`，本机和它机通过浏览器 IP 访问。采集阶段保持本机 CLI，Web UI 覆盖：上传、Pipeline 5 阶段调度与实时日志、任务历史、YAML 配置在线编辑。

## Architecture

```
浏览器 ── HTTP/WS ──→ FastAPI (单进程)
                        ├── Jinja2 模板渲染
                        ├── Scheduler (asyncio.create_task)
                        ├── ConnectionManager (WebSocket)
                        ├── import pipeline.* (直接调用)
                        └── 文件系统 (manifest / logs / configs)
```

- **单体应用**：一个 Python 进程跑全部
- **无外部依赖**：不引入 Redis、Celery、消息队列
- **Pipeline 调用**：直接 import `pipeline.pipeline.PipelineOrchestrator`，通过 `run_in_executor` 跑同步 stage
- **状态落盘**：manifest.json（阶段状态）+ logs/（运行日志）

## Stack

| 层 | 选择 |
|---|---|
| Web 框架 | FastAPI |
| 模板 | Jinja2 |
| 前端交互 | Alpine.js + HTMX |
| 认证 | itsdangerous 签名 cookie |
| 实时推送 | WebSocket |
| 系统监控 | psutil |
| 密码存储 | .env (WEB_PASSWORD) |

## Directory Structure

```
pipeline-tool/
├── pipeline/
│   ├── stages/
│   │   ├── context.py          # NEW — StageContext
│   │   ├── base.py             # MODIFIED — run() 加 context 参数
│   │   ├── masks.py            # MODIFIED — 签名同步
│   │   ├── hunyuangen.py       # MODIFIED — 签名同步
│   │   ├── scale.py            # MODIFIED — 签名同步
│   │   ├── package.py          # MODIFIED — 签名同步
│   │   └── foundationpose.py   # MODIFIED — 签名同步
│   └── ...                      # 其余文件不变
│
├── web/                         # NEW — Web UI 模块
│   ├── __init__.py
│   ├── app.py                   # FastAPI 入口, lifespan, auth middleware
│   ├── auth.py                  # itsdangerous 签名 cookie
│   ├── scheduler.py             # 任务调度器
│   ├── ws_manager.py            # WebSocket 连接管理
│   ├── log_setup.py             # PipelineLogger + WebSocketHandler
│   ├── job_store.py             # Job 元数据读写
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── pages.py             # HTML 页面路由
│   │   ├── tasks.py             # /api/tasks/*
│   │   ├── configs.py           # /api/configs/*
│   │   ├── upload.py            # /api/tasks/upload
│   │   ├── history.py           # /api/history/*
│   │   └── system.py            # /api/system
│   ├── templates/
│   │   ├── base.html
│   │   ├── login.html
│   │   ├── tasks.html
│   │   ├── task_detail.html
│   │   ├── upload.html
│   │   ├── config_editor.html
│   │   └── history.html
│   └── static/
│       └── style.css
│
├── logs/                        # NEW — 运行日志
│   └── <job_id>/
│       ├── metadata.json
│       └── run.log
│
├── configs/                     # UNCHANGED
├── tasks/                       # UNCHANGED
├── output/                      # UNCHANGED
├── run.sh
└── .env                         # WEB_PASSWORD=xxx
```

## Job Model

一个 task 可执行多次，每次运行是一个 Job。

```
tasks/mouse001/
├── ... (采集数据)
output/mouse001/
├── manifest.json
└── masks/ hunyuangen/ scale/ package/ foundationpose/

logs/
└── <job_id>/
    ├── metadata.json     # {job_id, task_name, preset, start_time, status, stages_completed, error_message, ...}
    └── run.log           # 完整日志
```

### JobInfo

```python
@dataclass
class JobInfo:
    job_id: str
    task_name: str
    preset: str
    start_time: float
    status: JobStatus           # pending | running | completed | failed | stopped
    current_stage: str | None
    stages_completed: list[str]
    asyncio_task: asyncio.Task | None
    process: Popen | None       # 子进程引用，用于 stop 时 terminate (Future: processes: set[Popen])
    error_message: str | None
    end_time: float | None
```

## API Design

### Auth

```
POST /api/login          # body: {password} → signed cookie
POST /api/logout
```

### Tasks

```
GET  /api/tasks                   # 扫描 tasks/ + manifest, 返回任务列表
GET  /api/tasks/{task_name}       # 任务详情 (manifest + latest_job_id + running_job_id)
POST /api/tasks/{task_name}/run   # body: {preset} → {job_id}
POST /api/tasks/{task_name}/reset # 仅删除 manifest.json，不删 output/、logs/、tasks/ 下任何文件
```

### Jobs

```
GET  /api/jobs/{job_id}           # 单个 Job 详情 (status, current_stage, stages_completed, ...)
POST /api/jobs/{job_id}/stop      # 停止指定 job
```

Task 详情返回示例：

```json
{
  "task_name": "mouse001",
  "stages": {"masks": "done", "hunyuan": "done", "scale": "running", ...},
  "latest_job_id": "abc123",
  "running_job_id": "abc123"
}
```

前端直接从 task 详情获取当前 job，不再需要猜。

### Upload

```
POST /api/tasks/upload            # multipart, 上传到 tasks/
```

约束：
- 最大上传大小: 500 MB（`server.max_body_size`）
- 允许扩展名: `.png`, `.jpg`, `.jpeg`, `.json`, `.txt`, `.yaml`, `.obj`
- **路径穿越防护**: 对每个文件名做 `os.path.basename()` 清洗，拒绝包含 `..` 的路径
- 上传完成后在 tasks/ 下创建以 folder name 命名的子目录

### Configs

```
GET  /api/configs                 # 列出 configs/ 下所有 YAML
GET  /api/configs/{name}          # 读取内容
PUT  /api/configs/{name}          # 保存, yaml.safe_load 校验, 非法返回 400
```

### History

```
GET    /api/history               # 列出所有 jobs (从 logs/ 读取)
GET    /api/history/{job_id}      # metadata + log 全文
DELETE /api/history/{job_id}      # 删除
```

### System

```
GET /api/system                   # {cpu_percent, memory_percent, disk_percent, gpu: {...} | null}
```

GPU 通过 `nvidia-smi` 获取。若 `nvidia-smi` 不存在（无 GPU 环境），返回 `"gpu": null`，不抛 500。

### WebSocket

```
WS /ws/jobs/{job_id}/logs         # 实时日志流, 一个 job 支持多连接
```

### Pages

```
GET /                    → 重定向到 /tasks
GET /login               → 登录页
GET /tasks               → 任务列表
GET /tasks/{task_name}   → 任务详情 + 调度
GET /upload              → 上传页
GET /configs             → 配置编辑器
GET /history             → 历史日志
```

## Core Components

### StageContext

```python
# pipeline/stages/context.py

@dataclass
class StageContext:
    logger: logging.Logger | None = None
    job_id: str | None = None
    stop_event: threading.Event | None = None
    progress_callback: Callable[[int, int, str], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def log(self, level: int, msg: str, *args, **kwargs) -> None: ...
    def report_progress(self, current: int, total: int, message: str = "") -> None: ...
    def is_stopped(self) -> bool: ...
```

- CLI 模式传 `None`，Web 模式传完整实例
- `progress_callback` 初期不实现，接口预留
- `metadata` 自由扩展：`context.metadata["user"]`, `context.metadata["gpu"]` 等
- 以后加字段不影响已有 stage

### Scheduler

```python
class Scheduler:
    _jobs: dict[str, JobInfo]
    _lock: asyncio.Lock
    _executor: ThreadPoolExecutor   # max_workers=2, 显式管理

    async def submit(task_name, preset, config_path=None) -> str:  # → job_id
    async def stop(job_id) -> bool:
    def get_status(job_id) -> JobInfo | None:
    async def list_running() -> list[JobInfo]:
    async def list_all() -> list[JobInfo]:
```

- `submit()` 内检查同 task 是否已 RUNNING，防重复提交
- `_run_job()` 用 `self._executor`（非默认线程池）跑同步 pipeline
- `stop()`: 先 `process.terminate()` 再 `task.cancel()`

### ConnectionManager

```python
class ConnectionManager:
    _connections: dict[str, set[WebSocket]]
    _lock: asyncio.Lock

    async def connect(job_id, ws): ...
    async def disconnect(job_id, ws): ...
    async def broadcast(job_id, message): ...
```

- `broadcast()` 内处理断线清理
- 一个 job 支持多个浏览器连接

### Logging

```python
class WebSocketLogHandler(logging.Handler):
    """emit() 通过 call_soon_threadsafe 桥接到 asyncio broadcast"""

def create_job_logger(job_id, ws_manager, project_root) -> logging.Logger:
    """三路输出: FileHandler + WebSocketHandler + ConsoleHandler"""
```

- 每个 job 独立 logger (`pipeline.job.<job_id>`)
- `propagate = False` 不向 root 冒泡
- `emit()` 是同步方法，`call_soon_threadsafe` 安全桥接
- finally 块关闭所有 handler，防泄漏

### Auth

- `itsdangerous.TimestampSigner` 签名 cookie
- Middleware: 非 `/login`, `/static` 请求检查 cookie
- 密码从 `.env` 的 `WEB_PASSWORD` 读取

## Concurrency Risks

| 风险 | 缓解 |
|------|------|
| 重复提交同一 task | submit() 持锁检查 RUNNING 状态 |
| stop / run 竞态 | asyncio.Lock 保护状态变更 |
| WebSocket 断线丢消息 | broadcast catch 异常后移除连接 |
| stage 阻塞事件循环 | 显式 ThreadPoolExecutor(max_workers=2) |
| handler 泄漏 | _run_job finally 关闭所有 handler |

## Recovery

服务重启时 Scheduler 内存状态丢失。启动恢复流程：

1. 扫描 `logs/*/metadata.json`
2. 对 `status == "running"` 的 job，统一标记为 `"failed"`，写入 `error_message: "Server restarted"`
3. manifest 不受影响（独立于 Job 生命周期）

保证重启后状态一致：不存在 "running but nobody is executing" 的僵尸状态。

## Visual Design

- **方向**: Precision Terminal — 暗色主题 + 精密仪器感
- **字体**: DM Mono (代码) + DM Sans (UI)
- **配色**: 青绿主色 / 琥珀运行态 / 红色错误态
- **布局**: 左侧固定侧边栏 (200px) + 右侧内容区
- 详见 visual companion mockup

## Scope

### In Scope
- 全局密码登录
- 任务面板 (列表 + 阶段状态灯)
- Pipeline 调度 (单 stage / 全 preset 运行 + 停止 + 重置)
- WebSocket 实时日志终端
- 文件夹上传
- YAML 配置在线编辑器 + 校验
- 历史日志查看
- 系统信息 (CPU/Mem/Disk/GPU)

### Out of Scope (Future)
- 多用户管理
- 自动 bbox 标注 (后续 pipeline stage)
- YOLO 训练集成 (后续)
- Progress 机制 (接口预留，不实现前端)

## Dependencies Added

```
fastapi
uvicorn
jinja2
python-multipart     # upload
itsdangerous         # signed cookie
psutil               # system info
pyyaml               # already exists
aiofiles             # async file reads
python-dotenv        # .env loading
```
