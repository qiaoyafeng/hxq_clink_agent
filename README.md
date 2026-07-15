# hxq-clink-agent

天润融通通话语音流实时推送 WebSocket Server。

接受天润融通系统作为 WebSocket Client 连接，接收通话 PCM 音频流（8kHz / 16bit），经 **ASR → LLM → TTS** 管线处理后，将 TTS 合成音频通过同一连接回传，实现实时双向语音对话。

## 架构概览

```
天润融通 (WS Client)
       │
       │ WebSocket: ws://host:port/realtime_voice?uniqueId=...&authString=...
       │
       ▼
┌─────────────────────────────────────────────────┐
│              hxq-clink-agent (Server)            │
│                                                  │
│  WSServer                                        │
│    └── Session (per connection)                  │
│          ├── AudioBuffer + VAD                   │
│          └── Pipeline                            │
│                ├── ASR  (语音→文本)              │
│                ├── LLM  (文本→回复)              │
│                └── TTS  (回复→PCM)               │
│                                                   │
│  回传: 按 4096B / 250ms 帧率发送 binary PCM      │
└─────────────────────────────────────────────────┘
```

## 通信协议

### 握手阶段

客户端发起 WebSocket 连接，URL 携带查询参数：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uniqueId` | String | 是 | 当前通话通道唯一标识 |
| `mainUniqueId` | String | 是 | 通话唯一标识 |
| `enterpriseId` | Integer | 是 | 企业 ID |
| `cno` | String | 是 | 座席工号 |
| `monitorSide` | Integer | 是 | 录音侧：0=混合 / 1=座席 / 2=客户 |
| `callStartTimestamp` | String | 是 | 通话开始时间戳（毫秒） |
| `customerNumber` | String | 是 | 客户电话号码 |
| `agentNumber` | String | 是 | 座席电话号码 |
| `sampleRate` | Integer | 是 | 采样率（8000） |
| `sampleWidth` | Integer | 是 | 采样宽度（16） |
| `timestamp` | String | 是 | WebSocket 连接建立时间（毫秒） |
| `authString` | String | 是 | 鉴权签名串 |

握手成功返回：
```json
{"event": "started", "sessionId": "<uuid>"}
```

### 实时通信阶段

- **接收音频**：客户端持续推送 binary 消息（PCM 原始数据）
- **回传音频**：服务端将 TTS 结果按 4096 字节 / 250ms 帧率回传 binary 消息
- **结束会话**：客户端发送 `{"action": "end"}` 文本消息

### 鉴权

`authString` 生成规则：

```
baseString = url_encode("appId,accessKeyId,timestamp")
signature  = HMAC-SHA1(secretKey, baseString) → Base64
authString = url_encode("appId,accessKeyId,timestamp,signature")
```

服务端使用本地 `HXQ_ACCESS_KEY_SECRET` 重建签名后与客户端传入值比对。

## 项目结构

```
src/hxq_clink_agent/
├── __init__.py
├── __main__.py           # 入口：启动 WS Server
├── config.py             # pydantic-settings 配置
├── auth.py               # authString 签名验证
├── ws_server.py          # WebSocket Server 主逻辑
├── session.py            # 单通话会话生命周期管理
├── audio_buffer.py       # PCM 缓冲 + 能量 VAD
├── pipeline.py           # ASR → LLM → TTS 管线编排
├── interfaces/           # 抽象接口（ASR / LLM / TTS）
│   ├── asr.py
│   ├── llm.py
│   └── tts.py
└── adapters/             # 具体实现（可替换）
    ├── asr_stub.py       # ASR 占位（开发联调用）
    ├── llm_stub.py       # LLM 占位
    └── tts_stub.py       # TTS 占位

Dockerfile               # 生产镜像构建
docker-compose.yml       # 编排配置
.dockerignore            # Docker 构建排除规则
```

## 环境要求

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/)（推荐的包管理工具）

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

复制示例配置文件并填入实际值：

```bash
cp .env.example .env
```

关键配置项：

```ini
# Server 监听
HXQ_WS_HOST=0.0.0.0
HXQ_WS_PORT=8765
HXQ_WS_PATH=/realtime_voice

# 鉴权（与天润融通控制台一致）
HXQ_APP_ID=<你的 appId>
HXQ_ACCESS_KEY_ID=<你的 accessKeyId>
HXQ_ACCESS_KEY_SECRET=<你的 accessKeySecret>

# PCM 参数
HXQ_PCM_SAMPLE_RATE=8000
HXQ_PCM_SAMPLE_WIDTH=16
HXQ_PCM_FRAME_SIZE=4096
HXQ_PCM_FRAME_INTERVAL=0.25

# VAD
HXQ_VAD_SILENCE_SEC=0.8
HXQ_VAD_ENERGY_THRESHOLD=500
```

### 3. 启动服务

```bash
uv run hxq-clink-agent
# 或
python -m hxq_clink_agent
```

### 4. 运行测试

```bash
uv run pytest tests/ -v
```

## Docker 部署

适用于生产环境或服务器部署，无需本地安装 Python / uv。

### 前置条件

- Docker >= 20.10
- Docker Compose v2（`docker compose` 命令）

### 构建镜像

```bash
docker compose build
```

或单独构建：

```bash
docker build -t hxq-clink-agent:latest .
```

### 启动服务

```bash
# 后台启动
docker compose up -d

# 查看日志
docker compose logs -f hxq-clink-agent
```

### 管理容器

```bash
# 查看运行状态
docker compose ps

# 停止服务
docker compose down

# 重建并重启（更新代码后）
docker compose up -d --build

# 重启服务（不重建镜像）
docker compose restart
```

### 自定义端口

如需修改映射端口，在 `docker-compose.yml` 中调整 `ports` 配置：

```yaml
ports:
  - "9000:8765"   # 宿主机 9000 → 容器 8765
```

### 环境变量覆盖

可通过 docker compose 命令行临时覆盖配置，无需修改 `.env` 文件：

```bash
docker compose run -e HXQ_WS_PORT=9000 -e HXQ_LOG_LEVEL=DEBUG hxq-clink-agent
```

### 健康检查

镜像内置了 TCP 端口健康检查（每 30 秒一次）。可通过以下命令查看健康状态：

```bash
docker inspect --format='{{.State.Health.Status}}' hxq-clink-agent
```

## 接入自有 AI 服务

当前 `adapters/` 目录使用 Stub 占位实现。接入真实服务时，实现对应接口即可：

```python
from hxq_clink_agent.interfaces import ASRInterface, LLMInterface, TTSInterface

class MyASR(ASRInterface):
    async def recognize(self, pcm: bytes, sample_rate: int = 8000) -> str:
        # 调用你的 ASR 服务
        ...

class MyLLM(LLMInterface):
    async def chat(self, text: str, history: list[dict[str, str]]) -> str:
        # 调用你的 LLM 服务
        ...

class MyTTS(TTSInterface):
    async def synthesize(self, text: str, sample_rate: int = 8000) -> bytes:
        # 调用你的 TTS 服务，返回 PCM 音频
        ...
```

然后在 `ws_server.py` 的 `_handle_connection` 中将 `ASRStub / LLMStub / TTSStub` 替换为你的实现类。

## 技术栈

| 依赖 | 用途 |
|------|------|
| `websockets >= 13.0` | WebSocket Server |
| `pydantic-settings >= 2.5.0` | 环境配置加载 |
| `loguru >= 0.7.2` | 结构化日志 |
| `python-dotenv >= 1.0.1` | `.env` 文件解析 |
| `pytest` / `pytest-asyncio` | 测试 |
| `hatchling` | 构建后端 |

## License

Private
