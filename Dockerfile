# ---------------------------------------------------------
# hxq-clink-agent Dockerfile
# 天润融通通话语音流实时推送 FastAPI Server (ASR-LLM-TTS)
# ---------------------------------------------------------

FROM python:3.13-slim

# 防止 Python 输出缓冲导致日志延迟
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

WORKDIR /app

# 将 apt 源替换为阿里云镜像，加速包下载
RUN sed -i 's#http://deb.debian.org/debian#http://mirrors.aliyun.com/debian#g; \
         s#http://security.debian.org/debian-security#http://mirrors.aliyun.com/debian-security#g' /etc/apt/sources.list.d/debian.sources

# 安装 uv 包管理工具（从 gitee 镜像安装）
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# 将 uv 加入 PATH
ENV PATH="/root/.local/bin:$PATH"

# 复制项目元数据和锁文件（利用 Docker 缓存层加速重建）
COPY pyproject.toml uv.lock README.md ./

# 复制源码
COPY src/ ./src/

# 使用 uv 安装依赖（--no-dev 跳过开发依赖，--frozen 使用锁文件）
RUN uv sync --no-dev --frozen

# -- 运行阶段 --------------------------------------------------
# 服务端口（通过 ENV 传入，与 .env 中 HXQ_PORT 保持一致，默认 8000）
ENV HXQ_PORT=8000
EXPOSE ${HXQ_PORT}

# 健康检查：通过 /health 端点（端口从 HXQ_PORT 环境变量读取）
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import http.client,os,sys; c=http.client.HTTPConnection('127.0.0.1',int(os.environ.get('HXQ_PORT','8000')),timeout=3); c.request('GET','/health'); r=c.getresponse(); sys.exit(0 if r.status==200 else 1)"

# 使用 uv run 启动项目
CMD ["uv", "run", "--no-dev", "python", "-m", "hxq_clink_agent"]
