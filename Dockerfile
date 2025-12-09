FROM python:3.13.5-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 工作目录
WORKDIR /app

# 编译器
RUN apt-get update && apt-get install -y build-essential
# 复制依赖列表和锁文件
COPY pyproject.toml uv.lock ./

# 安装依赖（使用 --frozen 确保使用锁文件中的版本）
RUN uv sync --frozen --no-dev

# 复制项目文件
COPY . .

EXPOSE 8000

ENTRYPOINT [ "uv", "run", "bot.py" ]