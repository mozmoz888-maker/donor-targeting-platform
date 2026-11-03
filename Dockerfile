FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY donor_targeting ./donor_targeting
COPY service ./service
COPY models ./models

# data/ (warehouse snapshot + artifacts) is mounted at runtime; precompute runs on boot
# if artifacts are missing, then the API serves. Memory limit is enforced by the platform
# (docker-compose: mem_limit 2g, cpus 4) matching the Render background worker.
ENV UV_COMPILE_BYTECODE=1
CMD ["sh", "-c", "test -f data/artifacts/meta.json || uv run python -m service.precompute; exec uv run uvicorn service.api:app --host 0.0.0.0 --port 8000"]
