FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[export]"
COPY configs ./configs
COPY data/docs ./data/docs
ENV MCAA_DB=/app/data/runtime/events.db MCAA_SCENARIO=technician
EXPOSE 8000
CMD ["mcaa-api"]
