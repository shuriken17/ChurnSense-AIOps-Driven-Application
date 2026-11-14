# Drift monitor image: the shared base image plus the application code.
# params.yaml and the reference profile are mounted by compose, so changing a threshold needs no rebuild.
ARG BASE_IMAGE=churnsense/base:dev
FROM ${BASE_IMAGE}
COPY src /app/src
ENV PYTHONPATH=/app/src
EXPOSE 8002
CMD ["python", "-m", "uvicorn", "churnsense.drift.monitor:create_app", "--factory", "--host", "0.0.0.0", "--port", "8002", "--workers", "1", "--no-access-log"]
