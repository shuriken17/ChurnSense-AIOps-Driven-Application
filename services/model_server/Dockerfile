# model server image: the shared base image plus the application code.
ARG BASE_IMAGE=churnsense/base:dev
FROM ${BASE_IMAGE}
COPY src /app/src
ENV PYTHONPATH=/app/src
EXPOSE 8001
CMD ["python", "-m", "uvicorn", "churnsense.serving.model_server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--no-access-log"]
