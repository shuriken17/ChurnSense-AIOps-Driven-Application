# backend image: the shared base image plus the application code.
ARG BASE_IMAGE=churnsense/base:dev
FROM ${BASE_IMAGE}
COPY src /app/src
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "churnsense.api.backend:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
