# Trainer image: the base image plus a tiny entry point.
# The repository is mounted at /workspace at run time (see compose.yaml), so code changes
# need no rebuild. The container runs as the host user, so files it writes are yours.
ARG BASE_IMAGE=churnsense/base:dev
FROM ${BASE_IMAGE}

COPY services/trainer/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]
CMD ["bash"]
