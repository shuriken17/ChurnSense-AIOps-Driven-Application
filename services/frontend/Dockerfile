# Frontend image: nginx with the static web interface baked in. It is a separate service from the
# backend; the two are joined only by the API address (API_BASE) and nginx's /api/ proxy.
FROM nginx:1.27-alpine
COPY frontend /usr/share/nginx/html
COPY services/nginx/nginx.conf /etc/nginx/nginx.conf
COPY services/frontend/40-config.sh /docker-entrypoint.d/40-config.sh
RUN chmod +x /docker-entrypoint.d/40-config.sh
