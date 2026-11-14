#!/usr/bin/env bash
# ChurnSense Batch 0: prepare the VM. Safe to run again (every step checks first).
#
#   ./batch0_setup.sh --dry-run    show what would change, change nothing
#   ./batch0_setup.sh              apply the changes
#
# What it does:
#   1. installs small helper packages if missing (jq, curl, gnupg, git, make)
#   2. adds log rotation to /etc/docker/daemon.json and KEEPS the settings already there
#   3. sets vm.swappiness=10 (persisted)
#   4. creates /srv/churnsense/... and /opt/churnsense/repo, owned by you
#   5. prints a report
# It never removes images, volumes or files.
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ $# -gt 0 ]]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

# Paths can be overridden for testing; the defaults are the real locations.
DAEMON_JSON="${DAEMON_JSON:-/etc/docker/daemon.json}"
SYSCTL_FILE="${SYSCTL_FILE:-/etc/sysctl.d/99-churnsense.conf}"
SRV_ROOT="${SRV_ROOT:-/srv/churnsense}"
REPO_DIR="${REPO_DIR:-/opt/churnsense/repo}"
MIN_FREE_GB="${MIN_FREE_GB:-15}"
OWNER="$(id -un)"
GROUP="$(id -gn)"

# The log rotation we want Docker to use (10 MB x 3 files per container).
DESIRED_JSON='{"log-driver":"json-file","log-opts":{"max-size":"10m","max-file":"3"}}'
# Packages the later batches need on the host.
WANTED_PKGS=(jq curl ca-certificates gnupg git make)
# Folders under /srv/churnsense (state, data and secrets live outside the repo).
SRV_DIRS=(dvc-remote mlflow/db mlflow/artifacts serving inference-logs state/runs secrets)
# Host ports ChurnSense will publish or use.
PORTS=(3000 5000 8000 8080 9090 9095)

log()  { printf '\n== %s\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '   WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# run: print the command, and run it unless this is a dry run.
run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '   [dry-run] %s\n' "$*"
    else
        printf '   + %s\n' "$*"
        "$@"
    fi
}

# --------------------------------------------------------------------------- preflight
preflight() {
    log "Preflight checks"
    if [[ "$EUID" -eq 0 ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            warn "running as root (allowed only for --dry-run)"
        else
            die "run this as your normal user, not root; it uses sudo where needed"
        fi
    fi

    local free_gb
    free_gb="$(df --output=avail -BG / | tail -n 1 | tr -dc '0-9')"
    info "free disk on /: ${free_gb} GB (need at least ${MIN_FREE_GB} GB)"
    if [[ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]]; then
        die "not enough free disk space; free some and run again"
    fi

    if command -v docker >/dev/null 2>&1; then
        info "docker: $(docker --version)"
        local running
        running="$(docker ps -q 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$running" -gt 0 && "${FORCE:-0}" != "1" ]]; then
            die "${running} container(s) are running; restarting Docker would stop them (set FORCE=1 to continue)"
        fi
    elif [[ "$DRY_RUN" -eq 1 ]]; then
        warn "docker is not installed here (fine for a dry run)"
    else
        die "docker is not installed"
    fi

    if [[ "$DRY_RUN" -eq 0 ]]; then
        info "sudo will ask for your password once"
        sudo -v
    fi
}

# --------------------------------------------------------------------------- 1. packages
install_packages() {
    log "1. Helper packages"
    local missing=() pkg
    for pkg in "${WANTED_PKGS[@]}"; do
        if dpkg -s "$pkg" >/dev/null 2>&1; then
            info "$pkg: present"
        else
            missing+=("$pkg")
        fi
    done
    if [[ "${#missing[@]}" -eq 0 ]]; then
        info "nothing to install"
        return
    fi
    info "missing: ${missing[*]}"
    run sudo apt-get update -qq
    run sudo apt-get install -y -qq "${missing[@]}"
}

# --------------------------------------------------------------------------- 2. docker
configure_docker_logging() {
    log "2. Docker log rotation (merge, keep existing settings)"
    command -v jq >/dev/null 2>&1 || { warn "jq is not installed yet; skipping (run again after step 1)"; return; }

    local current normalized_current merged
    if [[ -f "$DAEMON_JSON" ]]; then
        current="$(cat "$DAEMON_JSON")"
        # Validate first, so a broken file gives one clear message and nothing is touched.
        jq -e . <<<"$current" >/dev/null 2>&1 || die "$DAEMON_JSON is not valid JSON; fix it by hand first"
        info "existing $DAEMON_JSON: $(jq -c . <<<"$current")"
    else
        current='{}'
        info "no existing $DAEMON_JSON"
    fi
    normalized_current="$(jq -S . <<<"$current")"
    # '*' is a recursive merge: our keys are added, every other key stays untouched.
    merged="$(jq -S --argjson d "$DESIRED_JSON" '. * $d' <<<"$current")"

    if [[ "$merged" == "$normalized_current" ]]; then
        info "already configured, nothing to change"
        return
    fi
    info "new content: $(jq -c . <<<"$merged")"

    local backup=""
    if [[ -f "$DAEMON_JSON" ]]; then
        backup="${DAEMON_JSON}.bak.$(date +%Y%m%d%H%M%S)"
        run sudo cp -a "$DAEMON_JSON" "$backup"
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        info "[dry-run] would write the new content and restart Docker"
        return
    fi

    printf '%s\n' "$merged" | sudo tee "$DAEMON_JSON" >/dev/null
    info "restarting Docker (no containers are running)"
    if ! sudo systemctl restart docker; then
        warn "Docker failed to restart; restoring the old configuration"
        [[ -n "$backup" ]] && sudo cp -a "$backup" "$DAEMON_JSON"
        sudo systemctl restart docker || true
        die "rolled back; check: sudo journalctl -u docker -n 30"
    fi

    for _ in $(seq 1 30); do
        docker info >/dev/null 2>&1 && break
        sleep 1
    done
    local driver
    driver="$(docker info --format '{{.LoggingDriver}}' 2>/dev/null || true)"
    info "logging driver now: ${driver:-unknown}"
    [[ "$driver" == "json-file" ]] || die "Docker did not come back with json-file logging"

    # Quick network test: proves the mtu/dns settings still work after the restart.
    if docker run --rm hello-world >/dev/null 2>&1; then
        info "network test passed (pulled and ran hello-world)"
        docker rmi hello-world >/dev/null 2>&1 || true
    else
        warn "hello-world could not be pulled or run; check the network and ${DAEMON_JSON}"
    fi
}

# --------------------------------------------------------------------------- 3. swappiness
configure_swappiness() {
    log "3. vm.swappiness"
    local want="vm.swappiness=10"
    if [[ -f "$SYSCTL_FILE" ]] && grep -qx "$want" "$SYSCTL_FILE"; then
        info "$SYSCTL_FILE already has $want"
    else
        if [[ "$DRY_RUN" -eq 1 ]]; then
            info "[dry-run] would write '$want' to $SYSCTL_FILE"
        else
            printf '%s\n' "$want" | sudo tee "$SYSCTL_FILE" >/dev/null
            info "wrote $SYSCTL_FILE"
        fi
    fi
    if [[ "$DRY_RUN" -eq 0 ]]; then
        sudo sysctl -q -p "$SYSCTL_FILE"
    fi
    info "current value: $(cat /proc/sys/vm/swappiness 2>/dev/null || echo unknown)"
}

# --------------------------------------------------------------------------- 4. folders
create_folders() {
    log "4. Folders"
    local d
    for d in "${SRV_DIRS[@]}"; do
        run sudo mkdir -p "${SRV_ROOT}/${d}"
    done
    run sudo mkdir -p "$REPO_DIR"
    run sudo chown -R "${OWNER}:${GROUP}" "$SRV_ROOT" "$(dirname "$REPO_DIR")"
    run sudo chmod 700 "${SRV_ROOT}/secrets"
}

# --------------------------------------------------------------------------- 5. report
report() {
    log "5. Report"
    info "free disk on /:    $(df -h --output=avail / | tail -n 1 | tr -d ' ')"
    info "memory available:  $(free -h | awk '/^Mem:/ {print $7}') of $(free -h | awk '/^Mem:/ {print $2}')"
    info "swappiness:        $(cat /proc/sys/vm/swappiness 2>/dev/null || echo unknown)"
    if command -v docker >/dev/null 2>&1; then
        info "docker logging:    $(docker info --format '{{.LoggingDriver}}' 2>/dev/null || echo unknown)"
    fi
    info "clock in sync:     $(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown)"
    local busy=0 p
    for p in "${PORTS[@]}"; do
        if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${p}$"; then
            warn "port ${p} is already in use"
            busy=1
        fi
    done
    [[ "$busy" -eq 0 ]] && info "ports ${PORTS[*]}: all free"
    if [[ -d "$SRV_ROOT" ]]; then
        info "folders under ${SRV_ROOT}:"
        (cd "$SRV_ROOT" && find . -mindepth 1 -maxdepth 2 -type d | sort | sed 's/^/      /')
        info "owner of ${SRV_ROOT}: $(stat -c '%U:%G' "$SRV_ROOT")"
    fi
    [[ -d "$REPO_DIR" ]] && info "repo folder:       ${REPO_DIR} ($(stat -c '%U:%G' "$REPO_DIR"))"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '\nDry run finished. Nothing was changed.\n'
    else
        printf '\nBatch 0 finished.\n'
    fi
}

main() {
    preflight
    install_packages
    configure_docker_logging
    configure_swappiness
    create_folders
    report
}

main
