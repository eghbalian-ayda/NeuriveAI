#!/usr/bin/env bash
# NeuriveAI deployment script
# Usage: ./web/deploy.sh <command>

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_DIR="$SCRIPT_DIR"

MODEL_SERVER_SCRIPT="$WEB_DIR/model_server/main.py"
MODEL_SERVER_LOG="$WEB_DIR/model_server.log"
MODEL_SERVER_PID="$WEB_DIR/model_server.pid"

COMPOSE_FILE="$WEB_DIR/compose.yaml"

MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-8001}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-2}"

# Python in the neuriv conda env
PYTHON="/root/miniconda3/envs/neuriv/bin/python"

# ── Colours ────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
else
  RED=''; YELLOW=''; GREEN=''; CYAN=''; BOLD=''; RESET=''
fi

info()    { echo -e "${CYAN}▶ $*${RESET}"; }
ok()      { echo -e "${GREEN}✔ $*${RESET}"; }
warn()    { echo -e "${YELLOW}⚠ $*${RESET}"; }
err()     { echo -e "${RED}✖ $*${RESET}" >&2; }
header()  { echo -e "\n${BOLD}$*${RESET}"; }

# ── Helpers ────────────────────────────────────────────────────────────────
model_pid() {
  if [[ -f "$MODEL_SERVER_PID" ]]; then
    local pid
    pid=$(cat "$MODEL_SERVER_PID")
    if kill -0 "$pid" 2>/dev/null; then
      echo "$pid"
    fi
  fi
}

model_is_running() {
  [[ -n "$(model_pid)" ]]
}

compose_is_running() {
  podman-compose -f "$COMPOSE_FILE" ps --quiet 2>/dev/null | grep -q .
}

wait_for_http() {
  local url="$1" label="$2" timeout="${3:-30}"
  local i=0
  echo -n "  Waiting for $label"
  while ! curl -sf "$url" > /dev/null 2>&1; do
    sleep 1
    i=$((i + 1))
    echo -n "."
    if [[ $i -ge $timeout ]]; then
      echo ""
      return 1
    fi
  done
  echo ""
  return 0
}

# ── Model server operations ────────────────────────────────────────────────
model_start() {
  if model_is_running; then
    warn "Model server already running (pid $(model_pid))"
    return 0
  fi

  if [[ ! -f "$PYTHON" ]]; then
    err "Python not found at $PYTHON — check PYTHON variable in deploy.sh"
    exit 1
  fi

  info "Starting model server on :$MODEL_SERVER_PORT (GPU $CUDA_DEVICE)…"
  mkdir -p "$WEB_DIR/uploads"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  MODEL_SERVER_PORT="$MODEL_SERVER_PORT" \
  PIPELINE_DIR="$REPO_ROOT" \
  UPLOADS_DIR="$WEB_DIR/uploads" \
    nohup "$PYTHON" "$MODEL_SERVER_SCRIPT" \
      > "$MODEL_SERVER_LOG" 2>&1 &

  echo $! > "$MODEL_SERVER_PID"

  if wait_for_http "http://localhost:$MODEL_SERVER_PORT/health" "model server" 30; then
    ok "Model server running (pid $!, log: $MODEL_SERVER_LOG)"
  else
    err "Model server failed to start — check $MODEL_SERVER_LOG"
    cat "$MODEL_SERVER_LOG" | tail -20
    exit 1
  fi
}

model_stop() {
  local pid
  pid=$(model_pid)
  if [[ -z "$pid" ]]; then
    warn "Model server is not running"
    return 0
  fi
  info "Stopping model server (pid $pid)…"
  kill "$pid"
  # Wait up to 10s for graceful shutdown
  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1
    i=$((i + 1))
    if [[ $i -ge 10 ]]; then
      warn "Forcing kill…"
      kill -9 "$pid" 2>/dev/null || true
      break
    fi
  done
  rm -f "$MODEL_SERVER_PID"
  ok "Model server stopped"
}

model_logs() {
  if [[ ! -f "$MODEL_SERVER_LOG" ]]; then
    warn "No model server log at $MODEL_SERVER_LOG"
    return 0
  fi
  tail -f "$MODEL_SERVER_LOG"
}

# ── Docker backend operations ──────────────────────────────────────────────
backend_build() {
  info "Building Docker image…"
  podman-compose -f "$COMPOSE_FILE" build
  ok "Build complete"
}

backend_start() {
  info "Starting web backend (compose up)…"
  podman-compose -f "$COMPOSE_FILE" up -d

  if wait_for_http "http://localhost:$BACKEND_PORT/" "web backend" 30; then
    ok "Web backend running at http://localhost:$BACKEND_PORT"
  else
    err "Web backend failed to start — run: $0 logs web"
    exit 1
  fi
}

backend_stop() {
  info "Stopping web backend…"
  podman-compose -f "$COMPOSE_FILE" down
  ok "Web backend stopped"
}

backend_logs() {
  podman-compose -f "$COMPOSE_FILE" logs -f
}

# ── Commands ───────────────────────────────────────────────────────────────
cmd_start() {
  header "Starting NeuriveAI"
  model_start
  backend_start
  echo ""
  ok "NeuriveAI running → http://localhost:$BACKEND_PORT"
}

cmd_stop() {
  header "Stopping NeuriveAI"
  backend_stop
  model_stop
}

cmd_restart() {
  header "Restarting NeuriveAI"
  backend_stop  || true
  model_stop    || true
  model_start
  backend_start
  echo ""
  ok "NeuriveAI running → http://localhost:$BACKEND_PORT"
}

cmd_build() {
  header "Building Docker image"
  backend_build
}

cmd_deploy() {
  header "Deploying NeuriveAI (build + restart)"
  backend_build
  backend_stop  || true
  model_stop    || true
  model_start
  backend_start
  echo ""
  ok "Deploy complete → http://localhost:$BACKEND_PORT"
}

cmd_status() {
  header "NeuriveAI Status"

  # Model server
  echo -e "\n${BOLD}Model Server${RESET} (:$MODEL_SERVER_PORT)"
  if model_is_running; then
    local pid; pid=$(model_pid)
    echo -e "  ${GREEN}● running${RESET}  pid=$pid"
    local health; health=$(curl -sf "http://localhost:$MODEL_SERVER_PORT/health" 2>/dev/null || echo "unreachable")
    echo "  health: $health"
  else
    echo -e "  ${RED}○ stopped${RESET}"
    [[ -f "$MODEL_SERVER_LOG" ]] && echo "  last log: $(tail -1 "$MODEL_SERVER_LOG" 2>/dev/null)"
  fi

  # Web backend
  echo -e "\n${BOLD}Web Backend${RESET} (:$BACKEND_PORT)"
  local container_status
  container_status=$(podman-compose -f "$COMPOSE_FILE" ps 2>/dev/null || echo "")
  if echo "$container_status" | grep -q "Up\|running"; then
    echo -e "  ${GREEN}● running${RESET}"
    local http_status; http_status=$(curl -so /dev/null -w "%{http_code}" "http://localhost:$BACKEND_PORT/" 2>/dev/null || echo "unreachable")
    echo "  http: $http_status"
  else
    echo -e "  ${RED}○ stopped${RESET}"
  fi
  echo ""
}

cmd_health() {
  header "Health checks"
  echo ""

  echo -n "  Model server  http://localhost:$MODEL_SERVER_PORT/health  "
  local ms_health
  ms_health=$(curl -sf "http://localhost:$MODEL_SERVER_PORT/health" 2>/dev/null)
  if [[ -n "$ms_health" ]]; then
    echo -e "${GREEN}✔${RESET} $ms_health"
  else
    echo -e "${RED}✖ unreachable${RESET}"
  fi

  echo -n "  Web backend   http://localhost:$BACKEND_PORT/           "
  local be_status
  be_status=$(curl -so /dev/null -w "%{http_code}" "http://localhost:$BACKEND_PORT/" 2>/dev/null || echo "000")
  if [[ "$be_status" == "200" ]]; then
    echo -e "${GREEN}✔ HTTP $be_status${RESET}"
  else
    echo -e "${RED}✖ HTTP $be_status${RESET}"
  fi

  echo ""
  echo -n "  Demo endpoint http://localhost:$BACKEND_PORT/api/demo    "
  local demo
  demo=$(curl -sf "http://localhost:$BACKEND_PORT/api/demo" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"events={len(d['report']['events'])} profiles={len(d['report']['profiles'])}\")" 2>/dev/null || echo "")
  if [[ -n "$demo" ]]; then
    echo -e "${GREEN}✔ $demo${RESET}"
  else
    echo -e "${RED}✖ unreachable or error${RESET}"
  fi
  echo ""
}

cmd_logs() {
  local target="${1:-all}"
  case "$target" in
    model) model_logs ;;
    web)   backend_logs ;;
    all)
      header "Logs (Ctrl-C to exit)"
      # Tail model server log and compose logs side by side isn't great in bash,
      # so show model first then follow compose
      echo -e "${BOLD}── model server (last 30 lines) ──${RESET}"
      tail -30 "$MODEL_SERVER_LOG" 2>/dev/null || warn "No model server log"
      echo -e "\n${BOLD}── web backend (following) ──${RESET}"
      backend_logs
      ;;
    *) err "Unknown log target '$target'. Use: model | web | all" ; exit 1 ;;
  esac
}

cmd_model_start()  { model_start; }
cmd_model_stop()   { model_stop; }
cmd_model_restart(){ model_stop || true; model_start; }

# ── Usage ──────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF

${BOLD}NeuriveAI Deploy Script${RESET}

Usage: $(basename "$0") <command> [options]

${BOLD}Lifecycle:${RESET}
  start          Start model server + web backend
  stop           Stop both
  restart        Restart both
  deploy         Build image + full restart  (use after code changes)
  build          Rebuild Docker image only

${BOLD}Individual services:${RESET}
  model-start    Start model server only
  model-stop     Stop model server only
  model-restart  Restart model server only

${BOLD}Observability:${RESET}
  status         Show running state of both services
  health         Hit /health endpoints and verify demo data
  logs [model|web|all]   Tail logs (default: all)

${BOLD}Config (env overrides):${RESET}
  CUDA_VISIBLE_DEVICES   GPU to use          (default: 2)
  MODEL_SERVER_PORT      Model server port   (default: 8001)
  BACKEND_PORT           Web backend port    (default: 8000)

${BOLD}Examples:${RESET}
  ./web/deploy.sh deploy             # full build + start
  ./web/deploy.sh restart            # restart without rebuilding
  ./web/deploy.sh status             # check what's running
  ./web/deploy.sh health             # verify all endpoints
  ./web/deploy.sh logs model         # tail model server log
  CUDA_VISIBLE_DEVICES=1 ./web/deploy.sh start  # use different GPU

EOF
}

# ── Entry point ────────────────────────────────────────────────────────────
cd "$REPO_ROOT"

case "${1:-}" in
  start)         cmd_start ;;
  stop)          cmd_stop ;;
  restart)       cmd_restart ;;
  build)         cmd_build ;;
  deploy)        cmd_deploy ;;
  status)        cmd_status ;;
  health)        cmd_health ;;
  logs)          cmd_logs "${2:-all}" ;;
  model-start)   cmd_model_start ;;
  model-stop)    cmd_model_stop ;;
  model-restart) cmd_model_restart ;;
  ""|--help|-h)  usage ;;
  *)             err "Unknown command: ${1}"; usage; exit 1 ;;
esac
