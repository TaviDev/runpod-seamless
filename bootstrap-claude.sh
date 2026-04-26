#!/usr/bin/env bash
# bootstrap-claude.sh — get claude-code running on a fresh pod, then launch it.
# Mirrors install.sh Section 1's apt+node+claude steps but standalone, so
# claude can take over and drive install.sh from there.
#
# Usage on a clean pod (web terminal or SSH):
#   cd /workspace
#   git clone https://github.com/TaviDev/runpod-seamless.git .scripts
#   bash .scripts/bootstrap-claude.sh
#
# Idempotent. Ends by `exec claude` in /workspace if attached to a tty;
# otherwise prints the launch command. The env vars exported here only
# affect this script's subshell — install.sh writes /etc/profile.d/seamless.sh
# later for persistence across SSH sessions.

set -euo pipefail

WORKSPACE=/workspace
SCRIPTS_DIR=$WORKSPACE/.scripts
NPM_PREFIX=$WORKSPACE/npm-global
CLAUDE_CONFIG=$WORKSPACE/.claude
REPO_URL=https://github.com/TaviDev/runpod-seamless.git

log() { echo -e "\n\033[1;36m[bootstrap-claude] $*\033[0m"; }

# Clone runpod-seamless to /workspace/.scripts if missing — supports both
# running this script from inside an existing clone AND running it stand-alone
# (e.g. fetched directly via curl).
if [[ ! -d "$SCRIPTS_DIR/.git" ]]; then
  log "cloning runpod-seamless -> $SCRIPTS_DIR"
  mkdir -p "$WORKSPACE"
  git clone "$REPO_URL" "$SCRIPTS_DIR"
fi

# Node 20 (mirrors install.sh Section 1; install.sh will re-check, no-op).
if ! command -v node >/dev/null || [[ "$(node --version)" != v20.* ]]; then
  log "installing node 20"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends curl ca-certificates
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi

# Volume-backed npm prefix + claude-code CLI.
export NPM_CONFIG_PREFIX=$NPM_PREFIX
export PATH=$NPM_CONFIG_PREFIX/bin:$PATH
export CLAUDE_CONFIG_DIR=$CLAUDE_CONFIG
mkdir -p "$NPM_PREFIX" "$CLAUDE_CONFIG"

if ! command -v claude >/dev/null; then
  log "installing @anthropic-ai/claude-code"
  npm install -g @anthropic-ai/claude-code
fi

cd "$WORKSPACE"
log "ready. claude config: $CLAUDE_CONFIG_DIR  npm prefix: $NPM_PREFIX"
log "next: in claude, ask it to run  bash $SCRIPTS_DIR/install.sh  then  $WORKSPACE/start.sh test-all"

if [[ -t 0 && -t 1 ]]; then
  exec claude
else
  log "no tty attached — open a terminal on this pod and run:  claude"
fi
