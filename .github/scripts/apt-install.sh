#!/usr/bin/env bash
# apt-get update and install with hard limits, for the CI jobs that need system packages.
#
#   bash .github/scripts/apt-install.sh libvulkan-dev glslc mesa-vulkan-drivers
#
# On 2026-10-07 and 08 Ubuntu's mirrors stopped answering in the middle of
# `apt-get update`: jobs sat on it until their limit, 45 minutes or the 6-hour
# default. apt's own timeout (Acquire::http::Timeout, 30 s) gave up on one
# mirror and then hung on the next transfer all the same, so each command now
# runs under timeout(1) and is tried three times; HTTP pipelining, a known way
# for a mirror or proxy to stall apt, is off.
set -euo pipefail
printf '%s\n' 'Acquire::http::Timeout "30";' 'Acquire::https::Timeout "30";' \
  'Acquire::Retries "3";' 'Acquire::http::Pipeline-Depth "0";' |
  sudo tee /etc/apt/apt.conf.d/99colibri-ci > /dev/null

limited() {           # limited <seconds> <command...>: three tries, each under a hard limit
  local seconds=$1; shift
  for attempt in 1 2 3; do
    if sudo timeout --kill-after=30 "$seconds" "$@"; then return 0; fi
    echo "::warning::'$*' failed or ran past ${seconds} s (attempt $attempt of 3)"
    sleep 15
  done
  return 1
}

limited 300 apt-get update
limited 900 apt-get install -y --no-install-recommends "$@"
