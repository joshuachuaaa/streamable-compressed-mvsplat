#!/usr/bin/env bash
set -euo pipefail

python experiments/plot_fair_rd.py --all-metrics "$@"
