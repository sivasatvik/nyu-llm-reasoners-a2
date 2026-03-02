#!/bin/bash
# Run BF16 mixed-precision memory profiling for xl model.

set -euo pipefail

MIXED=1 bash student/benchmarking_scripts/run_memory_profiles_xl.sh
