#!/bin/bash
# Run BF16 mixed-precision memory profiling for 2.7B model.

set -euo pipefail

MIXED=1 bash student/benchmarking_scripts/run_memory_profiles_2_7b.sh
