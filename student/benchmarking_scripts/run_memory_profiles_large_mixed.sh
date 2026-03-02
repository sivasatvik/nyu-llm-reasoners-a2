#!/bin/bash
# Run BF16 mixed-precision memory profiling for large model.

set -euo pipefail

MIXED=1 bash student/benchmarking_scripts/run_memory_profiles_large.sh
