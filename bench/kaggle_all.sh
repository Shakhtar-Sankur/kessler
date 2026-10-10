#!/bin/bash
# Everything kessler measures, on Kaggle. Notebook settings: Accelerator GPU T4, Internet on.
# For the Space-Track parts, add secrets SPACETRACK_USER and SPACETRACK_PASS (Add-ons -> Secrets).
# One cell:
#   !cd /kaggle/working && rm -rf kessler && git clone -q https://github.com/Shakhtar-Sankur/kessler && bash kessler/bench/kaggle_all.sh
set -o pipefail
cd "$(dirname "$0")/.."
echo "== kessler $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pip install -q -r requirements.txt 2>&1 | grep -v -i "notice\|warning" | tail -2
make -s && make -s cuda CUDA_ARCH="-gencode arch=compute_75,code=sm_75" || exit 1
mkdir -p bench/results
echo "== tests (CPU and GPU)"
python -W ignore -m pytest -q -s tests/ 2>&1 | grep -E "GPU vs CPU|objects,|CARA values|passed|failed|Error"
echo "== M2: one day of the active catalogue, 5 km, GPU and CPU"
python bench/m2_screen.py --hours 24 --out bench/results/m2_t4.json 2>&1 | tail -40
echo "== M2: reproduce SOCRATES"
python bench/m2_socrates.py --n 100 --out bench/results/m2_socrates.json 2>&1 | tail -25
if python -c "from kessler.spacetrack import credentials; credentials()" 2>/dev/null; then
  echo "== M2: one day of the full Space-Track catalogue (debris included), GPU only"
  python bench/m2_screen.py --hours 24 --spacetrack --skip-cpu --out bench/results/m2_t4_spacetrack.json 2>&1 | tail -40
  echo "== M4: Starlink maneuvers, Dec 2025 - May 2026"
  python bench/m4_maneuvers.py --out bench/results/m4_maneuvers.json 2>&1 | tail -30
else
  echo "== Space-Track secrets not set: skipping the full catalogue and M4"
fi
mkdir -p /kaggle/working/kessler-results 2>/dev/null && cp bench/results/*.json /kaggle/working/kessler-results/ 2>/dev/null
echo "== done: send everything from '== kessler' to here, and the files in /kaggle/working/kessler-results"
