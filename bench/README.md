# Benchmarks

## Run on Kaggle (free T4 GPU)

New notebook, Settings → Accelerator → **GPU T4**, Internet on. One cell:

```
!git clone -q https://github.com/Shakhtar-Sankur/kessler
%cd kessler
!pip install -q sgp4 dsgp4 pytest
!make && make cuda CUDA_ARCH="-gencode arch=compute_75,code=sm_75"
!python -m pytest -q -s tests/
!python bench/m1_propagate.py --out bench/results/m1_t4.json
```

Send back the printed JSON (or `bench/results/m1_t4.json`).
