# Comparison Methods Nohup Scripts

This directory contains 12 nohup scripts for running comparison methods (GEARS, scDFM, PerturbDiff, Squidiff) with 3 different data splits (additive, holdout, LOCO).

## Data Splits

Based on the three CellFlow training scripts:

1. **Additive** (`train_cellflow_norman_scdfm_additive.py`)
   - Dataset: Norman 2019 (K562)
   - Split: 30% double perturbations as test, all single perturbations in train
   - Seed: 20240508, Split seed base: 42, Fold: 0

2. **Holdout** (`train_cellflow_norman_scdfm_holdout.py`)
   - Dataset: Norman 2019 (K562)
   - Split: Hold out 12 genes, test on held-out singles and all doubles involving them
   - Seed: 20240508, Split seed base: 42, Fold: 0

3. **LOCO** (`train_cellflow_loco_new.py`)
   - Dataset: Replogle (4 cell lines)
   - Split: Leave-One-Cell-Line-Out (hepg2), 30% train/test perturbations
   - Seed: 20240508

## Running Scripts

### Run single experiment:
```bash
cd /home/zhangshibo24s/cell_flow/comparison_methods/scripts_nohup
./run_gears_norman_additive.sh
```

### Run all experiments:
```bash
cd /home/zhangshibo24s/cell_flow/comparison_methods/scripts_nohup
for script in run_*.sh; do ./$script; done
```

### Monitor logs:
```bash
tail -f ../logs/gears_norman_additive.log
```

### Set GPU:
```bash
export CUDA_VISIBLE_DEVICES=0
./run_gears_norman_additive.sh
```

## Output Structure

Logs are saved to: `../logs/`
Results are saved to: `../outputs_{method}_{split}/`

## Summary Table

| Method | Norman Additive | Norman Holdout | LOCO |
|--------|-----------------|----------------|------|
| GEARS | `run_gears_norman_additive.sh` | `run_gears_norman_holdout.sh` | `run_gears_loco.sh` |
| scDFM | `run_scdfm_norman_additive.sh` | `run_scdfm_norman_holdout.sh` | `run_scdfm_loco.sh` |
| PerturbDiff | `run_perturbdiff_norman_additive.sh` | `run_perturbdiff_norman_holdout.sh` | `run_perturbdiff_loco.sh` |
| Squidiff | `run_squidiff_norman_additive.sh` | `run_squidiff_norman_holdout.sh` | `run_squidiff_loco.sh` |
