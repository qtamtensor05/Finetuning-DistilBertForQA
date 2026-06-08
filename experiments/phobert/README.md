# PhoBERT QA Experiments

Thu muc nay tach rieng thi nghiem PhoBERT, khong thay doi luong train DistilBERT hien tai.

## Train hai model PhoBERT

```bash
python3 experiments/phobert/train.py
```

Lenh tren doc `experiments/phobert/config.yaml` va train lan luot:

- `vinai/phobert-base`
- `vinai/phobert-base-v2`

Co the train rieng mot profile:

```bash
python3 experiments/phobert/train.py --profile train_phobert_base_vi
python3 experiments/phobert/train.py --profile train_phobert_base_v2_vi
```

## So sanh voi DistilBERT custom

Sau khi da co checkpoint DistilBERT va hai checkpoint PhoBERT:

```bash
python3 experiments/phobert/compare.py \
  --checkpoints \
  outputs/checkpoints_stage2_vi/best_model \
  outputs/checkpoints_phobert_base_vi/best_model \
  outputs/checkpoints_phobert_base_v2_vi/best_model
```

Ket qua duoc luu vao `outputs/compare_vi_models`.
