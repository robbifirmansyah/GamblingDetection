# 🎯 Quick Start Guide - Stratified Re-Splitting

## Apa yang Berubah?

### ❌ Masalah Lama:
- Holdout set berbeda distribusi → **Concept Drift**
- Model stuck di F1 = 0.89, tidak bisa tembus 0.90
- Overfitting ke train/test

### ✅ Solusi Baru:
- **Gabungkan semua data** → Shuffle → **Split ulang 70:20:10**
- Distribusi konsisten di semua splits
- Target F1 ≥ 0.90 lebih mudah dicapai

---

## 🚀 Cara Menjalankan

### Step 1: Buka Notebook
```bash
jupyter notebook Untitled.ipynb
```

### Step 2: Run Cells Secara Berurutan
1. **Cell 1-3**: Setup & imports ✅
2. **Cell 4**: Load data → **RE-SPLIT** disini! 🆕
3. **Cell 5-10**: EDA & visualizations
4. **Cell 11**: Feature Engineering (30+ features)
5. **Cell 12**: Extract features
6. **Cell 13**: Handle imbalanced data (SMOTE sweep)
7. **Cell 14**: Train models (LGB, XGB, Cat, RF)
8. **Cell 15-17**: Ensemble & tuning
9. **Cell 18-22**: Evaluation & results

### Step 3: Check Results
- Output di folder `outputs/`
- Model di `best_model.pkl`
- Metrics di `model_config.json`

---

## 📊 Expected Results

### Sebelum (Concept Drift):
```
Test F1:    0.91
Holdout F1: 0.89 ❌ (gap 0.02)
```

### Sesudah (Stratified):
```
Test F1:    0.91
Holdout F1: 0.90+ ✅ (gap 0.01)
```

---

## 💡 Yang Perlu Diperhatikan

### 1. Cell Penting (WAJIB RUN):
- **Cell 4**: Re-splitting disini
- **Cell 13**: SMOTE ratio sweep
- **Cell 16**: Ensemble tuning

### 2. Output yang Penting:
```
outputs/
├── class_distribution.png       # Check konsistensi!
├── ensemble_top_combinations.csv # Top configs
├── holdout_evaluation_best_model.png
└── model_comparison.png
```

### 3. Files yang Dihasilkan:
```
best_model.pkl          # Model terbaik
feature_extractor.pkl   # Feature pipeline
model_config.json       # Config & metrics
training_metrics.csv    # All models performance
```

---

## 🔍 Verification Checklist

Setelah run, pastikan:

- [ ] Semua splits punya distribusi mirip (ratio diff < 0.05)
- [ ] Test F1 dan Holdout F1 gap < 0.02
- [ ] Holdout F1 ≥ 0.90
- [ ] Model config tersimpan
- [ ] Visualizations generated

---

## ⏱️ Runtime Estimate

- Feature Extraction: ~10 min
- SMOTE sweep: ~5 min
- Model training: ~15 min
- Ensemble tuning: ~20 min
- **Total: ~50 minutes**

---

## 🆘 Troubleshooting

### Error: Unicode encoding
**Fix**: Sudah di-handle di cell 11 (safe encoding)

### Error: Memory
**Fix**: Skip BERT extraction (sudah diset skip)

### Warning: LightGBM/XGBoost
**Fix**: Ignore, model tetap jalan

---

## 📚 Documentation

Lengkap di:
- `README.md` - Full documentation
- `METHODOLOGY_CHANGE.md` - Detailed methodology
- `Untitled.ipynb` - Commented code

---

## 🎯 Goal

**Target**: F1 ≥ 0.90 pada holdout set

**Strategy**: Stratified re-splitting untuk eliminate concept drift

**Expected**: ✅ Target tercapai!

---

**Ready? Let's run it!** 🚀
