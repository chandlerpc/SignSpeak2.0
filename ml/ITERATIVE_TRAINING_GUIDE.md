# Iterative Training Workflow

This guide shows how to continuously improve your ASL model by collecting more data and retraining.

## Current Model Performance

- **Training data**: 500 samples per letter (26 letters = 13,000 total images)
- **Validation accuracy**: 99.81%
- **Working well**: A, C, E and most letters
- **Needs improvement**: B/O confusion, D/E confusion

## Workflow: Continuously Improve the Model

### Step 1: Identify Problem Areas

Test the current model and note which letters are being confused:
- Open the app (`npm run dev`)
- Sign each letter and note which ones predict incorrectly
- Common confusions: B↔O, D↔E, M↔N, U↔V

### Step 2: Collect More Diverse Samples

Focus on letters that need improvement:

```bash
# Open the Data Collector page
# Navigate to http://localhost:5173/collect
```

**Tips for better diversity**:
- **Vary hand position**: slightly left, right, up, down
- **Vary hand angle**: rotate palm slightly
- **Vary distance**: move hand closer/farther from camera
- **Vary lighting**: different times of day
- **Vary background**: different parts of your room

**How many new samples?**
- If a letter works well: 0-50 new samples
- If a letter has occasional errors: 100-200 new samples
- If a letter consistently fails: 200-300 new samples

### Step 3: Merge New Data with Existing Dataset

```bash
cd ml

# Merge new samples into existing dataset
py merge_training_data.py path/to/new_data.json

# This will:
# - Load existing training_data_deduplicated_500.json
# - Add new samples (avoiding duplicates)
# - Backup old dataset
# - Save merged dataset
```

### Step 4: Retrain the Model

```bash
cd ml

# Retrain with merged dataset
py retrain_simple.py training_data_deduplicated_500.json 2>&1 | tee training_v2.log

# Training will:
# - Load all training data (old + new)
# - Train for 50 epochs with early stopping
# - Save best model to checkpoints/simple_best.keras
# - Save final model to checkpoints/simple_final.keras
```

**Training takes**: ~15-30 minutes depending on total samples

### Step 5: Test the New Model

The model server automatically loads the latest model. Restart it:

```bash
# Kill old server (Ctrl+C in model server terminal)

# Start new server
cd ml
py model_server.py
```

Then test the app and check if problem letters improved!

### Step 6: Track Progress

Keep a log of each training iteration:

```bash
# Create a training log
echo "Version 2: Added 200 B samples, 150 D samples" >> training_history.txt
echo "Results: B accuracy improved from 60% to 85%" >> training_history.txt
echo "Validation accuracy: 99.85%" >> training_history.txt
echo "Date: $(date)" >> training_history.txt
echo "---" >> training_history.txt
```

### Step 7: Repeat

Repeat steps 1-6 until you're happy with the accuracy!

## Expected Improvements Per Iteration

| Iteration | Samples Added | Expected Improvement |
|-----------|---------------|---------------------|
| 1 (current) | 500/letter base | Baseline (99.81% val acc) |
| 2 | +100-200 problem letters | +2-5% on those letters |
| 3 | +100-200 more diversity | +2-5% more |
| 4+ | +50-100 edge cases | +1-3% more |

## Tips for Best Results

### 1. **Focus on Problem Letters**
Don't waste time collecting 500 more A samples if A already works perfectly. Focus on B, D, and any other problem letters.

### 2. **Add Diversity, Not Just Quantity**
100 samples with varied positions/angles/lighting is better than 500 identical samples.

### 3. **Test After Each Iteration**
Always test the new model before collecting more data. You might find new problem areas!

### 4. **Keep Old Model Versions**
Rename checkpoints before retraining:

```bash
cp checkpoints/simple_best.keras checkpoints/simple_best_v1.keras
cp checkpoints/simple_final.keras checkpoints/simple_final_v1.keras
```

### 5. **Monitor Overfitting**
Watch the training logs:
- Good: Training 55%, Validation 99% (augmentation working)
- Bad: Training 99%, Validation 60% (overfitting - need more diversity)

## Quick Commands Cheat Sheet

```bash
# Collect data
npm run dev
# Navigate to http://localhost:5173/collect

# Merge new data
cd ml
py merge_training_data.py new_samples.json

# Retrain
py retrain_simple.py training_data_deduplicated_500.json 2>&1 | tee training_v2.log

# Restart model server
# Ctrl+C to stop old server
py model_server.py

# Test app
# Navigate to http://localhost:5173
```

## Advanced: More Aggressive Data Augmentation

If you want the model to generalize better without collecting more data, increase augmentation in `retrain_simple.py`:

```python
# Line 153-160 in retrain_simple.py
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),         # Was 0.2
    layers.RandomZoom(0.3),             # Was 0.2
    layers.RandomTranslation(0.2, 0.2), # Was 0.1, 0.1
    layers.RandomBrightness(0.3),       # Was 0.2
    layers.RandomContrast(0.3),         # Was 0.2
])
```

Then retrain as usual.

## Goal

With 2-3 iterations of collecting diverse data and retraining, you should reach:
- **95%+ accuracy on all 26 letters**
- **High confidence predictions (>90%)**
- **Robust to hand position/angle variations**

Good luck! 🎉