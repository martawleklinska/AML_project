# AML Grooming Detection Pipeline

## Overview

This project implements a recall-oriented grooming detection pipeline for the PAN 2012 Sexual Predator Identification dataset.

The project uses four scripts:

```text
preprocess.py -> converts PAN 2012 XML files into JSON splits
visualize.py  -> inspects JSON data and creates debug plots
train.py      -> performs continued DeBERTa pretraining with masked language modeling
model.py      -> trains and evaluates the final binary grooming classifier
```

The main design goal is to reduce false negatives. In this task, missing a grooming conversation is considered more harmful than flagging an additional benign conversation. Therefore, the classifier uses positive-class weighting and threshold calibration based on F2 score.

---

## Pipeline

```text
PAN 2012 XML files
        |
        v
preprocess.py
        |
        v
pan12_dataset/train.json
pan12_dataset/val.json
pan12_dataset/test.json
        |
        +-------------------+
        |                   |
        v                   v
visualize.py                train.py
        |                   |
        v                   v
debug/                      runs/deberta_mlm/
                            |
                            v
                         model.py
                            |
                            v
                         results/
```

---

## Dataset splits

The preprocessing stage creates three JSON files:

```text
train.json -> training data
val.json   -> validation data
test.json  -> final test data
```

### train.json

Used for:

- classifier training;
- continued DeBERTa pretraining;
- estimating class weights.

The model updates its parameters on this split.

### val.json

Used for:

- validation during training;
- checkpoint selection;
- early stopping;
- decision-threshold calibration.

The model does not directly train on this split, but validation results influence model selection.

### test.json

Used only for final evaluation.

It must not be used for:

- pretraining;
- classifier training;
- hyperparameter tuning;
- threshold calibration.

Using `test.json` before final evaluation causes data leakage.

---

## Recommended directory structure

```text
project/
  data/
    train/
      pan12-sexual-predator-identification-training-corpus-2012-05-01.xml
      pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt
    test/
      pan12-sexual-predator-identification-test-corpus-2012-05-17.xml
      pan12-sexual-predator-identification-groundtruth-problem1.txt

  pan12_dataset/
    train.json
    val.json
    test.json

  runs/
    deberta_mlm/

  results/
    best_model.pt
    best_threshold.json
    test_results.json

  debug/
    analysis_train.json
    debug_train_all.txt
    plots/

  preprocess.py
  visualize.py
  train.py
  model.py
```

---

## Environment setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
python -m pip install numpy pandas scikit-learn
python -m pip install transformers accelerate sentencepiece
python -m pip install plotnine mizani matplotlib scipy statsmodels
python -m pip install thefuzz rapidfuzz
```

Install CUDA-enabled PyTorch for NVIDIA GPU:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Check CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected:

```text
True
NVIDIA GeForce RTX ...
```

---

## Step 1: preprocessing

Run:

```bash
python preprocess.py
```

### Input

```text
data/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml
data/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt
data/test/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml
data/test/pan12-sexual-predator-identification-groundtruth-problem1.txt
```

### Output

```text
pan12_dataset/train.json
pan12_dataset/val.json
pan12_dataset/test.json
```

### Function

`preprocess.py` converts raw PAN 2012 XML conversations into machine-learning-ready JSON files.

It performs:

- XML parsing;
- train/validation split creation;
- text normalisation;
- slang expansion;
- leet-speech normalisation;
- typo-tolerant fuzzy matching;
- rule-based grooming-signal detection;
- conversation-level risk scoring;
- insertion of structural metadata tokens.

### Example normalisation

```text
s3nd m3 4 p1c          -> send me a picture
d0nt t3ll y0ur p4r3nts -> do not tell your parents
whts ur discrod        -> what is your discord
```

### Rule tags

Messages can receive rule tags:

```text
[RULE:image_solicitation:3:regex]
[RULE:isolation:3:regex]
[RULE:platform_migration:3:fuzzy]
```

Tag format:

```text
[RULE:<category>:<severity>:<method>]
```

Severity levels:

```text
1 -> weak contextual signal
2 -> concerning signal
3 -> explicit high-risk signal
```

### Conversation prefix

Each flattened conversation starts with metadata:

```text
[RISK:HIGH] [CATS:isolation,image_solicitation] [ESC:1] [ARC:1] [DOM:0]
```

Meaning:

```text
[RISK:*] -> aggregated risk level
[CATS:*] -> detected rule categories
[ESC:*]  -> escalation flag
[ARC:*]  -> grooming-stage arc flag
[DOM:*]  -> message dominance flag
```

---

## Step 2: visualization

Run:

```bash
python visualize.py
```

### Input

```text
pan12_dataset/train.json
pan12_dataset/val.json
pan12_dataset/test.json
```

### Output

Usually:

```text
debug/
```

or:

```text
debug_p9/
```

Expected outputs:

```text
analysis_train.json
analysis_val.json
analysis_test.json

debug_train_all.txt
debug_val_all.txt
debug_test_all.txt

plots/*.png
plots/*.pdf
```

### Function

`visualize.py` parses the generated JSON files and extracts:

- risk level;
- label;
- rule categories;
- rule-hit counts;
- severity-3 counts;
- escalation flag;
- arc flag;
- dominance flag;
- regex/fuzzy method counts.

It also generates plots with `plotnine` / `p9.ggplot`.

Typical plots:

```text
risk_level_distribution
risk_label_heatmap
top_categories
category_share_by_label
total_rule_hits_distribution
unique_categories_distribution
severity3_distribution
structural_signal_rates
predator_message_share
regex_vs_fuzzy_hits
```

This step is used for debugging, quality control, and thesis figures.

---

## Step 3: continued DeBERTa pretraining

Run:

```bash
python train.py
```

### Input

```text
pan12_dataset/train.json
pan12_dataset/val.json
```

### Output

```text
runs/deberta_mlm/
```

### Function

`train.py` performs continued domain pretraining of DeBERTa using masked language modeling.

This is not full pretraining from scratch. It is domain adaptation of an existing model, usually:

```text
microsoft/deberta-v3-base
```

The objective is to adapt the language model to:

- chat-style text;
- PAN 2012 conversation structure;
- `[PRED]` and `[USER]` role tokens;
- rule tags;
- risk metadata tokens.

### Data policy

```text
train.json -> MLM training
val.json   -> MLM validation
test.json  -> not used
```

`test.json` must not be used here.

### Mechanism

The script:

1. Loads `train.json` and `val.json`.
2. Extracts conversation text.
3. Loads the base DeBERTa tokenizer.
4. Adds project-specific bracket tokens as special tokens.
5. Loads `AutoModelForMaskedLM`.
6. Resizes token embeddings if new tokens were added.
7. Splits long conversations into token chunks.
8. Applies random masking through the MLM data collator.
9. Trains the model.
10. Saves the adapted checkpoint.

After completion, the adapted model is stored in:

```text
runs/deberta_mlm/
```

This path is used by `model.py`.

---

## Step 4: final classifier training

Run:

```bash
python model.py
```

### Input

```text
pan12_dataset/train.json
pan12_dataset/val.json
pan12_dataset/test.json
runs/deberta_mlm/
```

### Required model setting

`model.py` should load the checkpoint created by `train.py`:

```python
BASE_MODEL = "runs/deberta_mlm"
```

This ensures that the final classifier uses the domain-adapted DeBERTa backbone.

### Output

```text
results/best_model.pt
results/best_threshold.json
results/test_results.json
```

### Classifier architecture

```text
DeBERTa backbone
        |
        v
[CLS] embedding
        |
        v
Dropout
        |
        v
Linear hidden_size -> 256
        |
        v
ReLU
        |
        v
Dropout
        |
        v
Linear 256 -> 1
        |
        v
binary logit
```

### Loss function

The classifier uses:

```text
BCEWithLogitsLoss
```

This combines sigmoid and binary cross entropy in one numerically stable operation.

### Class imbalance handling

The positive class is upweighted:

```text
pos_weight = scale * (N_negative / N_positive)
```

This increases the penalty for missing grooming conversations.

### Threshold calibration

The classifier does not use a fixed `0.5` threshold.

Instead, it searches thresholds on `val.json` and selects the one that maximises F2:

```text
threshold candidates: 0.05, 0.06, ..., 0.94
```

### Final evaluation

After training:

1. Load the best checkpoint.
2. Load the best validation threshold.
3. Evaluate once on `test.json`.
4. Save final metrics.

---

## Metrics

### Confusion matrix terms

```text
TP -> true positive: grooming correctly detected
FP -> false positive: benign conversation flagged as grooming
FN -> false negative: grooming missed
TN -> true negative: benign conversation correctly ignored
```

### Precision

```text
precision = TP / (TP + FP)
```

Precision answers:

```text
Of all conversations flagged as grooming, how many were truly grooming?
```

### Recall

```text
recall = TP / (TP + FN)
```

Recall answers:

```text
Of all grooming conversations, how many were detected?
```

### F2 score

```text
F2 = 5 * precision * recall / (4 * precision + recall)
```

F2 weights recall more strongly than precision.

This project uses F2 because false negatives are more harmful than false positives.

---

## Full execution order

Run:

```bash
python preprocess.py
python visualize.py
python train.py
python model.py
```

Recommended with a basic preprocessing check:

```bash
python preprocess.py
python -c "import json; print(len(json.load(open('pan12_dataset/train.json', encoding='utf-8'))))"

python visualize.py
python train.py
python model.py
```

---

## Expected final artifacts

```text
pan12_dataset/
  train.json
  val.json
  test.json

runs/
  deberta_mlm/

results/
  best_model.pt
  best_threshold.json
  test_results.json

debug/ or debug_p9/
  analysis_*.json
  debug_*.txt
  plots/
```

---

## Correct and incorrect data usage

### Correct

```text
train.json -> training
val.json   -> checkpoint selection and threshold calibration
test.json  -> final evaluation only
```

### Incorrect

```text
train.json + val.json + test.json -> pretraining
test.json -> threshold calibration
test.json -> repeated evaluation during development
```

This leaks test information into the model-selection process.

---

## Common errors

### `ModuleNotFoundError: No module named 'plotnine'`

```bash
python -m pip install plotnine pandas mizani matplotlib scipy statsmodels
```

### `ModuleNotFoundError: No module named 'pandas'`

```bash
python -m pip install pandas
```

### `ModuleNotFoundError: No module named 'transformers'`

```bash
python -m pip install transformers accelerate sentencepiece
```

### `torch.cuda.is_available()` returns `False`

Check PyTorch:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Reinstall CUDA-enabled PyTorch if needed:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### CUDA out of memory

Reduce batch size and increase gradient accumulation:

```python
BATCH_SIZE = 2
GRAD_ACCUM = 16
```

---

## Reproducibility notes

For final reporting, record:

```text
Python version
PyTorch version
Transformers version
CUDA availability
GPU model
random seed
train/val/test sizes
positive-class ratio
best threshold
test precision
test recall
test F2
TP / FP / FN
```

Exact reproducibility may still vary due to:

- CUDA kernels;
- GPU model;
- PyTorch version;
- Transformers version;
- nondeterministic operations.

---

## Limitations

1. Rule tags encode researcher-defined assumptions.
2. Fuzzy matching improves recall but can increase false positives.
3. Validation-based threshold selection depends on the quality of the validation split.
4. PAN 2012 is dated and may not reflect modern grooming tactics.
5. The test split must remain untouched until the final evaluation.

---

## Suggested report wording

This project implements a recall-oriented grooming detection pipeline for the PAN 2012 corpus. The preprocessing stage converts XML conversations into role-tagged JSON records, normalises slang and leet speech, applies regex and fuzzy rule matching, and injects structured risk tokens into the text. A DeBERTa-v3 model is then further pretrained on the domain corpus using masked language modeling, excluding the test split to avoid leakage. The adapted backbone is used by a binary classifier trained with positive-class weighting. The decision threshold is calibrated on the validation split by maximising F2 score, reflecting the higher cost of false negatives. Final performance is reported only once on the held-out test split.
