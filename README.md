
# Social Media Extremism Detection

This project implements an end-to-end Natural Language Processing (NLP) pipeline to detect extremist content in social media messages. It leverages Hugging Face Transformers and PyTorch to classify each message as either `NON_EXTREMIST` or `EXTREMIST`.

Two complementary approaches are implemented in a single notebook:

1. A lightweight **DistilBERT** baseline.
2. A domain-specialized **Twitter-RoBERTa** model with confidence thresholding for higher precision.

---

## Features

- Transformer-based text classification (DistilBERT & Twitter-RoBERTa).
- GPU support via automatic device detection (`cuda` if available, otherwise `cpu`).
- Robust preprocessing (missing text handling, label mapping).
- Confidence thresholding to reduce false positives on extremist content.
- Built-in evaluation: classification reports and confusion matrices.
- Ready-to-submit CSV files for competition / production pipelines.

---

## Models

### 1. DistilBERT (Baseline)

- **Architecture:** `distilbert-base-uncased`
- **Objective:** Provide a simple, stable, and memory-efficient baseline.
- **Training Configuration:**
  - Batch size: `8`
  - Epochs: `3`
  - Max sequence length: `64`
- **Methodology:** Standard fine-tuning on the training set.

Use this model when:
- You have limited compute resources.
- You need quick experimentation or a strong baseline for comparison.

---

### 2. Twitter-RoBERTa (Domain-Specialized)

- **Architecture:** `cardiffnlp/twitter-roberta-base-offensive`
- **Objective:** Exploit a model pre-trained on offensive Twitter data for better domain awareness.
- **Training Configuration:**
  - Batch size: `32`
  - Epochs: `6`
  - Max sequence length: `64`

#### Confidence Thresholding

To reduce over-flagging of content as extremist, the notebook implements a **strict confidence threshold**:

- If `P(EXTREMIST) ≥ 0.85` → label as `EXTREMIST`
- Otherwise → label as `NON_EXTREMIST`

This favors **precision** for the `EXTREMIST` class, which is often desirable in sensitive moderation tasks.

#### Visualization

The notebook includes:

- **Classification report** (precision, recall, F1-score) on a validation split.
- **Confusion matrix** (heatmap using Matplotlib/Seaborn) to visualize performance and threshold effects.

---

## Requirements

Install the required libraries with:

```bash
pip install torch transformers pandas numpy scikit-learn datasets matplotlib seaborn
````

The notebook automatically selects the device:

* Uses **GPU** if available (`cuda`).
* Falls back to **CPU** otherwise.

---

## Dataset

The project expects the following CSV files in the chosen input directory:

* `train.csv`

  * Required columns:

    * `ID`
    * `Original_Message`
    * `Extremism_Label` (`NON_EXTREMIST` / `EXTREMIST`)
* `test.csv`

  * Required columns:

    * `ID`
    * `Original_Message`

Preprocessing details:

* Missing text values are automatically filled.
* String labels are mapped to integers:

  * `NON_EXTREMIST` → `0`
  * `EXTREMIST` → `1`

Make sure your data matches this schema before running the notebook.

---

## Usage

1. **Set File Paths**

   Update the dataset paths in the notebook to point to your local data.


2. **Load and Preprocess Data**

   Run the data loading and preprocessing cells:

   * Reads `train.csv` and `test.csv`.
   * Handles missing values and label mapping.
   * Prepares datasets for Hugging Face `Dataset` / DataLoader usage.

3. **Train Models**

   * Run the **DistilBERT** section to train the baseline model.
   * Run the **Twitter-RoBERTa** section to train the domain-specialized model.
   * The notebook includes explicit memory management (e.g., `gc.collect()`, `torch.cuda.empty_cache()`) to mitigate RAM issues during training.

4. **Run Inference**

   * After training, run the inference cells to generate predictions on the `test.csv` data for each model.

---

## Outputs

The notebook generates the following submission files:

* `submission_distilbert.csv`
  Predictions from the standard DistilBERT baseline.

* `submission_roberta.csv`
  Predictions using the Twitter-RoBERTa model with the 0.85 confidence thresholding strategy.

Each file contains predicted labels for all rows in `test.csv`, keyed by `ID`.
