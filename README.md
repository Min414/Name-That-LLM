# Name That LLM: Classifying AI-Generated Text by Source Model

Can we tell which language model wrote a piece of text? This project trains classifiers on the Kaggle dataset [Name That LLM - OpenAI Generated Responses](https://www.kaggle.com/datasets/enaimaan/ntl-openai-generated-responses) (9,719 labeled responses).

## Task

- **Input:** response text (`text`)
- **Target:** source model (`model`): `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`
- **Class balance:** astra 2,704; luna 2,341; sol 2,338; terra 2,336

## Method

1. **Prompt-grouped split.** The same prompt appears multiple times, so data is split with `GroupShuffleSplit` on `prompt_group` (80/20) to prevent prompt leakage.
2. **Baseline:** word TF-IDF (1-2 grams) + logistic regression.
3. **Improved linear model:** word + character n-gram TF-IDF + LinearSVC.
4. **Transformer:** DistilBERT fine-tuned for 3 epochs (Google Colab T4).
5. **Two-group test:** relabel as gpt-5.6 family vs gpt-6.
6. **Robustness:** 5-fold `GroupKFold` cross-validation.

## Results

| Experiment | Accuracy | Macro F1 |
|---|---|---|
| 4-class, word TF-IDF + LogReg | 0.46 | 0.44 |
| 4-class, word + char n-grams + LinearSVC | 0.48 | 0.47 |
| 4-class, DistilBERT (3 epochs) | 0.46 | 0.45 |
| 2-class (5.6 vs 6), TF-IDF + LinearSVC | 0.78 | 0.74 |

Chance is about 0.25 for 4 classes. For 2 classes, always guessing "5.6" scores 0.72 accuracy, so class "6" F1 (0.62) is the more informative number.

**Grouped 5-fold CV (macro F1):**
- 4-class: 0.461 ± 0.013
- 2-class: 0.742 ± 0.004

## Findings

- Models are separable by **generation** (5.6 vs 6) but only weakly by **variant** within the 5.6 family. luna, sol and terra are mostly confused with each other.
- Three different approaches landed at 46-48% for 4 classes, which suggests a ceiling for text-only style signals.
- **Response length does not differ** between groups (about 117 vs 113 words on average).
- Top features suggest:
  - **gpt-6:** more direct and concrete; possessives and demonstratives (*its, their, these*), contrast words (*rather than, although, without*), second-person address (*if you, you're*).
  - **gpt-5.6:** more hedged and descriptive; modals (*may, may be*), plain *is/are/was*, simple connectors (*and, or, because, such as*).

## How to run

```bash
pip install pandas scikit-learn transformers datasets accelerate ftfy
```

1. Download the dataset from Kaggle and place the CSV in the working directory.
2. Run the notebook cells in order: load data, split, baseline, char n-grams, DistilBERT, 2-class test, cross-validation, feature inspection.
3. For DistilBERT, use a GPU runtime.

## Notes and limitations

- Some text has encoding artifacts (e.g. `â€™`); fix with `ftfy` or UTF-8 reading.
- Feature patterns are correlations on this dataset and may partly reflect the prompts used. They should not be generalized to all text from these models.
- Results come from one dataset; the `response_license` field is marked "not independently established", so check licensing before redistributing data.
- DistilBERT was trained briefly (3 epochs) with default settings; more tuning could change its result.

## Dataset credit

Dataset by [enaimaan](https://www.kaggle.com/datasets/enaimaan/ntl-openai-generated-responses) on Kaggle. Prompts derive from sources such as Databricks Dolly 15k (CC BY-SA 3.0).
