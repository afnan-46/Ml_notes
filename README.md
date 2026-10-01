# 🌳 Decision Tree & Random Forest — CSE457 Lab 1

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/afnan-46/REPO_NAME/blob/main/Lab1_DT_RF_with_Mushroom.ipynb)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

This repository holds my **CSE457 (Artificial Intelligence) Lab 1** at East West University. It does three things:

- builds a **decision tree from scratch with NumPy**,
- compares it with **scikit-learn's** decision tree and random forest,
- applies both models to the **UCI Mushroom dataset**, classifying mushrooms as edible or poisonous.

---

## 📂 Contents

| Part | Dataset | What it does |
|---|---|---|
| **A1** | Iris | Load the data, summary statistics, scatter matrix |
| **A2** | Iris | Decision tree **from scratch** (entropy + information gain), visualized with Graphviz |
| **A3** | Iris | `DecisionTreeClassifier` (entropy, depth 3): metrics, confusion matrix, tree plot |
| **A4** | Breast Cancer | `RandomForestClassifier` (100 trees): accuracy, classification report, plot of one tree |
| **B2** | Mushroom | Exploratory data analysis: missing values, class balance, per-feature distributions, Cramér's V |
| **B4** | Mushroom | Random forest accuracy for `n_estimators` = 1, 50, 100, 150, 200, 250 |
| **B5** | Mushroom | Random forest vs decision tree: metrics, false negatives, complexity, feature importance |

---

## 🧠 Decision Tree from Scratch

The custom `DecisionTree` class implements the ID3-style algorithm:

- **Entropy:** $H(S) = -\sum_i p_i \log_2 p_i$
- **Information gain:** $IG = H(S) - \frac{|S_L|}{|S|}H(S_L) - \frac{|S_R|}{|S|}H(S_R)$
- **Best split:** every unique value of every feature is tried as a `<=` threshold, and the one with the highest gain is kept.
- **Stopping rules:** a node becomes a leaf when it is pure, when `max_depth` is reached, or when no split improves the gain (majority-vote leaf).

On the Iris test set it reaches **precision, recall and F1 of 1.00** (macro-averaged).

---

## 🍄 Mushroom Classification

**Dataset:** [UCI Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom). It has 8,124 samples, 22 categorical features, and a binary label (edible or poisonous).

### EDA highlights
- Classes are nearly balanced: **51.8% edible, 48.2% poisonous**.
- `stalk-root` is missing in **30.5%** of rows. Missing values are kept as their own category instead of dropping the rows.
- `veil-type` has a single value for every row, so it is dropped.
- **`odor` is almost perfectly predictive** (Cramér's V = 0.97). For example, every mushroom with a foul odor is poisonous.

### Preprocessing
Features are **one-hot encoded**, which gives 116 binary features. Label encoding would invent an order between categories that doesn't exist. The data is split 80/20 with stratification.

### Results — `n_estimators` (full dataset)

| n_estimators | Test accuracy | 5-fold CV accuracy | Train time (s) |
|---:|:---:|:---:|---:|
| 1 | 1.000 | 1.000 | 0.02 |
| 50 | 1.000 | 1.000 | 0.12 |
| 100 | 1.000 | 1.000 | 0.24 |
| 150 | 1.000 | 1.000 | 0.47 |
| 200 | 1.000 | 1.000 | 0.67 |
| 250 | 1.000 | 1.000 | 0.71 |

The classes are almost perfectly separable, so **every setting reaches 100%**. Only training time grows, roughly linearly with the number of trees.

### Stress test — training on only 1% of the data (20 random splits)

| Model | Mean accuracy | Std |
|---|:---:|:---:|
| Random Forest, 1 tree | 0.932 | 0.036 |
| Random Forest, 50 trees | 0.970 | 0.012 |
| Random Forest, 100 trees | 0.971 | 0.012 |
| Random Forest, 150 trees | 0.972 | 0.011 |
| Random Forest, 200 trees | 0.972 | 0.011 |
| Random Forest, 250 trees | 0.972 | 0.011 |
| Single Decision Tree | 0.964 | 0.020 |

With scarce data the effect of ensembling becomes visible:

- **1 tree is the weakest and the least stable.** It is trained on a bootstrap sample and only sees a random subset of features at each split.
- **Accuracy jumps between 1 and 50 trees** because majority voting cancels out the errors of individual trees (variance reduction).
- **Accuracy plateaus after about 50–100 trees**, while the computational cost keeps rising.

### Decision Tree vs Random Forest

| Aspect | Decision Tree | Random Forest |
|---|---|---|
| Accuracy (full data) | 100% | 100% |
| Accuracy (1% data) | 96.4% ± 2.0 | 97.1% ± 1.2 |
| Poisonous predicted as edible | 0 | 0 |
| Model size | 25 nodes (depth 7) | 100 trees, about 6,160 nodes |
| Interpretability | ✅ Readable IF–THEN rules | ❌ A vote of many trees |
| Feature importance | Concentrated on `odor` | Spread across correlated features |

**Conclusion:** when there is enough data, both models are perfect, and the **decision tree is preferable for its simplicity and interpretability**. When data is limited, the **random forest is more accurate and more stable**.

---

## 🔧 Fixes to the Original Lab Notebook

- The scikit-learn split used `df.drop('target')`, but the Iris label column is called `'species'`, which caused a `KeyError`.
- The model variable `tree` overwrote `from sklearn import tree` and broke the random forest plot. It is renamed to `custom_tree`.
- The from-scratch tree crashed when no split improved the gain. It now falls back to a majority-vote leaf.
- The random forest accuracy and report were computed but never printed.
- A comment said "gini" while the code used `criterion='entropy'`.

---

## 🚀 How to Run

**Option 1 — Google Colab:** click the **Open in Colab** badge at the top.

**Option 2 — Locally:**
```bash
git clone https://github.com/afnan-46/REPO_NAME.git
cd REPO_NAME
pip install numpy pandas matplotlib seaborn scikit-learn scipy graphviz ucimlrepo
jupyter notebook Lab1_DT_RF_with_Mushroom.ipynb
```
> The `graphviz` Python package also needs the Graphviz system binary (`sudo apt install graphviz` or `brew install graphviz`).

The notebook downloads the Mushroom dataset automatically. It tries `ucimlrepo` first, then the UCI archive, then a GitHub mirror.

---

## 🛠️ Tech Stack
Python · NumPy · pandas · scikit-learn · Matplotlib · Seaborn · SciPy · Graphviz

## 🙏 Acknowledgements
- Base lab notebook: [raihanewubd/CSE457](https://github.com/raihanewubd/CSE457)
- Dataset: *Mushroom* (1981), UCI Machine Learning Repository. https://doi.org/10.24432/C5959T

## 👤 Author
**Afnan** — B.Sc. in Computer Science & Engineering, East West University
GitHub: [@afnan-46](https://github.com/afnan-46)
