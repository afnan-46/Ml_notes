# CSE 475 Lab 2: Unsupervised Learning and Clustering

This folder contains the materials for **Lab 2** of CSE 475, focused on applying unsupervised machine-learning techniques to datasets using clustering.

## Contents

| File | Description |
|---|---|
| `CSE_475_Lab_2_Unsupervised_learning_Clustering _with_2Data.ipynb` | Jupyter Notebook containing the data analysis, preprocessing, clustering workflow, visualizations, and observations. |
| `adult.csv` | Adult dataset used for clustering analysis. |
| `Mall_Customers_Clustering_Results.csv` | Output dataset containing clustering results for the mall-customer analysis. |

## Objectives

- Understand the workflow of unsupervised learning.
- Prepare and explore datasets before clustering.
- Group similar observations into clusters.
- Interpret clustering results through numerical summaries and visualizations.
- Compare patterns discovered in the supplied datasets.

## Requirements

Use Python 3 and install the packages used by the notebook. A typical setup includes:

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/afnan-46/Ml_notes.git
   ```

2. Move into this lab folder:

   ```bash
   cd Ml_notes/lAB_WORK2
   ```

3. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `CSE_475_Lab_2_Unsupervised_learning_Clustering _with_2Data.ipynb`.

5. Run the notebook cells in order. Keep `adult.csv` and `Mall_Customers_Clustering_Results.csv` in the same directory so the notebook can load them correctly.

## Expected Workflow

The notebook follows a typical clustering pipeline:

1. Load the dataset.
2. Inspect its structure and handle missing or unsuitable values.
3. Select relevant numeric features.
4. Scale or normalize features where necessary.
5. Apply clustering techniques.
6. Evaluate and interpret the identified groups.
7. Visualize patterns and save results.

## Dataset Notes

### Adult dataset

`adult.csv` is included as a local dataset for the lab's analysis. Review the notebook to see which columns are selected, how categorical values are prepared, and how the data is used in clustering.

### Mall customer results

`Mall_Customers_Clustering_Results.csv` stores clustering output from the mall-customer exercise. It can be used to inspect the cluster assigned to each record and compare customer-group characteristics.

## Reproducibility

- Run notebook cells sequentially from top to bottom.
- Use the included CSV files without changing their filenames or locations.
- If random initialization is used, set a `random_state` value in the notebook to make results repeatable.

## Author

Afnan Bd  
East West University
