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

   <img width="722" height="470" alt="image" src="https://github.com/user-attachments/assets/668462cc-13e9-48d1-a140-cb89fd68f15e" />
   <img width="1389" height="590" alt="image" src="https://github.com/user-attachments/assets/c302d32d-ab43-4d87-a0c0-97ecddde4d6f" />
   <img width="612" height="547" alt="image" src="https://github.com/user-attachments/assets/79af2f5e-81d0-4555-9dbd-f92cb5aaa0e5" />
   <img width="1790" height="590" alt="image" src="https://github.com/user-attachments/assets/f5576403-366c-4916-abc9-f913c65f34e6" />
   <img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/46efb25b-bf11-4d9f-be33-33ce5597c3c4" />
<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/ac761977-3417-48ec-94c3-3a4ae0ca4187" />
<img width="1890" height="490" alt="image" src="https://github.com/user-attachments/assets/591fa11c-4973-462a-a15d-6bb454c10c83" />
#
Cluster counts:

K-Means:
KMeans_Cluster
0    58
1    39
2    47
3    34
4    22
Name: count, dtype: int64

DBSCAN (-1 means noise):
DBSCAN_Cluster
-1    19
 0     9
 1    94
 2    40
 3    26
 4    12
Name: count, dtype: int64

GMM:
GMM_Cluster
0    69
1    39
2    36
3    35
4    21
PCA explained variance ratio: 71.77%

K-Means
Silhouette Score: 0.3498  (higher is better)
Davies-Bouldin Index: 1.0245  (lower is better)

DBSCAN
Silhouette Score: 0.0745  (higher is better)
Davies-Bouldin Index: 1.4907  (lower is better)

GMM
Silhouette Score: 0.3323  (higher is better)
Davies-Bouldin Index: 1.0383  (lower is better)

Comparison of valid clustering metrics:
  Algorithm  Silhouette Score  Davies-Bouldin Index
0   K-Means            0.3498                1.0245
1    DBSCAN            0.0745                1.4907
2       GMM            0.3323                1.0383







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
