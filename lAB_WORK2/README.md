CSE 475 Lab 2: Unsupervised Learning and Clustering
> A practical machine-learning lab that applies unsupervised learning techniques to discover meaningful patterns in data.
Overview
This project contains the work for CSE 475 Lab 2, focused on unsupervised learning and clustering. The accompanying Jupyter Notebook performs data exploration, preprocessing, clustering, visualization, and result analysis using the supplied datasets.
The project includes a mall-customer clustering result file and the Adult dataset for additional analysis.
Objectives
Understand the basic workflow of unsupervised machine learning.
Prepare datasets for clustering analysis.
Select relevant features and transform data when necessary.
Apply clustering techniques to group similar records.
Analyze and visualize the clusters produced by the model.
Interpret the patterns discovered from the data.
Project Structure
```text
lAB_WORK2/
├── CSE_475_Lab_2_Unsupervised_learning_Clustering _with_2Data.ipynb
├── adult.csv
├── Mall_Customers_Clustering_Results.csv
├── README.md
└── images/                         # Add exported notebook output images here
    ├── elbow-method.png            # Optional
    ├── cluster-visualization.png   # Optional
    └── result-summary.png          # Optional
```
File	Description
`CSE_475_Lab_2_Unsupervised_learning_Clustering _with_2Data.ipynb`	Jupyter Notebook containing the complete lab workflow, code, outputs, and analysis.
`adult.csv`	Adult dataset used in the lab.
`Mall_Customers_Clustering_Results.csv`	Saved clustering results from the mall-customer analysis.
`images/`	Recommended folder for plots, charts, and screenshots exported from the notebook.
Technologies
Python 3
Jupyter Notebook
pandas
NumPy
Matplotlib
Seaborn
scikit-learn
Installation
Install the required Python packages:
```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```
How to Run
Clone the repository:
```bash
   git clone https://github.com/afnan-46/Ml_notes.git
   ```
Open the lab directory:
```bash
   cd Ml_notes/lAB_WORK2
   ```
Start Jupyter Notebook:
```bash
   jupyter notebook
   ```
Open the following notebook and run all cells from top to bottom:
```text
   CSE_475_Lab_2_Unsupervised_learning_Clustering _with_2Data.ipynb
   ```
> Keep `adult.csv` and `Mall_Customers_Clustering_Results.csv` in this directory so the notebook can load them correctly.
Workflow
The notebook follows a typical unsupervised-learning pipeline:
Load and inspect the dataset.
Clean the data and handle missing or unsuitable values.
Select features relevant to clustering.
Encode categorical columns and scale numerical features when required.
Apply clustering algorithms.
Determine or compare suitable cluster counts.
Visualize the identified groups.
Save and interpret the final results.
Results
The final clustering output is available in:
```text
Mall_Customers_Clustering_Results.csv
```
This file can be used to inspect the cluster assignment associated with each customer record.
Add Your Output Images
Export the key charts from the notebook and save them inside the `images/` folder. After adding the image files, remove the HTML comments below to show the outputs in GitHub.
<!--
### Elbow Method

![Elbow Method](images/elbow-method.png)

Use this plot to justify the selected number of clusters.

### Cluster Visualization

![Cluster Visualization](images/cluster-visualization.png)

This visualization shows how the clustering model separates records into distinct groups.

### Result Summary

![Result Summary](images/result-summary.png)
Key Findings
Add the exact conclusions from the notebook here after checking the final outputs. For example:
Number of clusters selected: [Add value from notebook]
Features used for clustering: [Add feature names]
Best observed pattern: [Add your interpretation]
Output file: `Mall_Customers_Clustering_Results.csv`
> Replace the bracketed placeholders with the actual values produced by your notebook. This keeps the README accurate and prevents unsupported claims.
Reproducibility Notes
Run every notebook cell in sequence.
Do not rename or move the included CSV files unless you also update their paths in the notebook.
Set a fixed `random_state` in clustering code when reproducible results are required.
Keep versions of Python libraries consistent if results need to match exactly.
Author
Afnan Bd  
East West University  
Course: CSE 475
