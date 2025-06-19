# Transformers for Medical AI (Project 6)


We utilized the CheXpert dataset, a large collection of chest X-rays, to explore the diagnostic performance of three Vision Transformer models—ViT, BEiT, and Swin. These models were trained to classify 14 labeled thoracic diseases, and we further aimed to visualize the model’s decision-making process using heatmaps that highlight the affected areas in the images.


## Results

We conducted a comparative evaluation of three transformer-based models—**ViT (Vision Transformer)**, **Swin Transformer**, and **BEiT**—on the CheXpert dataset. The models were trained and tested using both the full and downsampled (small) versions of the dataset. 

![accuracy and loss](static/figure1.png)
![full vs downsampled](static/figure2.png)

> For a more detailed analysis and accompanying graphs, please refer to Here 👉[GitHub Page Link](https://leewoobin-ctrl.github.io/Project-6/)


---


## Project Installation Guide

Follow these steps to set up and run the project.

---

### Step 1: Dataset Download

### Downsampled Version

You can directly download the downsampled dataset from within the provided notebooks:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("ashery/chexpert")
print("Path to dataset files:", path)
 ```

 ### Full Version

The full dataset requires additional steps:

1. **Create an account** at [Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2).

2. **Install AzCopy**:
   Follow the guide at [Microsoft AzCopy Documentation](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10?tabs=dnf).

3. **Download the dataset**:
   Use the following AzCopy command:

     ```bash
     azcopy copy "https://aimistanforddatasets01.blob.core.windows.net/chexpertchestxrays-u20210408?sv=2019-02-02&sr=c&sig=26ZfteXX4RAEavJMrR1T%2BcNlzl7ljkVBziRWWifFLAg%3D&st=2025-06-19T04%3A45%3A11Z&se=2025-07-19T04%3A50%3A11Z&sp=rl" "<local-destination-path>" --recursive
     ```

Replace `<local-destination-path>` with your desired local directory path.

> **Note**: The dataset size is approximately **471.17 GB**, and the download may take considerable time.

### Step 2: Install Required Packages

### Using Conda

```
conda create --name project6 --file requirements.txt python=3.12
conda activate project6
conda install --file requirements.txt
```

### Using pip

```
python3 -m venv project6
source project6/bin/activate 
pip install -r requirements.txt
```

### Step 3: Run Jupyter Notebooks
Proceed with running the Jupyter notebooks located in the `notebooks` folder to train and evaluate the model.



### Step 4. Visit Our GitHub Page
You can access our project via the following link to learn more:  
👉 [GitHub Page Link](https://leewoobin-ctrl.github.io/Project-6/)

---

## Final Presentation Q&A
> Q1. "How did you handle potential overfitting observed in ViT attention heatmaps?" 

> A1. "Initially, the ViT model exhibited signs of overfitting when trained to predict only 4 labels, as the limited and binary classification task (negative/positive) introduced significant randomness, causing the model to rely on guesswork rather than genuine pattern learning. After properly training the model to predict all 14 labels—thus enriching the complexity and diversity of the training data—the ViT model showed substantially reduced overfitting. This comprehensive labeling encouraged the model to learn robust features rather than memorizing limited patterns, ultimately resolving the issue of overfitting observed earlier." 

> Q2. “Can you provide insights into why BEiT achieved superior performance compared to ViT and Swin Transformers?”  

> A2. "BEiT’s superior performance likely stems from its effective pre-training strategy, specifically masked image modeling, which encourages the model to learn robust and generalized representations from image data. Attention visualizations confirmed BEiT’s ability to consistently and precisely focus on clinically relevant regions, supporting the notion that BEiT develops better internal representations and decision-making capabilities tailored for medical diagnosis tasks."   

> Q3. "Can you provide more details on the benchmark comparison of the three transformer models?"  

> A3. "A comprehensive benchmark comparison of the three transformer-based models—ViT, Swin, and BEiT—was performed, evaluating their performance on accuracy, loss metrics, and interpretability through attention visualizations. A complete and detailed analysis of these comparisons, highlighting key strengths and weaknesses of each model, can be found our GitHub page(The Background part)"  


## Contributions
- **Zheng Hexing** (2023311430) - Investigated the Swin Transformer  
- **Chang Hwan Kim** (2024321234) - Implemented heatmap visualizations; contributed to the GitHub repository  
- **Maftuna Ziyamova** (2024311551) - Investigated Vision Transformers including ViT, BEIT, and Swin; contributed to heatmap analysis; presented the project; overviewed and improved the website/repo
- **Lee Woo Bin** (2025311560) - Investigated the BEIT Transformer; created and submitted the GitHub repository
