# Transformers for Medical AI(Project 6)


## Project Summary
We utilized the CheXpert dataset, a large collection of chest X-rays, to explore the diagnostic performance of three Vision Transformer models—ViT, BEiT, and Swin. These models were trained to classify 14 labeled thoracic diseases, and we further aimed to visualize the model’s decision-making process using heatmaps that highlight the affected areas in the images.


## Results

We conducted a comparative evaluation of three transformer-based models—**ViT (Vision Transformer)**, **Swin Transformer**, and **BEiT**—on the CheXpert dataset. The models were trained and tested using both the full and downsampled (small) versions of the dataset. Performance was assessed using three key criteria:

---

### 1. Accuracy

- **BEiT** achieved the highest **test accuracy**, indicating the best generalization ability to unseen data.
- **ViT** recorded the highest **training accuracy**, but its **test accuracy dropped significantly**, showing signs of **overfitting**.
- **Swin Transformer** showed **moderate performance** overall, balancing between ViT and BEiT.

---

### 2. Loss

- **BEiT** maintained the **lowest loss** on both the training and test sets throughout training epochs.
- **ViT** and **Swin** either showed **plateauing or increasing test loss**, indicating reduced learning effectiveness or potential overfitting.

---

### 3. Attention Visualization (Interpretability)

We visualized attention maps from both ViT and BEiT using animated GIFs.

- **ViT** exhibited **fragmented and inconsistent attention patterns**, sometimes focusing on irrelevant image regions such as blank spaces or corners.
- **BEiT**, in contrast, showed **structured and lesion-focused attention**, with attention heads consistently highlighting clinically relevant regions.

These visual differences demonstrate that BEiT offers **superior interpretability** and aligns better with diagnostic needs in medical imaging.

---

### Conclusion

> **BEiT** demonstrated the most balanced and reliable performance among the three models.  
> Thanks to its **masked image modeling pretraining**, BEiT achieved better generalization, lower loss, and more interpretable attention visualizations.  
> It is, therefore, the most suitable model for **medical imaging diagnosis** in this evaluation.
For a more detailed analysis and accompanying graphs, please refer to Here 👉[GitHub Page Link](https://leewoobin-ctrl.github.io/Project-6/)


## Installation Steps

### 1. Visit Our GitHub Page
You can access our project via the following link:  
👉 [GitHub Page Link](https://leewoobin-ctrl.github.io/Project-6/)

---

### 2. Available Resources on the Page

The GitHub Page provides links to:

- The **CheXpert small dataset** and **full dataset**
- The implementation code for all **three transformer models** we used (ViT, Swin, and BEiT)
- The final presentation report in **PDF format**

---

### 3. Assets Folder in the Repository

Except for the datasets,  
all other files mentioned above are also included in the `/Assets` folder of the `Project-6` GitHub repository.

## Final Presentation Q&A
Q1. "How did you handle potential overfitting observed in ViT attention heatmaps?" 

A1. "Initially, the ViT model exhibited signs of overfitting when trained to predict only 4 labels, as the limited and binary classification task (negative/positive) introduced significant randomness, causing the model to rely on guesswork rather than genuine pattern learning. After properly training the model to predict all 14 labels—thus enriching the complexity and diversity of the training data—the ViT model showed substantially reduced overfitting. This comprehensive labeling encouraged the model to learn robust features rather than memorizing limited patterns, ultimately resolving the issue of overfitting observed earlier." 

Q2. “Can you provide insights into why BEiT achieved superior performance compared to ViT and Swin Transformers?”  

A2. "BEiT’s superior performance likely stems from its effective pre-training strategy, specifically masked image modeling, which encourages the model to learn robust and generalized representations from image data. Attention visualizations confirmed BEiT’s ability to consistently and precisely focus on clinically relevant regions, supporting the notion that BEiT develops better internal representations and decision-making capabilities tailored for medical diagnosis tasks."   

Q3. "Can you provide more details on the benchmark comparison of the three transformer models?"  

A3. "A comprehensive benchmark comparison of the three transformer-based models—ViT, Swin, and BEiT—was performed, evaluating their performance on accuracy, loss metrics, and interpretability through attention visualizations. A complete and detailed analysis of these comparisons, highlighting key strengths and weaknesses of each model, can be found our GitHub page(The BackGround part)"  


## Team Members and Their contributions
Zheng Hexing(2023311430) : Investigated the SWIN Transformer  
Chang Hwan Kim(2024321234) : Implemented heatmap visualizations; contributed to the GitHub repository  
Maftuna Ziyamova(2024311551) : Investigated Vision Transformers including ViT, BEIT, and SWIN; contributed to heatmap analysis  
Lee Woo Bin(2025311560) : Investigated the BEIT Transformer; created and submitted the GitHub repository
