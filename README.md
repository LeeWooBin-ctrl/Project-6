# Transformers for Medical AI(Project 6)

## Project Summary
We utilized the CheXpert dataset, a large collection of chest X-rays, to explore the diagnostic performance of three Vision Transformer models—ViT, BEiT, and Swin. These models were trained to classify 14 labeled thoracic diseases, and we further aimed to visualize the model’s decision-making process using heatmaps that highlight the affected areas in the images.

## Project Result
Please refer to our GitHub page for detailed explanation  
Click Here 👉[GitHub Page Link](https://leewoobin-ctrl.github.io/Project-6/)

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
