# 🧬 RNA-seq Machine Learning Classification

This project applies machine learning to classify leukemia samples (ALL vs AML) using high-dimensional gene expression data.

## 📌 Objective
To demonstrate how classical ML models can be used to analyze transcriptomics data and identify biologically relevant features.

---

## 📊 Dataset
- Leukemia gene expression dataset (public benchmark)
- High-dimensional RNA-seq expression matrix
- Binary classification: ALL vs AML

---

## ⚙️ Methods
- Data preprocessing (standardization)
- Random Forest classifier
- Feature importance analysis
- Model evaluation using accuracy, confusion matrix

---

## 🧠 Key Techniques
- scikit-learn
- feature selection via tree-based importance
- dimensionality reduction (implicit via model learning)

---

## 📈 Results
- Classification accuracy: ~high (dataset-dependent)
- Identified top predictive gene features

---

## 🖼️ Visualizations

### Confusion Matrix
![Confusion Matrix](figures/confusion_matrix.png)

### Feature Importance
![Feature Importance](figures/feature_importance.png)

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/train.py
