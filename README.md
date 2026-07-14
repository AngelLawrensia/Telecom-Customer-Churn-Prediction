# Telecom Customer Churn Prediction using Machine Learning

📄 **Paper**: Will be added after publication.

---

## 📋 Overview
Customer churn prediction plays an important role in helping telecommunication companies retain customers and reduce customer acquisition costs. This project develops and optimizes machine learning models to predict customer churn using the IBM Telco Customer Churn Dataset within the CRISP-DM framework.

The project compares five supervised learning algorithms—Logistic Regression, K-Nearest Neighbors (KNN), Naïve Bayes, Random Forest, and XGBoost. To improve predictive performance, the workflow incorporates SMOTE for handling class imbalance, GridSearchCV with refinement tuning for hyperparameter optimization, and cross-validation for model evaluation. Model interpretability is enhanced using SHAP (SHapley Additive exPlanations), while the best-performing model is deployed through a Streamlit web application for interactive customer churn prediction.

Experimental results show that Random Forest achieved the best overall performance after optimization, obtaining an Accuracy of 92.76%, Precision of 83.66%, Recall of 90.37%, F1-score of 86.89%, and ROC-AUC of 97.24%. This project demonstrates an end-to-end machine learning workflow that combines predictive performance, model interpretability, and practical deployment to support data-driven customer retention strategies in the telecommunications industry.

### Key Features
### Key Features

- **End-to-End Machine Learning Pipeline**  
  Implements a complete machine learning workflow based on the CRISP-DM methodology, covering data understanding, preprocessing, modeling, evaluation, interpretation, and deployment.

- **Multi-Model Performance Comparison**  
  Evaluates and compares Logistic Regression, K-Nearest Neighbors (KNN), Naïve Bayes, Random Forest, and XGBoost to identify the most effective model for customer churn prediction.

- **Class Imbalance Handling**  
  Applies Synthetic Minority Over-sampling Technique (SMOTE) to improve the model's ability to identify minority-class (churn) customers.

- **Hyperparameter Optimization**  
  Uses GridSearchCV and refinement tuning to optimize model performance and improve generalization.

- **Explainable AI (XAI)**  
  Integrates SHAP (SHapley Additive exPlanations) to explain feature contributions and improve prediction transparency.

- **Interactive Streamlit Application**  
  Provides a user-friendly web interface for predicting customer churn from CSV or Excel files, including both single and batch prediction modes.

- **Prediction Probability & Visualization**  
  Displays churn probability together with SHAP visualizations to help users understand prediction results.

- **High Prediction Performance**  
  The optimized Random Forest model achieved 92.76% Accuracy, 83.66% Precision, 90.37% Recall, 86.89% F1-score, and 97.24% ROC-AUC on the IBM Telco Customer Churn Dataset.

---

## 🎯 Key Contributions

1. **Comprehensive Machine Learning Comparison**  
   Conducted a comparative analysis of five supervised machine learning algorithms—Logistic Regression, K-Nearest Neighbors (KNN), Naïve Bayes, Random Forest, and XGBoost—for customer churn prediction in the telecommunications industry.

2. **Integrated Model Optimization**  
   Improved prediction performance by combining SMOTE for class imbalance handling with GridSearchCV and refinement hyperparameter tuning within a unified CRISP-DM workflow.

3. **Explainable AI Integration**  
   Enhanced model transparency by incorporating SHAP (SHapley Additive exPlanations), enabling interpretation of feature contributions and supporting more explainable decision-making.

4. **Practical Deployment**  
   Developed an interactive Streamlit-based application that allows users to upload customer data, generate churn predictions, view prediction probabilities, and explore SHAP explanations.
   
---

## Project Architecture / Research Workflow

<p align="center">
  <img src="images/research_workflow.png" alt="Research Workflow" width="900">
</p>

The research follows the CRISP-DM methodology, consisting of six stages: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment. The workflow begins with customer churn data collection from the IBM Telco Customer Churn Dataset, followed by preprocessing, machine learning model development, performance evaluation, SHAP-based model interpretation, and deployment through a Streamlit web application.

---

## 🚀 Installation

**Section Description** Provide instructions for setting up the project environment.

For Example:

### Requirements

- Python 3.8+
- PyTorch 2.3.1+
- CUDA 11.8+
- APEX (for mixed precision training)
- etc.

### Setup

1. Clone the repository:
```bash
git clone <repository-url, for example: https://github.com/yourusername/yourprojectname.git>
cd ProjectName
```

2. Create a conda environment:
```bash
conda create -n projectname python=3.8
conda activate projectname
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

### Name of The Dataset

**Section Description** Provide information about the dataset used in the project. The dataset must be primary (data obtained for example (not limited to) from surveys, interviews, observations, experiments, sensors, web scrapping/crawling, API collection), not secondary or downloaded from other repository. 
What to include:
- Dataset name.
- Dataset source.
- Dataset link.
- Total number of samples.
- Data structure.
- File format.
- Example samples.

For example:
We provide six benchmark datasets for evaluating the Generalized Face Discovery task:

| Dataset | Known IDs | Unknown IDs | Train Samples | Test Samples |
|---------|-----------|-------------|---------------|--------------|
| YTF-500 | 250 | 250 | 48,089 | 11,779 |
| YTF-1000 | 500 | 500 | 96,002 | 23,523 |
| YTF-2000 | 1,000 | 1,000 | 190,248 | 46,615 |
| CASIA-500 | 250 | 250 | 46,991 | 11,999 |
| CASIA-1000 | 500 | 500 | 89,508 | 22,867 |
| CASIA-2000 | 1,000 | 1,000 | 184,432 | 47,114 |

Or you can just give the link of the dataset if you already upload on the repository like Kaggle or Mendeley Dataset.

### Download

**The dataset can be obtained at the link**
 
**Download links:**
**Datasets**: [Dataset link on repository or Drive]

Explain the structue of the dataset, for example:

Once downloaded, organize the data as follows:
```
YourProject/
├── positive_review2000/
│   ├── train/
│   └── test/
├── negative_review2000/
│   ├── train/
│   └── test/
├── neutral_review1000/
│   ├── train/
│   └── test/

```
---

## 🏋️ Training

**Section Description** For big data, explain how to train the model. What to include:
- Training scripts.
- Training parameters.
- Hyperparameters.
- Execution commands.
- GPU/CPU requirements.

Example Structure:
```bash
python train.py --config configs/train.yaml
```

---

## 📊 Results

**Section Description** This section presents the final outcomes, outputs, or achievements of the project. The content of this section may vary depending on the project type.  The goal is to demonstrate what has been successfully developed, implemented, evaluated, or achieved.
What to include:
- Final project outcomes.
- System implementation results.
- Experimental or testing results.
- Visualization results like evaluation graphs, screenshots of the system or dashboard.
- Performance analysis.
- User testing or usability results.
- Comparison with baseline or previous systems or benchmark comparison.
- Deployment results.

You can use table or visualization. 

---

## 🏗️ Project Structure

**Section Description** Explain you project structure.

For Example: 

```
FaceGCD/
├── data_loader/              # Dataset loaders and augmentations
│   ├── augmentations/        # Data augmentation strategies
│   ├── youtube_faces_*.py    # YTF dataset loaders
│   ├── casia_webface_*.py    # CASIA dataset loaders
│   └── data_loaders.py       # Main data loading utilities
├── model/                    # Model architectures
│   ├── dino_vision_transformer.py  # DINO ViT backbone
│   ├── prefix_generator.py   # HyperNetwork-based prefix generator
│   ├── ViT_face.py           # Face-specific ViT components
│   └── mobilenet.py          # Landmark CNN
├── trainer/                  # Training utilities
│   ├── trainer.py            # Main training loop
│   └── faster_mix_k_means_pytorch.py  # Semi-supervised K-Means
├── utils/                    # Utility functions
│   ├── cluster_utils.py      # Clustering utilities
│   ├── losses.py             # Loss functions
│   └── dino_utils.py         # DINO-specific utilities
├── shell/                    # Shell scripts for experiments
├── train.py                  # Training script
├── extract_features.py       # Feature extraction script
├── SSK.py                    # Semi-supervised K-Means evaluation
└── requirements.txt          # Python dependencies
```

---

## 📝 Citation

**Section Description** Use this section if the project is associated with a publication or research paper. What to include:
- Citation format.
- DOI.
- BibTeX entry.

For Example:
If you find this work useful for your research, please cite:

```bibtex
@article{oh2025facegcd,
  title={...},
  authors={...},
  journal={...},
  year={...}
}
```

---

## 🙏 Acknowledgments

**Section Description** Provide acknowledgements to contributors or supporting organizations. What to include:
- (Mandatory) Big Data Lab, Information Systems Study Program, Universitas Multimedia Nusantara (UMN).
- (if any) Sponsors/Partners.
- (if any) Research groups.
- (if any) Dataset providers.
- (if any) Collaborators.

---

## 📧 Contact

**Section Description** Provide maintainer or author contact information.

For Example:
For questions or issues, please:
- Open an issue on GitHub
- Contact: youremail@mail.com

---

## 📜 License

**Section Description** Please state your project license of any. For example:

This project is released under the MIT License. See [LICENSE](LICENSE) file for details.

---

## Tips

You can just download this Read Me template and modify it. 
---
