# Project Title

Write the project title clearly and concisely.
Example: Official PyTorch implementation of **"<Paper / Research Title>"**.

📄 **Paper**: [<your paper link, example:](https://www.researchgate.net/)

---

## 📋 Overview

**Section Description** This section provides a concise overview of the project. What to Include:
- Brief background information.
- Research or project objectives.
- Main dataset or data sources.
- Methods, frameworks, or algorithms used.
- Expected results or contributions.

### Key Features

**Section Description** This section highlights the main functionalities, capabilities, or technical components of the project. The goal is to help readers quickly understand what the system can do and what makes it technically useful. What to include:
- Main functionalities of the system.
- Important technical features.
- (optional if Big data) AI/ML capabilities.
- (optional if any) Automation or integration features.
- Performance-related features.

For Example:
- **Novel Task**: GFD requires recognizing both labeled and unlabeled known identities (IDs) while simultaneously discovering new, previously unseen IDs
- **Dynamic Prefix Generation**: Instance-specific feature extractors using lightweight, layer-wise prefixes generated on-the-fly by a HyperNetwork
- **State-of-the-art Performance**: Significantly outperforms existing GCD methods and ArcFace baseline on fine-grained face recognition tasks
- **High Cardinality Support**: Handles hundreds or thousands of visually similar face IDs effectively

---

## 🎯 Key Contributions

**Section Description** This section explains the main research, technical, or practical contributions of the project. This section focuses on:
- Novelty,
- Research contributions,
- Methodological improvements,
- Technical innovations,
- Real-world impact.

For example:
1. **Generalized Face Discovery (GFD)**: A new task formulation that bridges face identification and clustering in open-world scenarios
2. **Dynamic Prefix Mechanism**: HyperNetwork-based prefix generators that create instance-specific feature extractors without massive model capacity
3. **Comprehensive Benchmarks**: Six GFD benchmark datasets (YTF-500/1000/2000, CASIA-500/1000/2000)
4. **Strong Generalization**: Competitive performance on generic GCD benchmarks (CIFAR-100, ImageNet-100, CUB, etc.)

---

## Project Architecture / Research Workflow

**Section Description** This section explains the overall research pipeline or system workflow. What to include:
- Research methodology flowcharts
- Architecture diagram
- System pipeline illustrations

Can be include:
- Input and output for each stage.
- Tools, frameworks, or technologies used.
- Data flow or system process

For Example:

<p align="center">
  <img src="examplearchitecture.png" alt="Architecture of Your Project">
</p>

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
