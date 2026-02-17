### Report for HW 1
# Deep Learning Architecture Comparison

## Objective and Learning Outcomes

### Assignment Objective
Compare and evaluate three neural network architectures (MLP, CNN, and Vision Transformer) across three diverse datasets to understand how architectural design choices impact performance based on data modality and structure.

### NOTE: Setup.md includes how to setup all of the workflow and where the files are located. File ML_HW.ipynb contains all of the outputs, graphs, training time, evaluation outputs, and graphs. The work was originally compiled into the ipynb file, but now also restructured into seperate files for clarity on this git page. 

### Introduction and Objectives
Deep learning architectures have different strengths depending on the structure and characteristics of the data they are applied to. Selecting an appropriate model is therefore a critical step in building effective machine learning systems. While modern architectures such as convolutional neural networks (CNNs) and transformer-based models have achieved strong performance across many domains, their effectiveness is largely determined by how well their underlying inductive biases align with the data modality. Understanding these relationships is essential for both practical model selection and for developing intuition about how neural networks learn.

The purpose of this assignment is to compare multiple neural network architectures across datasets with different data types and levels of spatial structure. Specifically, a Multilayer Perceptron (MLP), a Convolutional Neural Network (CNN), and a Vision Transformer (ViT) were evaluated on tabular data (Adult Income), natural images (CIFAR-10), and medical histopathology images (PatchCamelyon). To ensure a fair comparison, consistent training procedures, data splits, and evaluation metrics were used across experiments. Rather than focusing only on accuracy, the assignment emphasized understanding why certain architectures work better for certain problems.

This study aims to analyze how architectural design choices influence model performance, training efficiency, and generalization. By examining accuracy, F1-score, and qualitative behavior across tasks, the experiments provide insight into when simpler architectures are sufficient and when specialized structures such as convolution or attention mechanisms become necessary. The results highlight the importance of matching model inductive bias to the underlying data characteristics in order to achieve reliable and efficient learning.Deep learning architectures have different strengths depending on the structure and characteristics of the data they are applied to. Selecting an appropriate model is therefore a critical step in building effective machine learning systems. While modern architectures such as convolutional neural networks (CNNs) and transformer-based models have achieved strong performance across many domains, their effectiveness is largely determined by how well their underlying inductive biases align with the data modality. Understanding these relationships is essential for both practical model selection and for developing intuition about how neural networks learn.

### Learning Outcomes

Through this work, several key learning outcomes were achieved. First, the experiments demonstrated the importance of matching the model architecture to the data type. Second, the assignment provided hands-on experience with building training pipelines, evaluating models using appropriate metrics, and maintaining consistent experimental settings for fair comparison. Finally, the project helped develop intuition about trade-offs between model complexity, training time, and generalization performance.

General Outcomes: 
1. **Architecture-Data Alignment**: Understand which architectures are best suited for different data types (tabular, natural images, medical images)
2. **Inductive Bias**: Learn how built-in assumptions (e.g., CNN's spatial locality, Transformer's attention) affect learning efficiency and generalization
3. **Practical Implementation**: Gain hands-on experience implementing, training, and evaluating deep learning models using PyTorch
4. **Experimental Methodology**: Develop skills in fair model comparison through consistent hyperparameters, proper data splits, and appropriate metrics
5. **Critical Analysis**: Interpret results to understand trade-offs between model complexity, training time, and performance

## Code Design and Modularity

### Architecture Overview - Code and Design Modularity
The code was structured in a modular way to make it easier to understand, reuse, and extend. Each major component of the workflow was separated into logical parts, including data loading, model, a config file, training, and evaluation. This separation allows individual components to be modified or replaced without affecting the entire pipeline.

The main experimentation and visualizations were originally implemented in a Jupyter Notebook (ML_HW.ipynb) to support iterative development and analysis. For better organization and clarity, the workflow was later restructured into separate scripts and modules. This design approach follows common deep learning practices and makes the project easier for others to explore, reproduce, and adapt for their own experiments. This is shown in the folders within this page. 

### Datasets and Architectures
## Three datasets were used to represent different data modalities:

Adult Income – Tabular data with structured numerical and categorical features

CIFAR-10 – Natural images (32×32 RGB) across 10 object classes

PatchCamelyon (PCam) – Medical histopathology images for tumor detection

To evaluate architectural differences, three model types were tested:

MLP – Fully connected network without spatial assumptions

CNN – Uses convolutional layers to capture spatial patterns and local features

Vision Transformer (ViT) – Uses self-attention to model global relationships across image patches

All models were trained using consistent preprocessing, data splits, and evaluation metrics (Accuracy and F1-score) to ensure fair comparison.

### Outcomes and Interpretation

## Results Summary

| Dataset | Architecture | Accuracy | F1-Score | Notes |
|---------|-------------|----------|----------|-------|
| **Adult Income** | MLP | 0.859 | 0.688 | Best for tabular data |
| **Adult Income** | CNN | 0.856 | 0.676 | Suboptimal - no spatial structure |
| **CIFAR-10** | MLP | 0.217 | 0.215 | Poor - loses spatial information |
| **CIFAR-10** | CNN | 0.384 | 0.387 | Better but needs deeper architecture |
| **PCam** | MLP | 0.499 | 0.000 | Failed - cannot capture patterns |
| **PCam** | CNN | 0.942 | 0.910 | Excellent - captures tissue structures |

### Key Insights

**Adult Income (Tabular):**
- MLP achieves 85.9% accuracy - optimal for non-spatial features
- CNN performs similarly (85.6%) but with unnecessary complexity
- F1-scores (0.68-0.67) reflect class imbalance (76% ≤50K, 24% >50K)

**CIFAR-10 (Natural Images):**
- MLP fails catastrophically (21.7%) - flattening destroys spatial relationships
- CNN reaches 38.4% - modest performance suggests need for deeper architecture
- Both well below random baseline would be 10% (10 classes)

**PatchCamelyon (Medical Images):**
- MLP at 49.9% accuracy (F1=0.000) indicates random guessing - model predicts single class
- CNN excels at 94.2% accuracy (F1=0.910) - successfully learns tissue patterns
- High F1-score confirms balanced performance across tumor/normal classes

### Architecture Effectiveness

| Data Type | Best Architecture | Worst Architecture | Reason |
|-----------|------------------|-------------------|---------|
| Tabular | **MLP** | CNN | No spatial structure to exploit |
| Natural Images | **CNN** | MLP | Spatial locality and translation invariance critical |
| Medical Images | **CNN** | MLP | Local tissue patterns require convolutional filters |

### My Interpretation of Results
The results demonstrate that model performance is strongly influenced by the alignment between the network architecture and the structure of the data. For tabular data (Adult Income), the MLP achieved the best performance because fully connected layers are well suited for independent feature relationships without spatial dependencies. The CNN produced similar accuracy but did not provide any meaningful advantage, highlighting that convolutional inductive biases are unnecessary when no spatial structure exists. The relatively lower F1-scores for both models reflect the class imbalance in the dataset rather than a limitation of the architectures themselves.

For image-based datasets, the importance of spatial feature learning becomes evident. On CIFAR-10, the MLP performed poorly because flattening the images removes spatial relationships between pixels, making it difficult for the model to learn meaningful visual patterns. The CNN improved performance by leveraging local receptive fields and weight sharing, although the modest accuracy suggests that a deeper or more advanced architecture would be required for competitive results on this dataset. This comparison reinforces the limitation of simple architectures for complex visual recognition tasks.

The contrast is most pronounced in the medical imaging task (PCam), where the CNN achieved very high accuracy and F1-score, while the MLP effectively failed by predicting a single class. This outcome highlights the critical role of convolutional feature extraction in detecting subtle local patterns such as tissue structures and tumor regions. Overall, the experiments emphasize that architectural inductive bias, rather than model complexity alone, is the key factor in achieving strong performance. Selecting an architecture that matches the data modality leads to better generalization, more efficient learning, and more reliable results.
