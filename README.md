### Report for HW 1
# Deep Learning Architecture Comparison

## Objective and Learning Outcomes

### Assignment Objective
Compare and evaluate three neural network architectures (MLP, CNN, and Vision Transformer) across three diverse datasets to understand how architectural design choices impact performance based on data modality and structure.

Setup.md includes how to setup all of the workflow and where the files are located. 

### Learning Outcomes
1. **Architecture-Data Alignment**: Understand which architectures are best suited for different data types (tabular, natural images, medical images)
2. **Inductive Bias**: Learn how built-in assumptions (e.g., CNN's spatial locality, Transformer's attention) affect learning efficiency and generalization
3. **Practical Implementation**: Gain hands-on experience implementing, training, and evaluating deep learning models using PyTorch
4. **Experimental Methodology**: Develop skills in fair model comparison through consistent hyperparameters, proper data splits, and appropriate metrics
5. **Critical Analysis**: Interpret results to understand trade-offs between model complexity, training time, and performance

## Code Design and Modularity

### Architecture Overview
The codebase follows a modular design pattern separating concerns for maintainability and reusability:

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

