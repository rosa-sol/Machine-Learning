### Report for HW 1
# Deep Learning Architecture Comparison

## Objective and Learning Outcomes

### Assignment Objective
Compare and evaluate three neural network architectures (MLP, CNN, and Vision Transformer) across three diverse datasets to understand how architectural design choices impact performance based on data modality and structure.

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

Dataset	Architecture	Accuracy	F1	Notes
0	Adult	MLP	0.859	0.6880	
1	Adult	CNN	0.856	0.6760	
2	CIFAR-100	MLP	0.217	0.2150	
3	CIFAR-100	CNN	0.384	0.3870	
4	PCAM	MLP	0.499	0.0000	
5	PCAM	CNN	0.942	0.9097	

