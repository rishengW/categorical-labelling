# Enhanced Wide & Deep Binary Classification Neural Network
This network is a simple application of the Wide and Deep Network, which is for the specified binary classification task, developed by Google Research. The architecture mainly incorporates the improved feature embedding and the refined attention block.

This model has four parts:
# 1. Wide Component
The wide branch is just a single linear layer applied directly to the categorical inputs after casting them to float. It converts the input, categorical feature IDs, into a single scalar per sample using a linear transformation, which helps capture memorization of straightforward patterns and some simple linear effects.

---

# 2. Deep Component

### 2.1 Feature Representation 
To build a more informative input representation for downstream learning. The deep component converts categorical information into richer feature representations and combines it with numerical information through the Feature Embedding technique. After feature combination, the model learns complex relationships among variables.
### 2.2 Nonlinear Learning
To capture deeper interactions that are difficult to model with simple direct rules.

---

# 3. Attention-based Refinement
To improve the quality of the learned representation and emphasize useful information in the feature space. This part refines the combined feature representation before further learning, which helps the model produce a more expressive and discriminative representation. Additionally, the normalization layer is used to improve stability and reliability.

---

## 4. Feature Fusion

### 4.1 Combination of Wide and Deep Information
This part integrates the outputs of the wide and deep components to combine simple memorized patterns with complex generalized patterns.
