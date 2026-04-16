# Enhanced Wide & Deep Binary Classification Neural Network
This network is a simple implication of the Wide and Deep Network, developed at an enhanced level by Google Research. The architecture mainly uses Logistic Regression and Neural Networks.

This model has four parts:
# 1. Wide Component
The wide branch is just a single linear layer applied directly to the categorical inputs after casting them to float. It converts the input, categorical feature IDs, into a single scalar per sample using a linear transformation, which helps capture memorization of straightforward patterns and some simple linear effects.

# 2. Deep Component

### 2.1 Feature Representation
To build a more informative input representation for downstream learning. The deep component converts categorical information into richer feature representations and combines it with numerical information. After feature combination, the model learns complex relationships among variables.
### 2.2 Nonlinear Learning
To capture deeper interactions that are difficult to model with simple direct rules.

---


