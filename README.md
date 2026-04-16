# Enhanced Wide & Deep Binary Classification Neural Network
This network is a simple implication of the Wide and Deep Network, developed at an enhanced level by Google Research. The architecture mainly uses Logistic Regression and Neural Networks.

This model has four parts:
# 1. Wide branch
The wide branch is just a single linear layer applied directly to the categorical inputs after casting them to float. It converts the input, categorical feature IDs, into a single scalar per sample using a linear transformation, which helps capture memorization and some simple linear effects.

# 2. Deep branch with embeddings
Feature Embedding creates one embedding table per categorical feature.
2.1. A feature-attention block
2.2. A fusion and output head


