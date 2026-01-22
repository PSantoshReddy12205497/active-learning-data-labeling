# Automated Data Labeling with Active Learning

This project implements an automated data labeling workflow using uncertainty-based active learning.  
It demonstrates how selecting the most uncertain samples for labeling improves model performance compared to random sampling.


# Problem Statement

Manual data labeling is expensive and time consuming.  
Active learning reduces labeling effort by querying only the most informative samples for annotation.


# Approach

1. Trained an initial model on a small labeled dataset  
2. Used model uncertainty to select the next sample for labeling  
3. Simulated human labeling using a scripted oracle  
4. Retrained the model iteratively  
5. Compared performance with random sampling  


# Active Learning Strategy

Model: Logistic Regression  
Sampling Method: Uncertainty Sampling  
Uncertainty Metric : 1 − max(predicted class probability)
