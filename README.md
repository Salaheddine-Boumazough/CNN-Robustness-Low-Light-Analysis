# Robustness Analysis of EfficientNetB0 to Domain Shift in Low-Light Conditions

## Description
This repository contains a research-focused study on the robustness of Convolutional Neural Networks (CNNs) when subjected to environmental domain shifts. Specifically, the project evaluates the EfficientNetB0 architecture's performance degradation in low-light conditions and explores the efficacy of targeted data augmentation as a mitigation strategy.

## Abstract
This study investigates how luminosity-based domain shifts impact the predictive accuracy of deep learning models. Using the CIFAR-10 dataset as a benchmark, we simulated varying degrees of low-light environments to quantify the performance gap between standard training and real-world degraded conditions. Our findings indicate that while baseline models suffer significant accuracy loss, the integration of custom low-light augmentation pipelines restores model reliability and improves feature extraction stability.

## Key Features
* Architecture: Implementation of EfficientNetB0 pre-trained on ImageNet.
* Dataset: Evaluation conducted using the CIFAR-10 dataset.
* Domain Shift Simulation: Custom scripts to simulate mild and combined low-light conditions for inference testing.
* Robust Training: Implementation of a specialized data augmentation pipeline focusing on brightness and contrast variability.

## Methodology
The research follows a three-stage experimental design:
1. Baseline Evaluation: Testing a model trained on standard lighting against standard test sets.
2. Robustness Testing: Subjecting the baseline model to simulated low-light domain shifts to identify performance bottlenecks.
3. Mitigation and Retraining: Training a second iteration of the model with a low-light augmentation strategy to evaluate recovery in accuracy and precision.

## Performance Metrics
The performance was evaluated using confusion matrices and classification reports to analyze class-specific sensitivities to light degradation.

| Scenario | Model Version | Key Finding |
| :--- | :--- | :--- |
| Standard Lighting | Baseline | High baseline accuracy across all 10 classes. |
| Mild Low-Light | Baseline | Initial drop in recall for classes with low color contrast. |
| Combined Low-Light | Baseline | Significant accuracy degradation and increased error rates. |
| Combined Low-Light | Robust (Retrained) | Substantial recovery in accuracy and more balanced precision scores. |

## Installation
Ensure you have Python installed, then install the required dependencies:
```bash
pip install -r requirements.txt
