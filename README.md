# Impact of Low-Light Data Augmentation on Emotion Recognition Robustness

This repository contains the implementation and experimental results for a study on the robustness of facial emotion recognition (FER) systems under significant illumination shifts. The research utilizes an EfficientNetB0 backbone to quantify performance degradation in low-light environments and evaluates mitigation strategies through consistent domain exposure and unsupervised adaptation.

## Abstract

Facial emotion recognition (FER) systems deployed in real-world environments often face uncontrolled lighting conditions that differ significantly from standard training benchmarks. This study systematically evaluates the robustness of the EfficientNetB0 architecture on the Emotions Dataset under controlled low-light variations. We demonstrate that while a baseline model achieves 85.0% accuracy on clean data, it suffers a catastrophic 31% drop to 54.0% when exposed to mild low-light test conditions. Furthermore, we find that standard geometric augmentation (rotation, zoom) fails to improve robustness to lighting-based domain shifts. Through failure analysis using Gradient-weighted Class Activation Mapping (Grad-CAM), we show that the model loses semantic facial features in the dark, instead anchoring predictions on high-contrast background artifacts. Finally, we propose a consistent low-light training strategy that recovers performance to 69.3%, significantly improving the recognition of 'Sad' and 'Happy' expressions in challenging lighting.

## Experimental Results

The following table summarizes the validation accuracy across the five controlled experimental setups:

| Model | Training Strategy | Test Condition | Accuracy |
| :--- | :--- | :--- | :--- |
| **Model A** | Baseline (No Augmentation) | Normal | 85.0%  |
| **Model B** | Standard Geometric | Normal | 82.1%  |
| **Model C** | Aggressive Low-Light Augmentation | Low-Light | 70.7%  |
| **Model D** | Baseline (Model A) | Low-Light | 54.0%  |
| **Model E** | Consistent Mild Low-Light | Low-Light | 69.3%  |

## Key Methodologies

### Low-Light Simulation Pipeline
To mimic realistic nighttime conditions, we implemented a custom preprocessing pipeline with the following parameters:
* ]Gamma Correction: 1.3 
* Brightness Reduction: 0.9 multiplicative factor 
* Sensor Noise: Gaussian noise with sigma = 0.03 

### Unsupervised Adaptation (AdaBN)
As an alternative to retraining, we investigated Adaptive Batch Normalization (AdaBN). By updating only the Batch Normalization statistics on the low-light test stream for a single epoch, we observed an accuracy recovery of +4.9% (improving performance from 37.9% to 42.8% without labels). This implies that much of the performance drop is due to covariate shift in feature statistics rather than a total loss of feature extraction capability.

## Visual Interpretability
Using Grad-CAM, we localized the model's attention during failure cases. In low-light images, the network ignores semantic landmarks like the mouth and eyebrows, instead focusing on high-contrast regions such as neck shadows and clothing collars.

## Author
Salaheddine Boumazough |
Computer Science |
Al Akhawayn University / University of Helsinki Exchange
