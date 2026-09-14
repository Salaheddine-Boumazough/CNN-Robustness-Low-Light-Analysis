# CNN Robustness Under Low-Light Illumination

This project studies a practical question in facial emotion recognition: how much performance is lost when a model trained on normally lit images is evaluated under darker, noisier conditions?

I trained four EfficientNetB0 classifiers on the same three-class dataset (`angry`, `happy`, and `sad`). The architecture, data split, optimizer settings, and evaluation procedure were kept fixed. The main experimental variable was the augmentation used during training.

The main result is that targeted low-light augmentation improved performance under moderate and severe illumination shifts without reducing clean-test accuracy. Standard geometric augmentation did not provide the same benefit. I also tested one-pass Adaptive Batch Normalization (AdaBN), but it reduced accuracy in this frozen-backbone setting.

## Experimental setup

- **Backbone:** EfficientNetB0 pretrained on ImageNet and kept frozen
- **Input size:** 224 x 224 RGB
- **Classes:** angry, happy, sad
- **Training images:** 6,799
- **Held-out test images:** 2,278
- **Validation split:** 20% of the training set
- **Batch size:** 32
- **Random seed:** 42
- **Environment used for the reported runs:** TensorFlow 2.20.0 on Google Colab

| Model | Training augmentation | Purpose |
|---|---|---|
| **A** | None | Clean baseline |
| **B** | Horizontal flip, +/-10 degree rotation, 10% translation, and 10% zoom | Standard geometric augmentation |
| **C** | Random gamma, exposure reduction, and Gaussian noise | Low-light-specific augmentation |
| **D** | Low-light and geometric transformations | Combined ablation |

For Model C, low-light augmentation was applied to 75% of training images. Gamma was sampled from 1.1 to 1.8, exposure from 0.60 to 0.95, and Gaussian noise standard deviation from 0.00 to 0.03 on the normalized `[0, 1]` scale. Validation and test images were not augmented during training.

## Paired robustness evaluation

Every model was evaluated on the same test images under four deterministic conditions. The Gaussian noise field and image order were shared across models, so differences are paired at the image level.

| Condition | Gamma | Exposure | Noise standard deviation |
|---|---:|---:|---:|
| Clean | 1.0 | 1.00 | 0.00 |
| Mild | 1.2 | 0.85 | 0.01 |
| Moderate | 1.5 | 0.70 | 0.02 |
| Severe | 1.8 | 0.55 | 0.03 |

The corruption is applied after scaling pixels to `[0, 1]`:

```text
corrupted = clip(exposure * image^gamma + Gaussian noise, 0, 1)
```

## Results

| Model | Clean accuracy | Mild accuracy | Moderate accuracy | Severe accuracy | Mean corrupted accuracy | Severe macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| **A: none** | **81.69%** | 81.39% | 77.96% | 71.42% | 76.92% | 69.31% |
| **B: geometry** | 78.05% | 76.08% | 76.21% | 69.71% | 74.00% | 68.43% |
| **C: low light** | 81.61% | **82.31%** | **82.88%** | **76.87%** | **80.68%** | **74.94%** |
| **D: combined** | 78.23% | 77.57% | 79.68% | 73.84% | 77.03% | 72.01% |

Model C retained essentially the same clean accuracy as Model A and performed better as the illumination shift became stronger. Its advantage over the baseline was:

- **Moderate:** +4.92 percentage points, 95% paired bootstrap CI `[+3.42, +6.45]`
- **Severe:** +5.44 percentage points, 95% paired bootstrap CI `[+3.77, +7.07]`

The confidence intervals were calculated with 2,000 paired bootstrap resamples. Model D also outperformed Model B under moderate and severe corruption, but its clean accuracy was lower than Models A and C. For that reason, Model C is the strongest overall model in this experiment, while Model D is retained as a useful combined-augmentation ablation.

## AdaBN result

I adapted Model A using one unlabeled pass over 1,359 mixed low-light images. Only the moving statistics in 49 Batch Normalization layers were updated; model weights and labels were not used.

AdaBN did not improve this model. Accuracy changed from 81.69% to 71.38% on clean images and from 71.42% to 69.49% under severe corruption. The paired confidence intervals were below zero at every test condition. One likely explanation is that updating the Batch Normalization statistics shifted the frozen EfficientNet representation without retraining the classifier head.

This negative result is kept because it helps define the limits of simple post-training adaptation in the chosen setup.

## Grad-CAM analysis

Grad-CAM was used as a qualitative comparison between Models A and C. Heatmaps were computed from EfficientNetB0's `top_conv` layer using the true-class logit as the target.

The notebook contains two sets of examples:

1. A fixed, class-balanced set selected without using severe-condition predictions.
2. Severe cases where Model C was correct and Model A was wrong, selected using the median true-class probability gain for each class.

In the selected recovery cases, Model C increased the true-class probability by 0.42 for `angry`, 0.40 for `happy`, and 0.52 for `sad`. Its activation often covered broader central facial regions, but several maps remained diffuse or included image boundaries. The heatmaps are therefore treated as illustrative evidence rather than a quantitative or causal result.

## Notebooks

Run the notebooks in this order. Each link opens directly in Google Colab.

| Step | Notebook | Colab |
|---:|---|---|
| 1 | [BaseModel_withoutAugmentation.ipynb](./BaseModel_withoutAugmentation.ipynb) | [Run Model A](https://colab.research.google.com/github/Salaheddine-Boumazough/CNN-Robustness-Low-Light-Analysis/blob/main/BaseModel_withoutAugmentation.ipynb) |
| 2 | [BaseModel_withAugmentation.ipynb](./BaseModel_withAugmentation.ipynb) | [Run Model B](https://colab.research.google.com/github/Salaheddine-Boumazough/CNN-Robustness-Low-Light-Analysis/blob/main/BaseModel_withAugmentation.ipynb) |
| 3 | [BaseModel_with_LowLight_Augmentation.ipynb](./BaseModel_with_LowLight_Augmentation.ipynb) | [Run Model C](https://colab.research.google.com/github/Salaheddine-Boumazough/CNN-Robustness-Low-Light-Analysis/blob/main/BaseModel_with_LowLight_Augmentation.ipynb) |
| 4 | [BaseModel_with_combined_lowlight.ipynb](./BaseModel_with_combined_lowlight.ipynb) | [Run Model D](https://colab.research.google.com/github/Salaheddine-Boumazough/CNN-Robustness-Low-Light-Analysis/blob/main/BaseModel_with_combined_lowlight.ipynb) |
| 5 | [LowLight_Robustness_Evaluation.ipynb](./LowLight_Robustness_Evaluation.ipynb) | [Run paired evaluation](https://colab.research.google.com/github/Salaheddine-Boumazough/CNN-Robustness-Low-Light-Analysis/blob/main/LowLight_Robustness_Evaluation.ipynb) |
| 6 | [AdaBN.ipynb](./AdaBN.ipynb) | [Run AdaBN](https://colab.research.google.com/github/Salaheddine-Boumazough/CNN-Robustness-Low-Light-Analysis/blob/main/AdaBN.ipynb) |
| 7 | [GradCam.ipynb](./GradCam.ipynb) | [Run Grad-CAM](https://colab.research.google.com/github/Salaheddine-Boumazough/CNN-Robustness-Low-Light-Analysis/blob/main/GradCam.ipynb) |

## Running the project

The dataset is not included in this repository. The notebooks expect it in Google Drive at:

```text
/content/drive/MyDrive/Emotions Dataset/
    train/
        angry/
        happy/
        sad/
    test/
        angry/
        happy/
        sad/
```

Trained models and generated tables, JSON summaries, and figures are saved under:

```text
/content/drive/MyDrive/CNN-Robustness-Low-Light-Analysis/outputs/
```

Models A-D must be run before the robustness, AdaBN, and Grad-CAM notebooks because the later notebooks load the saved model files.

## Scope and limitations

- The experiment uses one dataset and three emotion classes.
- The low-light conditions are synthetic and should not be treated as a replacement for evaluation on real nighttime images.
- The reported training runs use one seed. The paired test comparisons reduce evaluation noise, but repeated training seeds would be needed for stronger claims about training variability.
- Grad-CAM provides coarse qualitative localization and does not prove that a highlighted region caused a prediction.
- AdaBN was evaluated as a simple one-pass baseline; the negative result does not rule out other test-time adaptation methods.

## Author

Salaheddine Boumazough  
BSc Computer Science, Artificial Intelligence specialization  
Al Akhawayn University in Ifrane
