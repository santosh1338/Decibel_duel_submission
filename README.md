# Decibel Duel – Audio Classification & GAN Synthesis  
A machine learning project combining a **Convolutional Neural Network (CNN)** for labeled audio classification and a **Conditional Generative Adversarial Network (CGAN)** for generating synthetic audio spectrograms.  
This submission was developed for the *Decibel Duel* challenge, using a dataset containing **5 sound categories**.

---

## Project Overview

This repository contains two main components:

### **1. CNN-Based Audio Classifier (`CNN.py`)**
A convolutional neural network trained to classify audio clips into one of the following classes:

- dog_bark  
- drilling  
- engine_idling  
- siren  
- street_music  

#### **Features**
- Uses **Mel Spectrograms** + **MFCCs** as dual-channel input.  
- Includes **BatchNorm**, **AdaptiveAvgPool**, and **Dropout** for stability.  
- Train/validation split using `train_test_split`.  
- Outputs a `submission.csv` file mapping audio file IDs to predicted labels.  

---

### **2. Conditional GAN for Audio Generation (`GAN.py`)**
A CGAN trained on Mel-spectrograms to generate synthetic audio conditioned on class labels.

#### **Features**
- Generator and Discriminator both conditioned on one-hot labels.  
- Uses **Wasserstein GAN + Gradient Penalty (WGAN-GP)** for stable training.  
- Converts generated mel spectrograms back into waveform using:  
  - `InverseMelScale`  
  - `GriffinLim`  
- Saves generated audio per category at every epoch into `gan_generated_audio/`.

---
