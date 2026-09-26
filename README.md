# Breast Cancer Detection with Simple Neural Network (PyTorch)

A from-scratch implementation of a binary classification neural network using PyTorch to detect breast cancer (Malignant vs Benign) on the Wisconsin Breast Cancer dataset.

## Overview

This project implements a simple single-layer neural network (logistic regression) **without using `nn.Module` or optimizers**. Everything is built manually:

- Weights and bias initialization
- Forward pass with sigmoid activation
- Binary Cross-Entropy loss
- Manual gradient descent

## Dataset

- **Source**: [Wisconsin Breast Cancer Dataset](https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv)
- **Samples**: 569
- **Features**: 30 numerical features (mean, standard error, and worst values of radius, texture, perimeter, etc.)
- **Target**: `diagnosis` → `M` (Malignant) / `B` (Benign)

## Project Structure
