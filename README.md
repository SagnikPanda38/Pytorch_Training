# 🧠 PyTorch Training — From Scratch to `nn.Module`

A hands-on **PyTorch training repository** that demonstrates how a binary classification neural network can be implemented in two different ways:

* 🔧 **From Scratch using PyTorch Autograd**
* ⚡ **Using PyTorch's `nn.Module` API**

Both implementations use the **Breast Cancer dataset** and perform the same fundamental machine-learning workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
Train / Test Split
   ↓
Feature Scaling
   ↓
Label Encoding
   ↓
NumPy → PyTorch Tensors
   ↓
Neural Network
   ↓
Forward Pass
   ↓
Loss Calculation
   ↓
Backpropagation
   ↓
Parameter Updates
   ↓
Evaluation
```

The main goal of this repository is to understand what PyTorch is doing **under the hood** before moving to higher-level APIs.

---

# 📌 Repository Overview

This repository contains two implementations of a simple binary classification neural network.

| Implementation                                 | Main Concept                     | PyTorch Abstraction                        |
| ---------------------------------------------- | -------------------------------- | ------------------------------------------ |
| `Untitled9.ipynb`                              | Neural Network from scratch      | Manual weights, bias, loss and updates     |
| `pytorch_training_pipeline_using_nn_module.py` | Neural Network using PyTorch API | `nn.Module`, `nn.Linear`, `BCELoss`, `SGD` |

Both approaches ultimately implement a model similar to:

```text
Input Features
      │
      ▼
Linear Transformation
      │
      ▼
Sigmoid Activation
      │
      ▼
Binary Prediction
```

---

# 🎯 Learning Objectives

This repository is designed to build an understanding of:

* PyTorch tensors
* Tensor operations
* Neural-network parameters
* Weights and bias
* Forward propagation
* Sigmoid activation
* Binary cross-entropy
* Automatic differentiation
* Backpropagation
* Gradient descent
* PyTorch `nn.Module`
* Optimizers
* Model evaluation

The two implementations allow the training process to be understood at both the **low level** and the **high level**.

---

# 📂 Repository Structure

```text
pytorch-training/
│
├── Untitled9.ipynb
│
├── pytorch_training_pipeline_using_nn_module.py
│
└── README.md
```

### `Untitled9.ipynb`

Implements the neural network manually using PyTorch tensors and autograd.

The model explicitly defines:

* Weights
* Bias
* Forward propagation
* Sigmoid
* Binary cross-entropy loss
* Gradient calculation
* Gradient descent parameter updates

### `pytorch_training_pipeline_using_nn_module.py`

Implements the same basic idea using PyTorch's neural-network abstractions:

* `nn.Module`
* `nn.Linear`
* `nn.Sigmoid`
* `nn.BCELoss`
* `torch.optim.SGD`

The uploaded Python implementation defines the model as a custom `nn.Module` with a linear layer followed by sigmoid activation.

---

# 📊 Dataset

Both projects use the **Breast Cancer Wisconsin dataset** loaded from:

```text
Breast-Cancer-Detection/data.csv
```

The dataset is loaded directly using Pandas:

```python
df = pd.read_csv(
    'https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv'
)
```

The dataset contains numerical diagnostic features used for binary classification.

Two unnecessary columns are removed:

```python
df.drop(
    columns=['id', 'Unnamed: 32'],
    inplace=True
)
```

---

# 🔄 Common Data Preprocessing

Both implementations follow approximately the same preprocessing pipeline.

## 1. Load Dataset

```python
df = pd.read_csv(...)
```

---

## 2. Remove Unnecessary Columns

```python
df.drop(
    columns=['id', 'Unnamed: 32'],
    inplace=True
)
```

---

## 3. Train-Test Split

The data is divided into training and testing sets:

```python
x_train, x_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:],
    df.iloc[:, 0],
    test_size=0.2
)
```

The split is:

```text
80% → Training
20% → Testing
```

---

## 4. Feature Scaling

`StandardScaler` is used to standardize the input features:

```python
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

The scaler is fitted on the training set and then applied to the test set.

---

## 5. Label Encoding

The target labels are converted into numerical values:

```python
y_train = LabelEncoder().fit_transform(y_train)
y_test = LabelEncoder().fit_transform(y_test)
```

This allows the classification target to be represented numerically.

---

## 6. Convert NumPy Arrays to PyTorch Tensors

The data is converted into tensors before being passed into the neural network.

Example:

```python
x_train_torch = torch.from_numpy(x_train).float()
x_test_torch = torch.from_numpy(x_test).float()

y_train_torch = torch.from_numpy(y_train).float()
y_test_torch = torch.from_numpy(y_test).float()
```

---

# 🔧 Implementation 1 — Neural Network From Scratch

## File

```text
Untitled9.ipynb
```

This notebook demonstrates how to implement a simple neural network without using `nn.Module`.

Instead of relying on PyTorch's predefined layers and optimizer, the model explicitly creates its parameters.

---

# 🧠 Custom Model

The model is defined as:

```python
class SimpleNN():

    def __init__(self, x):

        self.weights = torch.rand(
            x.shape[1],
            1,
            dtype=torch.float64,
            requires_grad=True
        )

        self.bias = torch.zeros(
            1,
            dtype=torch.float64,
            requires_grad=True
        )
```

The model contains:

```text
Weights
   +
Bias
```

The weights and bias have:

```python
requires_grad=True
```

This tells PyTorch's **autograd system** to track operations involving these parameters and calculate their gradients.

---

# ➡️ Forward Propagation

The forward pass is manually implemented:

```python
def forward(self, x):

    z = torch.matmul(x, self.weights) + self.bias

    y_pred = torch.sigmoid(z)

    return y_pred
```

Mathematically:

```text
z = XW + b
```

Then:

```text
ŷ = sigmoid(z)
```

So the complete model is:

```text
Input X
   │
   ▼
XW + b
   │
   ▼
Sigmoid
   │
   ▼
Prediction ŷ
```

---

# 📉 Custom Loss Function

The notebook also manually implements binary cross-entropy:

```python
def loss_function(self, y_pred, y):

    epsilon = 1e-7

    y_pred = torch.clamp(
        y_pred,
        epsilon,
        1 - epsilon
    )

    loss = (
        -y * torch.log(y_pred)
        -(1-y) * torch.log(1-y_pred)
    )

    return loss.mean()
```

The binary cross-entropy equation is:

```text
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

The loss is averaged across the training samples.

---

# 🔁 Manual Training Loop

The notebook uses:

```python
learning_rate = 0.1
epochs = 30
```

The model is trained through:

```python
for epochs in range(epochs):

    y_pred = model.forward(
        x_train_torch.double()
    )

    loss = model.loss_function(
        y_pred,
        y_train_torch
    )

    loss.backward()

    with torch.no_grad():

        model.weights -= (
            learning_rate *
            model.weights.grad
        )

        model.bias -= (
            learning_rate *
            model.bias.grad
        )

        model.weights.grad.zero_()
        model.bias.grad.zero_()
```

---

# 🔍 What Happens During Training?

Each iteration performs:

### 1. Forward Pass

```text
X → XW + b → Sigmoid → ŷ
```

### 2. Loss Calculation

```text
ŷ + y → Binary Cross-Entropy → Loss
```

### 3. Backpropagation

```python
loss.backward()
```

PyTorch automatically calculates:

```text
∂Loss/∂Weights
∂Loss/∂Bias
```

### 4. Gradient Descent

The parameters are manually updated:

```text
W = W - η × ∂L/∂W

b = b - η × ∂L/∂b
```

where:

```text
η = learning rate
```

---

# 🧹 Gradient Reset

After updating the parameters, the gradients are manually cleared:

```python
model.weights.grad.zero_()
model.bias.grad.zero_()
```

This is important because PyTorch accumulates gradients by default.

---

# 📊 Evaluation

The notebook evaluates the model on the test set:

```python
with torch.no_grad():

    y_pred = model.forward(
        x_test_torch.double()
    )

    y_pred = (y_pred > 0.9).float()

    accuracy = (
        (y_pred == y_test_torch)
        .float()
        .mean()
    )

    print(
        f'Accuracy: {accuracy.item()}'
    )
```

The notebook uses a **0.9 threshold** when converting sigmoid outputs into binary predictions.

---

# ⚡ Implementation 2 — Using `nn.Module`

## File

```text
pytorch_training_pipeline_using_nn_module.py
```

The second implementation performs the same general task using PyTorch's built-in neural-network abstractions.

Instead of manually creating the weights and bias, the model uses:

```python
nn.Linear()
```

The model is defined as:

```python
class MySimpleNN(nn.Module):

    def __init__(self, num_features):

        super().__init__()

        self.linear = nn.Linear(
            num_features,
            1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, features):

        out = self.linear(features)
        out = self.sigmoid(out)

        return out
```

The implementation therefore follows:

```text
Input
  │
  ▼
nn.Linear
  │
  ▼
nn.Sigmoid
  │
  ▼
Output
```

This is the same basic mathematical model as the manually implemented network.

---

# ⚙️ Training Configuration

The `nn.Module` implementation uses:

```python
learning_rate = 0.1
epochs = 25
```

The loss function is:

```python
loss_function = nn.BCELoss()
```

The optimizer is:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)
```

---

# 🔁 `nn.Module` Training Loop

The training process is:

```python
for epoch in range(epochs):

    y_pred = model(X_train_tensor)

    loss = loss_function(
        y_pred,
        y_train_tensor.view(-1, 1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

The workflow is:

```text
Forward Pass
     ↓
Loss Calculation
     ↓
Zero Gradients
     ↓
Backpropagation
     ↓
Optimizer Step
```

---

# 🆚 From Scratch vs `nn.Module`

The key purpose of having both files in the same repository is to see the difference between **manually implementing a training pipeline** and using PyTorch's higher-level abstractions.

| Component       | From Scratch         | `nn.Module`                     |
| --------------- | -------------------- | ------------------------------- |
| Model class     | Custom Python class  | `nn.Module`                     |
| Weights         | Manually created     | `nn.Linear`                     |
| Bias            | Manually created     | `nn.Linear`                     |
| Forward pass    | Manually implemented | Custom `forward()`              |
| Sigmoid         | `torch.sigmoid()`    | `nn.Sigmoid()`                  |
| Loss            | Manually calculated  | `nn.BCELoss()`                  |
| Gradients       | PyTorch autograd     | PyTorch autograd                |
| Gradient update | Manual               | `optimizer.step()`              |
| Gradient reset  | Manual               | `optimizer.zero_grad()`         |
| Optimizer       | Manual SGD equation  | `torch.optim.SGD`               |
| Training loop   | Manual               | Manual                          |
| Main purpose    | Understand internals | Learn standard PyTorch workflow |

---

# 🧩 What PyTorch Is Doing for You

The first implementation makes the training process more explicit.

For example:

```python
model.weights -= learning_rate * model.weights.grad
```

is essentially performing the parameter update manually.

In the second implementation, this responsibility is delegated to:

```python
optimizer.step()
```

Similarly, the first implementation manually defines the loss:

```python
-y * torch.log(y_pred) - (1-y) * torch.log(1-y_pred)
```

while the second uses:

```python
nn.BCELoss()
```

This makes the second implementation shorter and more scalable.

---

# 🧠 Conceptual Relationship

The two implementations can be viewed as two levels of abstraction.

```text
                 PYTORCH
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
   Low-Level                  High-Level
   Approach                   Approach
        │                       │
        ▼                       ▼
 Manual Weights             nn.Linear
 Manual Loss                nn.BCELoss
 Manual Update              SGD Optimizer
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
          Same Core ML Concepts
```

---

# 📐 Mathematical Model

Both implementations effectively learn a logistic-regression-style binary classifier.

### Linear Transformation

```text
z = XW + b
```

### Sigmoid

```text
ŷ = 1 / (1 + e⁻ᶻ)
```

### Binary Cross-Entropy

```text
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

### Gradient Descent

```text
θ ← θ - η∇L
```

where:

* `X` = input features
* `W` = weights
* `b` = bias
* `ŷ` = predicted probability
* `y` = actual label
* `L` = loss
* `η` = learning rate
* `θ` = model parameters

---

# 📈 Evaluation

Both implementations calculate classification accuracy.

The `nn.Module` implementation uses:

```python
y_pred = (y_pred > 0.5).float()
```

The manual implementation uses:

```python
y_pred = (y_pred > 0.9).float()
```

Therefore, **the accuracy values from the two files should not be interpreted as a direct apples-to-apples comparison**, because the prediction thresholds differ.

The implementations also use different epoch counts:

```text
Manual implementation → 30 epochs
nn.Module implementation → 25 epochs
```

---

# 🛠️ Technologies

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Jupyter Notebook

---

# 📦 Installation

Install the required libraries:

```bash
pip install numpy pandas torch scikit-learn
```

---

# ▶️ Running the Project

## Run the Notebook

Open:

```text
Untitled9.ipynb
```

in:

* Jupyter Notebook
* JupyterLab
* VS Code
* Google Colab

Run the cells sequentially.

---

## Run the Python Implementation

Execute:

```bash
python pytorch_training_pipeline_using_nn_module.py
```

The script will:

1. Load the dataset
2. Preprocess the data
3. Create the neural network
4. Train the model
5. Print the loss for every epoch
6. Evaluate the model
7. Print the final accuracy

---

# 📚 Concepts Learned

By studying both files, you can understand the progression from basic tensor operations to standard PyTorch model development.

### Level 1 — Data

```text
Pandas
   ↓
NumPy
   ↓
PyTorch Tensor
```

### Level 2 — Neural Network

```text
Weights + Bias
      ↓
Linear Transformation
      ↓
Sigmoid
      ↓
Prediction
```

### Level 3 — Training

```text
Prediction
    ↓
Loss
    ↓
Gradient
    ↓
Parameter Update
```

### Level 4 — PyTorch Abstraction

```text
Manual Implementation
        ↓
nn.Module
        ↓
nn.Linear
        ↓
Loss Functions
        ↓
Optimizers
```

---

# 🚀 Future Improvements

The repository can be extended by adding:

* `DataLoader`
* Mini-batch training
* Validation dataset
* Multiple hidden layers
* ReLU activation
* Adam optimizer
* Learning-rate scheduling
* Confusion matrix
* Precision
* Recall
* F1-score
* ROC-AUC
* Training-loss visualization
* Validation-loss visualization
* GPU/CPU device handling
* Model saving and loading
* Reproducible random seeds
* Hyperparameter tuning

A natural next step would be to progress from:

```text
Manual Neural Network
        ↓
nn.Module
        ↓
Multi-Layer Perceptron
        ↓
DataLoader + Mini-Batches
        ↓
Validation Pipeline
        ↓
More Advanced PyTorch Models
```

---

# ⭐ Key Takeaway

This repository demonstrates the transition from understanding **how neural-network training works internally** to using the standard PyTorch API.

The first implementation shows the mechanics explicitly:

```text
Weights
   ↓
Forward Pass
   ↓
Loss
   ↓
Autograd
   ↓
Gradients
   ↓
Manual Parameter Update
```

The second implementation packages many of these operations into PyTorch's standard abstractions:

```text
nn.Module
   ↓
nn.Linear
   ↓
Loss Function
   ↓
Autograd
   ↓
Optimizer
```

Understanding both approaches makes it easier to understand what PyTorch's high-level APIs are actually doing rather than treating them as black boxes.

---

# 👨‍💻 Author

**Sagnik Panda**

CSE Student | Artificial Intelligence & Machine Learning

---

## 📜 License

This project is intended for **educational and learning purposes**.

