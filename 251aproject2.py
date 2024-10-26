#!/usr/bin/env python
# coding: utf-8

# # Project 2 - Coordinate Descent
# 
# ### Utilizes coordinate descent optimization for logistic regression in binary classification.
# ### Analyzes cyclic and random feature selection methods on the wine dataset.
# - Cyclic: updates one coordinate at a time sequentially.
# - Random: selects coordinates randomly for update.
# ### It also:
# - Benchmarks against standard logistic regression.
# - Explores sparse coordinate descent for solutions with limited non-zero coefficients.
# 
# 

# In[1]:


import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# In[2]:


pip install ucimlrepo


# ## Importing the dataset into the code

# In[3]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
X = wine.data.features 
y = wine.data.targets 
  
# metadata 
print(wine.metadata) 
  
# variable information 
print(wine.variables) 


# In[4]:


# Load the wine dataset and preprocess the data
wine = load_wine()
X, y = wine.data[:130, :], wine.target[:130]  # Using only the first two classes
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # Add intercept term
X_train, y_train = X, y  # Assign preprocessed data to training variables
scaler = StandardScaler()  # Initialize StandardScaler
X_train_scaled = scaler.fit_transform(X)  # Scale feature matrix


# ## 1. Standard Logistic Regression
# 
# - The methodology for standard logistic regression involves initial data preparation by loading and preprocessing the wine dataset, focusing on binary classification. 
# - After standardizing the features and initializing the logistic regression model with specific parameters, training commences using iterative optimization methods. 
# - The trained model predicts labels for the training data, and the logistic loss is computed to assess its performance. 
# - This methodology serves as a baseline for comparing alternative optimization techniques like cyclic coordinate descent and random feature selection.

# In[5]:


# Initialize the logistic regression model
binary_model = LogisticRegression(penalty="none", max_iter=1000)

# Fit the model to the training data
binary_model.fit(X_train_scaled, y_train)

# Predict the training labels
y_pred = binary_model.predict(X_train_scaled)

# Calculate the training loss
loss = log_loss(y_true=y_train, y_pred=y_pred)

# Print the training loss
print(f'Training loss for {loss}')


# ## 2. Coordinate descent method
# The methodology for Cyclic Coordinate Descent (CCD) optimization in logistic regression involves several key steps:
# 
# - Define the logistic loss function to quantify the difference between predicted probabilities and actual labels.
# - Calculate the gradient of the logistic loss function to determine the direction and magnitude of parameter adjustments.
# - Implement the CCD algorithm to iteratively update model parameters, selecting one feature at a time in a cyclic manner and adjusting the parameter based on the computed gradient.
# - Continuously monitor the logistic loss to assess optimization progress.
# - Train the logistic regression model using the CCD algorithm on standardized training data for iterative parameter refinement.
# - Evaluate the trained model's performance by computing the logistic loss on the training data to assess the effectiveness of the CCD method in optimizing logistic regression models for binary classification tasks.

# In[6]:


def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))


# In[7]:


def logistic_loss(X, y, w):
    """Logistic loss function."""
    n = len(y)
    y_pred = sigmoid(X.dot(w))
    loss = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / n
    return loss


# In[8]:


def logistic_gradient(X, y, w):
    """Gradient of logistic loss function."""
    n = len(y)
    y_pred = sigmoid(X.dot(w))
    gradient = X.T.dot(y_pred - y) / n
    return gradient


# In[9]:


def cyclic_coordinate_descent_logistic(X, y, max_epochs=10000, alpha=10):
    """Coordinate descent for logistic regression using cyclic method."""
    n, d = X.shape
    w = np.zeros(d)  # Initialize weight vector
    losses = []  # Track loss at each iteration
    
    for epoch in range(max_epochs):
        prev_loss = logistic_loss(X, y, w)
        # Select one coordinate for update in this epoch
        coordinate_index = epoch % d
        gradient_i = logistic_gradient(X, y, w)[coordinate_index]
        w[coordinate_index] -= alpha * gradient_i  # Update selected coordinate
        # Compute loss after updating one coordinate
        curr_loss = logistic_loss(X, y, w)
        losses.append(curr_loss)
        

    return w, losses

# Train logistic regression model using cyclic coordinate descent
w_cyclic, losses_cyclic = cyclic_coordinate_descent_logistic(X_train_scaled, y_train)

# Compute loss for cyclic method
loss_train_cyclic = logistic_loss(X_train_scaled, y_train, w_cyclic)

print("\nCyclic Coordinate Descent - Train Loss:", loss_train_cyclic)


# In[10]:


plt.plot([loss]*10000, linewidth=2.5, label='Loss for Logistic Regression', color='purple')
plt.plot(losses_cyclic, linewidth=2.5, label='Cyclic Coordinate Selection Loss', color='orange')
plt.xlabel('Number of Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()


# ## 3. Random feature Coordinate Descent
# The CCDR method involves these steps:
# 
# - Define the logistic loss function.
# - Compute the gradient.
# - Perform iterations over epochs.
# - Randomly select a coordinate for parameter update.
# - Adjust parameters based on the gradient.
# - Monitor the logistic loss.
# - Train the model using standardized data.
# - Evaluate performance by computing logistic loss on training data.

# In[11]:


import random

def cyclic_coordinate_descent_logistic_random(X, y, max_epochs=10000, alpha=10):
    """Coordinate descent for logistic regression using cyclic method."""
    n, d = X.shape
    w = np.zeros(d)  # Initialize weight vector
    losses = []  # Track loss at each iteration
    
    for epoch in range(max_epochs):
        prev_loss = logistic_loss(X, y, w)
        # Select one coordinate for update in this epoch
        coordinate_index = random.randint(0,X.shape[1]-1)
        gradient_i = logistic_gradient(X, y, w)[coordinate_index]
        w[coordinate_index] -= alpha * gradient_i  # Update selected coordinate
        # Compute loss after updating one coordinate
        curr_loss = logistic_loss(X, y, w)
        losses.append(curr_loss)
        

    return w, losses

# Train logistic regression model using cyclic coordinate descent
w_cyclic_random, losses_cyclic_random = cyclic_coordinate_descent_logistic_random(X_train_scaled, y_train)

# Compute loss for cyclic method
loss_train_cyclic_random = logistic_loss(X_train_scaled, y_train, w_cyclic_random)

print("\nCyclic Coordinate Descent Random - Train Loss:", loss_train_cyclic_random)


# In[12]:


plt.plot([loss]*10000, linewidth=2.5, label='Loss for Logistic Regression', color='purple')
plt.plot(losses_cyclic_random, linewidth=2.5, label='Random Coordinate Selection Loss', color='green')
plt.xlabel('Number of Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()


# In[13]:


plt.plot([loss]*10000, linewidth=2.5, label='Loss for Logistic Regression', color='purple')
plt.plot(losses_cyclic, linewidth=2.5, label='Cyclic Coordinate Selection Loss', color='orange')
plt.plot(losses_cyclic_random, linewidth=2.5, label='Random Coordinate Selection Loss', color='green')
plt.xlabel('Number of Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()


# ## Sparse Coordinate Descent
# - Sparse coordinate descent initializes w with zeros and randomly selects k coordinates without repetition.
# - It iterates over epochs, updating parameters based on selected coordinates.
# - At each index, it computes the gradient of the logistic loss function and updates the parameter using step size α.
# - The logistic loss assesses model performance, tracking progress.
# - Post-optimization, the model's performance is evaluated using loss values to gauge sparsity in w.
# - It can specify the desired sparsity level with input k.
# - Achieving the optimal k-sparse solution varies in convex cost functions.
# - Applied to the wine dataset, it initializes the model with sparse coordinate descent, varying k, and evaluates performance using loss values.

# In[14]:


def sparse_cyclic_coordinate_descent_logistic(X, y, k, max_epochs=10000, alpha=0.1):
    """Sparse coordinate descent for logistic regression."""
    n, d = X.shape
    w = np.zeros(d)  # Initialize weight vector
    losses = []  # Track loss at each iteration
    
    coordinate_indices = np.random.choice(d, k, replace=False)
    
    for epoch in range(max_epochs):
        prev_loss = logistic_loss(X, y, w)
        for coordinate_index in coordinate_indices:
            gradient_i = logistic_gradient(X, y, w)[coordinate_index]
            w[coordinate_index] -= alpha * gradient_i  # Update selected coordinate
        # Compute loss after updating k coordinates
        curr_loss = logistic_loss(X, y, w)
        losses.append(curr_loss)
        
    return w, losses


# In[15]:


# Define values of k to try
k_values = [1, 2, 4, 10]

# Dictionary to store loss values for different k
losses_dict = {}

# Try sparse coordinate descent for each value of k
for k in k_values:
    w_sparse, losses_sparse = sparse_cyclic_coordinate_descent_logistic(X_train_scaled, y_train, k)
    loss_train_sparse = log_loss(y_train, X_train_scaled.dot(w_sparse))
    losses_dict[k] = loss_train_sparse

# Print loss values for different k
print("Loss values for different values of k:")
for k, loss1 in losses_dict.items():
    print(f"k = {k}: Train Loss = {loss1}")

