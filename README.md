# Credit-Card-Fraud-Detection-

The dataset contains transaction details, where each record is classified as either fraudulent or non-fraudulent. Our goal is to build and evaluate models that can accurately detect fraudulent transactions.

### Dataset Overview

The dataset consists of **568,630 rows and 31 columns**. The columns include features `V1` to `V28`, the `Amount` of the transaction, and the target variable `Class`, where:

- **V1 to V28**: These are anonymized numerical features representing various aspects of the transaction (e.g., transaction patterns, behavior).
- **Amount**: The monetary value of the transaction.
- **Class**: The target variable indicating whether a transaction is fraudulent (1) or non-fraudulent (0).

The dataset is highly imbalanced, with a very small percentage of fraudulent transactions compared to non-fraudulent ones.

### Exploratory Data Analysis (EDA)

#### 1. Class Distribution

The target variable `Class` shows a significant class imbalance, with most transactions being non-fraudulent (0) and only a small fraction being fraudulent (1). Understanding this imbalance is crucial for evaluating the performance of machine learning models, as it can impact metrics like accuracy and precision.

To visualize this class distribution:

![Class Distribution](class_distribution.png)

The plot reveals that the dataset has a high class imbalance, with fraudulent transactions representing only a small portion of the total dataset.

#### 2. Outlier Detection

Outliers can influence model performance, especially for algorithms sensitive to extreme values (e.g., linear models, k-NN). Therefore, identifying and handling outliers is an important part of the preprocessing pipeline.

Two common methods were used to detect outliers:

- **Z-score Method**: The Z-score for each transaction's `Amount` was calculated, and outliers were defined as those with an absolute Z-score greater than 3. This method flagged extreme monetary values.
  
- **Interquartile Range (IQR) Method**: The IQR was calculated for the `Amount` variable, and transactions with values outside the range defined by Q1 - 1.5 \times IQR and Q3 + 1.5 \times IQR were flagged as outliers.

No significant outliers were detected using either of these methods, which suggests that the values for `Amount` are relatively stable and do not contain extreme anomalies that could affect the models.

#### 3. Feature Analysis

An important step in understanding the data is identifying which features are most significant in distinguishing between fraudulent and non-fraudulent transactions. **ANOVA** (Analysis of Variance) was used to test the statistical significance of the features.

From the results, we identified that **features `V1` through `V28`** were statistically significant in differentiating fraudulent transactions from non-fraudulent ones. These features are key in detecting fraud and are likely derived from underlying transaction patterns and behaviors.

#### 4. Correlation Matrix

We computed a correlation matrix to investigate the relationships between the features and the target variable (`Class`). A heatmap was generated to visualize these correlations, where higher correlation values (near 1 or -1) indicate stronger relationships between the features.

Here is the correlation matrix:

![Correlation Matrix](correlation_matrix.png)

The heatmap indicates several features with significant correlations to the target variable. These correlations suggest that the features may provide meaningful insights into the patterns of fraudulent transactions.

### Model Building

#### 1. Logistic Regression (L1 and L2 Regularization)

Logistic regression is a widely used model for binary classification tasks. We implemented logistic regression with **L1** (Lasso) and **L2** (Ridge) regularization techniques to prevent overfitting.

- **L1 Regularization** (Lasso): Lasso regularization encourages sparsity in the model by shrinking some coefficients to zero. This can help in feature selection by removing irrelevant features.
- **L2 Regularization** (Ridge): Ridge regularization penalizes large coefficients and helps prevent overfitting by distributing the penalty across all coefficients.

The results for both regularizations were evaluated using various metrics:

- **L1 Regularization**:
  - **Accuracy**: 97%
  - **Precision**:
    - Fraudulent: 98%
    - Non-Fraudulent: 95%
  - **Recall**:
    - Fraudulent: 95%
    - Non-Fraudulent: 98%

- **L2 Regularization**:
  - **Accuracy**: 97%
  - **Precision**:
    - Fraudulent: 98%
    - Non-Fraudulent: 95%
  - **Recall**:
    - Fraudulent: 95%
    - Non-Fraudulent: 98%

While the models performed similarly, L1 regularization showed a slightly better performance in terms of feature sparsity and interpretability.

#### 2. Random Forest Classifier

 
- **Accuracy**: 100%
- **Precision**:
  - Fraudulent: 100%
  - Non-Fraudulent: 100%
- **Recall**:
  - Fraudulent: 100%
  - Non-Fraudulent: 100%


#### 3. Decision Tree Classifier (Post-Pruning)

The Decision Tree algorithm was also evaluated. We used post-pruning to limit the depth of the tree and prevent overfitting, which often occurs in decision trees when they grow too deep.

- **Pruning**: A maximum depth of 5 was set for the decision tree to ensure the model was not overfitting to the data.
- **Performance**:
  - **Accuracy**: 96%
  - **Precision**:
    - Non-Fraudulent: 97%
    - Fraudulent: 95%
  - **Recall**:
    - Non-Fraudulent: 95%
    - Fraudulent: 97%



#### 4. Isolation Forest for Outlier Detection

An Isolation Forest was used to identify anomalies in the dataset. This model is particularly useful for detecting outliers (in this case, fraudulent transactions) in high-dimensional data.

- **Outlier Detection**:
  - **5,687 outliers** were detected, and **97%** of these outliers were fraudulent transactions.
  
The Isolation Forest is effective in identifying fraudulent transactions because frauds are often rare and can be considered anomalies in the transaction data.

#### 5. Deep Learning Models

Deep learning models, specifically **Artificial Neural Networks (ANN)**, can provide powerful classification capabilities, especially when dealing with high-dimensional data. We implemented a simple feedforward neural network using **TensorFlow** and **Keras** with the following architecture:

- **Architecture**: The model consists of:
  - **Input Layer**: 31 nodes corresponding to the 31 features (V1 to V28, Amount, and Class).
  - **Hidden Layer 1**: 64 neurons with ReLU activation.
  - **Hidden Layer 2**: 32 neurons with ReLU activation.
  - **Output Layer**: 1 neuron with a sigmoid activation function (binary classification: fraud vs. non-fraud).
  
- **Optimizer**: Adam optimizer was used to minimize the binary cross-entropy loss function.
- **Epochs**: 20 epochs with a batch size of 32.

**Training Performance**:
- **Accuracy**: 98%
- **Precision**:
  - Fraudulent: 97%
  - Non-Fraudulent: 99%
- **Recall**:
  - Fraudulent: 99%
  - Non-Fraudulent: 97%

The deep learning model performed exceptionally well, with a slightly better recall rate for fraudulent transactions compared to traditional machine learning models. The model is capable of learning complex patterns from the data, especially with the large number of features.

### Conclusion

By considering our example the company will able to save Rs 41,686,140 by reducing fraud losses.

# **Future Enhancements**  

While the model is already performing well, there are additional steps that can be explored to further enhance the project and derive deeper insights:  

1. **Feature Selection and Modeling on Top Features**  
   - Identify the top features contributing to fraud using feature importance techniques like SHAP, LIME, or permutation importance.  
   - Train and evaluate the model on these top features to test if a reduced feature set can improve interpretability and maintain accuracy.  

2. **Feature Engineering: Interaction Terms**  
   - Create interaction terms based on the top features selected.  
   - For instance, combining features that strongly influence fraud behavior could reveal hidden patterns.  
   - Evaluate the model's performance using these engineered features.  

3. **Explainable AI with SHAP or LIME**  
   - Use SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to understand individual feature contributions to predictions.  
   - This will help identify both global feature importance and specific feature impacts on individual fraud predictions, enhancing model transparency.  

4. **Model Deployment**  
   - Deploy the trained model using platforms like **Streamlit**, **Flask**, or **FastAPI**.  
   - Create an interactive interface where users can input transaction details and get real-time predictions on fraud likelihood.  
   - Showcase the deployment to highlight the practical usability of the model in a real-world scenario.  

---



