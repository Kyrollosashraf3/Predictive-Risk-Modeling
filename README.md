# Predictive Risk Modeling: Nepal Earthquake Damage Analysis

## Overview
This project focuses on predicting the severity of building damage utilizing the Nepal Earthquake Damage dataset from the devastating 2015 earthquake. The primary objective is to build robust Machine Learning models to identify which building features—such as structural design, location, and construction materials—contribute most significantly to severe earthquake damage.

By converting raw building and seismic data into a binary classification problem, this project highlights key **Data Science** and **Machine Learning** workflows, including Exploratory Data Analysis (EDA), advanced feature engineering, hyperparameter tuning, and model interpretation.

---

## 🔬 Exploratory Data Analysis (EDA)
Thorough data exploration was conducted to understand distributions and relationships between features:
- **Comprehensive Reporting**: Developed flexible reporting functions (`df_full_report`) to generate textual summaries (shape, missing values, duplicates) along with dynamically generated countplots and boxplots to visualize distributions and detect outliers `(plot_outliers)`.
- **Target Analysis**: Investigated the distribution of the target variable to understand the baseline class balance. Analyzed building heights and foundation types against severe damage ratios using pivot tables and visual charts.
- **Multicollinearity Check**: Used correlation heatmaps to identify and address highly correlated features (e.g., `count_floors_pre_eq` and `height_ft_pre_eq`).

---

## ⚙️ Feature Engineering & Data Preprocessing
Data cleaning and transformations were streamlined using a centralized `wrangle` function to prevent data leakage and prepare the dataset for predictive modeling:
1. **Target Variable Creation**: Grouped the multi-class `damage_grade` into a binary target `severe_damage` (1 for Grades 4 & 5, 0 for Grades 1, 2 & 3).
2. **Leakage Prevention**: Systematically dropped all `post_eq` features, which contain future information not available at prediction time.
3. **Dimensionality Reduction**: Removed high-cardinality identifiers (`building_id`) to prevent severe overfitting.
4. **Handling Multicollinearity**: Dropped redundant attributes (like `count_floors_pre_eq`) to stabilize the linear model's coefficients.
5. **Handling Missing Values**: Applied `dropna()` to formulate a clean pipeline.

---

## 🤖 Machine Learning Modeling

Two distinct classification approaches were employed, evaluated, and interpreted using an 80/20 train-test split (with a separate validation set for tree-based tuning). All algorithms were wrapped in Scikit-Learn `Pipelines` combining proper categorical encoding with the model estimators.

### 1. Logistic Regression
An interpretable, linear baseline model was established using a `OneHotEncoder` mapped to a `LogisticRegression` classifier.
- **Evaluation**: Accuracy scores were computed on both training and held-out test sets to ensure the model surpassed the majority-class baseline.
- **Feature Importance (Odds Ratios)**: 
  Extracted the coefficients and exponentiated them into **Odds Ratios**. Analyzed the most and least influential risk factors. 
  *Insight:* Features like specific foundation types (e.g., Mud mortar-Stone/Brick) significantly increased the likelihood of severe damage.

### 2. Decision Tree Classifier
A non-linear model was developed using an `OrdinalEncoder` paired with a `DecisionTreeClassifier`.
- **Hyperparameter Tuning**: Initially, the default model suffered from high variance (overfitting). A loop over `max_depth` (1 to 20) was orchestrated to map training vs. validation accuracy.
- **Validation Curve**: Plotted the tuning results to pinpoint the exact threshold where overfitting commenced, identifying the optimal tree depth (`max_depth=6`).
- **Model Interpretation**:
  - The final pruned tree was visualized using `plot_tree` to understand decision splits.
  - Calculated and plotted the **Gini Feature Importances**.
  - *Insight:* The model heavily utilized `district_id`, `ground_floor_type`, and `foundation_type`, affirming that structural materials and geographic location encapsulate the highest predictive power for building collapse.

---

## 🛠️ Tech Stack & Libraries
*   **Data Manipulation**: `pandas`, `numpy`
*   **Data Visualization**: `matplotlib`, `seaborn`
*   **Machine Learning**: `scikit-learn` (`LogisticRegression`, `DecisionTreeClassifier`, `Pipeline`, `train_test_split`)
*   **Feature Encoding**: `category_encoders` (`OneHotEncoder`, `OrdinalEncoder`)

---

## Conclusion
This project demonstrates an end-to-end predictive modeling workflow. By effectively engineering features and combating multicollinearity and data leakage, we successfully trained Logistic Regression and Decision Tree models. Model interpretability techniques (Odds Ratios and Gini Importances) not only validated the predictive performance but also extracted actionable domain insights into building vulnerability during seismic events.
