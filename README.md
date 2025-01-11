#Life Expectancy Prediction with Distributed Data Processing Using PySpark
This project aims to predict life expectancy based on various socio-economic and health-related factors using Apache Spark's distributed data processing capabilities. The project leverages the power of PySpark and Spark MLlib for distributed cleaning, preprocessing, and model training on a large-scale life expectancy dataset.

## Key Objectives
Applied distributed data preprocessing techniques using PySpark.
Trained six different machine learning models using Spark MLlib and evaluated their performances.
Analyzed Spark DAG visualizations to assess the performance bottlenecks and scaling potential.
## Dataset
The dataset contains various features such as:
Adult Mortality
Alcohol Consumption
BMI
Schooling
The target variable is Life Expectancy, a continuous variable.

## Preprocessing Steps
Remove Duplicates - Drop duplicate records.
Handle Missing Values - Replace missing values using the mean of columns.
Normalize Continuous Variables - Scale features using MinMaxScaler.
Outlier Removal - Remove data points that deviate by more than three standard deviations.
Feature Transformation - Apply log transformations on skewed columns like Measles.
Categorical Encoding & Aggregation - Encode categorical variables and calculate average life expectancy by country using window functions.
## Machine Learning Algorithms
The following models were trained and evaluated:

Linear Regression
Random Forest
Gradient Boosted Trees
Decision Tree
Lasso Regression
Logistic Regression (for classification tasks)
## Model Evaluation
The models were evaluated using several performance metrics including:

RMSE (Root Mean Squared Error)
R² (Coefficient of Determination)
Accuracy, Precision, and F1 Score (for Logistic Regression)
## Key Results:

Gradient Boosted Trees achieved the best performance with an RMSE of 2.69 and an R² of 0.92.
Logistic Regression showed impressive results in classification tasks with an accuracy of 91.5%.
## Spark DAG Visualizations
Spark's Directed Acyclic Graph (DAG) visualizations were analyzed to understand task execution flows and optimize resource utilization:

Job efficiency insights were gathered by analyzing Spark Web UI’s DAG visualizations.
Data shuffling during intermediate stages and optimization strategies like partition tuning and caching were explored.
## Comparative Analysis
A comparison between traditional machine learning models (Phase 2) and distributed models (Phase 3 using PySpark) showed that PySpark's distributed framework significantly improved computational efficiency and scalability, although some models had slightly higher MSE due to the nature of distributed processing.

## Conclusion
This study emphasizes the importance of distributed frameworks like PySpark for handling and processing large datasets efficiently. Despite some slight reductions in performance metrics, PySpark demonstrated its capability to scale computations effectively, making it a critical tool for large-scale machine learning tasks.


