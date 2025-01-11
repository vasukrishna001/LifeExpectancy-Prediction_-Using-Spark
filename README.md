# Life Expectancy Prediction with Distributed Data Processing and Streamlit Web Interface

This project predicts life expectancy based on various socio-economic and health-related factors using Apache Spark's distributed data processing capabilities. It leverages the PySpark library for preprocessing and model training, and integrates a Streamlit web-based user interface (UI) for real-time predictions.(file named as app.py, model.py, output.csv, random_forest.pkl, a video mp4 which describes on how all these files work)

## Key Objectives:
- Applied distributed data preprocessing techniques using PySpark.
- Trained and evaluated six different machine learning models using Spark MLlib.
- Created a web interface for life expectancy prediction using Streamlit.
- Integrated data visualizations, including a correlation heatmap and feature impact analysis, into the UI.

## Dataset:
The dataset used contains various features such as:
- **Adult Mortality**
- **Alcohol Consumption**
- **BMI**
- **Schooling**
  
The **target variable** is **Life Expectancy**, which is a continuous variable.

## Preprocessing Steps:
The following preprocessing steps were performed on the dataset:
1. **Remove Duplicates** - Duplicate records were removed using `dropDuplicates()`.
2. **Handle Missing Values** - Missing values in numeric columns were replaced by the mean using `fillna()`.
3. **Normalize Continuous Variables** - Features were scaled using MinMaxScaler.
4. **Outlier Removal** - Data points deviating by more than three standard deviations were removed.
5. **Feature Transformation** - Measles and other skewed columns were subjected to log transformations.
6. **Categorical Encoding & Aggregation** - Categorical variables were encoded, and the average life expectancy by country was calculated using window functions.

## Machine Learning Algorithms:
The following machine learning models were trained and evaluated using Spark MLlib:
- **Linear Regression**
- **Random Forest**
- **Gradient Boosted Trees**
- **Decision Tree**
- **Lasso Regression**
- **Logistic Regression** (for classification tasks)

### Key Results:
- **Gradient Boosted Trees**: Best model with an RMSE of 2.69 and an R² of 0.92.
- **Logistic Regression**: Achieved 91.5% accuracy in classification tasks, with high precision, recall, and F1 score.

## Spark DAG Visualizations:
Spark's Directed Acyclic Graph (DAG) visualizations were used to analyze task execution flows, job efficiency, and resource utilization. These visualizations provided insights into the distributed processing steps, data shuffling, and optimization strategies.

## Streamlit Web Interface:
The project also includes a **Streamlit** web interface for real-time life expectancy predictions. Here's how it works:

1. **Input Custom Features**: Users can input custom values for various features such as alcohol consumption, BMI, schooling, etc.
2. **Predictions**: The app predicts life expectancy based on the entered values.
3. **Validation**: Users can validate predictions by entering identical feature values from the preprocessed `output.csv` file and comparing the results.
4. **Correlation Heatmap**: The app displays a heatmap showing the relationships between different features, such as the positive correlation between "Income Composition of Resources" and life expectancy, and the negative correlation between "Adult Mortality" and life expectancy.
5. **Feature Impact Visualization**: A visualization was created for the impact of **Alcohol Consumption** on life expectancy predictions, showing that higher alcohol consumption leads to lower life expectancy.

## Running the Streamlit App:
To run the Streamlit app and interact with the web interface, execute the following command in the terminal:


After running the command, a URL will be generated that you can open in your browser to access the UI.

## Conclusion:
This project demonstrates the power of distributed data processing using PySpark and combines it with an interactive and user-friendly Streamlit web app for real-time predictions. It integrates key data visualizations, feature analysis, and machine learning to create a comprehensive tool for predicting and analyzing life expectancy based on socio-economic and health factors.

Thank you for checking out this project, and I hope you find it valuable for exploring machine learning, distributed systems, and web development!
