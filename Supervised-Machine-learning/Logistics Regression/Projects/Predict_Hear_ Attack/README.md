# Heart Attack Prediction

![Project Status](https://img.shields.io/badge/status-active-brightgreen) ![Python Version](https://img.shields.io/badge/python-3.x-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## Description
This project is a web-based application designed to predict the likelihood of a heart attack in patients based on various medical attributes. It utilizes a machine learning model to analyze the input data and provide a prediction, accessible through a user-friendly web interface.

## Problem Statement
The primary objective of this project is to leverage machine learning to provide a quick and accessible way to assess the risk of a heart attack. By analyzing key medical indicators, the application can assist in early-stage risk assessment, complementing professional medical advice.

## Getting Started
To get a local copy up and running, follow these simple steps.

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Installation
1. Clone the repo:
   ```sh
   git clone https://github.com/your_username/Heart-Attack-Prediction.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Heart-Attack-Prediction
   ```
3. Install the required packages:
   ```sh
   pip install -r requirement.txt
   ```

### Running the Application
1. Run the Flask application:
   ```sh
   python application.py
   ```
2. Open your web browser and go to `http://127.0.0.1:5000` to view the application.

## Usage
The application provides a web form where you can input the patient's medical data. Fill in the fields and click the "Predict" button to get the heart attack risk prediction.

## Dataset
The dataset used for training the model contains 76 attributes, but this project utilizes a subset of 14 key features for prediction. The "target" field indicates the presence of heart disease.

### Attribute Information
- **age**: Age of the patient
- **sex**: Sex of the patient (1 = male; 0 = female)
- **cp**: Chest pain type (4 values)
- **trestbps**: Resting blood pressure (in mm Hg on admission to the hospital)
- **chol**: Serum cholestoral in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- **restecg**: Resting electrocardiographic results (values 0, 1, 2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes; 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: The slope of the peak exercise ST segment
- **ca**: Number of major vessels (0-3) colored by flourosopy
- **thal**: 0 = normal; 1 = fixed defect; 2 = reversable defect
- **target**: 0 = less chance of heart attack; 1 = more chance of heart attack

## Methodology
The model was developed following these steps:

1.  **Data Loading and Exploration**: The `heart.csv` dataset was loaded, and an initial exploration was performed to check for missing values and understand the data types.
2.  **Exploratory Data Analysis (EDA)**: Visualizations such as count plots, pie charts, pair plots, and a correlation heatmap were used to understand the distribution of data and the relationships between different attributes.
3.  **Data Preprocessing**: The dataset was split into training (70%) and testing (30%) sets. The features were then scaled using `StandardScaler` to ensure that all features contribute equally to the model's performance.
4.  **Model Training and Hyperparameter Tuning**: A Logistic Regression model was trained on the preprocessed data. `GridSearchCV` was used to find the optimal hyperparameters for the model, resulting in the best parameters being `{'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}`.
5.  **Model Evaluation**: The model's performance was evaluated on the test set, achieving an accuracy of approximately 81.3%. The classification report and confusion matrix below provide a more detailed breakdown of the model's performance:

    **Classification Report:**
    ```
                  precision    recall  f1-score   support

               0       0.78      0.80      0.79        40
               1       0.84      0.82      0.83        51

        accuracy                           0.81        91
       macro avg       0.81      0.81      0.81        91
    weighted avg       0.81      0.81      0.81        91
    ```

    **Confusion Matrix:**
    ```
    [[32  8]
     [ 9 42]]
    ```

## Model
This project uses a **Logistic Regression** model. The trained `GridSearchCV` object, which includes the best estimator, is saved as `ridge.pkl` and is loaded by the Flask application to make predictions.

## Visualization
Here is a scatter plot visualizing the relationship between age and maximum heart rate, colored by the target variable (heart attack risk).

![Scatter Plot](Visualization%20Graph/Scatter%20Plot.png)

### Additional Visualizations
In addition to the scatter plot, the `application.ipynb` notebook in the `Notebbook` directory contains several other visualizations that provide deeper insights into the dataset:

- **Target Class Distribution**: A bar chart showing the distribution of patients with and without a high chance of a heart attack.
- **Gender Distribution**: A pie chart illustrating the gender distribution within the dataset.
- **Correlation Heatmap**: A heatmap that displays the correlation between all the attributes in the dataset, helping to identify significant relationships.

These visualizations are crucial for understanding the data and the model's behavior. For a detailed view, please refer to the notebook.

## Directory Structure
```
.Heart-Attack-Prediction/
├── Datasets/
│   └── heart.csv
├── Models/
│   ├── ridge.pkl
│   └── scaler.pkl
├── Notebbook/
│   └── ...
├── templates/
│   ├── index.html
│   └── home.html
├── Visualization Graph/
│   └── Scatter Plot.png
├── application.py
├── requirement.txt
└── README.md
```

## Dependencies
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) ![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.
