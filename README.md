# Car Evaluation using Support Vector Machines (SVM)

## Project Overview

This project focuses on building a Support Vector Machine (SVM) classifier to predict car evaluations based on a dataset of vehicle attributes. The dataset includes several categorical variables like `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`, and the target variable `class` which represents the car evaluation (e.g., `unacc`, `acc`, `good`, `vgood`). We apply preprocessing techniques like encoding, class balancing, and hyperparameter tuning via GridSearchCV to achieve the best possible model performance.

### Key Components:

- **Data Preprocessing:** Categorical encoding using `category_encoders.OrdinalEncoder`.
- **Class Imbalance Handling:** Managed via oversampling using `RandomOverSampler`.
- **Model Selection:** SVM is tuned using grid search with cross-validation.
- **Evaluation:** Confusion matrix, classification report, and performance plots are generated.
- **Visualization:** Results are visualized using Plotly for intuitive analysis.

## Dataset

The dataset used in this project is `car.csv`, which contains the following attributes:

- `buying`: Price of buying the car.
- `maint`: Maintenance cost of the car.
- `doors`: Number of doors.
- `persons`: Capacity in terms of the number of persons.
- `lug_boot`: Size of luggage boot.
- `safety`: Safety levels.
- `class`: Target variable representing car evaluation.

## Requirements

To install the required packages, run the following:

```bash
pip install -r requirements.txt
```

### Key Libraries:

- `pandas` for data manipulation.
- `category_encoders` for encoding categorical variables.
- `imblearn` for handling class imbalance.
- `scikit-learn` for machine learning algorithms and model evaluation.
- `plotly` for interactive visualizations.
- `joblib` for model persistence.

### Installing Dependencies

The required libraries can be installed using:

```bash
pip install pandas category_encoders imbalanced-learn scikit-learn plotly joblib
```

## Project Structure

- **`car.csv`**: Dataset file containing vehicle attributes and evaluation.
- **`CarEvaluation_SVM.ipynb`**: Jupyter Notebook containing the project code.
- **`best_model.pkl`**: Saved SVM model after hyperparameter tuning.
- **`X_test_data.csv` & `y_test_data.csv`**: Preprocessed test data saved for future evaluations.

## Data Preprocessing

1. **Encoding Categorical Variables**: The dataset contains categorical variables, which are encoded using ordinal encoding from `category_encoders`.
2. **Handling Class Imbalance**: Class imbalance in the target variable `class` is handled using `RandomOverSampler`.
3. **Splitting Data**: The data is split into training and testing sets using an 80-20 split with `train_test_split`.

## Model Training and Hyperparameter Tuning

We train the SVM model using GridSearchCV for hyperparameter tuning. The key hyperparameters tuned are:

- `C`: Regularization parameter.
- `gamma`: Kernel coefficient.
- `kernel`: Type of kernel used by the SVM (e.g., linear, polynomial, radial basis function).

```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly']
}
```

Grid search with 5-fold cross-validation is used to select the best model.

## Model Evaluation

After training, we evaluate the model using:

- **Accuracy**: Overall accuracy on the test set.
- **Confusion Matrix**: To understand the performance across different classes.
- **Classification Report**: Detailed performance metrics including precision, recall, and F1-score for each class.
- **Training Time**: The time taken to train the model for each epoch.

Results are visualized using Plotly for an interactive and clear understanding of the model's performance.

### Example Output

- **Best Parameters**: `{'C': 10, 'gamma': 1, 'kernel': 'rbf'}`
- **Accuracy**: `1.0`
- **Training Time per Epoch**: `0.29 seconds`

## Visualizations

- **Confusion Matrix**: Visualizes the performance across different classes.
- **Training and Testing Accuracy**: Plots showing accuracy with respect to different hyperparameter configurations.
- **Training Time per Epoch**: A line plot showing the time taken to train the model over multiple epochs.

## Saving the Model

The best model is saved as `best_model.pkl` using `joblib` for future use. The model can be loaded for inference or further evaluation.

```python
joblib.dump(best_model, 'best_model.pkl')
```

## How to Run the Code

1. Clone this repository:

```bash
git clone https://github.com/your-username/Car-Evaluation-SVM.git
```

2. Navigate to the project directory:

```bash
cd Car-Evaluation-SVM
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Jupyter notebook or Python script to train and evaluate the SVM model.

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Category Encoders Documentation](https://contrib.scikit-learn.org/category_encoders/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)

---

For further improvements, consider trying other classifiers or using techniques like SMOTE for handling class imbalance.
