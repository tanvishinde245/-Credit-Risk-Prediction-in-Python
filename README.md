# Credit Risk Prediction Project

This project uses machine learning algorithms to predict whether a loan applicant is likely to default. It applies classification techniques to a financial dataset and evaluates model performance using accuracy, cross-validation, and a confusion matrix.

---

## Dataset

- **File Used:** `bankloans.csv`
- **Attributes:**
  - `age`
  - `income`
  - `debtinc` (debt-to-income ratio)
  - Other financial indicators
  - `default` (target variable: 1 = default, 0 = no default)

---

## Technologies Used

- Python 3.12.8
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn (Random Forest, SVM, Logistic Regression, GridSearchCV)

---

## Workflow

1. **Data Loading and Cleaning**
   - Loaded using `pandas`
   - Dropped null values using `dropna()`
   - Visualized trends with Seaborn (e.g., `age` vs `income`, `age` vs `debtinc`)

2. **Data Preparation**
   - Separated features and target
   - Split into train/test using `train_test_split`
   - Scaled features with `StandardScaler`

3. **Modeling**

   ### Random Forest Classifier
   - 200 trees (`n_estimators=200`)
   - Accuracy: ~80%
   - Cross-validation (CV=10): ~78.39%

   ### Support Vector Machine (SVM)
   - Grid search used to tune `C`, `gamma`, and `kernel`
   - Best Parameters: `C=0.1`, `gamma=0.1`, `kernel='linear'`
   - Accuracy: Depends on tuned model

   ### Logistic Regression
   - Accuracy score
   - Visualized confusion matrix with heatmap

---

## Visualizations

- **Line Plot:** Age vs Income and Debt-to-Income Ratio
- **Confusion Matrix:** For evaluating logistic regression predictions

---

## Key Learnings

- Feature scaling is critical before SVM and Logistic Regression
- Random Forest gave the best out-of-box performance
- GridSearchCV significantly improves SVM performance with the right parameters
- Confusion matrix is helpful for evaluating classification errors

---

## How to Run

1. Clone the repository or download the `.py` or `.ipynb` file.
2. Ensure `bankloans.csv` is available in your path.
3. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
