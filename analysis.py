
#init
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

file_path = 'paneldata.csv'
data = pd.read_csv(file_path)

# categorizes rvus based on 33/66 percentile
rvu_thresholds = data['RVUs'].quantile([0.33, 0.66])

def categorize_rvu(rvu):
    if rvu <= rvu_thresholds[0.33]:
        return 'Small'
    elif rvu <= rvu_thresholds[0.66]:
        return 'Medium'
    else:
        return 'Large'

#create new row with categorization
data['RVU_Category'] = data['RVUs'].apply(categorize_rvu)

#defines what features for random forest model
X_rf = data[['Patient_Count', 'Total Appts in 2023']]
y_rf = data['RVU_Category']

#splits into training/testing sets, 20% is testing
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

#initializes with 100 trees then trains classifier on training data, then predicts based on that
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_rf_train, y_rf_train)
y_rf_pred = rf_classifier.predict(X_rf_test)

#accuracy calculation
rf_accuracy = accuracy_score(y_rf_test, y_rf_pred)
rf_classification_report = classification_report(y_rf_test, y_rf_pred)

print(f"Random Forest Classifier Accuracy: {rf_accuracy}")
print("Classification Report:")
print(rf_classification_report)

# multiple regression!!

# clean nans
data_reg = data.dropna(subset=['Patient_Count', 'Total Appts in 2023', 'RVUs'])
X_reg = data_reg[['Patient_Count', 'Total Appts in 2023']]
y_reg = data_reg['RVUs']

# split data again, still keeping 20%
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# run lin reg model then predict
reg_model = LinearRegression()
reg_model.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_model.predict(X_reg_test)

#mse & r^2 for these tests
reg_mse = mean_squared_error(y_reg_test, y_reg_pred)
reg_r2 = r2_score(y_reg_test, y_reg_pred)

print(f"Multiple Regression Mean Squared Error: {reg_mse}")
print(f"Multiple Regression R-squared: {reg_r2}")

#confusion matrix
conf_matrix = confusion_matrix(y_rf_test, y_rf_pred, labels=['Small', 'Medium', 'Large'])

# class labels for counts (true positive false positive etc). using dictionaires to store it
class_labels = ['Small', 'Medium', 'Large']
tp = {}
fp = {}
tn = {}
fn = {}

#iterate through and calculate
for i, label in enumerate(class_labels):
    tp[label] = conf_matrix[i, i]
    fp[label] = conf_matrix[:, i].sum() - conf_matrix[i, i]
    fn[label] = conf_matrix[i, :].sum() - conf_matrix[i, i]
    tn[label] = conf_matrix.sum() - (tp[label] + fp[label] + fn[label])

for label in class_labels:
    print(f"Class: {label}")
    print(f"True Positives (TP): {tp[label]}")
    print(f"False Positives (FP): {fp[label]}")
    print(f"True Negatives (TN): {tn[label]}")
    print(f"False Negatives (FN): {fn[label]}\n")

# plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Small', 'Medium', 'Large'], yticklabels=['Small', 'Medium', 'Large'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest Classifier')
plt.show()

# plotting actual vs predicted RVU values
plt.figure(figsize=(10, 6))
plt.scatter(y_reg_test, y_reg_test, color='blue', alpha=0.5, label='Actual Values')
plt.scatter(y_reg_test, y_reg_pred, color='red', alpha=0.5, label='Predicted Values')
for i in range(len(y_reg_test)):
    plt.plot([y_reg_test.iloc[i], y_reg_test.iloc[i]], [y_reg_test.iloc[i], y_reg_pred[i]], 'gray', lw=0.5)
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual RVUs')
plt.ylabel('Predicted RVUs')
plt.title('Actual vs Predicted RVUs')
plt.legend()
plt.show()



