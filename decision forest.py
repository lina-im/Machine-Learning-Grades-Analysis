import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#load the dataset
df = pd.read_csv("C:/Users/User/OneDrive/Documents/Desktop/student-mat.csv", sep = ';')

df['G3_class'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
#select target and features
target = 'G3_class'
protected = 'sex'
features = df.drop(columns=[target, 'G3'])
labels = df[target]

#identify categorical columns
categorical_cols = features.select_dtypes(include=['object']).columns

#split into train/test sets BEFORE preprocessing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

#ordinal encode categorical variables (fit on train, transform both)
encoder = OrdinalEncoder()
X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols].astype(str))
X_test[categorical_cols] = encoder.transform(X_test[categorical_cols].astype(str))

#normalize features (fit on train, transform both)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

#fill missing values with column mean (use train means to fill both)
train_means = X_train.mean()
X_train = X_train.fillna(train_means)
X_test = X_test.fillna(train_means)

#train the Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#predict and evaluate
y_pred = clf.predict(X_test)
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {overall_accuracy:.4f}")

#analyze bias by gender
y_pred_series = pd.Series(y_pred, index=y_test.index)

#now gender_test is aligned with y_test
gender_test = df.loc[y_test.index][protected]

#separate male and female samples using correct indexing
female_idx = gender_test[gender_test == 'F'].index
male_idx = gender_test[gender_test == 'M'].index

female_accuracy = accuracy_score(y_test.loc[female_idx], y_pred_series.loc[female_idx])
male_accuracy = accuracy_score(y_test.loc[male_idx], y_pred_series.loc[male_idx])

print(f"Female Accuracy: {female_accuracy:.4f}")
print(f"Male Accuracy: {male_accuracy:.4f}")

#plot results
plt.figure(figsize=(6, 4))
plt.bar(['Female', 'Male'], [female_accuracy, male_accuracy], color=['red', 'blue'])
plt.title('Decision Forest - Prediction Accuracy by Gender')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
