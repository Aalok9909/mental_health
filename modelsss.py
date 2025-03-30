# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import os

# # 🔹 STEP 1: Load the dataset
# file_path = "C:/Users/SHAH KANHAIYALAL/Desktop/full_setup/Updated_MindEase_Cleaned.csv"  # Change to correct path
# df = pd.read_csv(file_path)

# # 🔹 STEP 2: Define features (X) and target (y)
# target_column = "Mental_Health_Condition"
# X = df.drop(columns=[target_column])
# y, class_mapping = pd.factorize(df[target_column])  # Factorize labels

# # 🔹 STEP 3: Identify categorical and numerical columns
# categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
# numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# # 🔹 STEP 4: Preprocessing Pipeline
# preprocessor = ColumnTransformer(transformers=[
#     ("num", SimpleImputer(strategy="mean"), numerical_cols),  # Fill missing numerical values with mean
#     ("cat", Pipeline(steps=[
#         ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing categorical values
#         ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # One-hot encoding
#     ]), categorical_cols)
# ])

# # 🔹 STEP 5: Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # 🔹 STEP 6: Train the Random Forest Model
# rf_model = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("classifier", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
# ])
# rf_model.fit(X_train, y_train)

# # 🔹 STEP 7: Evaluate Model
# y_pred_rf = rf_model.predict(X_test)
# rf_accuracy = accuracy_score(y_test, y_pred_rf)
# print(f"🌲 Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# # 🔹 STEP 8: Save Model & Label Mapping
# import pickle

# # Save the mapping of numeric values to original labels
# label_mapping = dict(enumerate(pd.factorize(df["Mental_Health_Condition"])[1]))

# # Save the trained model along with the mapping
# model_data = {
#     "model": best_rf,  # Assuming 'best_rf' is the final trained model
#     "mapping": label_mapping
# }

# with open("mental_health_model.pkl", "wb") as model_file:
#     pickle.dump(model_data, model_file)

# print("✅ Model and label mapping saved successfully!")






# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import pickle

# # 🔹 STEP 1: Load the dataset
# file_path = "C:/Users/SHAH KANHAIYALAL/Desktop/full_setup/Updated_MindEase_Cleaned.csv"
# df = pd.read_csv(file_path)

# # 🔹 STEP 2: Define features (X) and target (y)
# target_column = "Mental_Health_Condition"
# X = df.drop(columns=[target_column])
# y = df[target_column]

# # 🔹 STEP 3: Convert target column (y) to numerical labels
# y = y.astype(str)  # Ensure consistency
# y_labels, y = pd.factorize(y)  # Convert categorical target to numbers

# # 🔹 Save label mapping
# label_mapping = dict(enumerate(y_labels))

# # 🔹 STEP 4: Identify categorical and numerical columns
# categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
# numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# # 🔹 Handle non-classified columns
# remaining_cols = set(X.columns) - set(categorical_cols) - set(numerical_cols)

# for col in remaining_cols:
#     try:
#         X[col] = pd.to_numeric(X[col], errors="coerce")
#         numerical_cols.append(col)
#     except:
#         categorical_cols.append(col)

# # Ensure all features are accounted for
# assert len(categorical_cols) + len(numerical_cols) == X.shape[1], "Feature mismatch!"

# # 🔹 STEP 5: Create preprocessing pipeline
# preprocessor = ColumnTransformer(transformers=[
#     ("num", SimpleImputer(strategy="mean"), numerical_cols),
#     ("cat", Pipeline(steps=[
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
#     ]), categorical_cols)
# ])

# # 🔹 STEP 6: Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # 🔹 STEP 7: Train the Random Forest Model
# rf_model = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("classifier", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
# ])

# rf_model.fit(X_train, y_train)

# # 🔹 STEP 8: Evaluate Model
# y_pred_rf = rf_model.predict(X_test)
# rf_accuracy = accuracy_score(y_test, y_pred_rf)
# print(f"🌲 Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# # 🔹 STEP 9: Save Model & Label Mapping
# model_data = {
#     "model": rf_model,
#     "mapping": label_mapping
# }

# with open("mental_health.pkl", "wb") as model_file:
#     pickle.dump(model_data, model_file)

# print("✅ Model and label mapping saved successfully!")



import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 🔹 STEP 1: Load Dataset
file_path = "Updated_MindEase_Cleaned.csv"  # Change this to your dataset path
df = pd.read_csv(file_path)

# 🔹 STEP 2: Define Features (X) and Target (y)
target_column = "Mental_Health_Condition"
X = df.drop(columns=[target_column])
y = df[target_column]

# 🔹 STEP 3: Convert Target to Numeric Labels
y, y_labels = pd.factorize(y)  # Convert categorical labels to numbers

# 🔹 STEP 4: Identify Categorical and Numerical Features
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 🔹 STEP 5: Create Preprocessing Pipeline
preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="mean"), numerical_cols),  # Fill missing numerical values
    ("cat", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing categorical values
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # One-hot encoding
    ]), categorical_cols)
])

# 🔹 STEP 6: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 STEP 7: Train the Model
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
])

rf_model.fit(X_train, y_train)

# 🔹 STEP 8: Evaluate the Model
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"🌲 Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# 🔹 STEP 9: Save the Model & Labels
with open("mental_health_model.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)

with open("mental_health_labels.pkl", "wb") as label_file:
    pickle.dump(y_labels, label_file)

print("✅ Model and labels saved successfully!")
