import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import joblib

print("Script started...")

try:
    df = pd.read_csv("train.csv")
except FileNotFoundError:
    print("Error: train.csv not found. Please make sure it's in the same folder.")
    exit()

df.drop("Loan_ID", axis=1, inplace=True)

for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']:
    df[col] = df[col].fillna(df[col].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

df["Dependents"] = df["Dependents"].replace('3+', '3').astype(int)
df["Loan_Status"] = df["Loan_Status"].map({'Y': 1, 'N': 0})
print("Data cleaning complete.")

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

categorical_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

preprocessor = ColumnTransformer(
    transformers=[
        ('onehotencoder', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features)
    ],
    remainder='passthrough' 
)
print("Preprocessor created correctly.")

model = LogisticRegression(max_iter=1500, random_state=42)

print("Fitting preprocessor and training the model...")
X_transformed = preprocessor.fit_transform(X)
model.fit(X_transformed, y)

joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(model, 'model.joblib')

print("\nSUCCESS! ✅ Preprocessor and Model have been saved successfully!")
print("You can now run the Streamlit app.")