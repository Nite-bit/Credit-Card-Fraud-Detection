import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# Custom feature names
custom_feature_names = [
    'Transaction Amount',
    'Time Since Last Transaction',
    'User Age',
    'Merchant Rating',
    'Transaction Type Score',
    'Device Trust Score',
    'Location Risk Score',
    'Previous Fraud Count',
    'Card Usage Frequency',
    'Account Age In Months'
]

# Generate synthetic data
X, y = make_classification(
    n_samples=1000,
    n_features=len(custom_feature_names),
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    weights=[0.9, 0.1],
    random_state=42
)

# Scale the 'Transaction Amount' (First feature) to a larger range (e.g., 100 to 100000)
X[:, 0] = np.abs(X[:, 0]) * 10000  # Transaction Amount, now scaled
X[:, 0] = np.round(X[:, 0])  # Round to the nearest integer

# Scale other features to a more realistic range, for example:
# User Age: Between 18 and 80
X[:, 2] = np.abs(X[:, 2]) * 62 + 18  # User Age, scaled between 18 and 80
X[:, 2] = np.round(X[:, 2])  # Round to nearest integer

# Merchant Rating: Between 0 and 5
X[:, 3] = np.abs(X[:, 3]) * 5  # Merchant Rating, scaled between 0 and 5
X[:, 3] = np.round(X[:, 3])  # Round to nearest integer

# Time Since Last Transaction: Between 1 and 365 days
X[:, 1] = np.abs(X[:, 1]) * 365  # Time Since Last Transaction, scaled to 1-365 days
X[:, 1] = np.round(X[:, 1])  # Round to nearest integer

# Device Trust Score: Between 0 and 1
X[:, 5] = np.abs(X[:, 5])  # Device Trust Score, stays between 0 and 1
X[:, 5] = np.round(X[:, 5])  # Round to nearest integer

# Location Risk Score: Between 0 and 10
X[:, 6] = np.abs(X[:, 6]) * 10  # Location Risk Score, scaled between 0 and 10
X[:, 6] = np.round(X[:, 6])  # Round to nearest integer

# Previous Fraud Count: Between 0 and 10
X[:, 7] = np.abs(X[:, 7]) * 10  # Previous Fraud Count, scaled between 0 and 10
X[:, 7] = np.round(X[:, 7])  # Round to nearest integer

# Card Usage Frequency: Between 1 and 1000
X[:, 8] = np.abs(X[:, 8]) * 1000  # Card Usage Frequency, scaled between 1 and 1000
X[:, 8] = np.round(X[:, 8])  # Round to nearest integer

# Account Age in Months: Between 1 and 240 months (20 years)
X[:, 9] = np.abs(X[:, 9]) * 240  # Account Age, scaled between 1 and 240 months
X[:, 9] = np.round(X[:, 9])  # Round to nearest integer

# Ensure the data is positive
X = np.abs(X)

# Create DataFrame
df = pd.DataFrame(X, columns=custom_feature_names)
df['Class'] = y
df.reset_index(inplace=True)
df.rename(columns={'index': 'TransactionID'}, inplace=True)

# Split the data into features and target
features = df.drop(columns=['Class'])
target = df['Class']

# Feature ranges for display in the GUI
feature_ranges = {col: (features[col].min(), features[col].max()) for col in features.columns if col != 'TransactionID'}

# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(
    features.drop(columns=['TransactionID']), target, test_size=0.2, random_state=42, stratify=target)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
xtrain = imputer.fit_transform(xtrain)
xtest = imputer.transform(xtest)

# Scale the data
scaler = MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# Train the model
model = XGBClassifier(eval_metric="logloss")
model.fit(xtrain, ytrain)

# Initialize Tkinter root window
root = tk.Tk()
root.title("🔐 Credit Card Fraud Detection")
root.geometry("750x900")
root.configure(bg="#d3d3d3")

# Fonts for the GUI
font_label = ("Verdana", 11, "bold")
font_entry = ("Verdana", 10)
font_button = ("Verdana", 11, "bold")
font_hint = ("Verdana", 9, "italic")

# Title Label
title_label = tk.Label(root, text="🔐 Fraud Detection System", font=("Verdana", 20, "bold"), bg="#d3d3d3", fg="black")
title_label.pack(pady=20)

# Transaction ID Frame
id_frame = tk.Frame(root, bg="#d3d3d3")
id_frame.pack(pady=5)
tk.Label(id_frame, text="🔎 Transaction ID:", font=font_label, bg="#d3d3d3", fg="black").grid(row=0, column=0, padx=8)
id_entry = tk.Entry(id_frame, font=font_entry, width=15, relief="solid", bd=1)
id_entry.grid(row=0, column=1, padx=5)
search_button = tk.Button(id_frame, text="Search", command=lambda: search_by_id(), font=font_button,
                        bg="#d3d3d3", fg="black", width=10, relief="raised", bd=2)
search_button.grid(row=0, column=2, padx=10)

# Frame for input fields
frame = tk.Frame(root, bg="#ecf0f1", padx=25, pady=20, relief="groove", borderwidth=4)
frame.pack(pady=20)

entries = []
labels = [col for col in features.columns if col != 'TransactionID']

def title_case(text):
    return ' '.join(word.capitalize() for word in text.split('_'))

for i, label in enumerate(labels):
    min_val, max_val = feature_ranges[label]
    range_hint = f"(Min: {min_val:.2f} | Max: {max_val:.2f})"

    tk.Label(frame, text=title_case(label), font=font_label, bg="#ecf0f1", fg="black", anchor="w").grid(row=i, column=0, padx=10, pady=6, sticky="w")
    tk.Label(frame, text=range_hint, font=font_hint, fg="black", bg="#ecf0f1").grid(row=i, column=1, sticky="w")
    entry = tk.Entry(frame, font=font_entry, width=25, relief="sunken", bd=1, bg="white", fg="black")
    entry.grid(row=i, column=2, padx=10, pady=6)
    entries.append(entry)

# Result Label
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Verdana", 15, "bold"), fg="black", bg="#d3d3d3")
result_label.pack(pady=10)

# Buttons Frame
button_frame = tk.Frame(root, bg="#d3d3d3")
button_frame.pack(pady=15)

# Predict Button
predict_button = tk.Button(
    button_frame, text="🛡 Predict", command=lambda: predict_fraud(),
    font=font_button, bg="#2ecc71", fg="black", relief="raised", padx=10, pady=5, width=20
)
predict_button.grid(row=0, column=0, padx=10)

# Clear Button
clear_button = tk.Button(
    button_frame, text="🧹 Clear", command=lambda: clear_fields(),
    font=font_button, bg="#e74c3c", fg="black", relief="raised", padx=10, pady=5, width=20
)
clear_button.grid(row=0, column=1, padx=10)

# Predict Fraud function
def predict_fraud():
    try:
        user_data = [float(entry.get()) for entry in entries]
        user_data = np.array(user_data).reshape(1, -1)
        user_data = scaler.transform(user_data)
        prediction = model.predict(user_data)[0]
        if prediction == 1:
            result_text.set("⚠ Fraudulent Transaction Detected!")
            result_label.config(fg="red")
            messagebox.showwarning("⚠ Alert", "This transaction appears to be FRAUDULENT!")
        else:
            result_text.set("✅ Transaction is Legitimate.")
            result_label.config(fg="green")
            messagebox.showinfo("Safe", "No fraud detected. Transaction is safe.")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

# Clear fields function
def clear_fields():
    for entry in entries:
        entry.delete(0, tk.END)
    result_text.set("")
    id_entry.delete(0, tk.END)

# Search by Transaction ID function
def search_by_id():
    try:
        trans_id = int(id_entry.get())
        row = df[df['TransactionID'] == trans_id]
        if row.empty:
            messagebox.showerror("Not Found", f"No transaction found with ID {trans_id}")
            return
        values = row[features.columns[features.columns != 'TransactionID']].iloc[0].to_list()
        for i, value in enumerate(values):
            entries[i].delete(0, tk.END)
            rounded_val = int(round(value)) if abs(value - round(value)) < 1e-5 or isinstance(value, (int, float)) else value
            entries[i].insert(0, str(rounded_val))
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid numeric Transaction ID.")

# Run the app
root.mainloop()
