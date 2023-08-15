from google.colab import files
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Upload a local Excel file
uploaded = files.upload()

# Choose the uploaded file name
uploaded_file_name = list(uploaded.keys())[0]

try:
    df = pd.read_excel(uploaded_file_name, sheet_name='cell_samples')
    print("Data loaded successfully.")
except Exception as e:
    print("Error:", e)
    df = None

if df is not None:
    # Assume columns 'Clump', 'UnifSize', ..., 'Mit', and 'Class' exist
    df.replace('?', pd.NA, inplace=True) 
    df.dropna(subset=['Class'], inplace=True)  # Drop rows with missing class labels
    
    # Convert relevant columns to numeric
    numeric_columns = ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Split the data into features and labels
    X = df.drop('Class', axis=1)  # Features
    y = df['Class']               # Labels

    # Handle NaN values in the features
    X.fillna(X.mean(), inplace=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train a Support Vector Machine (SVM) classifier
    classifier = SVC(kernel='linear', random_state=42)
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
else:
    print("DataFrame is empty due to the error.")
