from google.colab import files
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    df.replace('?', pd.NA, inplace=True)  # Replace '?' with NaN
    df.dropna(subset=['Class'], inplace=True)  # Drop rows with missing class labels
    
    # Convert relevant columns to numeric
    numeric_columns = ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # Create distribution graphs
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_columns, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

else:
    print("DataFrame is empty due to the error.")