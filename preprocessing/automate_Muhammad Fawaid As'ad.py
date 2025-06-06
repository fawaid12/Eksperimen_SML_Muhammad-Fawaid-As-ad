import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Ambil direktori saat ini (lokasi file .py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(path):
    full_path = os.path.join(BASE_DIR, "..", "diabetes.csv")
    return pd.read_csv(full_path)

def clean_data(df):
    # Menghapus data kosong
    df.dropna(axis=0, inplace=True)

    # Menghapus data duplikat
    df.drop_duplicates(inplace=True)

    # Ganti nilai 0 yang tidak logis dengan NaN
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)

    # Imputasi dengan median
    df.fillna(df.median(), inplace=True)
    
    return df

def scale_data(df):
    if 'Outcome' not in df.columns:
        raise ValueError("Kolom 'Outcome' tidak ditemukan pada dataset.")
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def preprocess_pipeline(input_path='diabetes.csv'):
    df = load_data(input_path)
    df_clean = clean_data(df)
    X_scaled, y = scale_data(df_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def save_split_data(X_train, X_test, y_train, y_test, output_dir='preprocessing'):
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

def main():
    X_train, X_test, y_train, y_test = preprocess_pipeline()
    save_split_data(X_train, X_test, y_train, y_test)
    print("Preprocessing selesai dan file disimpan.")

if __name__ == "__main__":
    main()
