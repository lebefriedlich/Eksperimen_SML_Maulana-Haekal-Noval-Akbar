"""
Pipeline Otomatisasi Preprocessing Dataset Heart Disease
Nama Siswa: Maulana Haekal Noval Akbar
Username: lebefriedlich
Course: Membangun Sistem Machine Learning - Dicoding
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path: str, output_dir: str):
    print(f"[INFO] Membaca raw dataset dari: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file tidak ditemukan: {input_path}")

    df_raw = pd.read_csv(input_path)
    print(f"[INFO] Dimensi dataset mentah: {df_raw.shape}")

    # 1. Bersihkan nama kolom
    df_clean = df_raw.copy()
    df_clean.columns = [c.strip().replace(" ", "_") for c in df_clean.columns]

    target_col = "Heart_Disease_Status"
    if target_col not in df_clean.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan dalam dataset!")

    feature_cols = [c for c in df_clean.columns if c != target_col]
    num_cols = df_clean[feature_cols].select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df_clean[feature_cols].select_dtypes(include=["object"]).columns

    # 2. Imputasi missing values
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # 3. Encoding target (Yes -> 1, No -> 0)
    if df_clean[target_col].dtype == object:
        df_clean[target_col] = df_clean[target_col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # 4. One-Hot Encoding fitur kategorikal
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    X_encoded = pd.get_dummies(X, drop_first=True)
    # Konversi boolean kolom dummies ke int jika ada
    X_encoded = X_encoded.astype({col: "int64" for col in X_encoded.select_dtypes(include="bool").columns})

    # Simpan full processed sebelum scaling untuk referensi
    full_processed_unscaled = pd.concat([X_encoded, y], axis=1)

    # 5. Split train-test dengan stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Standarisasi fitur dengan StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Bentuk dataframe terstandarisasi
    train_df = pd.DataFrame(X_train_scaled, columns=X_encoded.columns)
    train_df[target_col] = y_train.values

    test_df = pd.DataFrame(X_test_scaled, columns=X_encoded.columns)
    test_df[target_col] = y_test.values

    # Full processed scaled
    full_scaled = scaler.transform(X_encoded)
    full_processed = pd.DataFrame(full_scaled, columns=X_encoded.columns)
    full_processed[target_col] = y.values

    # 7. Simpan dataset
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "heart_train.csv")
    test_path = os.path.join(output_dir, "heart_test.csv")
    processed_path = os.path.join(output_dir, "heart_processed.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    full_processed.to_csv(processed_path, index=False)

    print(f"[SUCCESS] Dataset preprocessing selesai!")
    print(f" - Train dataset      : {train_path} ({train_df.shape})")
    print(f" - Test dataset       : {test_path} ({test_df.shape})")
    print(f" - Processed dataset  : {processed_path} ({full_processed.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate Data Preprocessing for Heart Disease Dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="heart_disease_raw/heart_disease.csv",
        help="Path ke file raw dataset CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="heart_disease_preprocessing",
        help="Direktori output untuk menyimpan file CSV hasil preprocessing",
    )

    args = parser.parse_args()
    preprocess_data(args.input, args.output)
