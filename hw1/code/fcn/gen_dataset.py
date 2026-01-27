import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

def load_and_split_adult_from_openml():
    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame.copy()
    df = df.replace('?', np.nan).dropna()

    df['target'] = df['class'].apply(lambda x: 1 if x.strip() in [">50K", ">50K."] else 0).astype(int)
    df.drop(columns=["class"], inplace=True)
    
    df_sampled = df.sample(4000, random_state=42).reset_index(drop=True)
    train_df = df_sampled.iloc[:2000].reset_index(drop=True)
    test_df = df_sampled.iloc[2000:].reset_index(drop=True)
    return train_df, test_df

def preprocess_numeric_data(df, numeric_cols):
    df_numeric = df[numeric_cols + ["target"]].copy()
    
    for col in numeric_cols:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    return df_numeric

def generate_adult_numeric_data_official():
    numeric_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    
    train_df, test_df = load_and_split_adult_from_openml()
    
    train_numeric = preprocess_numeric_data(train_df, numeric_cols)
    test_numeric = preprocess_numeric_data(test_df, numeric_cols)
    
    rename_dict = {old: f"feature{i+1}" for i, old in enumerate(numeric_cols)}
    train_numeric.rename(columns=rename_dict, inplace=True)
    test_numeric.rename(columns=rename_dict, inplace=True)
    
    train_sampled = train_numeric.sample(2000, random_state=42).reset_index(drop=True)
    test_sampled = test_numeric.sample(2000, random_state=42).reset_index(drop=True)
    
    train_sampled.to_csv("data/train.csv", index=False)
    test_sampled.to_csv("data/test.csv", index=False)
    print("Generated 'train.csv' and 'test.csv' using only numeric features.")


def main():
    import os
    os.makedirs("data", exist_ok=True)
    generate_adult_numeric_data_official()

if __name__ == "__main__":
    main()
