import pandas as pd
import os

RAW_DATA_PATH = "../data/raw/"
PROCESSED_DATA_PATH = "../data/processed/"

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def load_tsv(file_path):
    df = pd.read_csv(
        file_path,
        sep="\t",
        names=["tweet_id", "label"],
        dtype={"tweet_id": str}
    )
    return df

def preprocess_labels(df):
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["sexism", "racism", "hate", "none"])]
    return df

def main():
    df1 = load_tsv(RAW_DATA_PATH + "waseem_dataset.tsv")
    df2 = load_tsv(RAW_DATA_PATH + "extended_dataset.tsv")

    df1 = preprocess_labels(df1)
    df2 = preprocess_labels(df2)

    combined_df = pd.concat([df1, df2], ignore_index=True)

    print("Dataset summary:")
    print(combined_df["label"].value_counts())

    combined_df.to_csv(
        PROCESSED_DATA_PATH + "combined_dataset.csv",
        index=False
    )

    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
