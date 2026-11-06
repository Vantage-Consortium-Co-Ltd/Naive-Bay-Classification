from pathlib import Path
import pandas as pd


def load_data(train_path="Dataset/w_review_train.csv", test_path="Dataset/test_file.csv"):
    train_p, test_p = Path(train_path), Path(test_path)
    if train_p.exists() and test_p.exists():
        df_train = pd.read_csv(train_p, sep=";", header=None, names=["review", "rating"], encoding="utf-8")
        df_test = pd.read_csv(test_p, sep=";", encoding="utf-8")
        return {"train": df_train, "test": df_test}
    from datasets import load_dataset
    ds = load_dataset("Highgroundbkk/wongnai_reviews")
    df_train = pd.DataFrame({
        "review": ds["train"]["review_body"],
        "rating": [int(r) + 1 for r in ds["train"]["star_rating"]],
    })
    df_test = pd.DataFrame({
        "reviewID": range(1, len(ds["test"]["review_body"]) + 1),
        "review": ds["test"]["review_body"],
    })
    return {"train": df_train, "test": df_test}
