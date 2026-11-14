from datasets import load_dataset


def load_data():
    """Wongnai reviews -> (train, test) DataFrames with columns review, star (1-5).
    The dataset stores star_rating as 0-4, so add 1 to get the real star count."""
    data = load_dataset("Highgroundbkk/wongnai_reviews")
    out = []
    for split in ("train", "test"):
        df = data[split].to_pandas().rename(columns={"review_body": "review"})
        df["star"] = df.pop("star_rating") + 1
        out.append(df)
    return out
