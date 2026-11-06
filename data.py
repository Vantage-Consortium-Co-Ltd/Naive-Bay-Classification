"""Loading and Thai tokenization (pythainlp only)."""
import csv
import os
import pickle
from multiprocessing import Pool

from pythainlp import word_tokenize

csv.field_size_limit(10**9)
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_dataset")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def load_train(path=os.path.join(DIR, "w_review_train.csv")):
    texts, labels = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter=";"):
            if len(row) == 2:
                texts.append(row[0])
                labels.append(int(row[1]))
    return texts, labels


def load_test(path=os.path.join(DIR, "test_file.csv")):
    ids, texts = [], []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if len(row) == 2:
                ids.append(row[0])
                texts.append(row[1])
    return ids, texts


def tokenize(text):
    # newmm dictionary-based tokenizer; keep only tokens containing a letter/digit
    return [t for t in (w.strip().lower() for w in word_tokenize(text, engine="newmm", keep_whitespace=False))
            if any(ch.isalnum() for ch in t)]


def tokenize_all(texts, name):
    """Tokenize in parallel; cached on disk because newmm is slow on 60MB of text."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    with Pool() as pool:
        docs = pool.map(tokenize, texts, chunksize=200)
    with open(path, "wb") as f:
        pickle.dump(docs, f)
    return docs
