import pickle
import re
import time
from pathlib import Path

import numpy as np
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score

CACHE = Path(__file__).parent.parent / "cache"
STOPWORDS = set(thai_stopwords())
# str.isalnum() is False for Thai vowel/tone marks ("อร่อย".isalnum() -> False), so match by character range instead
WORD = re.compile(r"^[฀-๿a-z0-9]+$")


def preprocess_text(text):
    # Tokenize the text using PyThaiNLP (newmm: dictionary-based, much faster than "longest" on 46k reviews)
    tokens = word_tokenize(text.lower(), engine="newmm", keep_whitespace=False)

    # Keep Thai/English words and numbers, drop punctuation, emoji and stopwords
    return [t for t in tokens if WORD.match(t) and t not in STOPWORDS]


def preprocess_data(df, name):
    """Tokenize the 'review' column once and cache it, since tokenizing takes a few minutes."""
    path = CACHE / f"tokens_{name}.pkl"
    if path.exists():
        df["tokens"] = pickle.loads(path.read_bytes())
        return df
    df["tokens"] = df["review"].map(preprocess_text)
    CACHE.mkdir(exist_ok=True)
    path.write_bytes(pickle.dumps(df["tokens"].tolist()))
    return df


def vectorize(train, test, min_df=2):
    """Bag-of-words counts. The vocabulary comes from the train set only; words seen only once are dropped."""
    vec = CountVectorizer(analyzer=lambda tokens: tokens, min_df=min_df)
    return vec.fit_transform(train["tokens"]), vec.transform(test["tokens"]), vec


class NaiveBayes:
    """Multinomial Naive Bayes written with NumPy.

    P(word | class) = (count(word, class) + alpha) / (total words in class + alpha * |V|)
    alpha = 0 is plain maximum likelihood: a word never seen in a class makes that class impossible (log 0 = -inf).
    alpha = 1 is Laplace smoothing."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        counts = np.vstack([np.asarray(X[y == c].sum(0)).ravel() for c in self.classes])  # (K, V)
        with np.errstate(divide="ignore"):
            self.log_prior = np.log(np.array([(y == c).mean() for c in self.classes]))
            self.log_like = np.log(counts + self.alpha) - np.log(counts.sum(1, keepdims=True) + self.alpha * X.shape[1])
        return self

    def predict(self, X):
        # sparse X only multiplies the words a review contains, so unseen-word -inf only hits the classes that never saw it
        scores = np.asarray(X @ self.log_like.T) + self.log_prior
        return self.classes[scores.argmax(1)]


def split_data(train, test):
    return train["star"].to_numpy(), test["star"].to_numpy()


def train_model(X, y, alpha=1.0):
    return NaiveBayes(alpha).fit(X, y)


def evaluate_model(model, X, y):
    t0 = time.perf_counter(); pred = model.predict(X)
    return pred, dict(Acc=accuracy_score(y, pred), F1=f1_score(y, pred, average="macro"),
                      Predict_s=time.perf_counter() - t0)


def alpha_experiment(Xtr, ytr, Xte, yte, alphas=(0, 0.01, 0.1, 0.5, 1, 2, 5)):
    """Performance of Naive Bayes for each smoothing value alpha (alpha=1 is Laplace)."""
    rows = []
    for a in alphas:
        t0 = time.perf_counter(); model = train_model(Xtr, ytr, a)
        fit_s = time.perf_counter() - t0
        _, m = evaluate_model(model, Xte, yte)
        rows.append(dict(alpha=a, **m, Fit_s=fit_s))
    return rows
