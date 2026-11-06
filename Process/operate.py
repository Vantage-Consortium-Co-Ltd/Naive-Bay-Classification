import re
from multiprocessing import Pool

import numpy as np
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords

STOPWORDS = set(thai_stopwords())
HAS_WORD_CHAR = re.compile(r"[\u0E00-\u0E7Fa-zA-Z0-9]")
N_CLASSES = 5  # 1-5 stars; dataset labels are 0-4 and index the matrix rows directly


def preprocess_text(text):
    """Step 1: longest-matching tokenization; drop whitespace/punctuation tokens."""
    tokens = word_tokenize(text, engine="newmm")
    return [t.lower() for t in tokens if HAS_WORD_CHAR.search(t)]


def preprocess_all(texts):
    with Pool() as pool:
        return pool.map(preprocess_text, texts, chunksize=200)


def build_vocab(token_lists):
    """Step 2: vocabulary without stop words."""
    words = {t for tokens in token_lists for t in tokens} - STOPWORDS
    return {w: i for i, w in enumerate(sorted(words))}


def count_matrix(token_lists, labels, vocab):
    """Step 3-4: rows = star class, columns = vocab word, cells = word counts."""
    m = np.zeros((N_CLASSES, len(vocab)))
    for tokens, y in zip(token_lists, labels):
        for t in tokens:
            if t in vocab:
                m[y, vocab[t]] += 1
    return m


def train(counts, labels):
    """Step 4-5: Laplace smoothing (+1) then log P(word|class) and log P(class)."""
    smoothed = counts + 1
    log_likelihood = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
    prior = np.bincount(labels, minlength=N_CLASSES) / len(labels)
    return log_likelihood, np.log(prior)


def predict(token_lists, vocab, log_likelihood, log_prior):
    """Step 6: argmax over classes of log prior + sum of log likelihood of known words."""
    preds = []
    for tokens in token_lists:
        idx = [vocab[t] for t in tokens if t in vocab]
        preds.append(np.argmax(log_prior + log_likelihood[:, idx].sum(axis=1)))
    return np.array(preds)


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def naive_bayes(data):
    train_x = preprocess_all(data["train"]["review_body"])
    test_x = preprocess_all(data["test"]["review_body"])
    train_y = np.array(data["train"]["star_rating"])
    test_y = np.array(data["test"]["star_rating"])

    vocab = build_vocab(train_x)
    counts = count_matrix(train_x, train_y, vocab)
    log_likelihood, log_prior = train(counts, train_y)
    pred = predict(test_x, vocab, log_likelihood, log_prior)

    accuracy = float((pred == test_y).mean())
    return vocab, counts, confusion_matrix(test_y, pred), accuracy
