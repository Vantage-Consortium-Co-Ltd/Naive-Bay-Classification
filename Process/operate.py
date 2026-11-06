import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from pythainlp.tokenize import longest
from pythainlp.tokenize.longest import LongestMatchTokenizer, _TRAILING_CHAR

SEED = 42
ALPHA = 1.0
VAL_FRAC = 0.2
CLS = [1, 2, 3, 4, 5]


# Replace O(L^2) prefix scan with O(max_word_len) Trie.prefixes lookup (identical output, 100x faster)
def _is_next_word_valid_fast(self, text: str, begin_pos: int) -> bool:
    text = text[begin_pos:].strip()
    if not text or self._LongestMatchTokenizer__search_nonthai(text):
        return True
    return bool(self._LongestMatchTokenizer__trie.prefixes(text, 0))


def _longest_matching_fast(self, text: str, begin_pos: int) -> str:
    text = text[begin_pos:]
    match = self._LongestMatchTokenizer__search_nonthai(text)
    if match:
        return match
    word = word_valid = None
    for w in self._LongestMatchTokenizer__trie.prefixes(text, 0):
        word = w
        if _is_next_word_valid_fast(self, text, len(w)):
            word_valid = w
    if not word:
        return ""
    word_valid = word_valid or word
    n = len(word_valid)
    return text[:n + 1] if n < len(text) and text[n] in _TRAILING_CHAR else word_valid


LongestMatchTokenizer._LongestMatchTokenizer__is_next_word_valid = _is_next_word_valid_fast
LongestMatchTokenizer._LongestMatchTokenizer__longest_matching = _longest_matching_fast
longest._tokenizers.clear()


def preprocess_text(review: str) -> list[str]:
    return longest.segment(str(review).replace("\n", " ").strip())


def tokenize_with_cache(texts, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return joblib.load(cache_path)
    token_lists = []
    for i, review in enumerate(texts, 1):
        token_lists.append(preprocess_text(review))
        if i % 5000 == 0:
            print(f"tokenized {i}/{len(texts)} reviews")
    joblib.dump(token_lists, cache_path)
    return token_lists


def split_data(tok_train, y, val_frac=VAL_FRAC, seed=SEED):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in CLS:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = int(np.floor(val_frac * len(idx)))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    tok_tr = [tok_train[i] for i in train_idx]
    tok_val = [tok_train[i] for i in val_idx]
    return tok_tr, tok_val, y[train_idx], y[val_idx]


def train_model(tok_tr, y_tr, vocab, alpha=ALPHA):
    v_size = len(vocab)
    counts = np.zeros((5, v_size), dtype=np.int64)
    for tokens, rating in zip(tok_tr, y_tr):
        row = counts[rating - 1]
        for t in tokens:
            col = vocab.get(t)
            if col is not None:
                row[col] += 1
    class_doc_counts = np.array([(y_tr == c).sum() for c in CLS], dtype=np.float64)
    log_prior = np.log(class_doc_counts / len(y_tr))
    denomin = counts.sum(axis=1)[:, None]
    log_likelihood = np.log((counts + alpha) / (denomin + alpha * v_size))
    return counts, log_prior, log_likelihood


def predict_tokens(tokens, vocab, log_prior, log_likelihood):
    cols = [vocab[t] for t in tokens if t in vocab]
    log_post = log_prior + log_likelihood[:, cols].sum(axis=1)
    return CLS[int(np.argmax(log_post))]


def evaluate_model(tok_val, y_val, vocab, log_prior, log_likelihood):
    y_pred = np.array([predict_tokens(t, vocab, log_prior, log_likelihood) for t in tok_val], dtype=int)
    cm = np.zeros((5, 5), dtype=np.int64)
    for t, p in zip(y_val, y_pred):
        cm[t - 1, p - 1] += 1
    acc = float((y_pred == y_val).mean())
    rows = []
    for c in CLS:
        tp = float(cm[c - 1, c - 1])
        col_sum = float(cm[:, c - 1].sum())
        row_sum = float(cm[c - 1, :].sum())
        prec = tp / col_sum if col_sum > 0 else 0.0
        rec = tp / row_sum if row_sum > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rows.append({"rating": f"{c}★", "support": int(row_sum), "precision": prec, "recall": rec, "f1": f1})
    metrics_df = pd.DataFrame(rows)
    macro_f1 = float(metrics_df["f1"].mean())
    return cm, acc, macro_f1, metrics_df


def naive_bayes(data):
    df_train, df_test = data["train"], data["test"]
    tok_train = tokenize_with_cache(df_train["review"].tolist(), Path("cache/tokens_train.joblib"))
    tok_test = tokenize_with_cache(df_test["review"].tolist(), Path("cache/tokens_test.joblib"))
    vocab = {w: i for i, w in enumerate(sorted(set(w for tokens in tok_train for w in tokens)))}
    y = df_train["rating"].to_numpy(dtype=int)
    tok_tr, tok_val, y_tr, y_val = split_data(tok_train, y)
    counts, log_prior, log_likelihood = train_model(tok_tr, y_tr, vocab)
    cm, acc, macro_f1, metrics_df = evaluate_model(tok_val, y_val, vocab, log_prior, log_likelihood)
    print(f"Validation Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")
    print(metrics_df.to_string(index=False))

    test_preds = [predict_tokens(t, vocab, log_prior, log_likelihood) for t in tok_test]
    pred_df = pd.DataFrame({"reviewID": df_test["reviewID"].astype(int), "rating": test_preds})
    pred_df.to_csv("predictions.csv", index=False, encoding="utf-8")
    return tok_train, df_train["rating"], cm, acc, macro_f1
