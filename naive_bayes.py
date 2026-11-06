"""Multinomial Naive Bayes with Laplace smoothing, numpy only."""
import numpy as np


class Vocabulary:
    def __init__(self, min_count=1):
        self.min_count = min_count
        self.word2id = {}
        self.words = []

    def fit(self, docs):
        counts = {}
        for doc in docs:
            for w in doc:
                counts[w] = counts.get(w, 0) + 1
        for w, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            if c >= self.min_count:
                self.word2id[w] = len(self.words)
                self.words.append(w)
        return self

    def encode(self, doc):
        """Token list -> (word ids, counts); out-of-vocabulary words are dropped."""
        ids = [self.word2id[w] for w in doc if w in self.word2id]
        if not ids:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.unique(np.array(ids, dtype=np.int64), return_counts=True)

    def __len__(self):
        return len(self.words)


class NaiveBayes:
    """
    log P(c|d) ~ log P(c) + sum_w n(w,d) * log P(w|c)
    P(w|c) = (count(w,c) + alpha) / (total_words(c) + alpha * |V|)   (Laplace when alpha=1)
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, docs, labels, vocab, classes):
        self.vocab, self.classes = vocab, list(classes)
        V, K = len(vocab), len(self.classes)
        cls_index = {c: i for i, c in enumerate(self.classes)}
        self.counts = np.zeros((V, K), dtype=np.float64)  # count(w, c)
        doc_counts = np.zeros(K)
        for doc, y in zip(docs, labels):
            k = cls_index[y]
            doc_counts[k] += 1
            ids, n = vocab.encode(doc)
            self.counts[ids, k] += n
        self.log_prior = np.log(doc_counts / doc_counts.sum())
        self._smooth()
        return self

    def _smooth(self):
        V = self.counts.shape[0]
        total = self.counts.sum(axis=0)
        self.log_likelihood = np.log(self.counts + self.alpha) - np.log(total + self.alpha * V)

    def set_alpha(self, alpha):
        """Re-smooth the stored counts without recounting (used for alpha sweeps)."""
        self.alpha = alpha
        self._smooth()

    def scores(self, doc):
        ids, n = self.vocab.encode(doc)
        return self.log_prior + n @ self.log_likelihood[ids]

    def predict_one(self, doc):
        return self.classes[int(np.argmax(self.scores(doc)))]

    def predict(self, docs):
        return [self.predict_one(d) for d in docs]
