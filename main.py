import csv
import os
import time

import numpy as np

import data
import viz
from naive_bayes import NaiveBayes, Vocabulary

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CLASSES = [1, 2, 3, 4, 5]
os.makedirs(OUT, exist_ok=True)
log_lines = []


def log(s=""):
    print(s)
    log_lines.append(s)


def metrics(y_true, y_pred, classes=CLASSES):
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (np.array(y_true) - 1, np.array(y_pred) - 1), 1)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(axis=0), 1)
    rec = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return cm, tp.sum() / cm.sum(), prec, rec, f1


def report(name, y_true, y_pred):
    cm, acc, prec, rec, f1 = metrics(y_true, y_pred)
    log(f"{name}: accuracy={acc:.4f}  macro-F1={f1.mean():.4f}  "
        f"MAE={np.abs(np.array(y_true) - np.array(y_pred)).mean():.4f}")
    return cm, acc, prec, rec, f1


def main():
    # ---------- 1. load + tokenize ----------
    texts, labels = data.load_train()
    t0 = time.perf_counter()
    docs = data.tokenize_all(texts, "train_tokens")
    log(f"Loaded {len(texts)} reviews, tokenized in {time.perf_counter() - t0:.1f}s (cached after first run)")

    # stratified 80/20 split
    rng = np.random.default_rng(42)
    labels_arr = np.array(labels)
    val_idx = []
    for c in CLASSES:
        idx = rng.permutation(np.where(labels_arr == c)[0])
        val_idx.extend(idx[: len(idx) // 5])
    val_mask = np.zeros(len(labels), bool)
    val_mask[val_idx] = True
    tr_docs = [d for d, m in zip(docs, val_mask) if not m]
    tr_y = [y for y, m in zip(labels, val_mask) if not m]
    va_docs = [d for d, m in zip(docs, val_mask) if m]
    va_y = [y for y, m in zip(labels, val_mask) if m]
    log(f"Train {len(tr_docs)} / validation {len(va_docs)}")

    # ---------- 2. vocabulary + training ----------
    t0 = time.perf_counter()
    vocab = Vocabulary().fit(tr_docs)
    t_vocab = time.perf_counter() - t0
    log(f"Vocabulary size: {len(vocab)} (built from the training split only) in {t_vocab:.2f}s")

    t0 = time.perf_counter()
    model = NaiveBayes(alpha=1.0).fit(tr_docs, tr_y, vocab, CLASSES)
    t_train = time.perf_counter() - t0
    log(f"Training time (counts + Laplace smoothing): {t_train:.2f}s")

    # ---------- 3. evaluation ----------
    t0 = time.perf_counter()
    pred = model.predict(va_docs)
    t_pred = time.perf_counter() - t0
    log(f"Prediction: {len(va_docs)} docs in {t_pred:.2f}s ({len(va_docs) / t_pred:.0f} docs/s)")
    log()
    majority = max(CLASSES, key=tr_y.count)
    report(f"Baseline (always {majority}*)", va_y, [majority] * len(va_y))
    cm, acc, prec, rec, f1 = report("Naive Bayes (alpha=1)", va_y, pred)
    log("\nclass  precision  recall    F1    support")
    for i, c in enumerate(CLASSES):
        log(f"  {c}*    {prec[i]:.3f}     {rec[i]:.3f}   {f1[i]:.3f}   {cm[i].sum()}")

    # ---------- 4. alpha sweep (re-smooths stored counts; no recount) ----------
    log("\nalpha sweep (validation):")
    log("  alpha     accuracy  macro-F1")
    best = (0, 1.0)
    for a in [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        model.set_alpha(a)
        _, ac, _, _, f = metrics(va_y, model.predict(va_docs))
        log(f"  {a:<8}  {ac:.4f}    {f.mean():.4f}")
        if ac > best[0]:
            best = (ac, a)
    log(f"  best accuracy at alpha={best[1]}")
    model.set_alpha(1.0)

    # ---------- 5. learning curve / scaling ----------
    log("\nscaling with training size (alpha=1):")
    log("  fraction  docs    vocab   train(s)  accuracy")
    for frac in [0.1, 0.25, 0.5, 1.0]:
        n = int(len(tr_docs) * frac)
        t0 = time.perf_counter()
        v = Vocabulary().fit(tr_docs[:n])
        m = NaiveBayes(1.0).fit(tr_docs[:n], tr_y[:n], v, CLASSES)
        dt = time.perf_counter() - t0
        log(f"  {frac:<8}  {n:<6}  {len(v):<6}  {dt:<8.2f}  {metrics(va_y, m.predict(va_docs))[1]:.4f}")

    # ---------- 6. visualisations ----------
    viz.confusion_heatmap(cm, CLASSES, os.path.join(OUT, "confusion_heatmap.png"))
    viz.probability_table_heatmap(model.counts, vocab.words, CLASSES, model.log_prior, model.alpha,
                                  os.path.join(OUT, "probability_table_heatmap.png"),
                                  os.path.join(OUT, "probability_table.csv"))
    viz.word_heatmap(model.counts, vocab.words, CLASSES, os.path.join(OUT, "word_heatmap.png"))
    viz.class_word_clouds(model.counts, vocab.words, CLASSES, os.path.join(OUT, "wordclouds.png"))
    log("\nTop 10 class-specific words:")
    for c, t in zip(CLASSES, viz.top_words(model.counts, vocab.words, 10)):
        log(f"  {c}*: " + " ".join(w for w, _ in t))

    # ---------- 7. test-set submission (trained on all labelled data) ----------
    ids, test_texts = data.load_test()
    test_docs = data.tokenize_all(test_texts, "test_tokens")
    full_vocab = Vocabulary().fit(docs)
    full = NaiveBayes(1.0).fit(docs, labels, full_vocab, CLASSES)
    with open(os.path.join(OUT, "submission.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reviewID", "rating"])
        for i, p in zip(ids, full.predict(test_docs)):
            w.writerow([i, p])
    log(f"\nWrote submission for {len(ids)} test reviews")

    with open(os.path.join(OUT, "performance.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
