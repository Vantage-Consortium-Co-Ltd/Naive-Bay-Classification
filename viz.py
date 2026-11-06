"""Heat maps and word clouds drawn with matplotlib only (no wordcloud package)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

THAI_FONT = "Noto Sans Thai"
plt.rcParams["font.family"] = [THAI_FONT, "DejaVu Sans"]


def confusion_heatmap(cm, classes, path, title="Confusion matrix (validation set)"):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, sub, fmt in ((axes[0], cm, "counts", "{:d}"), (axes[1], norm, "row-normalised (recall)", "{:.2f}")):
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(classes)), [f"{c}★" for c in classes])
        ax.set_yticks(range(len(classes)), [f"{c}★" for c in classes])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(sub)
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = mat[i, j]
                ax.text(j, i, fmt.format(int(v) if mat is cm else v), ha="center", va="center",
                        color="white" if v > mat.max() * 0.55 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def discriminative_scores(counts, alpha=1.0):
    """score[w, c] = p(w|c) * log(p(w|c) / p(w|not c)); high = frequent AND specific to class c."""
    V = counts.shape[0]
    total = counts.sum(axis=0)
    p = (counts + alpha) / (total + alpha * V)
    rest = counts.sum(axis=1, keepdims=True) - counts
    q = (rest + alpha) / (rest.sum(axis=0) + alpha * V)
    return p * np.log(p / q)


def top_words(counts, words, k, min_count=20, min_len=2):
    sc = discriminative_scores(counts)
    ok = (counts.sum(axis=1) >= min_count) & np.array([len(w) >= min_len for w in words])
    out = []
    for c in range(counts.shape[1]):
        s = np.where(ok & (sc[:, c] > 0), sc[:, c], -np.inf)
        idx = np.argsort(-s)[:k]
        out.append([(words[i], float(s[i])) for i in idx if np.isfinite(s[i])])
    return out


def word_heatmap(counts, words, classes, path, per_class=6):
    """Rows = most class-specific words, columns = classes, colour = log-odds vs the other classes."""
    tops = top_words(counts, words, per_class)
    chosen = []
    for t in tops:
        for w, _ in t:
            if w not in chosen:
                chosen.append(w)
    idx = [words.index(w) for w in chosen]
    total = counts.sum(axis=0)
    p = (counts[idx] + 1) / (total + len(words))
    mat = np.log(p / p.mean(axis=1, keepdims=True))  # log-ratio vs the mean class
    lim = np.abs(mat).max()
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(chosen) + 1.5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(classes)), [f"{c}★" for c in classes])
    ax.set_yticks(range(len(chosen)), chosen)
    ax.set_title("Top class-specific words\nlog P(w|class) / mean over classes")
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def word_cloud(items, ax, W=900, H=600, seed=0, cmap="viridis"):
    """Greedy spiral placement; font size is proportional to score, boxes never overlap."""
    rng = np.random.default_rng(seed)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scores = np.array([s for _, s in items])
    sizes = 14 + 80 * (scores / scores.max()) ** 0.7
    colors = matplotlib.colormaps[cmap](np.linspace(1.0, 0.45, len(items)))
    placed = []  # (x0, y0, x1, y1) in data coords == pixels of the axes
    inv = ax.transData.inverted()
    for n, ((word, _), fs) in enumerate(zip(items, sizes)):
        t = ax.text(W / 2, H / 2, word, fontsize=fs, ha="center", va="center", color=colors[n])
        bb = t.get_window_extent(renderer)
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        w, h = (x1 - x0) + 6, (y1 - y0) + 6
        theta0 = rng.uniform(0, 2 * np.pi)
        for step in range(8000):
            r = 0.25 * step
            th = theta0 + 0.1 * step
            cx = W / 2 + r * np.cos(th) * 1.4  # wider than tall
            cy = H / 2 + r * np.sin(th)
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
                continue
            if all(box[2] < p[0] or box[0] > p[2] or box[3] < p[1] or box[1] > p[3] for p in placed):
                placed.append(box)
                t.set_position((cx, cy))
                break
        else:
            t.remove()


def class_word_clouds(counts, words, classes, path, k=70):
    tops = top_words(counts, words, k)
    cmaps = ["Reds", "Oranges", "YlGnBu", "GnBu", "Greens"]
    fig, axes = plt.subplots(3, 2, figsize=(16, 15), dpi=100)
    for ax in axes.flat[len(classes):]:
        ax.axis("off")
    for ax, c, items, cm in zip(axes.flat, classes, tops, cmaps):
        word_cloud(items, ax, cmap=cm)
        ax.set_title(f"{c} star{'s' if c > 1 else ''}", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    # one file per class as well
    for c, items, cm in zip(classes, tops, cmaps):
        f, ax = plt.subplots(figsize=(9, 6), dpi=100)
        ax.set_position([0, 0, 1, 1])
        word_cloud(items, ax, cmap=cm)
        f.savefig(path.replace(".png", f"_{c}star.png"), dpi=100)
        plt.close(f)


def probability_table_heatmap(counts, words, classes, log_prior, alpha, path, csv_path, n_freq=40, per_class=8):
    """Heat map of the Naive Bayes table P(word|class) with every cell annotated; full table saved to CSV."""
    V = counts.shape[0]
    P = (counts + alpha) / (counts.sum(axis=0) + alpha * V)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["word"] + [f"P(w|{c}star)" for c in classes])
        w.writerow(["<prior P(class)>"] + [f"{p:.6f}" for p in np.exp(log_prior)])
        for i, word in enumerate(words):
            w.writerow([word] + [f"{p:.3e}" for p in P[i]])

    freq = list(np.argsort(-counts.sum(axis=1))[:n_freq])
    spec = [words.index(wd) for t in top_words(counts, words, per_class) for wd, _ in t]
    groups = [("Most frequent words", freq), ("Most class-specific words", list(dict.fromkeys(spec)))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 0.3 * n_freq + 2))
    for ax, (title, idx) in zip(axes, groups):
        mat = P[idx] * 1000  # per 1,000 words
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=np.percentile(mat, 98))
        ax.set_xticks(range(len(classes)), [f"{c}★" for c in classes])
        ax.set_yticks(range(len(idx)), [words[i] for i in idx])
        ax.set_title(title)
        ax.xaxis.tick_top()
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[r, c] > im.norm.vmax * 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04, label="P(word | class) × 1000")
    prior = "  ".join(f"{c}★={np.exp(p):.3f}" for c, p in zip(classes, log_prior))
    fig.suptitle(f"Naive Bayes probability table (Laplace α={alpha:g})   priors: {prior}", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
