from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud

OUT = Path(__file__).parent.parent / "output"
FONT = "C:/Windows/Fonts/tahoma.ttf"  # WordCloud needs a font with Thai glyphs; change this path on macOS/Linux
plt.rcParams["font.family"] = "Tahoma"


def plot_heatmap(y_true, y_pred, labels, title):
    """Confusion matrix: rows = real stars, columns = predicted stars (normalised per row)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(xlabel="Predicted star", ylabel="True star", title=title)
    fig.tight_layout()
    fig.savefig(OUT / "heatmap.png", dpi=150)


def plot_wordclouds(df):
    """One word cloud for all reviews, then one per star (1-5)."""
    groups = [("ทุกคลาส", df)] + [(f"{s} ดาว", df[df["star"] == s]) for s in sorted(df["star"].unique())]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (name, part) in zip(axes.ravel(), groups):
        freq = Counter(t for tokens in part["tokens"] for t in tokens)
        wc = WordCloud(font_path=FONT, width=800, height=500, background_color="white",
                       max_words=100, regexp=r"\S+").generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{name} ({len(part):,} รีวิว)", fontsize=16)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "wordcloud.png", dpi=120)


def plot_alpha(res):
    """Accuracy / macro-F1 against the smoothing value alpha."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = res["alpha"].astype(str)
    ax.plot(x, res["Acc"], "o-", label="Accuracy")
    ax.plot(x, res["F1"], "s-", label="Macro F1")
    for i, r in res.iterrows():
        ax.annotate(f"{r.Acc:.3f}", (i, r.Acc), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.annotate(f"{r.F1:.3f}", (i, r.F1), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    ax.set(xlabel="alpha (1 = Laplace smoothing)", ylabel="score", ylim=(0, 1), title="Naive Bayes: effect of smoothing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "alpha.png", dpi=150)


def visualize_data(train, y_test, pred, labels, res):
    OUT.mkdir(exist_ok=True)
    plot_wordclouds(train)
    plot_heatmap(y_test, pred, labels, "Naive Bayes (Laplace, alpha=1) on test set")
    plot_alpha(res)
    print("saved", *sorted(p.name for p in OUT.glob("*.png")))
