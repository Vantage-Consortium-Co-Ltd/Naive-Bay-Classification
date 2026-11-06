import os

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# shortcut: Windows-only Thai font path, switch to a bundled font for other OSes
THAI_FONT = "C:/Windows/Fonts/leelawad.ttf"
OUT_DIR = "output"


def plot_heatmap(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ticks = range(5)
    labels = [f"{i + 1} star" for i in ticks]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in ticks:
        for j in ticks:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    ax.set_title("Confusion Matrix Heatmap")
    fig.savefig(f"{OUT_DIR}/heatmap.png", dpi=150, bbox_inches="tight")


def plot_wordclouds(vocab, counts):
    words = sorted(vocab, key=vocab.get)
    groups = {"all": counts.sum(axis=0)}
    groups.update({f"{i + 1}star": counts[i] for i in range(5)})
    for name, freq in groups.items():
        top = {w: f for w, f in zip(words, freq) if f > 0}
        wc = WordCloud(font_path=THAI_FONT, width=1000, height=500,
                       background_color="white").generate_from_frequencies(top)
        wc.to_file(f"{OUT_DIR}/wordcloud_{name}.png")


def visualize_data(vocab, counts, cm, accuracy):
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_heatmap(cm)
    plot_wordclouds(vocab, counts)
    print(f"Accuracy: {accuracy * 100:.2f}%")
