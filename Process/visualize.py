import os
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Leelawadee UI", "Tahoma", "DejaVu Sans"]

THAI_FONT = r"C:\Windows\Fonts\leelawui.ttf"
OUT_DIR = "figures"
CLS = [1, 2, 3, 4, 5]
SEED = 42


def plot_heatmap(cm, acc, macro_f1, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    tick_labels = [f"{c}★" for c in CLS]
    ax.set_xticks(range(5), tick_labels, fontname="DejaVu Sans")
    ax.set_yticks(range(5), tick_labels, fontname="DejaVu Sans")
    ax.set_xlabel("Predicted Rating")
    ax.set_ylabel("True Rating")
    ax.set_title(f"Confusion Matrix Heatmap (Acc={acc:.3f}, Macro-F1={macro_f1:.3f})")
    thresh = cm.max() / 2.0
    for i in range(5):
        for j in range(5):
            ax.text(
                j, i, f"{cm[i, j]}", ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_wordclouds(tok_train, ratings, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    freq = {"all": Counter(), "1": Counter(), "2": Counter(), "3": Counter(), "4": Counter(), "5": Counter()}
    for tokens, rating in zip(tok_train, ratings):
        filtered = [t for t in tokens if t.strip()]
        freq["all"].update(filtered)
        freq[str(int(rating))].update(filtered)
    freq = {k: dict(v) for k, v in freq.items()}

    wc_specs = [
        ("all", "รวมทุกคลาส", f"{out_dir}/wordcloud_all.png"),
        ("1", "1 ดาว", f"{out_dir}/wordcloud_1star.png"),
        ("2", "2 ดาว", f"{out_dir}/wordcloud_2star.png"),
        ("3", "3 ดาว", f"{out_dir}/wordcloud_3star.png"),
        ("4", "4 ดาว", f"{out_dir}/wordcloud_4star.png"),
        ("5", "5 ดาว", f"{out_dir}/wordcloud_5star.png"),
    ]
    for key, title, out_path in wc_specs:
        wc = WordCloud(
            font_path=THAI_FONT,
            width=1200,
            height=800,
            background_color="white",
            max_words=200,
            random_state=SEED,
        ).generate_from_frequencies(freq[key])
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=16)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def visualize_data(tok_train, ratings, cm, acc, macro_f1):
    plot_heatmap(cm, acc, macro_f1)
    plot_wordclouds(tok_train, ratings)
