"""
ML Assignment — Wongnai Reviews (Thai Sentiment Analysis)
Naive Bayes + Laplace Smoothing | Heat Map | Word Cloud
==========================================================
วิธีใช้: รัน cell ทีละ cell ใน Jupyter Notebook
"""

# ============================================================
# CELL 1: ติดตั้ง libraries
# ============================================================
# รันใน terminal หรือ notebook cell นี้ก่อน:
# pip install datasets pythainlp wordcloud matplotlib seaborn pandas tqdm

# ============================================================
# CELL 2: โหลด Dataset
# ============================================================
from datasets import load_dataset
import pandas as pd

print("กำลังโหลด dataset...")
dataset = load_dataset("Highgroundbkk/wongnai_reviews")
train_df = dataset["train"].to_pandas()
test_df  = dataset["test"].to_pandas()

print(f"Train: {len(train_df):,} แถว")
print(f"Test:  {len(test_df):,} แถว")
print(f"Columns: {list(train_df.columns)}")
print(train_df["star_rating"].value_counts().sort_index())

# ============================================================
# CELL 3: Preprocessing ด้วย pythainlp
# ============================================================
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from collections import Counter, defaultdict
from tqdm import tqdm

stop_words = set(thai_stopwords())

def preprocess(text):
    """ตัดคำ + ลบ stopwords + ลบคำสั้น"""
    tokens = word_tokenize(str(text), engine="newmm")
    tokens = [
        t.strip() for t in tokens
        if t.strip()                     # ไม่ใช่ช่องว่าง
        and t.strip() not in stop_words  # ไม่ใช่ stopword
        and len(t.strip()) > 1           # ยาวกว่า 1 ตัวอักษร
        and not t.strip().isdigit()      # ไม่ใช่ตัวเลขล้วน
    ]
    return tokens

# นับความถี่คำ แยกตามดาว
print("กำลัง preprocess train set (อาจใช้เวลา 2-5 นาที)...")
class_word_counts = defaultdict(Counter)   # {star: Counter({word: count})}
class_total_words = defaultdict(int)       # {star: total_word_count}

for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    tokens = preprocess(row["review_body"])
    star   = row["star_rating"]
    class_word_counts[star].update(tokens)
    class_total_words[star] += len(tokens)

# รวมทุก class
all_word_counts = Counter()
for star_counts in class_word_counts.values():
    all_word_counts.update(star_counts)

print("Preprocessing เสร็จแล้ว!")
print(f"Vocabulary size: {len(all_word_counts):,} คำ")

# ============================================================
# CELL 4: Heat Map (TF Table)
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np

# --- หา Thai font ---
# ลองหา font ในระบบ
thai_font = None
for font in fm.findSystemFonts():
    name = font.lower()
    if any(x in name for x in ["thsarabun", "noto", "tahoma", "arial"]):
        thai_font = font
        break

if thai_font:
    prop = fm.FontProperties(fname=thai_font)
    plt.rcParams["font.family"] = prop.get_name()
    print(f"ใช้ font: {thai_font}")
else:
    print("ไม่เจอ Thai font — คำในกราฟอาจแสดงผิด แนะนำโหลด THSarabunNew.ttf")

# --- สร้าง TF DataFrame ---
TOP_N = 40  # แสดง top 40 คำที่พบบ่อยที่สุด
top_words = [word for word, _ in all_word_counts.most_common(TOP_N)]

tf_data = {}
for star in [1, 2, 3, 4, 5]:
    total = class_total_words[star]
    tf_data[f"{star}★"] = [
        class_word_counts[star].get(w, 0) / total
        for w in top_words
    ]

tf_df = pd.DataFrame(tf_data, index=top_words)

# --- Plot Heatmap ---
fig, ax = plt.subplots(figsize=(10, 14))
sns.heatmap(
    tf_df,
    annot=True,
    fmt=".4f",
    cmap="YlOrRd",
    linewidths=0.3,
    ax=ax,
    cbar_kws={"label": "Term Frequency"},
)
ax.set_title(f"Heat Map: TF ของ {TOP_N} คำที่พบบ่อยที่สุด แยกตามดาว",
             fontsize=13, pad=12)
ax.set_xlabel("ระดับดาว", fontsize=11)
ax.set_ylabel("คำ", fontsize=11)
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("บันทึก heatmap.png แล้ว")

# ============================================================
# CELL 5: Word Cloud (6 รูป)
# ============================================================
from wordcloud import WordCloud

# --- หา font สำหรับ WordCloud ---
# ต้องการ .ttf ที่รองรับภาษาไทย
# ถ้ายังไม่มี: pip install thai-word-cloud  หรือโหลด THSarabunNew.ttf

wc_font = thai_font  # ใช้ font เดิมที่หาได้ด้านบน

def make_wordcloud(word_freq, title, font_path=None):
    """สร้าง WordCloud จาก dict {word: count}"""
    if not word_freq:
        return None
    kwargs = dict(
        width=1000, height=700,
        background_color="white",
        max_words=200,
        colormap="viridis",
        prefer_horizontal=0.7,
    )
    if font_path:
        kwargs["font_path"] = font_path
    wc = WordCloud(**kwargs)
    wc.generate_from_frequencies(word_freq)
    return wc

# --- สร้าง 6 Word Cloud ---
titles = ["รวม (All)"] + [f"{s} ดาว" for s in [1, 2, 3, 4, 5]]
freqs  = [dict(all_word_counts)] + [dict(class_word_counts[s]) for s in [1, 2, 3, 4, 5]]

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle("Word Cloud: Wongnai Reviews", fontsize=16, y=1.01)
axes = axes.flatten()

for i, (title, freq) in enumerate(zip(titles, freqs)):
    wc = make_wordcloud(freq, title, font_path=wc_font)
    if wc:
        axes[i].imshow(wc, interpolation="bilinear")
    axes[i].set_title(title, fontsize=13, pad=8)
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("wordclouds.png", dpi=150, bbox_inches="tight")
plt.show()
print("บันทึก wordclouds.png แล้ว")

# ============================================================
# CELL 6: Naive Bayes + Laplace Smoothing
# ============================================================
import math

STARS = [1, 2, 3, 4, 5]
vocab  = set(all_word_counts.keys())
V      = len(vocab)          # ขนาด vocabulary

# --- Prior probability (log) ---
total_docs = len(train_df)
class_doc_counts = train_df["star_rating"].value_counts().to_dict()
log_prior = {
    star: math.log(class_doc_counts[star] / total_docs)
    for star in STARS
}

# --- Log-likelihood ด้วย Laplace Smoothing ---
# P(word | class) = (count(word, class) + 1) / (total_words_in_class + |V|)
log_likelihood = {}
for star in STARS:
    log_likelihood[star] = {}
    denom = class_total_words[star] + V      # Laplace denominator
    for word in vocab:
        numerator = class_word_counts[star].get(word, 0) + 1   # Laplace +1
        log_likelihood[star][word] = math.log(numerator / denom)

# --- Predict function ---
def predict(text):
    tokens = preprocess(text)
    scores = {}
    for star in STARS:
        score = log_prior[star]
        denom = class_total_words[star] + V
        for token in tokens:
            if token in vocab:
                score += log_likelihood[star][token]
            else:
                # OOV word → Laplace: (0+1)/(total+V)
                score += math.log(1 / denom)
        scores[star] = score
    return max(scores, key=scores.get)

# ============================================================
# CELL 7: Performance Testing
# ============================================================
print("กำลัง test model...")

correct = 0
total   = len(test_df)

# นับ accuracy แยกตามดาวด้วย
per_class_correct = defaultdict(int)
per_class_total   = defaultdict(int)

for _, row in tqdm(test_df.iterrows(), total=total):
    pred = predict(row["review_body"])
    true = row["star_rating"]
    per_class_total[true] += 1
    if pred == true:
        correct += 1
        per_class_correct[true] += 1

# --- Report ---
overall_acc = correct / total

print("\n" + "="*45)
print("       PERFORMANCE TESTING RESULTS")
print("="*45)
print(f"Test samples : {total:,}")
print(f"Correct      : {correct:,}")
print(f"Overall Accuracy: {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
print("-"*45)
print("Accuracy แยกตามดาว:")
for star in STARS:
    acc = per_class_correct[star] / per_class_total[star]
    print(f"  {star} ดาว : {per_class_correct[star]:4d}/{per_class_total[star]:4d}"
          f"  = {acc:.4f}  ({acc*100:.1f}%)")
print("="*45)

# --- Bar chart accuracy ---
fig, ax = plt.subplots(figsize=(8, 5))
star_accs = [per_class_correct[s] / per_class_total[s] for s in STARS]
bars = ax.bar([f"{s}★" for s in STARS], star_accs,
              color=["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db"])
ax.axhline(overall_acc, color="black", linestyle="--", linewidth=1.5,
           label=f"Overall = {overall_acc*100:.2f}%")
ax.set_ylim(0, 1)
ax.set_xlabel("ระดับดาว")
ax.set_ylabel("Accuracy")
ax.set_title("Naive Bayes + Laplace Smoothing — Accuracy by Star Rating")
ax.legend()
for bar, acc in zip(bars, star_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("accuracy_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("บันทึก accuracy_chart.png แล้ว")
print("\nงานเสร็จ! ไฟล์ที่ได้: heatmap.png | wordclouds.png | accuracy_chart.png")
