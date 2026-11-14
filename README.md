# Naive Bayes: ทายดาวรีวิว Wongnai

ทายจำนวนดาว (1–5) ของรีวิวร้านอาหารภาษาไทยจากชุดข้อมูล [Wongnai reviews](https://huggingface.co/datasets/Highgroundbkk/wongnai_reviews)
ด้วย Multinomial Naive Bayes ที่เขียนเองด้วย NumPy และทดสอบผลของ Laplace smoothing

ตามโจทย์:
- Visualize heatmap (confusion matrix)
- Word cloud รวมทุกคลาส และแยก 1, 2, 3, 4, 5 ดาว
- ทดสอบประสิทธิภาพ Naive Bayes กับ Laplace smoothing

## วิธีรัน

```bash
pip install -r requirements.txt
python main.py
```

รอบแรกการตัดคำ 46,000 รีวิวใช้เวลาหลายนาที ผลตัดคำจะถูกเก็บไว้ใน `cache/` รอบถัดไปจึงเร็วขึ้น

## โครงสร้าง

| ไฟล์ | หน้าที่ |
|---|---|
| `main.py` | จุดรันหลัก |
| `Process/data.py` | โหลดข้อมูล แปลง `star_rating` จาก 0–4 เป็น 1–5 ดาว |
| `Process/operate.py` | ตัดคำ (PyThaiNLP `newmm`), สร้าง bag-of-words, คลาส `NaiveBayes`, ทดลองค่า alpha |
| `Process/visualize.py` | heatmap, word cloud, กราฟ alpha |

## หลักการ

$$P(\text{คลาส} \mid \text{รีวิว}) \propto P(\text{คลาส}) \prod_{\text{คำ}} P(\text{คำ} \mid \text{คลาส})$$

$$P(\text{คำ} \mid \text{คลาส}) = \frac{\text{จำนวนครั้งที่คำนั้นอยู่ในคลาส} + \alpha}{\text{จำนวนคำทั้งหมดในคลาส} + \alpha \cdot |V|}$$

- คำนวณในรูป log เพื่อไม่ให้ผลคูณของความน่าจะเป็นเล็ก ๆ หลายตัวกลายเป็น 0
- **α = 0** (ไม่ทำ smoothing): ถ้ามีคำที่ไม่เคยพบในคลาสนั้น ความน่าจะเป็นของทั้งคลาสจะเป็น 0 ทันที
- **α = 1** คือ **Laplace smoothing**: บวก 1 ให้ทุกคำ ไม่มีคำไหนมีความน่าจะเป็นเป็น 0

## ผลลัพธ์ (test set 6,203 รีวิว, คลังคำ 24,312 คำ)

Laplace (α = 1): **accuracy 0.510, macro-F1 0.343** ซึ่งตรงกับ `MultinomialNB` ของ scikit-learn 100%

| α | 0 | 0.01 | 0.1 | 0.5 | **1 (Laplace)** | 2 | 5 |
|---|---|---|---|---|---|---|---|
| Accuracy | 0.458 | 0.489 | 0.485 | 0.495 | **0.510** | 0.518 | 0.534 |
| Macro-F1 | 0.295 | 0.358 | 0.375 | 0.358 | **0.343** | 0.322 | 0.259 |

- α = 0 ได้ผลแย่สุด เพราะปัญหาความน่าจะเป็นเป็นศูนย์
- α ยิ่งสูง accuracy ยิ่งขึ้นแต่ F1 ลด เพราะโมเดลเอียงไปทาย 4 ดาว ซึ่งเป็นคลาสที่มีรีวิวมากที่สุด (47% ของ train)
- โมเดลไม่ทาย 1 ดาวเลย เพราะมีรีวิว 1 ดาวแค่ 1% ของข้อมูล และส่วนใหญ่ถูกทายเป็น 2 ดาว (ดู heatmap)

| ไฟล์ใน `output/` | เนื้อหา |
|---|---|
| `heatmap.png` | confusion matrix ของ α = 1 (normalize ตามแถว) |
| `wordcloud.png` | word cloud รวมทุกคลาส + แยก 1–5 ดาว |
| `alpha.png`, `alpha_results.csv` | accuracy / F1 ตามค่า α |

## หมายเหตุ

- ใช้ regex กรองเฉพาะคำไทย อังกฤษ และตัวเลข เพราะ `str.isalnum()` ให้ค่า False กับคำไทยที่มีสระบน/ล่างหรือวรรณยุกต์ (เช่น `"อร่อย".isalnum()` → False)
- ตัด stopword ภาษาไทยของ PyThaiNLP ออก และไม่ใช้คำที่พบแค่ครั้งเดียวใน train (`min_df=2`)
- word cloud ใช้ฟอนต์ Tahoma ที่ `C:/Windows/Fonts/tahoma.ttf` ถ้าใช้ macOS/Linux ให้แก้ `FONT` ใน `Process/visualize.py`
