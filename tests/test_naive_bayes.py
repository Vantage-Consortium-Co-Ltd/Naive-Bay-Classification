"""เทสต์ NaiveBayes ที่เขียนเอง เทียบกับ sklearn และการตัดคำภาษาไทย (ไม่ต้องโหลดชุดข้อมูล Wongnai)
รัน: pytest"""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB

from Process.operate import NaiveBayes, preprocess_text, vectorize


def random_counts(n=300, v=40, k=5, seed=0):
    """ข้อมูลนับคำสมมติ: แต่ละคลาสชอบใช้คำคนละชุด"""
    rng = np.random.default_rng(seed)
    y = rng.integers(1, k + 1, n)                       # ดาว 1-5 เหมือนข้อมูลจริง
    rates = rng.gamma(1.0, 1.0, (k, v))
    X = rng.poisson(rates[y - 1])
    return csr_matrix(X), y


# ---------- NaiveBayes ----------
@pytest.mark.parametrize('alpha', [0.01, 0.5, 1.0, 2.0])
def test_ทำนายตรงกับ_sklearn_MultinomialNB(alpha):
    X, y = random_counts()
    ours = NaiveBayes(alpha).fit(X, y)
    ref = MultinomialNB(alpha=alpha).fit(X, y)
    assert np.allclose(ours.log_like, ref.feature_log_prob_)
    assert np.allclose(ours.log_prior, ref.class_log_prior_)
    assert np.array_equal(ours.predict(X), ref.predict(X))


def test_ความน่าจะเป็นของคำในแต่ละคลาสรวมได้_1():
    X, y = random_counts()
    model = NaiveBayes(1.0).fit(X, y)
    assert np.allclose(np.exp(model.log_like).sum(axis=1), 1)


def test_laplace_ตามสูตร():
    # คลาส 1 ใช้คำ [2, 0], คลาส 2 ใช้คำ [0, 3], V = 2
    X = csr_matrix([[2, 0], [0, 3]])
    model = NaiveBayes(alpha=1).fit(X, np.array([1, 2]))
    # P(คำ 0 | คลาส 1) = (2 + 1) / (2 + 1 * 2) = 3/4
    assert np.exp(model.log_like[0, 0]) == pytest.approx(3 / 4)
    assert np.exp(model.log_like[0, 1]) == pytest.approx(1 / 4)
    assert np.exp(model.log_like[1, 0]) == pytest.approx(1 / 5)


def test_alpha_0_คำที่ไม่เคยเห็นในคลาสทำให้คลาสนั้นเป็นไปไม่ได้():
    X = csr_matrix([[5, 0], [0, 5]])
    model = NaiveBayes(alpha=0).fit(X, np.array([1, 2]))
    assert np.isneginf(model.log_like[0, 1])
    # รีวิวที่มีแต่คำ 1 -> คลาส 1 ได้ log = -inf ต้องได้คลาส 2
    assert model.predict(csr_matrix([[0, 3]])).tolist() == [2]
    # มีทั้งสองคำ -> ทุกคลาสเป็น -inf เท่ากันหมด ทายอะไรก็ได้ (นี่คือเหตุผลที่ต้องใช้ smoothing)
    scores = np.asarray(csr_matrix([[10, 1]]) @ model.log_like.T)
    assert np.isneginf(scores).all()


def test_alpha_1_ไม่มีคลาสไหนเป็นไปไม่ได้():
    model = NaiveBayes(alpha=1).fit(csr_matrix([[5, 0], [0, 5]]), np.array([1, 2]))
    assert np.isfinite(model.log_like).all()
    assert model.predict(csr_matrix([[10, 1]])).tolist() == [1]     # คำ 0 เยอะกว่า -> คลาส 1


def test_เรียนรู้ได้จริง_แม่นกว่าเดาสุ่มบนข้อมูลใหม่():
    X, y = random_counts(n=1000, seed=1)
    model = NaiveBayes(1.0).fit(X[:800], y[:800])
    acc = np.mean(model.predict(X[800:]) == y[800:])
    assert acc > 0.5        # 5 คลาส เดาสุ่มได้ ~0.2


# ---------- ตัดคำ + bag of words ----------
def test_ตัดคำไทยและทิ้งเครื่องหมายกับ_stopword():
    tokens = preprocess_text('อาหารอร่อยมาก!!! 😋 ราคา 100 บาท')
    assert 'อาหาร' in tokens and 'อร่อย' in tokens
    assert '100' in tokens
    assert not any(t in tokens for t in ['!!!', '😋', ' ', 'มาก'])   # "มาก" เป็น stopword


def test_ภาษาอังกฤษถูกแปลงเป็นตัวเล็ก():
    assert 'good' in preprocess_text('GOOD food')


def test_vectorize_คำศัพท์มาจาก_train_เท่านั้น():
    train = pd.DataFrame({'tokens': [['อร่อย', 'ดี'], ['อร่อย', 'แพง'], ['ดี']]})
    test = pd.DataFrame({'tokens': [['อร่อย', 'คำใหม่']]})
    Xtr, Xte, vec = vectorize(train, test, min_df=2)
    # min_df=2: "แพง" เจอแค่รีวิวเดียวถูกตัด, "คำใหม่" ไม่อยู่ใน train
    assert sorted(vec.get_feature_names_out()) == ['ดี', 'อร่อย']
    assert Xtr.shape == (3, 2) and Xte.shape == (1, 2)
    assert Xte.sum() == 1
