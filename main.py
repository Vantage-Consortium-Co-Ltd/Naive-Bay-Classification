"""Naive Bayes star-rating classifier on Wongnai reviews (Thai).

Run: python main.py
"""
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from Process.data import load_data
from Process.operate import preprocess_data, vectorize, split_data, train_model, evaluate_model, alpha_experiment
from Process.visualize import visualize_data, OUT


def main():
    train, test = load_data()
    train, test = preprocess_data(train, "train"), preprocess_data(test, "test")
    Xtr, Xte, vec = vectorize(train, test)
    ytr, yte = split_data(train, test)
    print(f"train={Xtr.shape[0]} test={Xte.shape[0]} vocab={Xtr.shape[1]}")

    model = train_model(Xtr, ytr, alpha=1.0)
    pred, m = evaluate_model(model, Xte, yte)
    print(f"Laplace (alpha=1): accuracy={m['Acc']:.4f} macro-F1={m['F1']:.4f}")

    # self-check: the NumPy model must give the same predictions as scikit-learn's MultinomialNB
    ref = MultinomialNB(alpha=1.0).fit(Xtr, ytr).predict(Xte)
    print(f"agrees with sklearn MultinomialNB on {(ref == pred).mean():.2%} of test reviews")

    res = pd.DataFrame(alpha_experiment(Xtr, ytr, Xte, yte))
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "alpha_results.csv", index=False)

    visualize_data(train, yte, pred, model.classes, res)


if __name__ == "__main__":
    main()
