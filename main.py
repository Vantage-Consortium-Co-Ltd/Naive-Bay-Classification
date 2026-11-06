from Process.data import load_data
from Process.operate import naive_bayes
from Process.visualize import visualize_data

if __name__ == "__main__":
    data = load_data()
    tok_train, ratings, cm, acc, macro_f1 = naive_bayes(data)
    visualize_data(tok_train, ratings, cm, acc, macro_f1)
