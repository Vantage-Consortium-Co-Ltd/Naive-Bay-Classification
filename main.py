from Process.data import load_data
from Process.operate import naive_bayes
from Process.visualize import visualize_data

if __name__ == "__main__":
    data = load_data()
    vocab, counts, cm, accuracy = naive_bayes(data)
    visualize_data(vocab, counts, cm, accuracy)
