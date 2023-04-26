import os
import glob


def load_imdb_data(data_dir):
    """
    加载 IMDB 数据集
    """
    # 加载训练数据
    train_pos_files = glob.glob(os.path.join(data_dir, "train/pos/*.txt"))
    train_neg_files = glob.glob(os.path.join(data_dir, "train/neg/*.txt"))
    train_texts = []
    train_labels = []
    for pos_file in train_pos_files:
        with open(pos_file, "r", encoding="utf-8") as f:
            train_texts.append(f.read())
            train_labels.append(1)
    for neg_file in train_neg_files:
        with open(neg_file, "r", encoding="utf-8") as f:
            train_texts.append(f.read())
            train_labels.append(0)

    # 加载测试数据
    test_pos_files = glob.glob(os.path.join(data_dir, "test/pos/*.txt"))
    test_neg_files = glob.glob(os.path.join(data_dir, "test/neg/*.txt"))
    test_texts = []
    test_labels = []
    for pos_file in test_pos_files:
        with open(pos_file, "r", encoding="utf-8") as f:
            test_texts.append(f.read())
            test_labels.append(1)
    for neg_file in test_neg_files:
        with open(neg_file, "r", encoding="utf-8") as f:
            test_texts.append(f.read())
            test_labels.append(0)

    return train_texts, train_labels, test_texts, test_labels


def main():
    train_texts, train_labels, test_texts, test_labels = load_imdb_data("E:/aclImdb")
    print(len(train_texts), len(train_labels), len(test_texts), len(test_labels))

    for i, (text, lb) in enumerate(zip(train_texts, train_labels)):
        print(f"-----------{i}------------")
        print(text)
        print(lb)

        if i > 10:
            break


if __name__ == '__main__':
    main()
