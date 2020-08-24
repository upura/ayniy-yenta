import pandas as pd


INPUT_DIR = "../input/data_v2/"

if __name__ == "__main__":
    train = pd.read_csv(INPUT_DIR + "train.csv")
    test = pd.read_csv(INPUT_DIR + "train.csv")

    print(train.shape, test.shape)
    # (903605, 2) (903605, 2)
    print(train["score"].value_counts())
    # 0.0    451258
    # 1.0    406956
    # 3.0     37661
    # 2.0      7730
