import pandas as pd

from ayniy.utils import Data

INPUT_DIR = "../input/data_v2/"


if __name__ == "__main__":
    train = pd.read_csv(INPUT_DIR + "train.csv")
    test = pd.read_csv(INPUT_DIR + "test.csv")
    Data.dump(train.drop("score", axis=1), "../input/X_train_fe000.pkl")
    Data.dump(train["score"], "../input/y_train_fe000.pkl")
    Data.dump(test, "../input/X_test_fe000.pkl")
