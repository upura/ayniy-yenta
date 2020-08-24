import pandas as pd

INPUT_DIR = "../input/data_v2/"


if __name__ == "__main__":
    test = pd.read_csv(INPUT_DIR + "test.csv")
    sub = test[["from-to"]]
    sub["score"] = 0.0
    print(sub.head())
    sub.to_csv(INPUT_DIR + "sample_submission.csv", index=False)
