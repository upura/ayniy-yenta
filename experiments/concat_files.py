import pandas as pd

from ayniy.utils import Data


def split_user_id(dataframe: pd.DataFrame) -> pd.DataFrame:
    _dataframe = dataframe.copy()
    _dataframe["userA"] = _dataframe["from-to"].str.split("-").map(lambda x: x[0])
    _dataframe["userB"] = _dataframe["from-to"].str.split("-").map(lambda x: x[1])
    return _dataframe


def add_user_ages(dataframe: pd.DataFrame) -> pd.DataFrame:
    df_user_ages = pd.read_csv(INPUT_DIR + "user_ages.csv", dtype={"user_id": str, "age": float})
    user_ages_map = df_user_ages.set_index("user_id")["age"].to_dict()
    _dataframe = dataframe.copy()
    _dataframe["userA_age"] = _dataframe["userA"].map(user_ages_map)
    _dataframe["userB_age"] = _dataframe["userB"].map(user_ages_map)
    return _dataframe


def add_user_purposes(dataframe: pd.DataFrame) -> pd.DataFrame:
    df_user_purposes = pd.read_csv(
        INPUT_DIR + "user_purposes.csv",
        dtype={
            "user_id": str,
            "purpose_id_1": int,
            "purpose_id_2": int,
            "purpose_id_3": int,
            "purpose_id_4": int,
            "purpose_id_5": int,
            "purpose_id_6": int,
            "purpose_id_7": int,
            "purpose_id_8": int,
            "purpose_id_9": int,
            "purpose_id_10": int,
            "purpose_id_11": int,
            "purpose_id_12": int,
            "purpose_id_13": int,
            "purpose_id_14": int,
            "purpose_id_15": int,
        },
    )
    _dataframe = dataframe.copy()
    _dataframe = pd.merge(
        _dataframe, df_user_purposes, left_on="userA", right_on="user_id", how="left"
    )
    _dataframe = pd.merge(
        _dataframe,
        df_user_purposes,
        left_on="userB",
        right_on="user_id",
        how="left",
        suffixes=("_userA", "_userB"),
    )
    return _dataframe


def add_user_vecs(dataframe: pd.DataFrame) -> pd.DataFrame:
    df_user_vecs = pd.read_csv(INPUT_DIR + "user_self_intro_vectors_300dims.csv")
    df_user_vecs["user_id"] = df_user_vecs["user_id"].astype("str")
    _dataframe = dataframe.copy()
    _dataframe = pd.merge(
        _dataframe, df_user_vecs, left_on="userA", right_on="user_id", how="left"
    )
    _dataframe = pd.merge(
        _dataframe,
        df_user_vecs,
        left_on="userB",
        right_on="user_id",
        how="left",
        suffixes=("_userA", "_userB"),
    )
    return _dataframe


INPUT_DIR = "../input/data_v2/"
IS_TRAIN = False

if __name__ == "__main__":

    if IS_TRAIN:
        train = pd.read_csv(INPUT_DIR + "train.csv")
        train = split_user_id(train)
        train = add_user_ages(train)
        train = add_user_purposes(train)
        train = add_user_vecs(train)
        Data.dump(
            train.drop(
                ["from-to", "score", "userA", "userB", "user_id_userA", "user_id_userB"], axis=1
            ),
            "../input/X_train_fe001.pkl",
        )
        Data.dump(train["score"], "../input/y_train_fe001.pkl")

    else:
        test = pd.read_csv(INPUT_DIR + "test.csv")
        test = split_user_id(test)
        test = add_user_ages(test)
        test = add_user_purposes(test)
        test = add_user_vecs(test)
        Data.dump(
            test.drop(["from-to", "userA", "userB", "user_id_userA", "user_id_userB"], axis=1),
            "../input/X_test_fe001.pkl",
        )
