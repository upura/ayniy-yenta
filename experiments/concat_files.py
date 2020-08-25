import gc

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
    del df_user_ages, user_ages_map
    gc.collect()
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
    del df_user_purposes
    gc.collect()
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
    del df_user_vecs
    gc.collect()
    return _dataframe


def add_user_strengths(dataframe: pd.DataFrame) -> pd.DataFrame:
    df_user_strengths = pd.read_csv(INPUT_DIR + "user_strengths.csv")
    df_user_strengths["user_id"] = df_user_strengths["user_id"].astype("str")
    _dataframe = dataframe.copy()
    _dataframe = pd.merge(
        _dataframe, df_user_strengths, left_on="userA", right_on="user_id", how="left"
    )
    _dataframe = pd.merge(
        _dataframe,
        df_user_strengths,
        left_on="userB",
        right_on="user_id",
        how="left",
        suffixes=("_userA", "_userB"),
    )
    del df_user_strengths
    gc.collect()
    return _dataframe


def add_user_works(dataframe: pd.DataFrame) -> pd.DataFrame:
    df_user_works = pd.read_csv(INPUT_DIR + "user_works.csv")
    df_user_works["user_id"] = df_user_works["user_id"].astype("str")

    table = df_user_works.groupby("user_id").agg({"max", "count"}).reset_index()
    table.columns = [
        "user_id",
        "company_id_count",
        "company_id_max",
        "industry_id_count",
        "industry_id_max",
        "over_1000_employees_count",
        "over_1000_employees_max",
    ]

    _dataframe = dataframe.copy()
    _dataframe = pd.merge(_dataframe, table, left_on="userA", right_on="user_id", how="left")
    _dataframe = pd.merge(
        _dataframe,
        table,
        left_on="userB",
        right_on="user_id",
        how="left",
        suffixes=("_userA", "_userB"),
    )

    del df_user_works, table
    gc.collect()
    return _dataframe


def add_user_skills(dataframe: pd.DataFrame) -> pd.DataFrame:
    df_user_skills = pd.read_csv(
        INPUT_DIR + "user_skills.csv", dtype={"user_id": str, "skill_id": int}
    )

    table = df_user_skills.groupby("user_id").agg({"max", "count"}).reset_index()
    table.columns = ["user_id", "skill_id_count", "skill_id_max"]

    _dataframe = dataframe.copy()
    _dataframe = pd.merge(_dataframe, table, left_on="userA", right_on="user_id", how="left")
    _dataframe = pd.merge(
        _dataframe,
        table,
        left_on="userB",
        right_on="user_id",
        how="left",
        suffixes=("_userA", "_userB"),
    )
    del df_user_skills, table
    gc.collect()
    return _dataframe


def add_user_educations(dataframe: pd.DataFrame) -> pd.DataFrame:
    df_user_educations = pd.read_csv(
        INPUT_DIR + "user_educations.csv",
        dtype={"user_id": str, "school_id": int, "degree_id": float},
    )

    table = df_user_educations.groupby("user_id").agg({"max", "count"}).reset_index()
    table.columns = [
        "user_id",
        "school_id_count",
        "school_id_max",
        "degree_id_count",
        "degree_id_max",
    ]

    _dataframe = dataframe.copy()
    _dataframe = pd.merge(_dataframe, table, left_on="userA", right_on="user_id", how="left")
    _dataframe = pd.merge(
        _dataframe,
        table,
        left_on="userB",
        right_on="user_id",
        how="left",
        suffixes=("_userA", "_userB"),
    )
    del df_user_educations, table
    gc.collect()
    return _dataframe


INPUT_DIR = "../input/data_v2/"
DELETE_COLS = ["from-to", "userA", "userB", "user_id_userA", "user_id_userB"]

if __name__ == "__main__":

    train = pd.read_csv(INPUT_DIR + "train.csv")
    train = split_user_id(train)
    train = add_user_ages(train)
    train = add_user_purposes(train)
    train = add_user_vecs(train)
    train = add_user_strengths(train)
    train = add_user_works(train)
    train = add_user_skills(train)
    train = add_user_educations(train)
    pd.Series(train.columns).to_csv("../input/col_names.csv", index=False)

    Data.dump(
        train.drop(DELETE_COLS + ["score"], axis=1), "../input/X_train_fe002.pkl",
    )
    Data.dump(train["score"], "../input/y_train_fe002.pkl")
    del train
    gc.collect()

    test = pd.read_csv(INPUT_DIR + "test.csv")
    test = split_user_id(test)
    test = add_user_ages(test)
    test = add_user_purposes(test)
    test = add_user_vecs(test)
    test = add_user_strengths(test)
    test = add_user_works(test)
    test = add_user_skills(test)
    test = add_user_educations(test)

    Data.dump(
        test.drop(DELETE_COLS, axis=1), "../input/X_test_fe002.pkl",
    )
