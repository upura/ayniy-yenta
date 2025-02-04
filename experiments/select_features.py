import argparse

import pandas as pd

from ayniy.utils import Data

INPUT_DIR = "../input/"


if __name__ == "__main__":
    """
    cd experiments
    python select_features.py --n 100
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--n')
    args = parser.parse_args()

    fe_id = 'fe002'
    run_id = 'run003'
    N_FEATURES = int(args.n)
    fe_name = f'{fe_id}_top{N_FEATURES}'

    X_train = Data.load(f'../input/X_train_{fe_id}.pkl')
    y_train = Data.load(f'../input/y_train_{fe_id}.pkl')
    X_test = Data.load(f'../input/X_test_{fe_id}.pkl')

    fi = pd.read_csv(f'../output/importance/{run_id}-fi.csv')['Feature'][:N_FEATURES]

    X_train = X_train[fi]
    X_test = X_test[fi]

    Data.dump(X_train, f'../input/X_train_{fe_name}.pkl')
    Data.dump(y_train, f'../input/y_train_{fe_name}.pkl')
    Data.dump(X_test, f'../input/X_test_{fe_name}.pkl')
