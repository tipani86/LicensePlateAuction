import os
import gzip
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.neural_network import *
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

##### Dataset settings #####

use_years = True
use_months = True

predict_second = 47     # What second (11:29:XX) to make the prediction in validation testing

# Years to skip when importing data (to filter out data from different statistical distribution)
skip_years = []
# skip_years = [2014, 2015]

validate = True            # Change to False to train with entire dataset!

# Doing training and validation split based on specific simulation month
ignore_years = [2020, 2021]
ignore_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Testing validation on/without some specific month (often before simulation of the said month)
# ignore_years = [2020]      # Comment to disable
# ignore_months = [11]       # Comment to disable

latest_year = 2021
latest_month = 8

############

# General training settings

lr = 0.03
epochs = 64
verbose = 0
if validate:
    verbose = 0

scaler = StandardScaler()
# model = GradientBoostingRegressor(learning_rate=lr, verbose=verbose, n_estimators=epochs)
# model = ARDRegression(n_iter=epochs)
model = BaggingRegressor(n_estimators=100)
# model = RandomForestRegressor()
# model = MLPRegressor(learning_rate_init=lr, hidden_layer_sizes=(16,), verbose=verbose, max_iter=epochs)

############

def _prepare_dataset(df, ignore_month=None, val=False):
    new_month = pd.get_dummies(df["Month"], prefix="month", prefix_sep="_ ")

    if val:
        i = 1
        while i <= 12:
            if ignore_month != i:
                new_month["month_ {}".format(i)] = 0
            i += 1

    timesteps = df[[
        "11:29:00", "11:29:01", "11:29:02", "11:29:03", "11:29:04", "11:29:05", "11:29:06", "11:29:07", "11:29:08", "11:29:09",
        "11:29:10", "11:29:11", "11:29:12", "11:29:13", "11:29:14", "11:29:15", "11:29:16", "11:29:17", "11:29:18", "11:29:19",
        "11:29:20", "11:29:21", "11:29:22", "11:29:23", "11:29:24", "11:29:25", "11:29:26", "11:29:27", "11:29:28", "11:29:29",
        "11:29:30", "11:29:31", "11:29:32", "11:29:33", "11:29:34", "11:29:35", "11:29:36", "11:29:37", "11:29:38", "11:29:39",
        "11:29:40", "11:29:41", "11:29:42", "11:29:43", "11:29:44", "11:29:45", "11:29:46", "11:29:47", "11:29:48", "11:29:49",
        "11:29:50", "11:29:51", "11:29:52", "11:29:53", "11:29:54", "11:29:55", "11:29:56", "11:29:57", "11:29:58", "11:29:59",
    ]]
    target = df[["11:30:00"]]

    # X = new_month.join(other_features).values  # Data seems to be good even without year and month info

    if use_years:
        other_features = df[["Year", "Plates", "Auctioners", "Success rate", "Startprice"]]
    else:
        other_features = df[["Plates", "Auctioners", "Success rate", "Startprice"]]             # No year option

    if use_months:
        features = new_month.join(other_features)
    else:
        features = other_features                   # No month option

    all_features = features.join(timesteps)

    return all_features, np.ravel(target)

def train(data_train, data_val=None, ignore_month=None):
    X_train, y_train = _prepare_dataset(data_train)

    # print("Normalizing data...")
    scaler.fit(X_train)

    # print("Fitting the model...")
    model.fit(scaler.transform(X_train), y_train)

    # print("Validating...")
    if validate:
        X_test, y_test = _prepare_dataset(data_val, ignore_month=ignore_month, val=True)
        monthly_startprice = X_test["Startprice"].to_numpy()
        monthly_actual = (y_test + monthly_startprice)[0]
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(X_test.T)
        # quit()
        pred = model.predict(scaler.transform(X_test))
        pred = int(np.round((pred[0] + 150 + monthly_startprice) / 100, 0) * 100)  # Aim for the middle of the 300 RMB range and round to closest 100
        pred = max(pred, X_test["11:29:{}".format(predict_second)].values + monthly_startprice + 300)

        # Testing custom strategies
        # pred = int(X_test["11:29:45"].values + monthly_startprice + 1000)    # 45s + 1000 strategy
        # pred = int(X_test["11:29:47"].values + monthly_startprice + 800)    # 47s + 800 strategy

        if pred - monthly_actual <= 300 and pred - monthly_actual >= 0:
            winning_bid = True
        else:
            winning_bid = False
        # print("Predicted endprice: {}. Actual endprice: {}. Winning bid: {}".format(pred, monthly_actual, winning_bid))
        return model, pred, monthly_actual, winning_bid
    return model

if __name__ == "__main__":
    print("Loading data ...")

    if len(skip_years) > 0:
        print("Note: Skipping year(s) {} from training data due to custom setting!".format(skip_years))

    data = pd.DataFrame()
    for row in pd.read_csv("data_csv.csv", sep=",", header=0, chunksize=1):

        if row.iloc[0][0] in skip_years:
            continue

        # """
        for i in range(predict_second + 1, 60):
            row["11:29:{}".format(i)] = 0  # Removing data after 29:XX as to not let model overfit
        # """

        data = data.append(row)
    data = data.fillna(value=0).reset_index(drop=True)

    if validate:
        print("Training with validation...")
        iter = 1
        total = len(ignore_years) * len(ignore_months)
        tic = time.time()
        ignore_year = None
        ignore_month = None
        simulation_results = []
        for year in ignore_years:
            ignore_year = year
            for month in ignore_months:
                ignore_month = month
                print("Ignoring the month of {}/{} in training ...".format(ignore_year, ignore_month))
                data_val = data.query("Year == {} & Month == {}".format(ignore_year, ignore_month))
                max_idx = int(data_val.index.values)    # You only know data up to this point: "future" data is discarded
                data_train = data.iloc[:max_idx].dropna()
                data_train = shuffle(data_train)

                model, predicted, monthly_actual, winning_bid = train(data_train, data_val, ignore_month)

                if not isinstance(predicted, int):
                    predicted = predicted[0]

                simulation_results.append({
                    "year": ignore_year,
                    "month": ignore_month,
                    "predicted": predicted,
                    "actual": monthly_actual,
                    "win_bid": winning_bid,
                })

                toc = time.time()
                elapsed = 1.0 * (toc - tic) / 60
                progress = 1.0 * iter / total * 100
                remaining = abs(((toc - tic) / 60) / (progress / 100) - elapsed)
                # print("{}% done. Elapsed: {} min(s). Remaining: {} min(s)".format(progress, elapsed, remaining))
                simulation_df = pd.DataFrame(simulation_results)
                successes = simulation_df.win_bid.sum()
                print("Success rate: {}/{} ({}%)".format(successes, len(simulation_df), round(100.0 * successes / len(simulation_df), 1)))
                if ignore_year == latest_year and ignore_month == latest_month:
                    break
                iter += 1
            if ignore_year == latest_year and ignore_month == latest_month:
                print("Reached the latest month. Stopping validation test (possibly early).")
                break
        print(simulation_df[["year", "month", "predicted", "actual", "win_bid"]].to_string(index=False))
        print("Final success rate: {}/{} ({}%)".format(successes, len(simulation_df), round(100.0 * successes / len(simulation_df), 1)))
        print("Final training with full data ...")
        data_train = shuffle(data)
        validate = False
        model = train(data_train)
        print("Saving the model ...")
        with gzip.open("model.pklz", "wb") as fp:
            pickle.dump(model, fp)
        print("Done!")
    else:
        print("Training with full data, no validation.")
        data_train = shuffle(data)
        print("Dataset size: {} months".format(len(data_train)))
        model = train(data_train)
        print("Saving the model ...")
        with gzip.open("model.pklz", "wb") as fp:
            pickle.dump(model, fp)
        print("Done!")