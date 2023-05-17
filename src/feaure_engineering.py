import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MaxAbsScaler


def feature_selection(
    data, selected_feature_count: int = 30, method: str = "ANOVA", seed=42
):
    np.random.seed(seed)
    data_after_selection = data.copy(deep=True)
    if method == "ANOVA":
        select = SelectKBest(score_func=f_classif, k=selected_feature_count)
    fitted = select.fit(
        data_after_selection.drop(columns=["Label"], axis=1),
        data_after_selection["Label"],
    )

    selected_features_boolean = select.get_support()
    features = list(data_after_selection.drop(columns=["Label"]).columns)
    selected_features = [
        features[j] for j in range(len(features)) if selected_features_boolean[j]
    ]
    return selected_features


"""
def transform():
    fitted = select.fit(
        data_after_selection.drop(columns=["Label"]),
        data_after_selection["Label"],
    )
    train_features = fitted.transform(data_after_selection[i][0].iloc[:, :-1])
    test_features = fitted.transform(data_after_selection[i][1].iloc[:, :-1])
        train_label = data_after_selection[i][0].Label
    test_label = data_after_selection[i][1].Label

    data_after_selection[i][0] = pd.DataFrame(
        data=train_features.astype("float32"),
        columns=selected_features,
        index=data_after_selection[i][0].index,
    )
    data_after_selection[i][0]["Label"] = train_label
    data_after_selection[i][1] = pd.DataFrame(
        data=test_features.astype("float32"),
        columns=selected_features,
        index=data_after_selection[i][1].index,
    )
    data_after_selection[i][1]["Label"] = test_label
"""


def scaling(data, split_index: int, method: str = "MaxAbs", seed=42):
    np.random.seed(seed)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    if method == "MaxAbs":
        scaler = MaxAbsScaler()
    scaler.fit(train_data)
    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)
    train_data = pd.DataFrame(
        data=scaled_train,
        columns=train_data.columns,
        index=train_data.index,
    )
    train_data["Label"] = train_data["Label"] * 2
    train_data["Label"] = train_data["Label"].astype("int")
    test_data = pd.DataFrame(
        data=scaled_test,
        columns=test_data.columns,
        index=test_data.index,
    )
    test_data["Label"] = test_data["Label"] * 2
    test_data["Label"] = test_data["Label"].astype("int")

    return pd.concat([train_data, test_data])
