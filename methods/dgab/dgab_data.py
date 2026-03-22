import os
import pickle

import dgl
import dgl.data.utils
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dgab_data(dataset: str, test_size: float, seed: int = 42):
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
    feat_data, labels, train_idx, test_idx, g, cat_features = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    if dataset == "S-FFSD":
        cat_features = ["Target", "Location", "Type"]

        df = pd.read_csv(prefix + "S-FFSD_full.csv")
        df = df.loc[:, ~df.columns.str.contains("Unnamed")]
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)
        out = []
        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in data.groupby(column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend(
                    [
                        sorted_idxs[i]
                        for i in range(df_len)
                        for j in range(edge_per_trans)
                        if i + j < df_len
                    ]
                )
                tgt.extend(
                    [
                        sorted_idxs[i + j]
                        for i in range(df_len)
                        for j in range(edge_per_trans)
                        if i + j < df_len
                    ]
                )
            alls.extend(src)
            allt.extend(tgt)
        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))

        cal_list = ["Source", "Target", "Location", "Type"]
        for col in data.columns:
            if col not in cal_list and col != "Labels" and col != "Time":
                min_val = np.min(data[col])
                shift = abs(min_val) + 1e-6 if min_val <= 0 else 1e-6
                data[col] = np.log(data[col] + shift)
                data[col] = (data[col] - data[col].min()) / (
                    data[col].max() - data[col].min()
                )
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        index = list(range(len(labels)))
        g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata["feat"] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.save_graphs(graph_path, [g])

        train_idx, test_idx, y_train, y_test = train_test_split(
            index,
            labels,
            stratify=labels,
            test_size=test_size / 2,
            random_state=seed,
            shuffle=True,
        )

    elif dataset == "yelp":
        cat_features = []
        data_file = loadmat(prefix + "YelpChi.mat")
        labels = pd.DataFrame(data_file["label"].flatten())[0]
        feat_data = pd.DataFrame(data_file["features"].todense().A)
        for col in feat_data.columns:
            min_val = np.min(feat_data[col])
            shift = abs(min_val) + 1e-6 if min_val <= 0 else 1e-6
            feat_data[col] = np.log(feat_data[col] + shift)
            feat_data[col] = (feat_data[col] - feat_data[col].min()) / (
                feat_data[col].max() - feat_data[col].min()
            )
        with open(prefix + "yelp_homo_adjlists.pickle", "rb") as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(
            index,
            labels,
            stratify=labels,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata["feat"] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.save_graphs(graph_path, [g])

    elif dataset == "amazon":
        cat_features = []
        data_file = loadmat(prefix + "Amazon.mat")
        labels = pd.DataFrame(data_file["label"].flatten())[0]
        feat_data = pd.DataFrame(data_file["features"].todense().A)
        for col in feat_data.columns:
            min_val = np.min(feat_data[col])
            shift = abs(min_val) + 1e-6 if min_val <= 0 else 1e-6
            feat_data[col] = np.log(feat_data[col] + shift)
            feat_data[col] = (feat_data[col] - feat_data[col].min()) / (
                feat_data[col].max() - feat_data[col].min()
            )

        with open(prefix + "amz_homo_adjlists.pickle", "rb") as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(3305, len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(
            index,
            labels[3305:],
            stratify=labels[3305:],
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata["feat"] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.save_graphs(graph_path, [g])

    return feat_data, labels, train_idx, test_idx, g, cat_features
