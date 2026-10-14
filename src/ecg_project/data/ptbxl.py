#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ptbxl_preprocess.py —— 直接读取 PTB-XL 原始数据集并生成训练/验证/测试数组。

逻辑移植自 timeseries notebook:
  * wfdb 读取 filename_lr / filename_hr 指向的原始波形
  * scp_statements.csv 聚合 diagnostic/subdiagnostic/superdiagnostic/form/rhythm/all 标签
  * MultiLabelBinarizer 二值化多标签
  * 默认过滤 NORM + 异常诊断类同时出现的冲突记录
  * 按 strat_fold 切分,并只用训练集拟合 StandardScaler
"""

import ast
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from tqdm.auto import tqdm


TASK_TO_CTYPE = {
    "super": "superdiagnostic",
    "sub": "subdiagnostic",
    "diag": "diagnostic",
    "form": "form",
    "rhythm": "rhythm",
    "all": "all",
}


def _join(root, name):
    return os.path.join(root, name)


def _load_npy_or_pickle(path):
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def _read_ptbxl_database(root):
    csv_path = _join(root, "ptbxl_database.csv")
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    return df


def load_raw_data_ptbxl(df, sampling_rate, root, use_cache=True):
    """读取 PTB-XL 原始波形,返回 (data, filtered_df)。data 形状为 (N, L, 12)。"""
    if sampling_rate == 100:
        file_col, cache_name = "filename_lr", "raw100.npy"
    elif sampling_rate == 500:
        file_col, cache_name = "filename_hr", "raw500.npy"
    else:
        raise ValueError("sampling_rate must be 100 or 500")

    cache_path = _join(root, cache_name)
    if use_cache and os.path.exists(cache_path):
        data = _load_npy_or_pickle(cache_path)
        if len(data) == len(df):
            return np.asarray(data), df
        print(f"[ptbxl] 缓存 {cache_name} 样本数({len(data)})与 metadata({len(df)})不一致,重新读取。")

    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("首次读取 PTB-XL 原始波形需要安装 wfdb: pip install wfdb") from exc

    signals, valid_ids = [], []
    for ecg_id, fname in tqdm(zip(df.index, df[file_col]), total=len(df),
                              desc=f"[ptbxl] Loading {sampling_rate}Hz"):
        try:
            signal, _ = wfdb.rdsamp(_join(root, fname))
        except FileNotFoundError:
            continue
        signals.append(signal)
        valid_ids.append(ecg_id)

    if not signals:
        raise RuntimeError(f"No PTB-XL waveform files were loaded from {root}")

    data = np.asarray(signals, dtype=np.float32)
    filtered_df = df.loc[valid_ids].copy()
    if use_cache:
        np.save(cache_path, data)
    return data, filtered_df


def drop_norm_abnormal_conflicts(df, root):
    """过滤 notebook 中定义的 NORM + abnormal_classes 冲突记录。"""
    scp = pd.read_csv(_join(root, "scp_statements.csv"))
    if "Unnamed: 0" in scp.columns:
        scp = scp.rename(columns={"Unnamed: 0": "scp_code"})
    elif scp.columns[0] != "scp_code":
        scp = scp.rename(columns={scp.columns[0]: "scp_code"})

    code_to_class = pd.Series(scp["diagnostic_class"].values, index=scp["scp_code"]).to_dict()
    abnormal_classes = {"MI", "STTC", "CD", "HYP"}

    def is_conflict(codes):
        classes = {code_to_class.get(code) for code in codes.keys()}
        classes = {c for c in classes if pd.notna(c)}
        return "NORM" in classes and bool(classes.intersection(abnormal_classes))

    mask = df.scp_codes.apply(is_conflict)
    if mask.any():
        print(f"[ptbxl] 过滤 NORM+异常冲突记录: {int(mask.sum())} 条")
    return df.loc[~mask].copy(), mask.to_numpy()


def compute_label_aggregations(df, root, ctype):
    """利用 scp_statements.csv 将原始 scp_codes 聚合为目标标签集合。"""
    df = df.copy()
    df["scp_codes_len"] = df.scp_codes.apply(len)
    aggregation_df = pd.read_csv(_join(root, "scp_statements.csv"), index_col=0)

    if ctype in ("diagnostic", "subdiagnostic", "superdiagnostic"):
        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]

        def aggregate_all_diagnostic(y_dic):
            return list({key for key in y_dic.keys() if key in diag_agg_df.index})

        def aggregate_subdiagnostic(y_dic):
            out = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != "nan":
                        out.append(c)
            return list(set(out))

        def aggregate_superdiagnostic(y_dic):
            out = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != "nan":
                        out.append(c)
            return list(set(out))

        if ctype == "diagnostic":
            df["diagnostic"] = df.scp_codes.apply(aggregate_all_diagnostic)
            df["diagnostic_len"] = df.diagnostic.apply(len)
        elif ctype == "subdiagnostic":
            df["subdiagnostic"] = df.scp_codes.apply(aggregate_subdiagnostic)
            df["subdiagnostic_len"] = df.subdiagnostic.apply(len)
        else:
            df["superdiagnostic"] = df.scp_codes.apply(aggregate_superdiagnostic)
            df["superdiagnostic_len"] = df.superdiagnostic.apply(len)

    elif ctype == "form":
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            return list({key for key in y_dic.keys() if key in form_agg_df.index})

        df["form"] = df.scp_codes.apply(aggregate_form)
        df["form_len"] = df.form.apply(len)

    elif ctype == "rhythm":
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            return list({key for key in y_dic.keys() if key in rhythm_agg_df.index})

        df["rhythm"] = df.scp_codes.apply(aggregate_rhythm)
        df["rhythm_len"] = df.rhythm.apply(len)

    elif ctype == "all":
        df["all_scp"] = df.scp_codes.apply(lambda x: list(set(x.keys())))
        df["all_scp_len"] = df.all_scp.apply(len)
    else:
        raise ValueError(f"Unknown PTB-XL ctype: {ctype}")
    return df


def _filter_rare_labels(labels, min_samples):
    if min_samples <= 0:
        return labels
    flattened = [item for row in labels.values for item in row]
    if not flattened:
        return labels
    counts = pd.Series(flattened).value_counts()
    keep = set(counts[counts > min_samples].index.values)
    return labels.apply(lambda x: list(set(x).intersection(keep)))


def select_data(data, labels, ctype, min_samples):
    """过滤空标签样本并用 MultiLabelBinarizer 生成 multi-hot 标签。"""
    column = "all_scp" if ctype == "all" else ctype
    len_col = f"{column}_len"
    labels = labels.copy()
    labels[column] = _filter_rare_labels(labels[column], min_samples)
    labels[len_col] = labels[column].apply(len)

    mask = labels[len_col] > 0
    selected_labels = labels.loc[mask].copy()
    selected_data = data[mask.to_numpy()]

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(selected_labels[column].values).astype(np.float32)
    return selected_data.astype(np.float32), selected_labels, y, list(mlb.classes_)


def _prepare_label_space(labels, ctype, min_samples):
    """Filter empty/rare labels and fit one label space shared by all splits."""
    column = "all_scp" if ctype == "all" else ctype
    len_col = f"{column}_len"
    labels = labels.copy()
    labels[column] = _filter_rare_labels(labels[column], min_samples)
    labels[len_col] = labels[column].apply(len)
    labels = labels.loc[labels[len_col] > 0].copy()
    mlb = MultiLabelBinarizer()
    mlb.fit(labels[column].values)
    return labels, column, mlb


def _standardize_split(X, scaler):
    X_out = []
    for x in X:
        shape = x.shape
        X_out.append(scaler.transform(x.reshape(-1, 1)).reshape(shape))
    return np.asarray(X_out, dtype=np.float32)


def standardize_by_train(X_train, X_val, X_test):
    scaler = StandardScaler()
    scaler.fit(np.vstack(X_train).reshape(-1, 1).astype(float))
    return (_standardize_split(X_train, scaler),
            _standardize_split(X_val, scaler),
            _standardize_split(X_test, scaler))


def prepare_ptbxl_splits(root, task="super", sampling_rate=100, min_samples=0,
                         train_fold=8, val_fold=9, test_fold=10,
                         drop_conflict=True, use_cache=True, standardize=True,
                         smoke_limit=None):
    """返回 dict: X_train/y_train/X_val/y_val/X_test/y_test/label_names。"""
    ctype = TASK_TO_CTYPE[task]
    raw_labels = _read_ptbxl_database(root)

    if smoke_limit:
        if drop_conflict:
            raw_labels, _ = drop_norm_abnormal_conflicts(raw_labels, root)
        labels = compute_label_aggregations(raw_labels, root, ctype)
        labels, label_column, mlb = _prepare_label_space(labels, ctype, min_samples)

        split_queries = {
            "train": labels.strat_fold <= train_fold,
            "val": labels.strat_fold == val_fold,
            "test": labels.strat_fold == test_fold,
        }
        result = {"label_names": list(mlb.classes_)}
        for split_name, split_mask in split_queries.items():
            split_labels = labels.loc[split_mask].head(smoke_limit).copy()
            split_data, split_labels = load_raw_data_ptbxl(
                split_labels,
                sampling_rate,
                root,
                use_cache=False,
            )
            result[f"X_{split_name}"] = split_data.astype(np.float32)
            result[f"y_{split_name}"] = mlb.transform(
                split_labels[label_column].values
            ).astype(np.float32)

        if standardize:
            result["X_train"], result["X_val"], result["X_test"] = standardize_by_train(
                result["X_train"], result["X_val"], result["X_test"]
            )
        print(
            f"[ptbxl:smoke] labels={len(result['label_names'])} "
            f"train={result['X_train'].shape} val={result['X_val'].shape} "
            f"test={result['X_test'].shape}"
        )
        return result

    data, raw_labels = load_raw_data_ptbxl(raw_labels, sampling_rate, root, use_cache=use_cache)

    if drop_conflict:
        filtered_labels, conflict_mask = drop_norm_abnormal_conflicts(raw_labels, root)
        data = data[~conflict_mask]
        raw_labels = filtered_labels

    labels = compute_label_aggregations(raw_labels, root, ctype)
    data, labels, y, label_names = select_data(data, labels, ctype, min_samples)

    test_mask = labels.strat_fold == test_fold
    val_mask = labels.strat_fold == val_fold
    train_mask = labels.strat_fold <= train_fold

    X_train, y_train = data[train_mask.to_numpy()], y[train_mask.to_numpy()]
    X_val, y_val = data[val_mask.to_numpy()], y[val_mask.to_numpy()]
    X_test, y_test = data[test_mask.to_numpy()], y[test_mask.to_numpy()]

    if standardize:
        X_train, X_val, X_test = standardize_by_train(X_train, X_val, X_test)

    print(f"[ptbxl] task={task}/{ctype} | labels={len(label_names)} {label_names}")
    print(f"[ptbxl] Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "label_names": label_names,
    }
