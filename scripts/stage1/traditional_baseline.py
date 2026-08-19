"""传统特征基线：静态/动态响应的手工特征 + 随机森林 / 逻辑回归。

给深度模型一个"非深度学习上限"的参照：如果手工特征 + 树模型就能达到
接近深度模型的性能，说明输入里的信号其实很简单，深度模型的优势需要
更严格论证；反过来则说明多温度位移/振动特征确实需要深度编码。
特征提取刻意做得小而确定、可解释，不与深度编码器"比容量"。
纪律：所有统计量只在训练集拟合，阈值只在验证集选择。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

from stage1.stage1_common import (  # noqa: E402
    SPLIT_NAMES,
    find_best_thresholds,
    multilabel_metrics,
    save_json,
    validate_data_and_manifest,
)


def _load_dataset(data_dir):
    try:
        from DL_model.dataset import StructuralDataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("traditional baseline requires the project PyTorch environment") from exc
    return StructuralDataset(data_dir, normalize=False, fit_normalizer=False)


def _static_features(array):
    """静态多温度位移特征：对每个测点/通道沿温度轴取均值、方差、极值与线性趋势斜率。"""
    # 输入 [温度步, 测点, 通道]；把温度当作一条轴，压缩成每个测点/通道的统计量。
    array = np.asarray(array, dtype=np.float32)
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    minimum = array.min(axis=0)
    maximum = array.max(axis=0)
    time = np.arange(array.shape[0], dtype=np.float32)
    centered_time = time - time.mean()
    slope = np.sum(centered_time.reshape((-1,) + (1,) * (array.ndim - 1)) * (array - mean), axis=0)
    slope /= np.sum(np.square(centered_time))
    return np.concatenate([mean, std, minimum, maximum, maximum - minimum, slope], axis=None)


def _dynamic_features(array):
    """动态加速度特征：均值、方差、RMS、绝对峰值与 95 分位（稳健的幅值/能量汇总）。"""
    # 输入 [时间, 测点, 通道]；RMS 表征能量，95 分位比 max 对离群点更稳健。
    array = np.asarray(array, dtype=np.float32)
    abs_array = np.abs(array)
    return np.concatenate(
        [
            array.mean(axis=0),
            array.std(axis=0),
            np.sqrt(np.mean(np.square(array), axis=0)),
            abs_array.max(axis=0),
            np.percentile(abs_array, 95, axis=0),
        ],
        axis=None,
    )


def _fit_predict(x_train, y_train, x_eval, seed, estimator_name):
    """按材料标签逐个训练二分类器，输出每个标签的正类概率；单类标签自动退回先验。"""
    try:
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("traditional baseline requires scikit-learn") from exc

    if estimator_name == "logistic":
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_eval = scaler.transform(x_eval)

    probabilities = []
    for label_index in range(y_train.shape[1]):
        target = y_train[:, label_index]
        if len(np.unique(target)) < 2:
            model = DummyClassifier(strategy="prior")
        elif estimator_name == "logistic":
            model = LogisticRegression(
                max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
        model.fit(x_train, target)
        classes = np.asarray(model.classes_)
        if classes.size == 1:
            probability = np.ones(len(x_eval), dtype=np.float32) if classes[0] == 1 else np.zeros(len(x_eval), dtype=np.float32)
        else:
            positive_index = int(np.flatnonzero(classes == 1)[0])
            probability = model.predict_proba(x_eval)[:, positive_index]
        probabilities.append(probability)
    return np.stack(probabilities, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Run a traditional response-feature baseline.")
    parser.add_argument("data_dir")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("static", "dynamic"), required=True)
    parser.add_argument("--estimator", choices=("random_forest", "logistic"), default="random_forest")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _manifest, splits = validate_data_and_manifest(args.data_dir, args.split_manifest)
    dataset = _load_dataset(args.data_dir)
    # 全量提取特征（与 seed 无关，当前实现会对每个 mode/seed 重复提取一次，仅速度问题）。
    features = []
    for index, sample in enumerate(dataset.samples):
        disp, _strain, ace = dataset.response_arrays(sample)
        features.append(_static_features(disp) if args.mode == "static" else _dynamic_features(ace))
        if (index + 1) % 250 == 0:
            print(f"  extracted features: {index + 1}/{len(dataset.samples)}", flush=True)
    x = np.asarray(features, dtype=np.float32)
    y = dataset.targets.cpu().numpy().astype(np.float32)

    train = np.asarray(splits["train"], dtype=int)
    payload = {
        "experiment": "stage1.traditional",
        "mode": args.mode,
        "estimator": args.estimator,
        "seed": args.seed,
        "data_dir": os.path.abspath(args.data_dir),
        "split_manifest": os.path.abspath(args.split_manifest),
        "feature_count": int(x.shape[1]),
        "metrics": {},
    }
    probabilities_by_split = {}
    for split_name in SPLIT_NAMES:
        indices = np.asarray(splits[split_name], dtype=int)
        probabilities_by_split[split_name] = _fit_predict(
            x[train], y[train], x[indices], args.seed, args.estimator
        )
    # 阈值只在验证集上按 F1 挑选；测试集只用这套固定阈值，禁止事后重调。
    val_indices = np.asarray(splits["val"], dtype=int)
    thresholds = find_best_thresholds(y[val_indices], probabilities_by_split["val"])
    payload["validation_thresholds"] = thresholds.astype(float).tolist()
    for split_name in SPLIT_NAMES:
        indices = np.asarray(splits[split_name], dtype=int)
        payload["metrics"][split_name] = multilabel_metrics(
            y[indices], probabilities_by_split[split_name], threshold=thresholds
        )
    save_json(args.output, payload)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
