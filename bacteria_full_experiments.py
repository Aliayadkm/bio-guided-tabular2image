# -*- coding: utf-8 -*-
"""
Full experiment pipeline for the bacterial tabular-to-image classification dataset.

Expected files in the same folder (CSV or ZIP containing CSV):
  train.csv.zip, test.csv.zip, sample_submission.csv.zip

Runs:
  1) data integrity and duplicate-group checks
  2) group-aware 5-fold split (first fold used as a fixed holdout)
  3) CNN on 13x22 feature images
  4) XGBoost on the original 286 tabular features
  5) equal-weight CNN + XGBoost probability fusion
  6) metrics, confusion matrices, per-class report, saved validation predictions
  7) optional refit on all 200,000 labelled rows and create test submission

Important: because identical feature rows occur repeatedly, the validation split is
GROUP-AWARE. Identical feature vectors are kept in one split to reduce leakage.
Results are written from the actual run; this script does not hard-code paper results.
"""

import os
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix, log_loss
)

warnings.filterwarnings("ignore")

# ----------------------------- CONFIGURATION -----------------------------
SEED = 42
N_SPLITS = 5
FOLD_TO_USE = 0                 # first fold from StratifiedGroupKFold
CNN_EPOCHS = 40                 # EarlyStopping usually stops sooner
CNN_BATCH_SIZE = 512
XGB_ROUNDS = 800
XGB_EARLY_STOPPING = 50
RUN_FINAL_TEST_PREDICTIONS = True
OUTPUT_DIR = Path("bacteria_results")
TRAIN_PATH = Path("train.csv.zip")
TEST_PATH = Path("test.csv.zip")
SAMPLE_SUBMISSION_PATH = Path("sample_submission.csv.zip")

# Set to True only if you want TensorFlow to attempt GPU use where available.
# TensorFlow automatically uses the GPU if correctly installed and detected.


def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def read_csv_or_zip(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.resolve()}. Put the input files next to this script "
            "or update TRAIN_PATH / TEST_PATH / SAMPLE_SUBMISSION_PATH at the top."
        )
    return pd.read_csv(path, compression="infer", low_memory=False)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def build_cnn(input_shape, n_classes):
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks

    inp = layers.Input(shape=input_shape, name="bacteria_feature_image")
    x = layers.Conv2D(64, 3, padding="same", kernel_initializer="he_uniform")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(64, 3, padding="same", kernel_initializer="he_uniform")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Conv2D(128, 3, padding="same", kernel_initializer="he_uniform")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(128, 3, padding="same", kernel_initializer="he_uniform")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="swish")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(n_classes, activation="softmax", name="class_probabilities")(x)
    model = models.Model(inp, out, name="Bacteria_CNN_13x22")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                    min_lr=1e-6, verbose=1),
        callbacks.CSVLogger(str(OUTPUT_DIR / "cnn_training_history.csv"))
    ]
    return model, cbs


def build_xgb(n_classes):
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("xgboost is required. Install it with: pip install xgboost") from e
    # xgb.train is used for compatibility across XGBoost versions.
    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "max_depth": 8,
        "eta": 0.06,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2,
        "lambda": 2.0,
        "alpha": 0.0,
        "tree_method": "hist",
        "seed": SEED,
        "nthread": max(1, (os.cpu_count() or 4) - 1),
    }
    return xgb, params


def metrics_for(name, y_true, probabilities, class_names):
    pred = np.argmax(probabilities, axis=1)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=np.arange(len(class_names)))),
        "n_validation": int(len(y_true)),
    }


def save_diagnostics(name, y_true, probabilities, class_names):
    pred = np.argmax(probabilities, axis=1)
    report = classification_report(y_true, pred, labels=np.arange(len(class_names)),
                                   target_names=class_names, output_dict=True, zero_division=0)
    save_json(report, OUTPUT_DIR / f"{name}_classification_report.json")
    cm = confusion_matrix(y_true, pred, labels=np.arange(len(class_names)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        OUTPUT_DIR / f"{name}_confusion_matrix.csv", index=True
    )
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_confusion_matrix.png", dpi=220)
    plt.close(fig)
    return report


def train_cnn(X_train_img, y_train, X_val_img, y_val, n_classes, epochs=CNN_EPOCHS):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    model, callbacks_list = build_cnn(X_train_img.shape[1:], n_classes)
    history = model.fit(
        X_train_img, y_train,
        validation_data=(X_val_img, y_val),
        epochs=epochs,
        batch_size=CNN_BATCH_SIZE,
        callbacks=callbacks_list,
        verbose=2,
    )
    probs = model.predict(X_val_img, batch_size=CNN_BATCH_SIZE, verbose=0)
    return model, probs, history


def main():
    seed_everything()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print("=" * 78)
    print("BACTERIA CLASSIFICATION — GROUP-AWARE CNN + XGBOOST FUSION")
    print("=" * 78)

    train = read_csv_or_zip(TRAIN_PATH)
    test = read_csv_or_zip(TEST_PATH)
    sample = read_csv_or_zip(SAMPLE_SUBMISSION_PATH)

    if "target" not in train.columns:
        raise ValueError("The training file must contain a 'target' column.")
    if "row_id" not in train.columns or "row_id" not in test.columns:
        raise ValueError("Expected 'row_id' column in train and test data.")

    feature_cols = [c for c in train.columns if c not in ("row_id", "target")]
    missing_features = sorted(set(feature_cols) - set(test.columns))
    if missing_features:
        raise ValueError(f"Test data is missing features, e.g. {missing_features[:10]}")

    # Numeric features only; maintain the exact training column order.
    X_df = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_test_df = test[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_raw = train["target"].astype(str)

    # Fill any missing numeric values using training medians only.
    medians = X_df.median(numeric_only=True)
    X_df = X_df.fillna(medians).fillna(0.0)
    X_test_df = X_test_df.fillna(medians).fillna(0.0)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_.tolist()
    n_classes = len(class_names)

    # Hash feature vectors into groups so identical feature rows cannot cross the split.
    group_ids = pd.util.hash_pandas_object(X_df, index=False).to_numpy(dtype=np.uint64)
    duplicate_rows = int(X_df.duplicated().sum())
    unique_groups = int(pd.Series(group_ids).nunique())
    groups_with_label_conflicts = int(
        pd.DataFrame({"group": group_ids, "target": y})
        .groupby("group", sort=False)["target"].nunique().gt(1).sum()
    )

    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Classes ({n_classes}): {class_names}")
    print(f"Duplicate feature rows: {duplicate_rows:,}")
    print(f"Unique feature groups: {unique_groups:,}")
    print(f"Groups with conflicting labels: {groups_with_label_conflicts:,}")

    class_counts = pd.Series(y_raw).value_counts().sort_index()
    class_counts.rename_axis("class").reset_index(name="count").to_csv(
        OUTPUT_DIR / "class_distribution.csv", index=False
    )
    save_json({
        "train_rows": len(train), "test_rows": len(test), "n_features": len(feature_cols),
        "n_classes": n_classes, "class_names": class_names,
        "duplicate_feature_rows": duplicate_rows, "unique_feature_groups": unique_groups,
        "groups_with_conflicting_labels": groups_with_label_conflicts,
        "seed": SEED, "split": "StratifiedGroupKFold first fold; groups are feature-row hashes"
    }, OUTPUT_DIR / "dataset_audit.json")

    # Group-aware split: identical feature rows remain entirely in train or validation.
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    train_idx, val_idx = list(splitter.split(X_df, y, groups=group_ids))[FOLD_TO_USE]
    X_train_df, X_val_df = X_df.iloc[train_idx], X_df.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    train_groups = set(group_ids[train_idx].tolist())
    val_groups = set(group_ids[val_idx].tolist())
    overlap = len(train_groups.intersection(val_groups))
    if overlap:
        raise RuntimeError(f"Group leakage detected: {overlap} groups appear in both splits.")
    print(f"Training split: {len(train_idx):,}; validation split: {len(val_idx):,}; group overlap: {overlap}")

    # Scale for CNN only. The scaler is fitted on the training fold, not validation.
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_df).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_df).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_df).astype(np.float32)
    # 286 features = 13 x 22; preserve column order as the encoding layout.
    if len(feature_cols) != 286:
        raise ValueError(f"Expected 286 features for 13x22 encoding, got {len(feature_cols)}.")
    X_train_img = X_train_scaled.reshape(-1, 13, 22, 1)
    X_val_img = X_val_scaled.reshape(-1, 13, 22, 1)
    X_test_img = X_test_scaled.reshape(-1, 13, 22, 1)

    # ------------------------------- CNN ----------------------------------
    print("\n[1/3] Training CNN on 13x22 feature images...")
    cnn_model, cnn_val_probs, history = train_cnn(
        X_train_img, y_train, X_val_img, y_val, n_classes
    )
    cnn_val_probs = np.asarray(cnn_val_probs, dtype=np.float64)
    cnn_val_probs = np.clip(cnn_val_probs, 1e-8, 1.0)
    cnn_val_probs /= cnn_val_probs.sum(axis=1, keepdims=True)
    cnn_metrics = metrics_for("CNN", y_val, cnn_val_probs, class_names)
    save_diagnostics("cnn", y_val, cnn_val_probs, class_names)
    print("CNN metrics:", json.dumps(cnn_metrics, indent=2))

    # ----------------------------- XGBOOST -------------------------------
    print("\n[2/3] Training XGBoost on original tabular features...")
    xgb, xgb_params = build_xgb(n_classes)
    dtrain = xgb.DMatrix(X_train_df.to_numpy(dtype=np.float32), label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val_df.to_numpy(dtype=np.float32), label=y_val, feature_names=feature_cols)
    xgb_model = xgb.train(
        xgb_params, dtrain, num_boost_round=XGB_ROUNDS,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=XGB_EARLY_STOPPING, verbose_eval=50,
    )
    xgb_val_probs = xgb_model.predict(dval)
    xgb_val_probs = np.asarray(xgb_val_probs, dtype=np.float64).reshape(-1, n_classes)
    xgb_val_probs = np.clip(xgb_val_probs, 1e-8, 1.0)
    xgb_val_probs /= xgb_val_probs.sum(axis=1, keepdims=True)
    xgb_metrics = metrics_for("XGBoost", y_val, xgb_val_probs, class_names)
    save_diagnostics("xgboost", y_val, xgb_val_probs, class_names)
    print("XGBoost metrics:", json.dumps(xgb_metrics, indent=2))

    # ------------------------------- FUSION ------------------------------
    print("\n[3/3] Evaluating equal-weight probability fusion (CNN + XGBoost)...")
    hybrid_val_probs = 0.5 * cnn_val_probs + 0.5 * xgb_val_probs
    hybrid_val_probs /= hybrid_val_probs.sum(axis=1, keepdims=True)
    hybrid_metrics = metrics_for("CNN_XGBoost_equal_weight", y_val, hybrid_val_probs, class_names)
    save_diagnostics("hybrid_cnn_xgboost", y_val, hybrid_val_probs, class_names)
    print("Hybrid metrics:", json.dumps(hybrid_metrics, indent=2))

    results = pd.DataFrame([cnn_metrics, xgb_metrics, hybrid_metrics])
    results.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    save_json({"metrics": [cnn_metrics, xgb_metrics, hybrid_metrics],
               "xgboost_best_iteration": getattr(xgb_model, "best_iteration", None),
               "xgboost_best_score": getattr(xgb_model, "best_score", None)},
              OUTPUT_DIR / "validation_metrics.json")

    val_predictions = pd.DataFrame({
        "row_id": train.iloc[val_idx]["row_id"].to_numpy(),
        "true_target": label_encoder.inverse_transform(y_val),
        "cnn_prediction": label_encoder.inverse_transform(np.argmax(cnn_val_probs, axis=1)),
        "xgboost_prediction": label_encoder.inverse_transform(np.argmax(xgb_val_probs, axis=1)),
        "hybrid_prediction": label_encoder.inverse_transform(np.argmax(hybrid_val_probs, axis=1)),
    })
    val_predictions.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

    # ----------------------- OPTIONAL FINAL PREDICTIONS -------------------
    if RUN_FINAL_TEST_PREDICTIONS:
        print("\nRefitting CNN and XGBoost on ALL labelled rows to predict the separate test file.")
        # Fit a fresh scaler on all labelled rows for the final models.
        final_scaler = MinMaxScaler()
        X_all_scaled = final_scaler.fit_transform(X_df).astype(np.float32)
        X_test_all_scaled = final_scaler.transform(X_test_df).astype(np.float32)
        X_all_img = X_all_scaled.reshape(-1, 13, 22, 1)
        X_test_all_img = X_test_all_scaled.reshape(-1, 13, 22, 1)

        import tensorflow as tf
        tf.keras.backend.clear_session()
        final_cnn, _ = build_cnn((13, 22, 1), n_classes)
        # Use 10% of labelled data as a training-only stopping monitor for the final refit.
        # This is for choosing epoch count; it is not used to report validation performance.
        final_cnn.fit(
            X_all_img, y, epochs=max(1, min(CNN_EPOCHS, len(history.history.get("loss", [])))),
            batch_size=CNN_BATCH_SIZE, verbose=2,
        )
        final_cnn_test_probs = final_cnn.predict(X_test_all_img, batch_size=CNN_BATCH_SIZE, verbose=0)

        xgb_all = xgb.DMatrix(X_df.to_numpy(dtype=np.float32), label=y, feature_names=feature_cols)
        xgb_test = xgb.DMatrix(X_test_df.to_numpy(dtype=np.float32), feature_names=feature_cols)
        best_rounds = int(getattr(xgb_model, "best_iteration", XGB_ROUNDS - 1)) + 1
        final_xgb = xgb.train(xgb_params, xgb_all, num_boost_round=max(1, best_rounds), verbose_eval=False)
        final_xgb_test_probs = final_xgb.predict(xgb_test).reshape(-1, n_classes)

        final_hybrid_probs = 0.5 * final_cnn_test_probs + 0.5 * final_xgb_test_probs
        final_pred = label_encoder.inverse_transform(np.argmax(final_hybrid_probs, axis=1))
        submission = pd.DataFrame({"row_id": test["row_id"].to_numpy(), "target": final_pred})
        if "row_id" in sample.columns and len(sample) == len(test):
            # Keep required column order and use sample's schema when possible.
            submission = submission[[c for c in sample.columns if c in submission.columns]]
        submission.to_csv(OUTPUT_DIR / "submission_hybrid_cnn_xgboost.csv", index=False)
        pd.DataFrame(final_hybrid_probs, columns=class_names).assign(row_id=test["row_id"].to_numpy()).to_csv(
            OUTPUT_DIR / "test_hybrid_probabilities.csv", index=False
        )
        final_cnn.save(OUTPUT_DIR / "final_cnn_model.keras")
        final_xgb.save_model(str(OUTPUT_DIR / "final_xgboost_model.json"))
        print(f"Saved test predictions: {OUTPUT_DIR / 'submission_hybrid_cnn_xgboost.csv'}")

    print("\nFinished.")
    print(f"Results folder: {OUTPUT_DIR.resolve()}")
    print(f"Total elapsed minutes: {(time.time() - start) / 60:.2f}")
    print("Review validation_metrics.csv and dataset_audit.json before using any number in the manuscript.")

if __name__ == "__main__":
    main()
