# -*- coding: utf-8 -*-
"""
Bacterial classification: reproducible validation + final test predictions.
Place this file in the same folder as:
    train.csv.zip, test.csv.zip, sample_submission.csv.zip
(or the corresponding .csv files).

Run:
    python bacteria_one_click.py

Optional quick test (does NOT produce final research results):
    set BACTERIA_SMOKE=1   (Windows CMD)
    python bacteria_one_click.py

Outputs are written to ./bacteria_results/
NOTE: test.csv has no labels; do not report test accuracy.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")

import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, log_loss, classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

SEED = 42
TEST_SIZE = 0.20
SMOKE = os.environ.get("BACTERIA_SMOKE", "0") == "1"
SMOKE_N = 12000
CNN_EPOCHS = 8 if SMOKE else 30
BATCH_SIZE = 256
XGB_ROUNDS = 100 if SMOKE else 800
OUT = Path("bacteria_results")
OUT.mkdir(exist_ok=True)

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
seed_everything()

def read_table(stem):
    candidates = [Path(f"{stem}.csv.zip"), Path(f"{stem}.zip"), Path(f"{stem}.csv")]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Cannot find {stem}.csv.zip or {stem}.csv in: {Path.cwd()}\n"
        "Put the data files beside this script."
    )

def identify_target(df):
    likely = ["label", "target", "class", "y", "species", "bacteria", "category"]
    lower = {str(c).strip().lower(): c for c in df.columns}
    for key in likely:
        if key in lower:
            return lower[key]
    # Usually the final non-feature column is the label. Prefer a low-cardinality column.
    candidates = []
    for c in df.columns:
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= min(100, max(2, int(len(df) * 0.02))):
            candidates.append((nunique, c))
    if not candidates:
        raise ValueError("Could not identify the target column. Set TARGET_COLUMN manually in the script.")
    candidates.sort(key=lambda z: z[0])
    return candidates[0][1]

def feature_matrix(train, test, target):
    y_raw = train[target].copy()
    drop_cols = [target]
    # Remove obvious row identifiers only if they are unique and not useful numeric features.
    for c in list(train.columns):
        if c == target or c not in test.columns:
            continue
        cl = str(c).lower()
        if cl in {"id", "index", "unnamed: 0", "row_id"}:
            drop_cols.append(c)
    Xdf = train.drop(columns=drop_cols, errors="ignore")
    Tdf = test.drop(columns=[c for c in drop_cols if c != target], errors="ignore")
    common = [c for c in Xdf.columns if c in Tdf.columns]
    if not common:
        raise ValueError("No matching feature columns between train and test.")
    Xdf, Tdf = Xdf[common].copy(), Tdf[common].copy()
    # Keep numeric columns; bacterial k-mer datasets should be numeric.
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    Tdf = Tdf.apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)
    Tdf = Tdf.replace([np.inf, -np.inf], np.nan)
    return Xdf, Tdf, y_raw, common

def make_groups(X):
    # Identical feature rows are assigned to the same group to avoid duplicate leakage.
    hashed = pd.util.hash_pandas_object(X, index=False).astype("uint64")
    return hashed.to_numpy()

def metrics_for(name, y_true, probs, classes):
    pred = np.argmax(probs, axis=1)
    result = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, probs, labels=np.arange(len(classes)))),
    }
    pd.DataFrame(classification_report(
        y_true, pred, labels=np.arange(len(classes)), target_names=[str(c) for c in classes],
        output_dict=True, zero_division=0
    )).T.to_csv(OUT / f"{name}_classification_report.csv")
    pd.DataFrame(confusion_matrix(y_true, pred, labels=np.arange(len(classes))),
                 index=[str(c) for c in classes], columns=[str(c) for c in classes]
                 ).to_csv(OUT / f"{name}_confusion_matrix.csv")
    return result, pred

def main():
    print("=" * 72)
    print("BACTERIAL CLASSIFICATION - ONE-CLICK PIPELINE")
    print("Smoke test:", SMOKE, "| Seed:", SEED)
    print("=" * 72)

    train = read_table("train")
    test = read_table("test")
    sample = read_table("sample_submission") if (
        Path("sample_submission.csv.zip").exists() or Path("sample_submission.csv").exists()
    ) else None

    target = identify_target(train)
    print("Detected target column:", target)
    print("Train shape:", train.shape, "| Test shape:", test.shape)

    Xdf, Tdf, y_raw, feature_names = feature_matrix(train, test, target)
    le = LabelEncoder()
    y_all = le.fit_transform(y_raw.astype(str))
    classes = le.classes_
    print("Features:", len(feature_names), "| Classes:", len(classes))
    print("Class counts:", dict(pd.Series(y_raw.astype(str)).value_counts()))

    if SMOKE and len(Xdf) > SMOKE_N:
        # Stratified subsample for a fast pipeline check only.
        Xdf, _, y_all, _, idx, _ = train_test_split(
            Xdf, y_all, np.arange(len(Xdf)), train_size=SMOKE_N,
            random_state=SEED, stratify=y_all
        )
        # Reset after split; use original row indices for duplicate groups.
        groups_all = make_groups(Xdf)
    else:
        groups_all = make_groups(Xdf)

    # Split by identical feature rows. If group-aware split cannot retain every class,
    # fall back to stratified split and report that fact.
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr_idx, va_idx = next(splitter.split(Xdf, y_all, groups=groups_all))
    missing_train = set(np.unique(y_all)) - set(np.unique(y_all[tr_idx]))
    missing_val = set(np.unique(y_all)) - set(np.unique(y_all[va_idx]))
    if missing_train or missing_val:
        print("WARNING: group split misses classes; using stratified split instead.")
        tr_idx, va_idx = train_test_split(
            np.arange(len(y_all)), test_size=TEST_SIZE, random_state=SEED, stratify=y_all
        )

    X_train_raw = Xdf.iloc[tr_idx].to_numpy(dtype=np.float32)
    X_val_raw = Xdf.iloc[va_idx].to_numpy(dtype=np.float32)
    y_train, y_val = y_all[tr_idx], y_all[va_idx]

    # Fit imputation on training split only: no validation/test leakage.
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw).astype(np.float32)
    X_val = imputer.transform(X_val_raw).astype(np.float32)
    X_test = imputer.transform(Tdf.to_numpy(dtype=np.float32)).astype(np.float32)
    print(f"Split: train={len(tr_idx):,}, validation={len(va_idx):,}; "
          f"duplicate-group overlap={len(set(groups_all[tr_idx]) & set(groups_all[va_idx]))}")

    all_metrics = []
    probs_by_model = {}

    # Model 1: raw-feature XGBoost (fast, strong baseline).
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=XGB_ROUNDS, max_depth=8, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            objective="multi:softprob", num_class=len(classes),
            eval_metric="mlogloss", tree_method="hist", random_state=SEED,
            n_jobs=max(1, (os.cpu_count() or 2) - 1)
        )
        print("\n[1/4] Training raw-feature XGBoost...")
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        p_xgb_val = xgb.predict_proba(X_val)
        # Ensure columns follow encoded class order.
        if p_xgb_val.shape[1] != len(classes):
            raise ValueError("XGBoost probability columns do not match class count.")
        m, pred = metrics_for("raw_xgboost", y_val, p_xgb_val, classes)
        all_metrics.append(m); probs_by_model["raw_xgboost"] = p_xgb_val
        print("XGBoost validation:", json.dumps(m, indent=2))
    except Exception as e:
        xgb = None
        p_xgb_val = None
        print("XGBoost skipped/error:", repr(e))

    # Model 2: CNN on row-major 13x22 image, only when TensorFlow is available.
    cnn = embedding_model = scaler = None
    p_cnn_val = p_seq_val = p_fusion_val = None
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, callbacks
        tf.keras.utils.set_random_seed(SEED)
        # Fit scaling on training split only.
        scaler = MinMaxScaler()
        Xtr_scaled = scaler.fit_transform(X_train).astype(np.float32)
        Xva_scaled = scaler.transform(X_val).astype(np.float32)
        Xte_scaled = scaler.transform(X_test).astype(np.float32)
        # 286 features -> 13 x 22. If feature count differs, pad to a rectangle.
        n_features = Xtr_scaled.shape[1]
        rows = int(np.floor(np.sqrt(n_features)))
        cols = int(np.ceil(n_features / rows))
        rows = max(rows, 1)
        pad = rows * cols - n_features
        def to_image(a):
            if pad:
                a = np.pad(a, ((0, 0), (0, pad)), mode="constant")
            return a.reshape(-1, rows, cols, 1)
        Xtr_img, Xva_img, Xte_img = to_image(Xtr_scaled), to_image(Xva_scaled), to_image(Xte_scaled)

        inputs = layers.Input(shape=(rows, cols, 1))
        z = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        z = layers.Conv2D(32, 3, padding="same", activation="relu")(z)
        z = layers.MaxPooling2D()(z)
        z = layers.Dropout(0.20)(z)
        z = layers.Conv2D(64, 3, padding="same", activation="relu")(z)
        z = layers.Conv2D(64, 3, padding="same", activation="relu")(z)
        z = layers.GlobalAveragePooling2D()(z)
        emb = layers.Dense(128, activation="relu", name="cnn_embedding_128d")(z)
        z = layers.Dropout(0.25)(emb)
        outputs = layers.Dense(len(classes), activation="softmax")(z)
        cnn = models.Model(inputs, outputs)
        embedding_model = models.Model(inputs, emb)
        cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                    loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
        ]
        print(f"\n[2/4] Training CNN on {rows}x{cols} feature grid...")
        cnn.fit(Xtr_img, y_train, validation_data=(Xva_img, y_val),
                epochs=CNN_EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=2)
        p_cnn_val = cnn.predict(Xva_img, batch_size=BATCH_SIZE, verbose=0)
        m, _ = metrics_for("cnn_row_major", y_val, p_cnn_val, classes)
        all_metrics.append(m); probs_by_model["cnn_row_major"] = p_cnn_val
        print("CNN validation:", json.dumps(m, indent=2))

        # Model 3: sequential CNN embedding -> XGBoost.
        # Train the downstream model on frozen embeddings from the training-fold CNN.
        if xgb is not None:
            print("\n[3/4] Training XGBoost on 128-D CNN embeddings...")
            Etr = embedding_model.predict(Xtr_img, batch_size=BATCH_SIZE, verbose=0)
            Eva = embedding_model.predict(Xva_img, batch_size=BATCH_SIZE, verbose=0)
            seq_xgb = XGBClassifier(
                n_estimators=XGB_ROUNDS, max_depth=5, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0,
                objective="multi:softprob", num_class=len(classes),
                eval_metric="mlogloss", tree_method="hist", random_state=SEED,
                n_jobs=max(1, (os.cpu_count() or 2) - 1)
            )
            seq_xgb.fit(Etr, y_train, eval_set=[(Eva, y_val)], verbose=False)
            p_seq_val = seq_xgb.predict_proba(Eva)
            m, _ = metrics_for("cnn_embedding_xgboost", y_val, p_seq_val, classes)
            all_metrics.append(m); probs_by_model["cnn_embedding_xgboost"] = p_seq_val
            print("Sequential hybrid validation:", json.dumps(m, indent=2))
        else:
            seq_xgb = None

        # Model 4: probability fusion CNN + raw XGBoost.
        if p_xgb_val is not None:
            p_fusion_val = (p_cnn_val + p_xgb_val) / 2.0
            p_fusion_val = p_fusion_val / p_fusion_val.sum(axis=1, keepdims=True)
            m, _ = metrics_for("probability_fusion", y_val, p_fusion_val, classes)
            all_metrics.append(m); probs_by_model["probability_fusion"] = p_fusion_val
            print("Probability fusion validation:", json.dumps(m, indent=2))
    except Exception as e:
        print("CNN / embedding models skipped/error:", repr(e))
        seq_xgb = None

    # Save validation predictions and metrics.
    pd.DataFrame(all_metrics).sort_values("macro_f1", ascending=False).to_csv(
        OUT / "validation_metrics.csv", index=False
    )
    (OUT / "validation_metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )
    if probs_by_model:
        pred_df = pd.DataFrame({"row_index_in_validation": va_idx, "true_label": le.inverse_transform(y_val)})
        for name, probs in probs_by_model.items():
            pred_df[name + "_prediction"] = le.inverse_transform(np.argmax(probs, axis=1))
            pred_df[name + "_confidence"] = np.max(probs, axis=1)
        pred_df.to_csv(OUT / "validation_predictions.csv", index=False)

    # Final training on all labeled rows and predict unlabeled test rows.
    # This stage creates submission predictions only; no test metrics are computed.
    print("\n[FINAL] Fit best available models on all labeled rows and predict test...")
    final_imputer = SimpleImputer(strategy="median")
    X_all = final_imputer.fit_transform(Xdf.to_numpy(dtype=np.float32)).astype(np.float32)
    X_test_final = final_imputer.transform(Tdf.to_numpy(dtype=np.float32)).astype(np.float32)

    final_probs = None
    final_model_name = None
    if xgb is not None:
        try:
            from xgboost import XGBClassifier
            final_xgb = XGBClassifier(
                n_estimators=XGB_ROUNDS, max_depth=8, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                objective="multi:softprob", num_class=len(classes),
                eval_metric="mlogloss", tree_method="hist", random_state=SEED,
                n_jobs=max(1, (os.cpu_count() or 2) - 1)
            )
            final_xgb.fit(X_all, y_all, verbose=False)
            final_probs = final_xgb.predict_proba(X_test_final)
            final_model_name = "raw_xgboost"
            import joblib
            joblib.dump(final_xgb, OUT / "raw_xgboost_final.joblib")
            joblib.dump(final_imputer, OUT / "final_imputer.joblib")
        except Exception as e:
            print("Final XGBoost fit failed:", repr(e))

    # Train a final CNN as well if TensorFlow is available; use raw XGBoost as fallback.
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, callbacks
        tf.keras.utils.set_random_seed(SEED)
        final_scaler = MinMaxScaler()
        Xa_s = final_scaler.fit_transform(X_all).astype(np.float32)
        Xt_s = final_scaler.transform(X_test_final).astype(np.float32)
        nf = Xa_s.shape[1]
        rr = max(1, int(np.floor(np.sqrt(nf))))
        cc = int(np.ceil(nf / rr))
        pp = rr * cc - nf
        def img_final(a):
            if pp:
                a = np.pad(a, ((0, 0), (0, pp)), mode="constant")
            return a.reshape(-1, rr, cc, 1)
        Xa_img, Xt_img = img_final(Xa_s), img_final(Xt_s)
        inp = layers.Input(shape=(rr, cc, 1))
        z = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
        z = layers.Conv2D(32, 3, padding="same", activation="relu")(z)
        z = layers.MaxPooling2D()(z)
        z = layers.Dropout(0.20)(z)
        z = layers.Conv2D(64, 3, padding="same", activation="relu")(z)
        z = layers.Conv2D(64, 3, padding="same", activation="relu")(z)
        z = layers.GlobalAveragePooling2D()(z)
        emb = layers.Dense(128, activation="relu", name="cnn_embedding_128d")(z)
        z = layers.Dropout(0.25)(emb)
        out = layers.Dense(len(classes), activation="softmax")(z)
        final_cnn = models.Model(inp, out)
        final_emb = models.Model(inp, emb)
        final_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                          loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        # Use fixed epoch count selected conservatively; validation was used above.
        final_cnn.fit(Xa_img, y_all, epochs=CNN_EPOCHS, batch_size=BATCH_SIZE, verbose=2)
        p_cnn_test = final_cnn.predict(Xt_img, batch_size=BATCH_SIZE, verbose=0)
        import joblib
        joblib.dump(final_scaler, OUT / "final_minmax_scaler.joblib")
        final_cnn.save(OUT / "final_cnn.keras")
        if final_probs is not None:
            final_probs = (final_probs + p_cnn_test) / 2.0
            final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
            final_model_name = "probability_fusion"
        else:
            final_probs, final_model_name = p_cnn_test, "cnn_row_major"
    except Exception as e:
        print("Final CNN skipped/error; use XGBoost if available:", repr(e))

    if final_probs is None:
        raise RuntimeError(
            "No model completed successfully. Check the error above and ensure xgboost is installed."
        )

    pred_encoded = np.argmax(final_probs, axis=1)
    pred_labels = le.inverse_transform(pred_encoded)
    submission = sample.copy() if sample is not None else pd.DataFrame(index=np.arange(len(test)))
    if len(submission) != len(test):
        submission = pd.DataFrame(index=np.arange(len(test)))
    # Fill a likely target column if present, otherwise use the last sample-submission column.
    target_out = target if target in submission.columns else None
    if target_out is None:
        non_id = [c for c in submission.columns if str(c).lower() not in {"id", "index", "unnamed: 0"}]
        target_out = non_id[-1] if non_id else "label"
    submission[target_out] = pred_labels
    submission.to_csv(OUT / "submission_predictions.csv", index=False)
    pd.DataFrame(final_probs, columns=[str(c) for c in classes]).to_csv(
        OUT / "test_predicted_probabilities.csv", index=False
    )
    (OUT / "run_summary.json").write_text(json.dumps({
        "seed": SEED, "smoke_test": SMOKE, "target_column": str(target),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "n_features": int(len(feature_names)), "classes": [str(c) for c in classes],
        "final_prediction_model": final_model_name,
        "validation_metrics_file": "validation_metrics.csv",
        "warning": "Test labels are unavailable; test accuracy was not calculated."
    }, indent=2), encoding="utf-8")
    print("\nDONE.")
    print("Validation metrics:", OUT / "validation_metrics.csv")
    print("Test predictions:", OUT / "submission_predictions.csv")
    print("Final model used:", final_model_name)
    print("IMPORTANT: test labels are unavailable, so no test accuracy is reported.")

if __name__ == "__main__":
    main()
