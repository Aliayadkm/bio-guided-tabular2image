# -*- coding: utf-8 -*-
"""
Full-scale (200,000 rows) test of the PAPER'S OWN pipeline on the SAME group-aware split as
bacteria_full_experiments.py (same seed, same StratifiedGroupKFold, same fold, same CNN architecture).

What it adds to the first script:
  (1) biological feature ordering (Pearson -> UPGMA -> optimal leaf ordering) computed ONLY on the training fold
  (2) the same CNN trained on the biologically ordered 13x22 image         -> isolates the encoding effect
  (3) the paper's Hybrid: 128-d CNN embedding -> XGBoost (fixed rounds, no validation-based stopping)
  (4) comparison with the first run's predictions (row-major CNN, XGBoost on raw features, equal-weight fusion):
      accuracy with bootstrap CIs, exact McNemar tests, paired bootstrap CIs of accuracy differences.

Needs: train.csv.zip (or train.csv) and, to compare with the first run, bacteria_results/validation_predictions.csv.
Resumable: each step is saved to OUT and skipped if already present.
Smoke test (minutes): set environment variable BACTERIA_SMOKE=1.
"""
import os, re, json, time, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, binomtest
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss

warnings.filterwarnings("ignore")
SEED, N_SPLITS, FOLD_TO_USE = 42, 5, 0
SMOKE = os.environ.get("BACTERIA_SMOKE") == "1"
CNN_EPOCHS = 1 if SMOKE else 40
CNN_BATCH_SIZE = 512
XGB_ROUNDS = 20 if SMOKE else 400          # fixed: no early stopping on the validation set
TRAIN_PATH = Path(os.environ.get("TRAIN_PATH", "train.csv.zip"))
FIRST_RUN_DIR = Path(os.environ.get("FIRST_RUN_DIR", "bacteria_results"))
OUT = Path("bacteria_results_paper_pipeline_smoke" if SMOKE else "bacteria_results_paper_pipeline")
N_BOOT = 200 if SMOKE else 2000


def seed_everything():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED); np.random.seed(SEED)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def gc_content(names):
    out = []
    for n in names:
        d = {k: int(v) for k, v in re.findall(r"([ATGC])(\d+)", n)}
        out.append((d.get("G", 0) + d.get("C", 0)) / 10.0)
    return np.array(out)


def biological_order(X_train):
    """Order learnt on the TRAINING FOLD only (no labels used)."""
    R = np.nan_to_num(np.corrcoef(X_train, rowvar=False), nan=0.0)
    D = 1.0 - R
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    cond = squareform(np.clip(D, 0, None), checks=False)
    Z = optimal_leaf_ordering(linkage(cond, method="average"), cond)
    return leaves_list(Z)


def build_cnn(n_classes, input_shape=(13, 22, 1)):
    """Identical to the CNN of bacteria_full_experiments.py; the 128-d dense layer is named 'embedding'."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    inp = layers.Input(shape=input_shape)
    x = inp
    for filters, pool in [(64, False), (64, True), (128, False), (128, False)]:
        x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_uniform")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)
        if pool:
            x = layers.MaxPooling2D(pool_size=2)(x)
            x = layers.Dropout(0.20)(x)
    x = layers.GlobalAveragePooling2D()(x)
    emb = layers.Dense(128, activation="swish", name="embedding")(x)
    x = layers.Dropout(0.25)(emb)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
           callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
           callbacks.CSVLogger(str(OUT / "bio_cnn_training_history.csv"))]
    return model, models.Model(inp, emb), cbs


def mcnemar_exact(ok_a, ok_b):
    b = int((ok_a & ~ok_b).sum()); c = int((~ok_a & ok_b).sum())
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return b, c, p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seed_everything()
    train = pd.read_csv(TRAIN_PATH, compression="infer", low_memory=False)
    if SMOKE:
        train = train.sample(6000, random_state=SEED).reset_index(drop=True)
    feat = [c for c in train.columns if c not in ("row_id", "target")]
    X_df = train[feat].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.median()).fillna(0.0)
    le = LabelEncoder(); y = le.fit_transform(train["target"].astype(str)); K = len(le.classes_)
    groups = pd.util.hash_pandas_object(X_df, index=False).to_numpy(dtype=np.uint64)
    tr_idx, va_idx = list(StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X_df, y, groups=groups))[FOLD_TO_USE]
    assert not (set(groups[tr_idx].tolist()) & set(groups[va_idx].tolist())), "group leakage"
    ytr, yva = y[tr_idx], y[va_idx]
    Xtr, Xva = X_df.iloc[tr_idx].to_numpy(np.float64), X_df.iloc[va_idx].to_numpy(np.float64)
    print(f"train {len(tr_idx):,}  validation {len(va_idx):,}  (group overlap 0)  smoke={SMOKE}")

    # ---- (1) biological ordering from the training fold only ----
    order_path = OUT / "bio_order.npy"
    if order_path.exists():
        order = np.load(order_path)
    else:
        order = biological_order(Xtr); np.save(order_path, order)
    gc = gc_content(feat)
    rho, p = spearmanr(np.argsort(order), gc)
    print(f"ordering learnt on the training fold: Spearman(position, GC) = {rho:.3f} (p={p:.2e})")
    save_json({"gc_spearman_rho": float(rho), "gc_spearman_p": float(p)}, OUT / "ordering_gc_validation.json")

    sc = MinMaxScaler().fit(Xtr)
    img = lambda X: sc.transform(X)[:, order].astype(np.float32).reshape(-1, 13, 22, 1)
    Itr, Iva = img(Xtr), img(Xva)

    # ---- (2) CNN on the biologically ordered image ----
    probs_path = OUT / "bio_cnn_val_probs.npy"
    if not probs_path.exists():
        import tensorflow as tf
        tf.keras.backend.clear_session()
        model, emb_model, cbs = build_cnn(K)
        model.fit(Itr, ytr, validation_data=(Iva, yva), epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE, callbacks=cbs, verbose=2)
        np.save(probs_path, model.predict(Iva, batch_size=CNN_BATCH_SIZE, verbose=0))
        np.save(OUT / "emb_train.npy", emb_model.predict(Itr, batch_size=CNN_BATCH_SIZE, verbose=0))
        np.save(OUT / "emb_val.npy", emb_model.predict(Iva, batch_size=CNN_BATCH_SIZE, verbose=0))
        model.save(OUT / "bio_cnn_model.keras")
    p_bio = np.load(probs_path)

    # ---- (3) paper's Hybrid: CNN embedding -> XGBoost ----
    hyb_path = OUT / "hybrid_val_probs.npy"
    if not hyb_path.exists():
        import xgboost as xgb
        etr, eva = np.load(OUT / "emb_train.npy"), np.load(OUT / "emb_val.npy")
        params = {"objective": "multi:softprob", "num_class": K, "eval_metric": "mlogloss", "max_depth": 6, "eta": 0.1,
                  "subsample": 0.85, "colsample_bytree": 0.85, "tree_method": "hist", "seed": SEED,
                  "nthread": max(1, (os.cpu_count() or 4) - 1)}
        bst = xgb.train(params, xgb.DMatrix(etr, label=ytr), num_boost_round=XGB_ROUNDS)
        np.save(hyb_path, bst.predict(xgb.DMatrix(eva)).reshape(-1, K))
    p_hyb = np.load(hyb_path)

    # ---- (4) comparison with the first run ----
    preds = {"CNN bio-ordered (this run)": p_bio.argmax(1), "Hybrid bio-CNN-embedding+XGB (this run)": p_hyb.argmax(1)}
    first = FIRST_RUN_DIR / "validation_predictions.csv"
    if first.exists() and not SMOKE:
        fp = pd.read_csv(first)
        fp = pd.read_csv(first)

        # Match predictions by validation-row index
        fp = fp.set_index("row_index_in_validation")

        rid = np.arange(len(yva))
        fp = fp.reindex(rid)

        assert fp["true_label"].notna().all(), (
            "Some validation rows are missing from validation_predictions.csv"
        )

        assert (
                fp["true_label"].astype(str).to_numpy()
                == le.inverse_transform(yva)
        ).all(), "Validation labels do not match"

        mapping = [
            ("cnn_row_major_prediction", "CNN row-major (first run)"),
            ("raw_xgboost_prediction", "XGBoost raw features (first run)"),
            ("probability_fusion_prediction", "Equal-weight fusion CNN+XGB (first run)")
        ]

        for col, name in mapping:
            preds[name] = le.transform(fp[col].astype(str).to_numpy())
    else:
        print("first-run predictions not used (missing or smoke test)")

    rng = np.random.default_rng(SEED); n = len(yva)
    rows = []
    for name, pr in preds.items():
        ok = pr == yva
        boots = [ok[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
        rows.append({"model": name, "accuracy": float(ok.mean()), "ci95_low": float(np.percentile(boots, 2.5)),
                     "ci95_high": float(np.percentile(boots, 97.5)), "macro_f1": float(f1_score(yva, pr, average="macro"))})
    pd.DataFrame(rows).to_csv(OUT / "accuracy_with_ci.csv", index=False)

    names = list(preds)
    pair_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b_ = names[i], names[j]
            oka, okb = preds[a] == yva, preds[b_] == yva
            b, c, p = mcnemar_exact(oka, okb)
            diffs = []
            for _ in range(N_BOOT):
                ix = rng.integers(0, n, n); diffs.append(oka[ix].mean() - okb[ix].mean())
            pair_rows.append({"A": a, "B": b_, "acc_A_minus_acc_B": float(oka.mean() - okb.mean()),
                              "ci95_low": float(np.percentile(diffs, 2.5)), "ci95_high": float(np.percentile(diffs, 97.5)),
                              "A_only_correct": b, "B_only_correct": c, "p_exact_mcnemar": p})
    pairs = pd.DataFrame(pair_rows)
    m = len(pairs)
    order_p = np.argsort(pairs["p_exact_mcnemar"].to_numpy()); holm = np.empty(m); run = 0.0
    for rank, k in enumerate(order_p):
        run = max(run, min(1.0, pairs["p_exact_mcnemar"].iloc[k] * (m - rank))); holm[k] = run
    pairs["p_holm"] = holm
    pairs.to_csv(OUT / "paired_comparisons.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
    print("\n=== ACCURACY (group-aware validation fold, bootstrap 95% CI) ===")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\n=== PAIRED COMPARISONS (exact McNemar, Holm over all pairs) ===")
    print(pairs.round(4).to_string(index=False))
    print(f"\nElapsed minutes: {(time.time() - t0) / 60:.1f}.  Outputs in {OUT.resolve()}")


if __name__ == "__main__":
    main()
