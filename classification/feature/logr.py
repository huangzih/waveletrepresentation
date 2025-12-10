import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import normaltest, kstest, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

CSV_SEP = ";"
RANDOM_STATE = 42
MODEL_ROOT = "./models/logreg"


def TMSD(traj, t_lags):
    traj = np.asarray(traj, dtype=float)
    t_lags = np.asarray(t_lags, dtype=int)
    ttt = np.zeros_like(t_lags, dtype=float)
    n = len(traj)
    for idx, t in enumerate(t_lags):
        if t <= 0 or t >= n:
            continue
        diff = traj[:-t] - traj[t:]
        if len(diff) > 0:
            ttt[idx] = np.mean(diff ** 2)
    return ttt


def aging(traj, twind):
    traj = np.asarray(traj, dtype=float)
    twind = np.asarray(twind, dtype=int)
    age = np.zeros(len(twind))
    for i, it in enumerate(twind):
        it = max(2, min(len(traj), it))
        traj_seg = traj[:it]
        age[i] = TMSD(traj_seg, np.array([1]))[0]
    return age


def bound(val):
    if np.isnan(val) or np.isinf(val):
        return 0.0
    if val >= 100:
        val = 100
    elif val <= -100:
        val = -100
    return float(val)


def asymmetry_1D(traj):
    mean_x = np.mean(traj)
    std_x = np.std(traj)
    if std_x == 0:
        return 0.0
    return np.mean((traj - mean_x) ** 3) / (std_x ** 3)


def efficiency_1D(traj):
    if len(traj) < 2:
        return 0.0
    net_disp = traj[-1] - traj[0]
    total_path = np.sum(np.abs(np.diff(traj)))
    if total_path == 0:
        return 0.0
    return net_disp / total_path


def fractal_dimension_1D(traj, scales=(2, 4, 8, 16, 32)):
    L = len(traj)
    if L < 2:
        return 1.0
    box_counts = []
    used_scales = []
    for s in scales:
        if s > L:
            break
        size = L // s
        local_ranges = []
        for i in range(s):
            segment = traj[i * size:(i + 1) * size]
            local_ranges.append(segment.max() - segment.min())
        box_counts.append(np.sum(local_ranges))
        used_scales.append(s)
    if len(box_counts) <= 1:
        return 1.0
    log_scales = np.log(np.array(used_scales))
    log_counts = np.log(box_counts)
    A = np.vstack([log_scales, np.ones(len(log_scales))]).T
    slope, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]
    return slope


def straightness_1D(traj):
    return abs(efficiency_1D(traj))


def trappedness_1D(traj, threshold=0.1):
    mn, mx = np.min(traj), np.max(traj)
    if (mx - mn) < 2 * threshold:
        return 1.0
    lower_bound = mn + threshold
    upper_bound = mx - threshold
    frac = np.mean((traj >= lower_bound) & (traj <= upper_bound))
    return float(frac)


def noah_moses_joseph_exponents_1D(traj):
    L = len(traj)
    if L < 5:
        return (0.0, 0.0, 0.0)
    max_lag = min(10, L - 1)
    lags = np.arange(1, max_lag + 1)
    variances, ranges, msds = [], [], []
    for lag in lags:
        diffs = traj[lag:] - traj[:-lag]
        variances.append(np.var(diffs))
        ranges.append(np.max(diffs) - np.min(diffs))
        msds.append(np.mean(diffs ** 2))
    log_lags = np.log(lags)
    A = np.vstack([log_lags, np.ones(len(log_lags))]).T
    noah_slope = np.linalg.lstsq(A, np.log(variances), rcond=None)[0][0]
    moses_slope = np.linalg.lstsq(A, np.log(ranges), rcond=None)[0][0]
    joseph_slope = np.linalg.lstsq(A, np.log(msds), rcond=None)[0][0]
    return (noah_slope, moses_slope, joseph_slope)


def empirical_vacf_1d(traj, lag=1):
    v = np.diff(traj)
    if len(v) <= lag:
        return 0.0
    v0 = v[:-lag]
    v1 = v[lag:]
    return float(np.mean(v0 * v1))


def maximal_excursion_1d(traj):
    disp = traj - traj[0]
    return float(np.max(np.abs(disp)))


def gaussianity_mean(traj, t_lags):
    g_list = []
    for t in t_lags:
        t = int(t)
        if t <= 0 or t >= len(traj):
            continue
        diffs = traj[t:] - traj[:-t]
        r2 = np.mean(diffs ** 2)
        r4 = np.mean(diffs ** 4)
        if r2 > 0:
            g = -1 + 2 * r4 / (3 * (r2 ** 2))
            g_list.append(g)
    if not g_list:
        return 0.0
    return float(np.mean(g_list))


def msd_ratio(msd_vals, t_lags):
    msd_vals = np.asarray(msd_vals, float)
    t_lags = np.asarray(t_lags, float)
    if len(msd_vals) < 2:
        return 0.0
    n1 = t_lags[:-1]
    n2 = t_lags[1:]
    r_n1 = msd_vals[:-1]
    r_n2 = msd_vals[1:]
    valid = (r_n2 > 0) & (n2 != 0)
    if not np.any(valid):
        return 0.0
    val = np.mean(r_n1[valid] / r_n2[valid] - n1[valid] / n2[valid])
    return float(val)


def p_variation_simple(traj, p=1.5):
    if len(traj) < 2:
        return 0.0
    v = np.diff(traj)
    return float(np.mean(np.abs(v) ** p))


def residence_times_features(traj, thresh_factor=1.0):
    v = np.abs(np.diff(traj))
    if len(v) == 0:
        return 0.0, 0.0
    thresh = thresh_factor * np.median(v)
    states = (v > thresh).astype(int)
    durations = []
    current = states[0]
    length = 1
    for s in states[1:]:
        if s == current:
            length += 1
        else:
            durations.append(length)
            current = s
            length = 1
    durations.append(length)
    durations = np.array(durations, float)
    avg_res_time = float(np.mean(durations))
    if len(durations) > 1:
        res_state = float(np.max(durations) - np.min(durations))
    else:
        res_state = 0.0
    return res_state, avg_res_time


def standardized_max_distance(traj):
    n = len(traj)
    if n == 0:
        return 0.0
    disp = traj - traj[0]
    max_dist = np.max(np.abs(disp))
    denom = np.std(traj) * np.sqrt(n)
    if denom == 0:
        return 0.0
    return float(max_dist / denom)


def mean_maximal_excursion(traj):
    disp = np.abs(traj - traj[0])
    running_max = np.maximum.accumulate(disp)
    return float(np.mean(running_max))


def detrend_moving_average_features(traj, window=10):
    traj = np.asarray(traj, float)
    n = len(traj)
    if n < window + 1:
        return 0.0, 0.0, 0.0
    means, stds, ma = [], [], []
    for i in range(n - window + 1):
        seg = traj[i:i + window]
        m = np.mean(seg)
        s = np.std(seg)
        means.append(m)
        stds.append(s)
        ma.append(m)
    ma = np.array(ma)
    residuals = traj[window - 1:] - ma
    detrend_ma = float(np.std(residuals))
    avg_mw = float(np.mean(means))
    max_std = float(np.max(stds))
    return detrend_ma, avg_mw, max_std


def trajec_feature(traces):
    traces = np.asarray(traces, float)
    n_points = len(traces)
    if n_points < 3:
        return np.zeros(30, dtype=float)

    n_points_float = float(n_points)

    mean_x = np.mean(traces)
    std_x = np.std(traces)
    if std_x != 0:
        kurtosis_x = bound(np.mean((traces - mean_x) ** 4) / (std_x ** 4))
    else:
        kurtosis_x = 0.0

    vtraces = np.diff(traces)
    if len(vtraces) > 0:
        avg_step = bound(np.mean(np.abs(vtraces)))
    else:
        avg_step = 0.0

    if len(vtraces) > 20:
        _, nom_p = normaltest(vtraces)
        nom_stat = bound(np.log(nom_p + 1e-12))
    else:
        nom_stat = -10.0

    if len(vtraces) > 1 and np.std(vtraces) > 0:
        _, ks_p = kstest(
            vtraces,
            "norm",
            args=(np.mean(vtraces), np.std(vtraces)),
        )
        ks_stat = bound(np.log(ks_p + 1e-12))
    else:
        ks_stat = -10.0

    tlag = np.linspace(1, max(2, n_points - 1), 10, dtype=int)
    tlag = np.unique(tlag)
    tlag = tlag[(tlag > 0) & (tlag < n_points)]

    msd = TMSD(traces, tlag)
    valid_mask = msd > 1e-12

    if np.sum(valid_mask) < 3:
        anomalous_exp = 0.0
        diff_coeff = 0.0
        msd_rat = 0.0
        exponent_power = 0.0
        q_power = 0.0
        inter_spread = 0.0
    else:
        msd_safe = msd[valid_mask]
        tlag_safe = tlag[valid_mask]

        log_t = np.log(tlag_safe)
        log_msd = np.log(msd_safe)

        poly1 = np.polyfit(log_t, log_msd, 1)
        nw1 = poly1[0]
        nw0 = poly1[1]

        anomalous_exp = bound(nw1)
        diff_coeff = bound(np.exp(nw0))

        poly2 = np.polyfit(log_t, log_msd, 2)
        nw2 = poly2[0]
        exponent_power = bound(nw2)

        msd_rat = bound(msd_ratio(msd, tlag))

        pred_logmsd = nw1 * log_t + nw0
        ss_res = np.sum((log_msd - pred_logmsd) ** 2)
        ss_tot = np.sum((log_msd - np.mean(log_msd)) ** 2)
        q_power = bound(1 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

        inter_spread = bound(np.mean(msd_safe))

    asym = bound(asymmetry_1D(traces))
    effi = bound(efficiency_1D(traces))
    fdim = bound(fractal_dimension_1D(traces))
    stra = bound(straightness_1D(traces))
    trap = bound(trappedness_1D(traces, threshold=0.1))

    noah_exp, moses_exp, joseph_exp = noah_moses_joseph_exponents_1D(traces)
    noah_exp = bound(noah_exp)
    moses_exp = bound(moses_exp)
    joseph_exp = bound(joseph_exp)

    vacf1 = bound(empirical_vacf_1d(traces, lag=1))
    max_exc = bound(maximal_excursion_1d(traces))
    p_var = bound(p_variation_simple(traces, p=1.5))
    gauss_mean = bound(gaussianity_mean(traces, tlag))
    std_max_dist = bound(standardized_max_distance(traces))

    res_state, avg_res_time = residence_times_features(traces)
    res_state = bound(res_state)
    avg_res_time = bound(avg_res_time)
    mean_max_exc = bound(mean_maximal_excursion(traces))

    detrend_ma, avg_mw, max_std = detrend_moving_average_features(
        traces,
        window=min(10, max(2, n_points // 3)),
    )
    detrend_ma = bound(detrend_ma)
    avg_mw = bound(avg_mw)
    max_std = bound(max_std)

    feature = np.array(
        [
            anomalous_exp,
            diff_coeff,
            msd_rat,
            effi,
            stra,
            vacf1,
            max_exc,
            p_var,
            asym,
            fdim,
            gauss_mean,
            kurtosis_x,
            trap,
            std_max_dist,
            exponent_power,
            res_state,
            avg_res_time,
            avg_step,
            q_power,
            inter_spread,
            n_points_float,
            mean_max_exc,
            nom_stat,
            ks_stat,
            noah_exp,
            moses_exp,
            joseph_exp,
            detrend_ma,
            avg_mw,
            max_std,
        ],
        dtype=float,
    )
    return feature


def trajec_feature_nd_split(traj_matrix):
    feats = [trajec_feature(traj_matrix[d]) for d in range(traj_matrix.shape[0])]
    return np.concatenate(feats, axis=0)


def parse_trajectory_string(s, seq_length):
    values = np.array([float(v) for v in s.split(",")], dtype=np.float32)
    if len(values) != seq_length:
        raise ValueError(
            "Sequence length mismatch: expected {}, got {}.".format(seq_length, len(values))
        )
    return values


def extract_features_from_dataframe(df, seq_length, dimension):
    all_dim_cols = ["pos_x", "pos_y", "pos_z"]
    if dimension < 1 or dimension > 3:
        raise ValueError("dimension must be 1, 2 or 3, got {}.".format(dimension))

    dim_cols = all_dim_cols[:dimension]
    for col in dim_cols:
        if col not in df.columns:
            raise KeyError(
                "dimension={} but column '{}' not found in DataFrame.".format(dimension, col)
            )

    if "label" not in df.columns:
        raise KeyError("Input DataFrame must contain 'label' column.")

    features_list = []
    labels_list = []

    for _, row in df.iterrows():
        trajs = []
        for col in dim_cols:
            trajs.append(parse_trajectory_string(row[col], seq_length))
        traj_matrix = np.stack(trajs, axis=0)
        feat = trajec_feature_nd_split(traj_matrix)
        features_list.append(feat)
        labels_list.append(row["label"])

    X = np.vstack(features_list).astype(np.float32)
    y = np.asarray(labels_list)
    return X, y


def train_logreg(seq_length, dimension, train_path, valid_path):
    df_train = pd.read_csv(train_path, sep=CSV_SEP)
    df_valid = pd.read_csv(valid_path, sep=CSV_SEP)

    print("Train samples:", len(df_train))
    print("Valid samples:", len(df_valid))

    X_train, y_train = extract_features_from_dataframe(df_train, seq_length, dimension)
    X_valid, y_valid = extract_features_from_dataframe(df_valid, seq_length, dimension)

    print("Train data shape:", X_train.shape, y_train.shape)
    print("Valid data shape:", X_valid.shape, y_valid.shape)

    model_dir = os.path.join(MODEL_ROOT, "len{}_dim{}".format(seq_length, dimension))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "logreg_model.joblib")

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred_valid = clf.predict(X_valid)
    acc_valid = accuracy_score(y_valid, y_pred_valid)
    f1_valid_micro = f1_score(y_valid, y_pred_valid, average="micro")
    print("Accuracy: {:.4f}, F1-micro: {:.4f}".format(acc_valid, f1_valid_micro))
    joblib.dump(clf, model_path)
    print("Model saved to:", model_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("length", type=int)
    parser.add_argument("dimension", type=int, choices=[1, 2, 3])
    parser.add_argument("train_path", type=str)
    parser.add_argument("valid_path", type=str)
    args = parser.parse_args()
    train_logreg(args.length, args.dimension, args.train_path, args.valid_path)


if __name__ == "__main__":
    main()