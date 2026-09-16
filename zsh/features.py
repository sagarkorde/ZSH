"""Feature audit, selection and preprocessing (log transform + robust scaling)."""
import numpy as np
import pandas as pd

from .config import CFG


def mode_share(series):
    vc = series.value_counts(dropna=False)
    return float(vc.iloc[0] / len(series))


def select_features(dev, rng, log=print):
    """Apply the pre-declared selection rules (ANALYSIS_PLAN §3.1) on DEV rows."""
    fc = CFG["features"]
    cand = list(fc["priority"])
    report = {"candidates": cand + fc["extra_audit_only"], "mode_share": {}, "removed": {}}

    for f in cand + fc["extra_audit_only"]:
        report["mode_share"][f] = mode_share(dev[f])
    kept_after_mode = []
    for f in cand:
        if report["mode_share"][f] >= fc["mode_share_max"]:
            report["removed"][f] = f"most frequent value covers {report['mode_share'][f]:.5f} of DEV rows"
        else:
            kept_after_mode.append(f)
    for f in fc["extra_audit_only"]:
        report["removed"][f] = (f"most frequent value covers {report['mode_share'][f]:.5f} of DEV rows"
                                if report["mode_share"][f] >= fc["mode_share_max"]
                                else "audit-only column (not a v2 candidate)")

    n = min(fc["spearman_rows"], len(dev))
    idx = rng.choice(len(dev), size=n, replace=False)
    ranks = dev.iloc[idx][kept_after_mode].rank(method="average")
    rho = ranks.corr(method="pearson")
    report["spearman"] = rho.round(4).to_dict()

    selected = []
    for f in kept_after_mode:
        clash = [(g, float(rho.loc[f, g])) for g in selected if abs(rho.loc[f, g]) >= fc["spearman_max"]]
        if clash:
            g, r = max(clash, key=lambda t: abs(t[1]))
            report["removed"][f] = f"|Spearman rho| = {abs(r):.4f} with kept feature {g}"
        else:
            selected.append(f)
    report["selected"] = selected
    log(f"selected {len(selected)} features: {selected}")
    return selected, report


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


class Preprocessor:
    """sign(x)*log(1+|x|) and median/IQR scaling for continuous features; binaries untouched."""

    def __init__(self, features, binary=None):
        self.features = list(features)
        binary = set(CFG["features"]["binary"] if binary is None else binary)
        self.is_binary = np.array([f in binary for f in self.features])

    def _raw(self, df):
        X = df[self.features].to_numpy(np.float64, copy=True)
        cont = ~self.is_binary
        X[:, cont] = signed_log(X[:, cont])
        return X

    def fit(self, df, sample_weight=None):
        X = self._raw(df)
        d = X.shape[1]
        self.center_ = np.zeros(d)
        self.scale_ = np.ones(d)
        for j in np.where(~self.is_binary)[0]:
            x = X[:, j]
            if sample_weight is None:
                q25, med, q75 = np.percentile(x, [25, 50, 75])
                sd = x.std()
            else:
                q25, med, q75 = _weighted_quantiles(x, sample_weight, [0.25, 0.5, 0.75])
                mu = np.average(x, weights=sample_weight)
                sd = np.sqrt(np.average((x - mu) ** 2, weights=sample_weight))
            iqr = q75 - q25
            self.center_[j] = med
            if iqr > 0:
                self.scale_[j] = iqr
            elif sd > 0:
                self.scale_[j] = 1.349 * sd
            else:
                self.scale_[j] = 1.0
        return self

    def transform(self, df):
        X = self._raw(df)
        X = (X - self.center_) / self.scale_
        return X.astype(np.float32)

    def to_dict(self):
        return {"features": self.features, "binary": self.is_binary.tolist(),
                "center": self.center_.tolist(), "scale": self.scale_.tolist()}


def _weighted_quantiles(x, w, qs):
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws)
    cw /= cw[-1]
    return [float(xs[np.searchsorted(cw, q)]) for q in qs]
