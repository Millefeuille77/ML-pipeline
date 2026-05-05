"""Cluster the 30 SKUs by demand behavior using K-Means.

Sweep k in [3, 6] and select the best k by silhouette score. Assign
human-readable labels from centroid inspection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.api.schemas import ClusterResult, ClusterSummary
from src.models.feature_engineering import LIFECYCLE_ENCODING
from src.utils.helpers import safe_divide
from src.utils.logger import get_logger

logger = get_logger(__name__)

_K_MIN: Final[int] = 3
_K_MAX: Final[int] = 6
_RANDOM_STATE: Final[int] = 42

_CLUSTER_FEATURE_COLS: Final[list[str]] = [
    "avg_weekly_demand", "demand_coefficient_of_variation", "seasonality_strength",
    "trend_slope", "promo_sensitivity", "lifecycle_stage_encoded", "price_point_normalized",
]


@dataclass
class ClusterModel:
    """Trained clustering bundle plus per-SKU labels."""

    estimator: KMeans
    scaler: StandardScaler
    sku_labels: dict[str, int]
    cluster_descriptions: dict[int, str]


def _demand_features(sku_df: pd.DataFrame) -> pd.DataFrame:
    """Return (sku, avg_weekly_demand, std_demand) aggregated per SKU."""
    avg = sku_df.groupby("sku", as_index=False)["units_sold"].mean().rename(
        columns={"units_sold": "avg_weekly_demand"}
    )
    std = sku_df.groupby("sku", as_index=False)["units_sold"].std().rename(
        columns={"units_sold": "std_demand"}
    )
    return avg.merge(std, on="sku")


def _seasonality_feature(sku_df: pd.DataFrame) -> pd.DataFrame:
    """Return (sku, seasonality_strength) = max_monthly_avg / min_monthly_avg."""
    monthly = sku_df.groupby(["sku", "month"], as_index=False)["units_sold"].mean()
    agg = monthly.groupby("sku")["units_sold"].agg(["max", "min"]).reset_index()
    agg["seasonality_strength"] = (agg["max"] / agg["min"].replace(0, np.nan)).fillna(1.0)
    return agg[["sku", "seasonality_strength"]]


def _trend_feature(sku_df: pd.DataFrame) -> pd.DataFrame:
    """Return (sku, trend_slope) via linear regression over the full series."""
    return (
        sku_df.sort_values(["sku", "week"])
        .groupby("sku")
        .apply(lambda g: _slope(g["units_sold"].to_numpy()), include_groups=False)
        .reset_index(name="trend_slope")
    )


def _promo_feature(sku_df: pd.DataFrame) -> pd.DataFrame:
    """Return (sku, promo_sensitivity) = promo-mean / non-promo-mean."""
    result = (
        sku_df.groupby("sku")[["units_sold", "promotion_flag"]]
        .apply(_promo_sensitivity)
        .reset_index()
        .rename(columns={0: "promo_sensitivity"})
    )
    if "promo_sensitivity" not in result.columns:
        result = result.rename(columns={result.columns[-1]: "promo_sensitivity"})
    return result


def _lifecycle_feature(sku_df: pd.DataFrame) -> pd.DataFrame:
    """Return (sku, lifecycle_stage_encoded) using modal lifecycle per SKU."""
    lc = (
        sku_df.groupby("sku")["lifecycle_stage"]
        .agg(lambda s: s.mode().iloc[0] if not s.empty else "Mature")
        .reset_index(name="lifecycle_stage_mode")
    )
    lc["lifecycle_stage_encoded"] = lc["lifecycle_stage_mode"].map(LIFECYCLE_ENCODING).fillna(1).astype(int)
    return lc[["sku", "lifecycle_stage_encoded"]]


def _build_sku_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the weekly DataFrame into one feature row per SKU."""
    frame = weekly_df.copy()
    demand = _demand_features(frame)
    seasonality = _seasonality_feature(frame)
    trend = _trend_feature(frame)
    promo = _promo_feature(frame)
    lifecycle = _lifecycle_feature(frame)
    avg_price = frame.groupby("sku", as_index=False)["price_unit"].mean().rename(
        columns={"price_unit": "avg_price"}
    )
    merged = (
        demand.merge(seasonality, on="sku").merge(trend, on="sku")
        .merge(promo, on="sku").merge(lifecycle, on="sku").merge(avg_price, on="sku")
    )
    merged["demand_coefficient_of_variation"] = merged.apply(
        lambda r: safe_divide(r["std_demand"], r["avg_weekly_demand"], 0.0), axis=1
    )
    price_max = merged["avg_price"].max() or 1.0
    merged["price_point_normalized"] = merged["avg_price"] / price_max
    return merged[["sku", *_CLUSTER_FEATURE_COLS]].fillna(0.0)


def _slope(series: np.ndarray) -> float:
    """Linear regression slope of `series` against integer time index."""
    if len(series) < 2:
        return 0.0
    indices = np.arange(len(series), dtype=float)
    slope, _intercept = np.polyfit(indices, series.astype(float), 1)
    return float(slope)


def _promo_sensitivity(group: pd.DataFrame) -> float:
    """Ratio of promo-week mean to non-promo-week mean of `units_sold`."""
    promo_mean = group.loc[group["promotion_flag"] == 1, "units_sold"].mean()
    base_mean = group.loc[group["promotion_flag"] == 0, "units_sold"].mean()
    if not np.isfinite(promo_mean) or not np.isfinite(base_mean):
        return 1.0
    return safe_divide(float(promo_mean), float(base_mean), 1.0)


def _select_best_k(
    scaled: np.ndarray, k_min: int, k_max: int
) -> tuple[int, KMeans, float]:
    """Sweep k in [k_min, k_max], return (best_k, best_estimator, best_score).

    Raises:
        RuntimeError: If no valid k produces a multi-cluster partition.
    """
    n_samples = scaled.shape[0]
    best: tuple[float, KMeans] | None = None
    for k in range(k_min, min(k_max, n_samples - 1) + 1):
        if k >= n_samples:
            continue
        estimator = KMeans(n_clusters=k, random_state=_RANDOM_STATE, n_init=10)
        labels = estimator.fit_predict(scaled)
        unique_labels = set(labels)
        if len(unique_labels) < 2 or len(unique_labels) >= n_samples:
            continue
        score = silhouette_score(scaled, labels)
        if best is None or score > best[0]:
            best = (score, estimator)
    if best is None:
        raise RuntimeError("clustering failed: no valid k found")
    return best[1].n_clusters, best[1], best[0]


def fit_clustering(weekly_df: pd.DataFrame) -> ClusterModel:
    """Fit K-Means after sweeping k in [3, 6] and selecting by silhouette.

    Raises:
        ValueError: If fewer than 4 SKUs are present.
    """
    sku_features = _build_sku_features(weekly_df)
    if len(sku_features) < _K_MIN + 1:
        raise ValueError(f"need at least {_K_MIN + 1} SKUs to cluster, got {len(sku_features)}")
    matrix = sku_features[_CLUSTER_FEATURE_COLS].to_numpy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    chosen_k, estimator, best_score = _select_best_k(scaled, _K_MIN, _K_MAX)
    labels = estimator.predict(scaled)
    sku_labels = {sku: int(label) for sku, label in zip(sku_features["sku"], labels)}
    descriptions = _label_clusters(sku_features.assign(cluster=labels))
    logger.info("clustering_fit", extra={"chosen_k": chosen_k, "silhouette": round(best_score, 4)})
    return ClusterModel(estimator=estimator, scaler=scaler,
                        sku_labels=sku_labels, cluster_descriptions=descriptions)


def _label_clusters(frame: pd.DataFrame) -> dict[int, str]:
    """Assign human-readable labels to numeric clusters."""
    descriptions: dict[int, str] = {}
    for cluster_id, group in frame.groupby("cluster"):
        avg_demand = group["avg_weekly_demand"].mean()
        cv = group["demand_coefficient_of_variation"].mean()
        seasonality = group["seasonality_strength"].mean()
        promo = group["promo_sensitivity"].mean()
        lifecycle = group["lifecycle_stage_encoded"].mean()
        if avg_demand > frame["avg_weekly_demand"].median() and cv < 0.3:
            descriptions[int(cluster_id)] = "High-Volume Steady"
        elif seasonality > 1.5:
            descriptions[int(cluster_id)] = "Seasonal Responders"
        elif promo > 1.3:
            descriptions[int(cluster_id)] = "Promo-Dependent"
        elif lifecycle < 0.7:
            descriptions[int(cluster_id)] = "Declining Niche"
        else:
            descriptions[int(cluster_id)] = "Mid-Tier Consistent"
    return descriptions


def get_cluster(sku: str, model: ClusterModel) -> ClusterResult:
    """Return cluster membership info for a single SKU.

    Raises:
        KeyError: If `sku` is unknown to the model.
    """
    if sku not in model.sku_labels:
        raise KeyError(f"sku {sku!r} not in clustering model")
    cluster_id = model.sku_labels[sku]
    description = model.cluster_descriptions.get(cluster_id, "Unlabeled")
    similar = [other for other, label in model.sku_labels.items() if label == cluster_id and other != sku]
    return ClusterResult(sku=sku, cluster_label=f"cluster_{cluster_id}",
                         cluster_description=description, similar_skus=similar)


def get_all_clusters(model: ClusterModel, weekly_df: pd.DataFrame) -> list[ClusterSummary]:
    """Return a summary row per cluster."""
    summaries: list[ClusterSummary] = []
    cluster_to_skus: dict[int, list[str]] = {}
    for sku, label in model.sku_labels.items():
        cluster_to_skus.setdefault(label, []).append(sku)
    for cluster_id, members in cluster_to_skus.items():
        avg_demand = float(weekly_df.loc[weekly_df["sku"].isin(members), "units_sold"].mean() or 0.0)
        summaries.append(ClusterSummary(
            cluster_label=f"cluster_{cluster_id}",
            cluster_description=model.cluster_descriptions.get(cluster_id, "Unlabeled"),
            sku_count=len(members), members=sorted(members), avg_weekly_demand=avg_demand,
        ))
    summaries.sort(key=lambda s: s.cluster_label)
    return summaries
