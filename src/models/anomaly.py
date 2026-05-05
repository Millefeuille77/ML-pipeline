"""Statistical anomaly detection rules tuned to the FMCG dataset."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

import numpy as np
import pandas as pd

from src.api.schemas import Alert
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SPIKE_SIGMA: Final[float] = 2.5
_DROP_SIGMA: Final[float] = 2.0
_STOCK_RISK_WEEKS: Final[float] = 2.0
_PRICE_DEVIATION_PCT: Final[float] = 0.15


def detect_alerts(weekly_df: pd.DataFrame) -> list[Alert]:
    """Run all anomaly rules over `weekly_df` and emit Alert objects.

    Args:
        weekly_df: Weekly DataFrame (must include lag/rolling features).

    Returns:
        List of Alert instances. Empty if no rules triggered.
    """
    alerts: list[Alert] = []
    detected_at = datetime.now(timezone.utc)
    for _index, row in weekly_df.iterrows():
        alerts.extend(_evaluate_row(row, detected_at))
    logger.info("alerts_evaluated", extra={"alert_count": len(alerts), "row_count": len(weekly_df)})
    return alerts


def _evaluate_row(row: pd.Series, detected_at: datetime) -> list[Alert]:
    """Apply every rule to a single row; return triggered alerts."""
    triggered: list[Alert] = []
    units_sold = float(row.get("units_sold", 0.0))
    rolling_mean = float(row.get("rolling_mean_4", 0.0))
    rolling_std = float(row.get("rolling_std_4", 0.0))
    stock = float(row.get("stock_available", 0.0))
    price = float(row.get("price_unit", 0.0))
    price_avg = float(row.get("price_avg", price))
    base = _alert_base(row, detected_at)
    if rolling_mean > 0 and units_sold > rolling_mean + _SPIKE_SIGMA * rolling_std:
        triggered.append(Alert(**base, alert_type="demand_spike", severity="HIGH",
                               message=f"Units sold {units_sold:.1f} exceeded mean+2.5σ ({rolling_mean:.1f}+{_SPIKE_SIGMA}*{rolling_std:.1f}).",
                               recommended_action="Review supply lines; may indicate viral demand or data issue."))
    if (
        rolling_mean > 0
        and units_sold >= 0
        and units_sold < rolling_mean - _DROP_SIGMA * rolling_std
    ):
        triggered.append(Alert(**base, alert_type="demand_drop", severity="MEDIUM",
                               message=f"Units sold {units_sold:.1f} below mean-2σ ({rolling_mean:.1f}-{_DROP_SIGMA}*{rolling_std:.1f}).",
                               recommended_action="Investigate channel listing or stockouts."))
    if units_sold < 0:
        triggered.append(Alert(**base, alert_type="return_anomaly", severity="LOW",
                               message=f"Negative units_sold ({units_sold:.1f}) — likely a return batch.",
                               recommended_action="Verify return reason in upstream POS."))
    if rolling_mean > 0 and stock / rolling_mean < _STOCK_RISK_WEEKS:
        triggered.append(Alert(**base, alert_type="stock_risk", severity="HIGH",
                               message=f"Stock cover {stock / rolling_mean:.1f} weeks below {_STOCK_RISK_WEEKS}-week threshold.",
                               recommended_action="Trigger replenishment order."))
    if price_avg > 0 and abs(price - price_avg) / price_avg > _PRICE_DEVIATION_PCT:
        triggered.append(Alert(**base, alert_type="price_anomaly", severity="LOW",
                               message=f"Price {price:.2f} deviates >15% from 4-week avg {price_avg:.2f}.",
                               recommended_action="Confirm pricing change is intentional."))
    return triggered


def _alert_base(row: pd.Series, detected_at: datetime) -> dict[str, object]:
    """Common identifier fields shared by every alert variant."""
    return {
        "sku": str(row["sku"]),
        "channel": row["channel"],
        "region": row["region"],
        "detected_at": detected_at,
    }


def detect_promo_cannibalization(weekly_df: pd.DataFrame) -> list[Alert]:
    """Detect SKUs whose promoted weeks underperform their non-promoted weeks.

    Args:
        weekly_df: Weekly DataFrame including `promotion_flag` and `units_sold`.

    Returns:
        List of MEDIUM-severity Alert instances.
    """
    alerts: list[Alert] = []
    detected_at = datetime.now(timezone.utc)
    grouped = weekly_df.groupby(["sku", "channel", "region"])
    for (sku, channel, region), group in grouped:
        promo = group.loc[group["promotion_flag"] == 1, "units_sold"]
        non_promo = group.loc[group["promotion_flag"] == 0, "units_sold"]
        if promo.empty or non_promo.empty:
            continue
        if float(promo.mean()) >= float(non_promo.mean()):
            continue
        alerts.append(
            Alert(
                sku=sku,
                channel=channel,
                region=region,
                alert_type="promo_cannibalization",
                severity="MEDIUM",
                message=(
                    f"Promo mean {float(promo.mean()):.1f} below non-promo mean "
                    f"{float(non_promo.mean()):.1f}; consider lifting promo."
                ),
                recommended_action="Review promo ROI and channel mix.",
                detected_at=detected_at,
            )
        )
    return alerts


def filter_alerts(
    alerts: list[Alert],
    sku: str | None = None,
    channel: str | None = None,
    region: str | None = None,
) -> list[Alert]:
    """Filter alerts by optional sku/channel/region predicates."""
    out: list[Alert] = []
    for alert in alerts:
        if sku is not None and alert.sku != sku:
            continue
        if channel is not None and alert.channel != channel:
            continue
        if region is not None and alert.region != region:
            continue
        out.append(alert)
    return out


def check_alerts(
    weekly_df: pd.DataFrame,
    sku: str | None = None,
    channel: str | None = None,
    region: str | None = None,
) -> list[Alert]:
    """End-to-end: run all rules then filter to the requested scope."""
    rule_alerts = detect_alerts(weekly_df)
    cannibalization = detect_promo_cannibalization(weekly_df)
    return filter_alerts(rule_alerts + cannibalization, sku=sku, channel=channel, region=region)
