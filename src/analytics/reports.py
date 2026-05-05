"""Natural-language insights stitched together from EDA outputs."""
from __future__ import annotations

from typing import Final

import pandas as pd

from src.analytics.eda import (
    lifecycle_distribution,
    promo_impact_analysis,
    sales_by_category,
)
from src.api.schemas import Alert, ForecastResult
from src.utils.helpers import safe_divide
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROMO_LIFT_HIGH_PCT: Final[float] = 15.0
_TOP_INSIGHT_LIMIT: Final[int] = 5


def generate_insights(
    weekly_df: pd.DataFrame,
    forecasts: list[ForecastResult],
    alerts: list[Alert],
) -> list[str]:
    """Produce a list of human-readable insights for stakeholder dashboards.

    Args:
        weekly_df: Weekly modeling DataFrame.
        forecasts: Recent forecast outputs (may be empty).
        alerts: Detected alerts (may be empty).

    Returns:
        List of one-line insight strings, capped to a small number.
    """
    insights: list[str] = []
    insights.extend(_category_growth_insights(weekly_df))
    insights.extend(_promo_effectiveness_insights(weekly_df))
    insights.extend(_lifecycle_insights(weekly_df))
    insights.extend(_alert_insights(alerts))
    insights.extend(_forecast_insights(forecasts))
    return insights[:_TOP_INSIGHT_LIMIT]


def _category_growth_insights(weekly_df: pd.DataFrame) -> list[str]:
    """Compare each category's last-month vs prior-month total."""
    if weekly_df.empty:
        return []
    monthly = sales_by_category(weekly_df, period="monthly")
    if monthly.empty:
        return []
    pivoted = monthly.pivot(index="category", columns="period", values="units_sold").fillna(0.0)
    if pivoted.shape[1] < 2:
        return []
    last_two = pivoted.iloc[:, -2:]
    insights: list[str] = []
    for category in pivoted.index:
        previous = float(last_two.iloc[:, 0].loc[category])
        current = float(last_two.iloc[:, 1].loc[category])
        delta_pct = 100.0 * safe_divide(current - previous, previous, 0.0)
        if abs(delta_pct) >= 5.0:
            direction = "up" if delta_pct > 0 else "down"
            insights.append(
                f"{category} category demand {direction} {abs(delta_pct):.1f}% MoM "
                f"({previous:.0f} -> {current:.0f} units)."
            )
    return insights


def _promo_effectiveness_insights(weekly_df: pd.DataFrame) -> list[str]:
    """Highlight categories where promotions generate strong lift."""
    if weekly_df.empty or "category" not in weekly_df.columns:
        return []
    impact = promo_impact_analysis(weekly_df)
    if impact.empty:
        return []
    sku_to_category = weekly_df.drop_duplicates(subset=["sku"]).set_index("sku")["category"]
    impact = impact.merge(sku_to_category, on="sku", how="left")
    grouped = impact.groupby("category")["promo_lift_pct"].mean().sort_values(ascending=False)
    insights: list[str] = []
    for category, mean_lift in grouped.items():
        if mean_lift >= _PROMO_LIFT_HIGH_PCT:
            insights.append(
                f"Promotions in {category} produce a strong {mean_lift:.1f}% average lift."
            )
    return insights


def _lifecycle_insights(weekly_df: pd.DataFrame) -> list[str]:
    """Surface SKUs in Decline as candidates for markdown."""
    if weekly_df.empty:
        return []
    distribution = lifecycle_distribution(weekly_df)
    insights: list[str] = []
    if distribution.get("Decline", 0) > 0:
        insights.append(
            f"{distribution['Decline']} SKUs in Decline lifecycle — consider markdown or rationalization."
        )
    if distribution.get("Growth", 0) > 0:
        insights.append(
            f"{distribution['Growth']} SKUs in Growth lifecycle — prioritize stocking and channel coverage."
        )
    return insights


def _alert_insights(alerts: list[Alert]) -> list[str]:
    """Summarize the most severe alerts."""
    if not alerts:
        return []
    high_alerts = [alert for alert in alerts if alert.severity == "HIGH"]
    if not high_alerts:
        return []
    top = high_alerts[:3]
    summary = "; ".join(f"{alert.sku}/{alert.alert_type}" for alert in top)
    return [f"{len(high_alerts)} HIGH-severity alerts active (e.g. {summary})."]


def _forecast_insights(forecasts: list[ForecastResult]) -> list[str]:
    """Highlight the top forecasted-demand SKU."""
    if not forecasts:
        return []
    ranked = sorted(forecasts, key=lambda forecast: sum(forecast.predicted_units), reverse=True)
    top = ranked[0]
    total = sum(top.predicted_units)
    return [
        f"Highest forecasted demand: {top.sku} ({top.channel} / {top.region}) at "
        f"{total:.0f} units over {len(top.weeks)} weeks."
    ]
