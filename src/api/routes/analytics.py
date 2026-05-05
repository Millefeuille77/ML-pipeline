"""Analytics endpoints driven by SQL aggregates over the loaded fact tables."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Final

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import text

from src.api.schemas import (
    Category,
    ChannelComparisonRow,
    InventoryRiskItem,
    LifecycleDistribution,
    Metric,
    SalesSummary,
    TopProduct,
)
from src.database.connection import get_engine
from src.utils.helpers import safe_divide
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])

_TOP_PRODUCT_MAX: Final[int] = 100
_INVENTORY_THRESHOLD_MAX: Final[int] = 30
_TREND_MAX_MONTHS: Final[int] = 24
_DEFAULT_TOP_N: Final[int] = 10


@router.get("/sales-summary", response_model=SalesSummary)
def sales_summary(
    start_date: Annotated[date, Query(description="Inclusive start date")],
    end_date: Annotated[date, Query(description="Inclusive end date")],
    category: Annotated[Category | None, Query(description="Optional category filter")] = None,
) -> SalesSummary:
    """Aggregate sales between two dates with optional category filter."""
    if start_date > end_date:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="start_date after end_date")
    engine = get_engine()
    sql = text(
        """
        SELECT
            COALESCE(SUM(d.units_sold), 0) AS total_units,
            COALESCE(SUM(d.units_sold * d.price_unit), 0) AS total_revenue,
            COUNT(*) AS record_count
        FROM daily_sales d
        JOIN products p ON p.sku = d.sku
        WHERE d.sale_date BETWEEN :start AND :end
          AND (:category IS NULL OR p.category = :category)
        """
    )
    breakdown_sql = text(
        """
        SELECT p.category AS category, COALESCE(SUM(d.units_sold), 0) AS units_sold
        FROM daily_sales d
        JOIN products p ON p.sku = d.sku
        WHERE d.sale_date BETWEEN :start AND :end
          AND (:category IS NULL OR p.category = :category)
        GROUP BY p.category
        ORDER BY p.category
        """
    )
    with engine.connect() as connection:
        row = connection.execute(
            sql,
            {"start": start_date, "end": end_date, "category": category},
        ).fetchone()
        breakdown_rows = connection.execute(
            breakdown_sql,
            {"start": start_date, "end": end_date, "category": category},
        ).mappings().all()
    return SalesSummary(
        start_date=start_date,
        end_date=end_date,
        total_units_sold=float(row[0] or 0.0) if row else 0.0,
        total_revenue=float(row[1] or 0.0) if row else 0.0,
        record_count=int(row[2] or 0) if row else 0,
        breakdown=[dict(record) for record in breakdown_rows],
    )


_TOP_SQL_UNITS: Final[str] = """
    SELECT p.sku AS sku,
           p.category AS category,
           SUM(d.units_sold) AS metric_value
    FROM daily_sales d
    JOIN products p ON p.sku = d.sku
    GROUP BY p.sku, p.category
    ORDER BY metric_value DESC
    LIMIT :limit
"""
_TOP_SQL_REVENUE: Final[str] = """
    SELECT p.sku AS sku,
           p.category AS category,
           SUM(d.units_sold * d.price_unit) AS metric_value
    FROM daily_sales d
    JOIN products p ON p.sku = d.sku
    GROUP BY p.sku, p.category
    ORDER BY metric_value DESC
    LIMIT :limit
"""


@router.get("/top-products", response_model=list[TopProduct])
def top_products(
    n: Annotated[int, Query(ge=1, le=_TOP_PRODUCT_MAX)] = _DEFAULT_TOP_N,
    metric: Annotated[Metric, Query(description="Ranking metric")] = "units_sold",
) -> list[TopProduct]:
    """Top N products ranked by `units_sold` or revenue."""
    engine = get_engine()
    sql = text(_TOP_SQL_REVENUE if metric == "revenue" else _TOP_SQL_UNITS)
    with engine.connect() as connection:
        rows = connection.execute(sql, {"limit": n}).mappings().all()
    return [
        TopProduct(
            sku=row["sku"],
            category=row["category"],
            metric_value=float(row["metric_value"] or 0.0),
            metric_name=metric,
        )
        for row in rows
    ]


@router.get("/inventory-risk", response_model=list[InventoryRiskItem])
def inventory_risk(
    threshold_days: Annotated[int, Query(ge=1, le=_INVENTORY_THRESHOLD_MAX)] = 7,
) -> list[InventoryRiskItem]:
    """SKUs whose stock cover is shorter than `threshold_days`."""
    engine = get_engine()
    sql = text(
        """
        SELECT sku, channel, region,
               AVG(stock_available) AS stock_available,
               AVG(rolling_mean_4) AS avg_weekly_demand
        FROM weekly_features
        WHERE rolling_mean_4 IS NOT NULL AND rolling_mean_4 > 0
        GROUP BY sku, channel, region
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(sql).mappings().all()
    risks: list[InventoryRiskItem] = []
    for row in rows:
        weekly_demand = float(row["avg_weekly_demand"] or 0.0)
        stock = float(row["stock_available"] or 0.0)
        days_of_supply = 7.0 * safe_divide(stock, weekly_demand, 0.0)
        if days_of_supply < threshold_days:
            risks.append(
                InventoryRiskItem(
                    sku=row["sku"],
                    channel=row["channel"],
                    region=row["region"],
                    stock_available=stock,
                    avg_weekly_demand=weekly_demand,
                    days_of_supply=days_of_supply,
                    threshold_days=threshold_days,
                )
            )
    risks.sort(key=lambda item: item.days_of_supply)
    return risks


@router.get("/category-trends")
def category_trends(
    months: Annotated[int, Query(ge=1, le=_TREND_MAX_MONTHS)] = 6,
) -> list[dict]:
    """Total monthly units_sold per category over the trailing `months` months."""
    engine = get_engine()
    sql = text(
        """
        SELECT p.category AS category,
               DATE_TRUNC('month', d.sale_date)::date AS month,
               SUM(d.units_sold) AS total_units_sold
        FROM daily_sales d
        JOIN products p ON p.sku = d.sku
        GROUP BY p.category, DATE_TRUNC('month', d.sale_date)
        ORDER BY DATE_TRUNC('month', d.sale_date) DESC
        LIMIT :limit
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(sql, {"limit": months * 5}).mappings().all()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    frame = frame.sort_values(["category", "month"]).reset_index(drop=True)
    # WHY: NUMERIC columns from Postgres arrive as decimal.Decimal which
    # cannot multiply with Python float; cast first so pct_change yields float.
    frame["total_units_sold"] = frame["total_units_sold"].astype(float)
    frame["delta_pct"] = frame.groupby("category")["total_units_sold"].pct_change() * 100.0
    return frame.to_dict(orient="records")


@router.get("/lifecycle-distribution", response_model=LifecycleDistribution)
def lifecycle_dist() -> LifecycleDistribution:
    """Counts of SKUs per Growth/Mature/Decline lifecycle stage."""
    engine = get_engine()
    sql = text(
        """
        SELECT lifecycle_stage AS lifecycle_stage, COUNT(DISTINCT sku) AS sku_count
        FROM weekly_features
        GROUP BY lifecycle_stage
        """
    )
    counts: dict[str, int] = {"Growth": 0, "Mature": 0, "Decline": 0}
    with engine.connect() as connection:
        for row in connection.execute(sql).mappings().all():
            counts[row["lifecycle_stage"]] = int(row["sku_count"] or 0)
    return LifecycleDistribution(
        growth=counts["Growth"],
        mature=counts["Mature"],
        decline=counts["Decline"],
    )


@router.get("/channel-comparison", response_model=list[ChannelComparisonRow])
def channel_comparison() -> list[ChannelComparisonRow]:
    """Per-channel KPIs including promotion lift."""
    engine = get_engine()
    sql = text(
        """
        SELECT channel,
               promotion_flag,
               SUM(units_sold) AS total_units_sold,
               AVG(price_unit) AS avg_price_unit
        FROM daily_sales
        GROUP BY channel, promotion_flag
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(sql).mappings().all()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    grouped = frame.groupby("channel")
    out: list[ChannelComparisonRow] = []
    for channel, group in grouped:
        total_units = float(group["total_units_sold"].sum() or 0.0)
        avg_price = float(group["avg_price_unit"].mean() or 0.0)
        promo_units = float(group.loc[group["promotion_flag"] == 1, "total_units_sold"].sum() or 0.0)
        non_promo_units = float(group.loc[group["promotion_flag"] == 0, "total_units_sold"].sum() or 0.0)
        promo_lift = 100.0 * safe_divide(promo_units - non_promo_units, non_promo_units, 0.0)
        out.append(
            ChannelComparisonRow(
                channel=channel,  # type: ignore[arg-type]
                total_units_sold=total_units,
                avg_price_unit=avg_price,
                promo_lift_pct=promo_lift,
            )
        )
    return out
