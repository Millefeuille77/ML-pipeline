"""Pydantic request/response models — single source of truth for API contracts.

Every field name and Literal value matches the actual dataset columns in
`data/raw/` exactly. Any change here ripples to backend routes, ETL loaders,
and ML model outputs — keep this file authoritative and minimal.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.validators import SKU_PATTERN

Category = Literal["Milk", "Yogurt", "ReadyMeal", "Juice", "SnackBar"]
Channel = Literal["Retail", "Discount", "E-commerce"]
Region = Literal["PL-Central", "PL-North", "PL-South"]
PackType = Literal["Multipack", "Single", "Carton"]
LifecycleStage = Literal["Growth", "Mature", "Decline"]
AlertSeverity = Literal["LOW", "MEDIUM", "HIGH"]
AlertType = Literal[
    "demand_spike",
    "demand_drop",
    "return_anomaly",
    "stock_risk",
    "price_anomaly",
    "promo_cannibalization",
]
Metric = Literal["units_sold", "revenue"]

SKU_FIELD = Field(
    ...,
    description="Stock-keeping unit, format XX-NNN (e.g. MI-006).",
    pattern=SKU_PATTERN.pattern,
    min_length=6,
    max_length=6,
)


class ProductInfo(BaseModel):
    """Static product master record."""

    model_config = ConfigDict(from_attributes=True)

    sku: str = SKU_FIELD
    brand: str
    segment: str
    category: Category
    pack_type: PackType


class SalesRecord(BaseModel):
    """One row of the daily fact table (`FMCG_2022_2024.csv`)."""

    model_config = ConfigDict(from_attributes=True)

    date: date
    sku: str = SKU_FIELD
    channel: Channel
    region: Region
    price_unit: float = Field(..., gt=0.0)
    promotion_flag: int = Field(..., ge=0, le=1)
    delivery_days: int = Field(..., ge=1, le=5)
    stock_available: float
    delivered_qty: float
    units_sold: float


class WeeklyAggregate(BaseModel):
    """One row of the weekly modeling table (`weekly_df_final_for_modeling.csv`)."""

    model_config = ConfigDict(from_attributes=True)

    sku: str = SKU_FIELD
    week: date
    channel: Channel
    region: Region
    units_sold: float
    stock_available: float
    promotion_flag: int = Field(..., ge=0, le=1)
    price_unit: float = Field(..., gt=0.0)
    delivery_days: float
    is_holiday_peak: int = Field(..., ge=0, le=1)
    week_number: int = Field(..., ge=1, le=53)
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=2020, le=2030)
    is_holiday_week: int = Field(..., ge=0, le=1)
    is_summer: int = Field(..., ge=0, le=1)
    is_winter: int = Field(..., ge=0, le=1)
    sku_age: int = Field(..., ge=0)
    lifecycle_stage: LifecycleStage
    lag_1: float | None = None
    lag_2: float | None = None
    rolling_mean_4: float | None = None
    rolling_std_4: float | None = None
    momentum: float | None = None
    target_next_week: float | None = None


class EnrichmentRecord(BaseModel):
    """Enrichment columns generalized from `df_weekly_MI-006_enriched.csv`."""

    model_config = ConfigDict(from_attributes=True)

    sku: str = SKU_FIELD
    week: date
    channel: Channel
    region: Region
    price_avg: float = Field(..., ge=0.0)
    promo_rate: float = Field(..., ge=0.0, le=1.0)
    stock_avg: float
    deliveries: int = Field(..., ge=0)
    avg_temp: float
    inflation_index: float = Field(..., ge=0.0)
    school_in_session: int = Field(..., ge=0, le=1)
    category_trend: float
    event_score: float


class ForecastRequest(BaseModel):
    """Body for forecast endpoints."""

    sku: str = SKU_FIELD
    channel: Channel
    region: Region
    horizon_weeks: int = Field(default=4, ge=1, le=12)


class ForecastResult(BaseModel):
    """Forecast output for a single (sku, channel, region)."""

    sku: str = SKU_FIELD
    channel: Channel
    region: Region
    weeks: list[date]
    predicted_units: list[float]
    confidence_lower: list[float]
    confidence_upper: list[float]
    model_version: str

    @field_validator("predicted_units", "confidence_lower", "confidence_upper")
    @classmethod
    def _matches_weeks_length(cls, value: list[float], info) -> list[float]:
        """Ensure value-array length matches `weeks` length when both present."""
        weeks = info.data.get("weeks")
        if weeks is not None and len(weeks) != len(value):
            raise ValueError(
                f"length mismatch: {info.field_name} has {len(value)} entries "
                f"but weeks has {len(weeks)}."
            )
        return value


class ClusterResult(BaseModel):
    """Membership info for a single SKU within the K-Means clustering."""

    sku: str = SKU_FIELD
    cluster_label: str
    cluster_description: str
    similar_skus: list[str]


class ClusterSummary(BaseModel):
    """Aggregate description of one cluster."""

    cluster_label: str
    cluster_description: str
    sku_count: int = Field(..., ge=0)
    members: list[str]
    avg_weekly_demand: float


class Alert(BaseModel):
    """Anomaly / business-rule alert for a (sku, channel, region) combination."""

    sku: str = SKU_FIELD
    channel: Channel
    region: Region
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    recommended_action: str
    detected_at: datetime


class BatchPredictionResult(BaseModel):
    """Result envelope for a batch parquet inference run."""

    batch_id: str
    predictions: list[ForecastResult]
    created_at: datetime


class HealthResponse(BaseModel):
    """Liveness/readiness response surfaced from `/api/v1/health`."""

    db_status: Literal["ok", "degraded", "down"]
    model_status: Literal["ok", "missing", "error"]
    uptime_seconds: float = Field(..., ge=0.0)
    version: str
    row_counts: dict[str, int]


class SalesSummary(BaseModel):
    """Aggregated sales summary for `/analytics/sales-summary`."""

    start_date: date
    end_date: date
    total_units_sold: float
    total_revenue: float
    record_count: int = Field(..., ge=0)
    breakdown: list[dict]


class TopProduct(BaseModel):
    """One row of `/analytics/top-products`."""

    sku: str = SKU_FIELD
    category: Category
    metric_value: float
    metric_name: Metric


class InventoryRiskItem(BaseModel):
    """One row of `/analytics/inventory-risk`."""

    sku: str = SKU_FIELD
    channel: Channel
    region: Region
    stock_available: float
    avg_weekly_demand: float
    days_of_supply: float
    threshold_days: int


class CategoryTrendPoint(BaseModel):
    """One point of `/analytics/category-trends`."""

    category: Category
    month: date
    total_units_sold: float
    delta_pct: float | None = None


class LifecycleDistribution(BaseModel):
    """Counts of SKUs per lifecycle stage."""

    growth: int = Field(..., ge=0)
    mature: int = Field(..., ge=0)
    decline: int = Field(..., ge=0)


class ChannelComparisonRow(BaseModel):
    """One row of `/analytics/channel-comparison`."""

    channel: Channel
    total_units_sold: float
    avg_price_unit: float
    promo_lift_pct: float


class ErrorResponse(BaseModel):
    """Standard error envelope returned to clients."""

    error: str
    detail: str
    request_id: str
