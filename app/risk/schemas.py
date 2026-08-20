"""Pydantic schemas for PR Risk Analytics API and services."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TopFactorDTO(BaseModel):
    feature: str
    value: float
    impact: float
    direction: str = Field(description="'raises_risk' or 'lowers_risk'")


class RiskPredictRequest(BaseModel):
    repository_id: str
    pr_number: int
    features: dict[str, float]
    snapshot_at: datetime | None = None
    stage: str = Field(default="live", pattern="^(initial|live|historical)$")
    persist: bool = True


class RiskPredictionResponse(BaseModel):
    repository_id: str
    pr_number: int
    risk_probability: float = Field(ge=0.0, le=1.0)
    risk_level: str = Field(pattern="^(LOW|MEDIUM|HIGH)$")
    model_version: str
    top_factors: list[TopFactorDTO]
    predicted_at: datetime
    stage: str = "live"


class StaleCheckRequest(BaseModel):
    repository_id: str
    pr_number: int
    is_open: bool
    hours_since_last_activity: float = Field(ge=0.0)
    threshold_hours: float | None = Field(default=None, ge=1.0)


class StaleCheckResponse(BaseModel):
    repository_id: str
    pr_number: int
    is_stale: bool
    threshold_hours: float
    hours_since_last_activity: float
    reason: str


class OutcomeRecordRequest(BaseModel):
    repository_id: str
    pr_number: int
    merged_at: datetime | None = None
    observed_until: datetime
    is_risky: bool
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class OutcomeRecordResponse(BaseModel):
    status: str
    repository_id: str
    pr_number: int
    is_risky: bool


class DXRequest(BaseModel):
    median_first_review_hours: float = Field(ge=0.0)
    median_pr_cycle_hours: float = Field(ge=0.0)
    stale_pr_rate: float = Field(ge=0.0, le=1.0)
    change_failure_rate: float = Field(ge=0.0, le=1.0)
    ci_success_rate: float = Field(ge=0.0, le=1.0)
    weights: dict[str, float] | None = None


class DXResponse(BaseModel):
    score: float = Field(ge=0.0, le=100.0)
    components: dict[str, float]
    weights: dict[str, float]


class ModelMetadataResponse(BaseModel):
    model_name: str
    model_version: str
    trained_at: datetime
    feature_schema_version: str
    feature_names: list[str]
    thresholds: dict[str, float]
    metrics: dict[str, Any]
    is_demo: bool = False
