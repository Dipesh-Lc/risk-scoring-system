"""
Pydantic schemas for API request/response validation.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """
    Input schema: raw UCI credit card features.
    All field descriptions match the UCI data dictionary.
    """

    LIMIT_BAL: float = Field(..., description="Credit limit in NT dollars", ge=0)
    SEX: int = Field(..., description="1=male, 2=female", ge=1, le=2)
    EDUCATION: int = Field(
        ..., description="1=grad school, 2=university, 3=high school, 4=other", ge=0, le=6
    )
    MARRIAGE: int = Field(..., description="1=married, 2=single, 3=other", ge=0, le=3)
    AGE: int = Field(..., description="Age in years", ge=18, le=100)

    PAY_0: int = Field(
        ..., description="Repayment status in September (-1=on time, 1–9=months late)", ge=-2, le=9
    )
    PAY_2: int = Field(..., description="Repayment status in August", ge=-2, le=9)
    PAY_3: int = Field(..., description="Repayment status in July", ge=-2, le=9)
    PAY_4: int = Field(..., description="Repayment status in June", ge=-2, le=9)
    PAY_5: int = Field(..., description="Repayment status in May", ge=-2, le=9)
    PAY_6: int = Field(..., description="Repayment status in April", ge=-2, le=9)

    BILL_AMT1: float = Field(..., description="Bill statement amount September (NT$)")
    BILL_AMT2: float = Field(..., description="Bill statement amount August (NT$)")
    BILL_AMT3: float = Field(..., description="Bill statement amount July (NT$)")
    BILL_AMT4: float = Field(..., description="Bill statement amount June (NT$)")
    BILL_AMT5: float = Field(..., description="Bill statement amount May (NT$)")
    BILL_AMT6: float = Field(..., description="Bill statement amount April (NT$)")

    PAY_AMT1: float = Field(..., description="Payment amount September (NT$)", ge=0)
    PAY_AMT2: float = Field(..., description="Payment amount August (NT$)", ge=0)
    PAY_AMT3: float = Field(..., description="Payment amount July (NT$)", ge=0)
    PAY_AMT4: float = Field(..., description="Payment amount June (NT$)", ge=0)
    PAY_AMT5: float = Field(..., description="Payment amount May (NT$)", ge=0)
    PAY_AMT6: float = Field(..., description="Payment amount April (NT$)", ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "LIMIT_BAL": 50000,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 35,
                "PAY_0": 0,
                "PAY_2": 0,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
                "BILL_AMT1": 15000,
                "BILL_AMT2": 14000,
                "BILL_AMT3": 13000,
                "BILL_AMT4": 12000,
                "BILL_AMT5": 11000,
                "BILL_AMT6": 10000,
                "PAY_AMT1": 2000,
                "PAY_AMT2": 1800,
                "PAY_AMT3": 1600,
                "PAY_AMT4": 1500,
                "PAY_AMT5": 1400,
                "PAY_AMT6": 1300,
            }
        }
    }


class DriverExplanation(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    direction: str  # "increases_risk" | "decreases_risk"


class ScoreResponse(BaseModel):
    default_probability: float = Field(..., ge=0, le=1)
    risk_score: int = Field(..., ge=0, le=100)
    risk_band: str = Field(..., description="Low | Medium | High")
    top_drivers: List[DriverExplanation] = Field(default_factory=list)


class BatchScoreRequest(BaseModel):
    records: List[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: Optional[str] = None
