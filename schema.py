from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum

class SexEnum(str, Enum):
    male = "male"
    female = "female"

class EmbarkedEnum(str, Enum):
    S = "S"
    C = "C"
    Q = "Q"

class TitanicFeatures(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: SexEnum
    Age: float = Field(..., ge=0, le=80)
    Fare: float = Field(..., ge=0, le=500)
    SibSp: int = Field(..., ge=0, le=8)
    Parch: int = Field(..., ge=0, le=6)
    Embarked: EmbarkedEnum

class PredictionResponse(BaseModel):
    Survived: int
    Survival_Probability: float
    Death_Probability: float
