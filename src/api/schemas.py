from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512)


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    model_version: str
