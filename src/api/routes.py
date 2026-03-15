from fastapi import APIRouter, Depends
from .dependencies import get_model
from .schemas import PredictRequest, PredictResponse

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(req:PredictRequest, model=Depends(get_model)):
    sentiment, confidence = model.predict(req.text)
    return PredictResponse(text=req.text, sentiment=sentiment, 
                           confidence=round(confidence, 4), 
                           model_version="v1.0")
    
@router.get("/health")
def health():
    return {"status": "healthy"}
