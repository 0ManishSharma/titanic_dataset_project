from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.titanic_project.pipelines.prediction_pipeline import TitanicPredictor
from schema import TitanicFeatures, PredictionResponse

app = FastAPI(title="Titanic Survival API")
predictor = None  # Lazy load

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_model():
    """Load model on startup (fixes multiprocessing)"""
    global predictor
    try:
        predictor = TitanicPredictor()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")

@app.get("/")
async def root():
    return {"message": "Titanic Survival API ✅"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/predict")
async def predict(features: dict):
    """Predict survival"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        result = predictor.predict(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)