import os
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from image_analyzer import analyze_chart_image
from predictor import predictor_instance

app = FastAPI(
    title="CryptoVision API",
    description="ML-Powered Cryptocurrency Market Analyzer API",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "CryptoVision ML Backend",
        "model_loaded": predictor_instance.model is not None
    }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/api/metrics")
def get_model_metrics():
    metrics_file = os.path.join(BASE_DIR, "models", "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return json.load(f)
    else:
        raise HTTPException(status_code=404, detail="Model metrics file not found. Train models first.")

@app.post("/api/analyze-chart")
async def analyze_chart(
    file: Optional[UploadFile] = File(None),
    cryptocurrency: Optional[str] = Form("BTC/USDT"),
    exchange: Optional[str] = Form("Binance"),
    timeframe: Optional[str] = Form("1D"),
    current_price: Optional[float] = Form(64500.0)
):
    if file:
        contents = await file.read()
        price_val = current_price if current_price else 64500.0
        cv_result = analyze_chart_image(contents, price_val)
        
    try:
        prediction_result = predictor_instance.predict_market(
            current_price=current_price if current_price else 64500.0,
            cv_analysis=cv_result
        )
        
        return {
            "success": True,
            "inputs": {
                "cryptocurrency": cryptocurrency,
                "exchange": exchange,
                "timeframe": timeframe,
                "current_price": current_price,
                "has_chart_image": file is not None
            },
            "cv_analysis": cv_result,
            "analysis_result": prediction_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
