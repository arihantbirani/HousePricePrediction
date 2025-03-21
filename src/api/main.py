from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Optional
import logging
from src.utils.preprocessor import preprocess_input
import os
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using machine learning",
    version="1.0.0"
)

# Define the input data model
class HouseData(BaseModel):
    MSSubClass: int
    MSZoning: str
    LotFrontage: Optional[float]
    LotArea: int
    Street: str
    Alley: Optional[str]
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: Optional[str]
    MasVnrArea: Optional[float]
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: Optional[str]
    BsmtCond: Optional[str]
    BsmtExposure: Optional[str]
    BsmtFinType1: Optional[str]
    BsmtFinSF1: Optional[float]
    BsmtFinType2: Optional[str]
    BsmtFinSF2: Optional[float]
    BsmtUnfSF: Optional[float]
    TotalBsmtSF: Optional[float]
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: Optional[str]
    FirstFlrSF: int
    SecondFlrSF: int
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: Optional[float]
    BsmtHalfBath: Optional[float]
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: Optional[str]
    GarageType: Optional[str]
    GarageYrBlt: Optional[float]
    GarageFinish: Optional[str]
    GarageCars: Optional[float]
    GarageArea: Optional[float]
    GarageQual: Optional[str]
    GarageCond: Optional[str]
    PavedDrive: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    ThreeSsnPorch: int
    ScreenPorch: int
    PoolArea: int
    PoolQC: Optional[str]
    Fence: Optional[str]
    MiscFeature: Optional[str]
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str

# Load model and preprocessor at startup
@app.on_event("startup")
async def startup_event():
    try:
        # Load model and preprocessor from local path
        model_path = "models/house_price_model.joblib"
        preprocessor_path = "models/preprocessor.joblib"
        
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            logger.warning("Model or preprocessor files not found. Please ensure they exist in the models directory.")
            return
            
        # Load the model and preprocessor
        global model, preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        logger.info("Model and preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model and preprocessor: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "Welcome to the House Price Prediction API"}

@app.post("/api/predict")
async def predict_price(house_data: HouseData):
    try:
        # Preprocess the input data
        processed_data = preprocess_input(house_data.dict())
        
        # Make prediction
        prediction = model.predict(processed_data)
        
        return {
            "predicted_price": float(prediction[0]),
            "confidence": "high"  # You might want to implement confidence scoring
        }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 