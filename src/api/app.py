from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import joblib
import json
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using machine learning models",
    version="1.0.0"
)

# Add CORS middleware with more permissive settings for Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Load the model and preprocessor
try:
    # Load best model info
    model_info_path = ROOT_DIR / "models/best_model_info.json"
    with open(model_info_path, 'r') as f:
        best_model_info = json.load(f)
    best_model_name = best_model_info['best_model_name']
    
    model_path = ROOT_DIR / f"models/{best_model_name}.joblib"
    preprocessor_path = ROOT_DIR / "data/features/preprocessor.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    logger.info(f"Successfully loaded model ({best_model_name}) and preprocessor")
except Exception as e:
    logger.error(f"Error loading model or preprocessor: {e}")
    raise

class HouseFeatures(BaseModel):
    # Optional fields that were previously causing issues
    Id: Optional[int] = Field(default=1, description="ID")
    FirstFlrSF: Optional[float] = Field(default=1000.0, alias='1stFlrSF', description="First Floor square feet")
    SecondFlrSF: Optional[float] = Field(default=500.0, alias='2ndFlrSF', description="Second floor square feet")
    ThreeSsnPorch: Optional[float] = Field(default=0.0, alias='3SsnPorch', description="Three season porch area in square feet")
    
    # Numerical features with default values
    GrLivArea: float = Field(default=1500.0, description="Above ground living area")
    TotalBsmtSF: float = Field(default=1000.0, description="Total basement square footage")
    OverallQual: int = Field(default=5, ge=1, le=10, description="Overall material and finish quality")
    OverallCond: int = Field(default=5, ge=1, le=10, description="Overall condition rating")
    YearBuilt: int = Field(default=1970, ge=1800, le=2024, description="Original construction date")
    FullBath: int = Field(default=2, ge=0, description="Full bathrooms above grade")
    HalfBath: int = Field(default=1, ge=0, description="Half baths above grade")
    BedroomAbvGr: int = Field(default=3, ge=0, description="Bedrooms above grade")
    TotRmsAbvGrd: int = Field(default=6, ge=0, description="Total rooms above grade")
    GarageCars: int = Field(default=2, ge=0, description="Size of garage in car capacity")
    
    # Additional numerical features with default values
    LotFrontage: float = Field(default=60.0, description="Linear feet of street connected to property")
    LotArea: int = Field(default=8000, description="Lot size in square feet")
    YearRemodAdd: int = Field(default=1970, description="Remodel date (same as construction date if no remodeling)")
    MasVnrArea: float = Field(default=0.0, description="Masonry veneer area in square feet")
    BsmtFinSF1: float = Field(default=0.0, description="Type 1 finished square feet")
    BsmtFinSF2: float = Field(default=0.0, description="Type 2 finished square feet")
    BsmtUnfSF: float = Field(default=0.0, description="Unfinished square feet of basement area")
    LowQualFinSF: float = Field(default=0.0, description="Low quality finished square feet (all floors)")
    BsmtFullBath: int = Field(default=0, description="Basement full bathrooms")
    BsmtHalfBath: int = Field(default=0, description="Basement half bathrooms")
    KitchenAbvGr: int = Field(default=1, description="Kitchens above grade")
    Fireplaces: int = Field(default=0, description="Number of fireplaces")
    GarageYrBlt: float = Field(default=1970.0, description="Year garage was built")
    GarageArea: float = Field(default=400.0, description="Size of garage in square feet")
    WoodDeckSF: float = Field(default=0.0, description="Wood deck area in square feet")
    OpenPorchSF: float = Field(default=0.0, description="Open porch area in square feet")
    EnclosedPorch: float = Field(default=0.0, description="Enclosed porch area in square feet")
    ScreenPorch: float = Field(default=0.0, description="Screen porch area in square feet")
    PoolArea: float = Field(default=0.0, description="Pool area in square feet")
    MiscVal: int = Field(default=0, description="Value of miscellaneous feature")
    MoSold: int = Field(default=6, description="Month Sold (MM)")
    YrSold: int = Field(default=2024, description="Year Sold (YYYY)")
    
    # Categorical features with default values
    MSSubClass: str = Field(default="60", description="Building class")
    MSZoning: str = Field(default="RL", description="General zoning classification")
    Street: str = Field(default="Pave", description="Type of road access")
    Alley: str = Field(default="NA", description="Type of alley access")
    LotShape: str = Field(default="Reg", description="General shape of property")
    LandContour: str = Field(default="Lvl", description="Flatness of the property")
    Utilities: str = Field(default="AllPub", description="Type of utilities available")
    LotConfig: str = Field(default="Inside", description="Lot configuration")
    LandSlope: str = Field(default="Gtl", description="Slope of property")
    Neighborhood: str = Field(default="NAmes", description="Physical locations within Ames city limits")
    Condition1: str = Field(default="Norm", description="Proximity to various conditions")
    Condition2: str = Field(default="Norm", description="Proximity to various conditions (if more than one is present)")
    BldgType: str = Field(default="1Fam", description="Type of dwelling")
    HouseStyle: str = Field(default="2Story", description="Style of dwelling")
    RoofStyle: str = Field(default="Gable", description="Type of roof")
    RoofMatl: str = Field(default="CompShg", description="Roof material")
    Exterior1st: str = Field(default="VinylSd", description="Exterior covering on house")
    Exterior2nd: str = Field(default="VinylSd", description="Exterior covering on house (if more than one material)")
    MasVnrType: str = Field(default="None", description="Masonry veneer type")
    ExterQual: str = Field(default="TA", description="Exterior material quality")
    ExterCond: str = Field(default="TA", description="Present condition of the material on the exterior")
    Foundation: str = Field(default="PConc", description="Type of foundation")
    BsmtQual: str = Field(default="TA", description="Height of the basement")
    BsmtCond: str = Field(default="TA", description="General condition of the basement")
    BsmtExposure: str = Field(default="No", description="Walkout or garden level walls")
    BsmtFinType1: str = Field(default="Unf", description="Quality of basement finished area")
    BsmtFinType2: str = Field(default="Unf", description="Quality of second finished area (if present)")
    Heating: str = Field(default="GasA", description="Type of heating")
    HeatingQC: str = Field(default="TA", description="Heating quality and condition")
    CentralAir: str = Field(default="Y", description="Central air conditioning")
    Electrical: str = Field(default="SBrkr", description="Electrical system")
    KitchenQual: str = Field(default="TA", description="Kitchen quality")
    Functional: str = Field(default="Typ", description="Home functionality")
    FireplaceQu: str = Field(default="NA", description="Fireplace quality")
    GarageType: str = Field(default="Attchd", description="Garage location")
    GarageFinish: str = Field(default="Unf", description="Interior finish of the garage")
    GarageQual: str = Field(default="TA", description="Garage quality")
    GarageCond: str = Field(default="TA", description="Garage condition")
    PavedDrive: str = Field(default="Y", description="Paved driveway")
    PoolQC: str = Field(default="NA", description="Pool quality")
    Fence: str = Field(default="NA", description="Fence quality")
    MiscFeature: str = Field(default="NA", description="Miscellaneous feature not covered in other categories")
    SaleType: str = Field(default="WD", description="Type of sale")
    SaleCondition: str = Field(default="Normal", description="Condition of sale")

    class Config:
        populate_by_name = True  # Updated from allow_population_by_field_name

@app.get("/")
async def root():
    """Serve the HTML interface"""
    return FileResponse("templates/index.html")

@app.get("/api")
async def api_info():
    """
    API information endpoint
    """
    return {
        "message": "Welcome to the House Price Prediction API",
        "version": "1.0.0",
        "status": "ready" if model is not None else "model_not_loaded",
        "model_type": type(model).__name__,
        "feature_count": len(preprocessor.get_feature_names_out())
    }

@app.get("/api/features")
async def get_features():
    """Get list of required features"""
    return {
        "features": [
            field.name for field in HouseFeatures.__fields__.values()
        ]
    }

@app.post("/api/predict")
async def predict(features: HouseFeatures):
    """Make a prediction for house price"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    try:
        # Convert features to DataFrame and ensure required fields are present
        data_dict = features.dict()
        
        # Ensure these fields are present with their default values
        required_fields = {
            '1stFlrSF': 1000.0,
            '2ndFlrSF': 500.0,
            '3SsnPorch': 0.0
        }
        
        # Add any missing required fields with their default values
        for field, default_value in required_fields.items():
            if field not in data_dict:
                data_dict[field] = default_value
        
        input_data = pd.DataFrame([data_dict])
        
        # Apply preprocessing
        X = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return {
            "predicted_price": float(prediction),
            "predicted_price_formatted": f"${prediction:,.2f}"
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model_info")
async def get_model_info():
    """Get information about the model"""
    return {
        "model_type": type(model).__name__,
        "features": preprocessor.get_feature_names_out().tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 