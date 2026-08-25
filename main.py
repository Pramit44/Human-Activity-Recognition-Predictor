from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
import numpy as np
import pandas as pd 
import joblib 
from tensorflow.keras.models import load_model
from warnings import filterwarnings
filterwarnings('ignore')
app = FastAPI(title="Welcome to HAR (Human Activity Recognition) Predictor", description="This API is designed to predict human activities based on sensor data. It uses a pre-trained machine learning model to classify activities such as walking, sitting, standing, etc., based on input features extracted from sensor readings.", version="1.0.0")

model = load_model('har_model.keras') 
encoder = joblib.load('encoder.pkl')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.txt')):  
        raise HTTPException(status_code=400, detail="File must be in .csv or .txt format")
    
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Empty CSV file")
        
    try:
        features = df.copy()
        predictions = model.predict(features)
        predicted_indices = np.argmax(predictions, axis=1)
        predicted_labels = encoder.inverse_transform(predicted_indices)
        
        df['Predicted_Activity'] = predicted_labels  
        output = df.to_csv(index=False)
        
        return StreamingResponse(
            iter([output]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=HAR_predictions.csv"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))