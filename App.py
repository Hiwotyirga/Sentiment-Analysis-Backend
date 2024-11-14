from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import joblib
# import time
from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from scipy.sparse import hstack
import numpy as np  # Added numpy for reshaping favorite_count

app = FastAPI()

# Configure the PostgreSQL database URL
DATABASE_URL = "postgresql+psycopg2://postgres:admin@localhost/sentiment_analysis"

# SQLAlchemy configuration
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define the database table structure
class SentimentRecord(Base):
    __tablename__ = "sentiment_records"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String(500), index=True)  # Setting a limit for text length
    sentiment = Column(String(20))
    confidence = Column(Float)

# Create the table if it doesn't exist
Base.metadata.create_all(bind=engine)

# Load the model and vectorizer on startup
model = None
vectorizer = None

@app.on_event("startup")
async def load_model_and_vectorizer():
    global model, vectorizer
    try:
        model = joblib.load("lr_model .pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        print("Model and vectorizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")

# Define request body format
class TextInput(BaseModel):
    text: str
    favorite_count: int  # Include favorite_count field

# Sentiment mapping
sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Sentiment analysis endpoint
@app.post("/analyze")
async def analyze_sentiment(input: TextInput, db: SessionLocal = Depends(get_db)):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model or vectorizer could not be loaded.")

    try:
        text = input.text
        favorite_count = input.favorite_count
        
        # Transform the text using the vectorizer
        transformed_text = vectorizer.transform([text])
        
        # Reshape favorite_count to (1, 1) and stack with transformed text
        favorite_count_reshaped = np.array([[favorite_count]])
        input_features = hstack([transformed_text, favorite_count_reshaped])  # Combined shape should be (1, 5001)
        
        # Predict using the model
        prediction = model.predict(input_features)[0]
        
        # Get probabilities for confidence score
        probabilities = model.predict_proba(input_features)
        confidence = probabilities[0][prediction]  # Confidence for predicted class
        
        sentiment = sentiment_map.get(prediction, "unknown")

        # Save result to database
        sentiment_record = SentimentRecord(text=text, sentiment=sentiment, confidence=confidence)
        db.add(sentiment_record)
        db.commit()
        db.refresh(sentiment_record)

        return {"sentiment": sentiment, "confidence": confidence}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Use `SessionLocal` dependency
