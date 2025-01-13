from pydantic import BaseModel
from datetime import datetime

class SalesPredictionInput(BaseModel):
    store: int
    day_of_week: int
    date: str # Datetime
    promo: int
    state_holiday: str
    school_holiday: int

class SalesPredictionOutput(BaseModel):
    predicted_sales: float
