from pydantic import BaseModel
from datetime import datetime
from typing import List, Union, Optional

class SalesPredictionInput(BaseModel):
    Id: int
    Store: int
    DayOfWeek: int
    Date: str # Datetime
    Open: int
    Promo: int
    StateHoliday: str
    SchoolHoliday: str

class SalesPredictionOutput(BaseModel):
    PredictedSales: Union[float, List[float]]
