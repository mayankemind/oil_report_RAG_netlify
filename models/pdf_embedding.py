from pydantic import BaseModel
from typing import List, Dict

class PDFEmbedding(BaseModel):
    page_number:int
    content:str
    embedding: list[float]

# Data Model for Requests
class QueryRequest(BaseModel):
    query: str
