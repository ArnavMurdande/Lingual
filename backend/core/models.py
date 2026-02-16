from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class SummaryRequest(BaseModel):
    text: str
    sentences_count: int = 5
    summary_ratio: Optional[float] = None # Percentage of original content (0.1 to 1.0)
    refinement: bool = False  # Legacy support (Hybrid is now a mode)
    language: str = "en"      # Target language code
    mode: str = "extractive"  # extractive, abstractive, hybrid
    length: int = 5           # Maps to sentences count or token limit

class SentenceScore(BaseModel):
    text: str
    index: int
    score: float
    rank: int
    reasons: Dict[str, float]  # Breakdown: tfidf, textrank, embedding (centrality)

class SummaryResponse(BaseModel):
    original_text: str
    summary_text: str
    extractive_summary: List[str]
    refined_summary: Optional[str] = None
    ranking_data: List[SentenceScore]
    metrics: Dict[str, Any]  # Compression ratio, etc.
