from pydantic import BaseModel, Field
from typing import List, Optional

class SearchResponse(BaseModel):
    """Refined list of file paths matching the query."""
    file_paths: List[str] = Field(..., description="List of absolute file paths relevant to the query.")
    node_id: str = Field(..., description="ID of the node returning these results.")

class AnalysisResult(BaseModel):
    """Analysis result containing statistical summary."""
    summary: str = Field(..., description="Text summary of the statistical analysis.")
    raw_stats: Optional[dict] = Field(default=None, description="Raw statistical data (p-values, counts, etc.)")
    node_id: str = Field(..., description="ID of the node providing this analysis.")
