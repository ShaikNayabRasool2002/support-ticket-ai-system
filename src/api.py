import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime

from src.database import db_manager
from src.anomaly_detector import anomaly_detector
from src.llm_engine import query_engine

app = FastAPI(
    title="AI Support Ticket System API",
    description="End-to-End AI System for Support Ticket Ingestion, Natural Language Text-to-SQL, and Statistical Anomaly Detection.",
    version="1.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "How many tickets are currently open?"}, min_length=2)

class QueryResponse(BaseModel):
    query: str
    answer: str
    sql: str
    explanation: Optional[str] = None
    data: List[Dict[str, Any]]
    total_records: int
    execution_time_ms: float
    provider: str

class AnomalyItem(BaseModel):
    ticket_id: str
    created_at: str
    category: str
    priority: str
    status: str
    agent_id: str
    issue_summary: str
    anomaly_type: str
    metric_name: str
    actual_value: float
    threshold: float
    baseline_median: float
    severity: str
    reason: str

class AnomalyResponse(BaseModel):
    total_anomalies: int
    reference_timestamp: str
    resolution_time_outliers_count: int
    unresolved_sla_breaches_count: int
    critical_count: int
    high_count: int
    anomalies: List[AnomalyItem]

class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    total_tickets: int
    llm_provider: str
    timestamp: str

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns the operational status, database connectivity, and configured LLM provider."""
    metrics = db_manager.get_summary_metrics()
    return HealthResponse(
        status="healthy",
        database_connected=True,
        total_tickets=metrics.get("total_tickets", 0),
        llm_provider=query_engine.get_active_provider(),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.post("/query", response_model=QueryResponse, tags=["Natural Language"])
def handle_natural_language_query(req: QueryRequest):
    """
    Accepts a natural language question about support tickets, converts it to safe SQL / anomaly analysis,
    and returns synthesized answers and raw records.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    result = query_engine.process_query(req.query)
    return QueryResponse(**result)

@app.get("/anomalies", response_model=AnomalyResponse, tags=["Anomaly Detection"])
def detect_anomalies(
    category: Optional[str] = Query(None, description="Filter by category (Billing, Technical, General)"),
    min_severity: Optional[str] = Query(None, description="Filter by minimum severity (HIGH, CRITICAL)"),
    days_window: Optional[int] = Query(None, description="Lookback window in days (e.g. 7 for past week)")
):
    """
    Detects statistical outliers in resolution time (IQR) and unresolved high/critical priority tickets over 24h.
    """
    results = anomaly_detector.detect_all_anomalies(
        category=category,
        min_severity=min_severity,
        days_window=days_window
    )
    return AnomalyResponse(**results)

@app.get("/metrics", tags=["Analytics"])
def get_metrics():
    """Returns high-level ticket KPIs for dashboard display."""
    return db_manager.get_summary_metrics()

@app.get("/tickets", tags=["Tickets"])
def list_tickets(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    category: Optional[str] = None,
    agent_id: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """Paginated list of support tickets with optional filters."""
    conditions = []
    params = []
    if status:
        conditions.append("status = ?")
        params.append(status)
    if priority:
        conditions.append("priority = ?")
        params.append(priority)
    if category:
        conditions.append("category = ?")
        params.append(category)
    if agent_id:
        conditions.append("agent_id = ?")
        params.append(agent_id)

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    offset = (page - 1) * page_size

    count_sql = f"SELECT COUNT(*) as total FROM tickets{where_clause}"
    count_res = db_manager.execute_safe_query(count_sql, tuple(params) if params else None)
    total_items = count_res[0]["total"]

    data_sql = f"SELECT * FROM tickets{where_clause} ORDER BY created_at DESC LIMIT {page_size} OFFSET {offset}"
    items = db_manager.execute_safe_query(data_sql, tuple(params) if params else None)

    return {
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": (total_items + page_size - 1) // page_size,
        "items": items
    }

# UI Route
UI_DIR = os.path.join(os.path.dirname(__file__), "ui")
if os.path.exists(UI_DIR):
    @app.get("/", response_class=HTMLResponse, tags=["UI"])
    def get_ui():
        index_file = os.path.join(UI_DIR, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>Support Ticket AI UI is under construction. Visit /docs for Swagger API.</h1>"
