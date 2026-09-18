import pytest
from fastapi.testclient import TestClient
from src.database import db_manager
from src.anomaly_detector import anomaly_detector
from src.llm_engine import query_engine
from src.api import app

client = TestClient(app)

def test_database_ingestion():
    """Verify that all 500 rows are loaded and metrics match schema."""
    count = db_manager.init_db()
    assert count == 500
    metrics = db_manager.get_summary_metrics()
    assert metrics["total_tickets"] == 500
    assert metrics["open_tickets"] == 111
    assert metrics["resolved_tickets"] == 327
    assert metrics["escalated_tickets"] == 62
    assert metrics["avg_csat"] > 0

def test_database_safe_query_guard():
    """Verify that destructive SQL operations are rejected."""
    with pytest.raises(ValueError, match="Only SELECT or WITH queries are permitted"):
        db_manager.execute_safe_query("DELETE FROM tickets")

    with pytest.raises(ValueError, match="Only SELECT or WITH queries are permitted"):
        db_manager.execute_safe_query("DROP TABLE tickets")

    with pytest.raises(ValueError, match="Multiple SQL statements are not permitted"):
        db_manager.execute_safe_query("SELECT 1; SELECT 2")

def test_anomaly_detection_iqr():
    """Verify IQR resolution outliers are correctly identified."""
    anomalies = anomaly_detector.detect_resolution_time_anomalies()
    assert len(anomalies) > 0
    # Every outlier's actual resolution time must exceed its category threshold
    for a in anomalies:
        assert a["actual_value"] > a["threshold"]
        assert a["severity"] in ["HIGH", "CRITICAL"]

def test_anomaly_detection_unresolved_sla():
    """Verify unresolved high/critical tickets over 24 hours."""
    unresolved_anomalies = anomaly_detector.detect_unresolved_high_priority_anomalies()
    assert len(unresolved_anomalies) > 0
    for a in unresolved_anomalies:
        assert a["actual_value"] > 24.0
        assert a["status"] in ["Open", "Escalated"]
        assert a["priority"] in ["Critical", "High"]

def test_all_sample_queries():
    """Test all sample queries specified in the assessment requirements."""
    # 1. Open tickets
    res1 = query_engine.process_query("How many tickets are currently open?")
    assert res1["total_records"] == 1
    assert res1["data"][0]["count"] == 111
    assert "111" in res1["answer"]

    # 2. Agent with most resolved tickets this month
    res2 = query_engine.process_query("Which agent resolved the most tickets this month?")
    assert res2["total_records"] == 1
    assert "agent_id" in res2["data"][0]
    assert "AGT-01" in res2["answer"]

    # 3. Critical tickets not resolved within 12 hours
    res3 = query_engine.process_query("Show me all Critical tickets not resolved within 12 hours.")
    assert res3["total_records"] > 0
    for r in res3["data"]:
        assert r["priority"] == "Critical"

    # 4. Average customer rating for Technical tickets
    res4 = query_engine.process_query("What is the average customer rating for Technical category tickets?")
    assert res4["total_records"] == 1
    assert res4["data"][0]["avg_rating"] == 3.74

    # 5. Anomalies in resolution times this week
    res5 = query_engine.process_query("Are there any anomalies in resolution times this week?")
    assert "detected" in res5["answer"].lower() or "anomalies" in res5["answer"].lower()
    assert res5["total_records"] > 0

    # 6. Agent with lowest customer rating
    res6 = query_engine.process_query("Which agent has the lowest average customer rating?")
    assert res6["total_records"] == 1
    assert "agent_id" in res6["data"][0]
    assert "3.48" in res6["answer"]

    # 7. Unresolved critical tickets
    res7 = query_engine.process_query("How many critical tickets are unresolved?")
    assert res7["total_records"] == 1
    assert res7["data"][0]["count"] == 31

def test_api_health():
    """Test GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_tickets"] == 500
    assert "llm_provider" in data

def test_api_query_endpoint():
    """Test POST /query endpoint."""
    response = client.post("/query", json={"query": "How many tickets are currently open?"})
    assert response.status_code == 200
    data = response.json()
    assert "111" in data["answer"]
    assert "SELECT" in data["sql"]
    assert data["execution_time_ms"] >= 0

def test_api_anomalies_endpoint():
    """Test GET /anomalies endpoint."""
    response = client.get("/anomalies?min_severity=CRITICAL")
    assert response.status_code == 200
    data = response.json()
    assert data["total_anomalies"] > 0
    for a in data["anomalies"]:
        assert a["severity"] == "CRITICAL"

def test_api_metrics_endpoint():
    """Test GET /metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_tickets"] == 500
    assert data["open_tickets"] == 111

def test_api_tickets_endpoint():
    """Test GET /tickets paginated endpoint."""
    response = client.get("/tickets?page=1&page_size=10&status=Open")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 10
    assert data["total_items"] == 111

def test_api_ui_endpoint():
    """Test GET / UI dashboard endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Support Ticket Intelligence" in response.text
