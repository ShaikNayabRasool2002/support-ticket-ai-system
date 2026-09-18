import os
import re
import json
import httpx
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from src.database import db_manager
from src.anomaly_detector import anomaly_detector

# Schema prompt for LLM SQL generation
DATABASE_SCHEMA_DESCRIPTION = """
Table: tickets
Columns:
- ticket_id (TEXT, Primary Key): e.g. 'TKT-001'
- created_at (TIMESTAMP): format 'YYYY-MM-DD HH:MM:SS', range 2024-01-01 to 2024-03-30
- category (TEXT): 'Billing', 'Technical', 'General'
- priority (TEXT): 'Low', 'Medium', 'High', 'Critical'
- status (TEXT): 'Open', 'Resolved', 'Escalated'
- response_time_hrs (REAL): Hours to first response
- resolution_time_hrs (REAL): Hours to resolution (NULL if not resolved)
- agent_id (TEXT): e.g. 'AGT-01' to 'AGT-12'
- customer_rating (INTEGER): 1 to 5 (NULL if not resolved)
- issue_summary (TEXT): Description of the issue

Rules:
1. Return ONLY valid SQLite SELECT queries.
2. For unresolved tickets, use status IN ('Open', 'Escalated') or status != 'Resolved'.
3. Customer satisfaction ratings apply to resolved tickets with customer_rating IS NOT NULL.
4. For relative date calculations (like 'this month' or 'last month'), the dataset spans January 2024 to March 2024. The latest month is March 2024 ('2024-03').
"""

class NLQueryEngine:
    """
    Hybrid Natural Language Query Engine.
    Combines external LLMs (Groq, Ollama, HuggingFace) with a built-in semantic Text-to-SQL engine.
    """

    def __init__(self, db=db_manager, detector=anomaly_detector):
        self.db = db
        self.detector = detector
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.hf_token = os.getenv("HF_TOKEN", "")

    def get_active_provider(self) -> str:
        if self.groq_api_key:
            return "Groq (Cloud Free Tier)"
        elif self.hf_token:
            return "HuggingFace Inference API"
        return "Built-in Zero-Cost Semantic NLP Engine (Local / Offline)"

    def call_groq_llm(self, prompt: str) -> Optional[str]:
        """Calls Groq Cloud free-tier API using LLaMA-3.3-70B."""
        if not self.groq_api_key:
            return None
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": f"You are an expert SQLite data analyst. Convert natural language queries to read-only SQLite SQL.\n{DATABASE_SCHEMA_DESCRIPTION}\nRespond with JSON: {{\"sql\": \"SELECT ...\", \"explanation\": \"...\"}}."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            response = httpx.post(url, headers=headers, json=payload, timeout=12.0)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[Groq LLM Notice] Fallback triggered: {e}")
        return None

    def call_ollama_llm(self, prompt: str) -> Optional[str]:
        """Calls local Ollama server if running."""
        try:
            url = f"{self.ollama_base_url}/api/generate"
            payload = {
                "model": "llama3",
                "prompt": f"Convert to SQLite SELECT query: {prompt}\n{DATABASE_SCHEMA_DESCRIPTION}\nReturn only JSON with key 'sql'.",
                "stream": False,
                "format": "json"
            }
            response = httpx.post(url, json=payload, timeout=5.0)
            if response.status_code == 200:
                return response.json().get("response")
        except Exception:
            pass
        return None

    def semantic_text_to_sql(self, query: str) -> Tuple[str, str]:
        """
        Advanced, zero-cost semantic NLP parser and SQL generator.
        Extracts entities (category, priority, status, agent, date, metrics, aggregations)
        and constructs dynamic, accurate SQLite queries for ANY user question.
        """
        raw_q = query.strip()
        q = raw_q.lower()

        # 1. Normalize common typos and abbreviations
        typo_map = {
            r'\btotla\b': 'total',
            r'\btotall\b': 'total',
            r'\bticketss\b': 'tickets',
            r'\btiket\b': 'ticket',
            r'\btickts\b': 'tickets',
            r'\bresovled\b': 'resolved',
            r'\bcritcal\b': 'critical',
            r'\btechncal\b': 'technical',
            r'\btehnical\b': 'technical',
            r'\btech\b': 'technical',
            r'\bcsat\b': 'customer rating',
            r'\bsatisfaction\b': 'customer rating',
        }
        for pattern, replacement in typo_map.items():
            q = re.sub(pattern, replacement, q)

        # 2. Extract Entities
        # Categories
        cat_match = None
        if "billing" in q:
            cat_match = "Billing"
        elif "technical" in q:
            cat_match = "Technical"
        elif "general" in q:
            cat_match = "General"

        # Priorities
        priority_match = None
        if "critical" in q:
            priority_match = "Critical"
        elif "high" in q:
            priority_match = "High"
        elif "medium" in q:
            priority_match = "Medium"
        elif "low" in q and "lowest" not in q and "slow" not in q:
            priority_match = "Low"

        # Statuses
        status_match = None
        if "unresolved" in q or "not resolved" in q:
            status_match = "Unresolved"
        elif "escalated" in q or "escalate" in q:
            status_match = "Escalated"
        elif re.search(r'\b(?:resolved|closed|completed)\b', q):
            status_match = "Resolved"
        elif "open" in q or "pending" in q:
            status_match = "Open"

        # Agent ID
        agent_match = None
        ag_search = re.search(r'\b(?:agt[-_]?|agent\s*)(\d{1,2})\b', q)
        if ag_search:
            agent_match = f"AGT-{int(ag_search.group(1)):02d}"

        # Rating
        rating_match = None
        rat_search = re.search(r'(?:rating|rated|stars?)\s*(?:of\s*)?([1-5])\b', q)
        if rat_search:
            rating_match = int(rat_search.group(1))

        # Month
        month_match = None
        if "january" in q or "jan " in q or "jan\b" in q:
            month_match = "2024-01"
        elif "february" in q or "feb " in q or "feb\b" in q:
            month_match = "2024-02"
        elif "march" in q or "mar " in q or "mar\b" in q or "this month" in q:
            month_match = "2024-03"

        # Build conditions
        where_clauses = []
        if cat_match:
            where_clauses.append(f"category = '{cat_match}'")
        if priority_match:
            where_clauses.append(f"priority = '{priority_match}'")
        if status_match:
            if status_match == "Unresolved":
                where_clauses.append("status IN ('Open', 'Escalated')")
            else:
                where_clauses.append(f"status = '{status_match}'")
        if agent_match:
            where_clauses.append(f"agent_id = '{agent_match}'")
        if rating_match is not None:
            where_clauses.append(f"customer_rating = {rating_match}")
        if month_match:
            where_clauses.append(f"strftime('%Y-%m', created_at) = '{month_match}'")

        # 3. Agent Ranking Queries (Best, Worst, Most Resolved, Lowest Rating, etc.)
        if "agent" in q:
            if any(w in q for w in ["most", "highest", "top", "max"]) and any(w in q for w in ["resolved", "ticket", "close"]):
                m_filter = f" AND strftime('%Y-%m', created_at) = '{month_match}'" if month_match else ""
                sql = f"SELECT agent_id, COUNT(*) as resolved_count FROM tickets WHERE status = 'Resolved'{m_filter} GROUP BY agent_id ORDER BY resolved_count DESC LIMIT 1"
                period = f" in {month_match}" if month_match else " overall"
                return sql, f"Finding the agent who resolved the most tickets{period}."

            if any(w in q for w in ["lowest", "worst", "poorest"]) and any(w in q for w in ["rating", "customer rating", "score"]):
                sql = "SELECT agent_id, ROUND(AVG(customer_rating), 2) as avg_rating, COUNT(customer_rating) as rated_tickets FROM tickets WHERE customer_rating IS NOT NULL GROUP BY agent_id ORDER BY avg_rating ASC LIMIT 1"
                return sql, "Finding the agent with the lowest average customer rating."

            if any(w in q for w in ["highest", "best", "top"]) and any(w in q for w in ["rating", "customer rating", "score"]):
                sql = "SELECT agent_id, ROUND(AVG(customer_rating), 2) as avg_rating, COUNT(customer_rating) as rated_tickets FROM tickets WHERE customer_rating IS NOT NULL GROUP BY agent_id ORDER BY avg_rating DESC LIMIT 1"
                return sql, "Finding the agent with the highest average customer rating."

            if any(w in q for w in ["fastest", "quickest"]) and "response" in q:
                sql = "SELECT agent_id, ROUND(AVG(response_time_hrs), 2) as avg_response_hrs FROM tickets GROUP BY agent_id ORDER BY avg_response_hrs ASC LIMIT 1"
                return sql, "Finding the agent with the fastest average response time."

            if any(w in q for w in ["list", "all", "performance", "ranking"]) and "agent" in q:
                sql = "SELECT agent_id, COUNT(*) as total_handled, SUM(CASE WHEN status='Resolved' THEN 1 ELSE 0 END) as resolved_count, ROUND(AVG(customer_rating), 2) as avg_csat FROM tickets GROUP BY agent_id ORDER BY resolved_count DESC"
                return sql, "Displaying complete performance breakdown across all support agents."

        # 4. Critical SLA hours query (e.g. "not resolved within 12 hours")
        hours_match = re.search(r'(\d+)\s*(?:hours|hrs)', q)
        if hours_match and any(w in q for w in ["not resolved", "unresolved", "longer than", "within", "took more", "exceeded"]):
            hrs = float(hours_match.group(1))
            crit_clause = "priority = 'Critical' AND " if "critical" in q else ""
            sql = f"""
                SELECT ticket_id, created_at, category, priority, status, resolution_time_hrs, agent_id, issue_summary
                FROM tickets 
                WHERE {crit_clause}(resolution_time_hrs > {hrs} OR (status IN ('Open', 'Escalated') AND resolution_time_hrs IS NULL))
                ORDER BY created_at DESC
            """.strip()
            return sql, f"Finding tickets that were not resolved within {hrs} hours."

        # 5. Average / Max / Min Metrics (Response time, Resolution time, CSAT)
        if any(w in q for w in ["average", "mean", "avg", "highest", "longest", "maximum", "max", "lowest", "fastest", "minimum", "min"]):
            func = "AVG"
            if any(w in q for w in ["highest", "longest", "maximum", "max", "slowest"]):
                func = "MAX"
            elif any(w in q for w in ["lowest", "fastest", "minimum", "min", "quickest"]):
                func = "MIN"

            target_col = None
            col_label = ""
            if "rating" in q or "customer rating" in q:
                target_col = "customer_rating"
                col_label = "customer rating"
                where_clauses.append("customer_rating IS NOT NULL")
            elif "resolution" in q or "resolve" in q:
                target_col = "resolution_time_hrs"
                col_label = "resolution time (hrs)"
                where_clauses.append("resolution_time_hrs IS NOT NULL")
            elif "response" in q:
                target_col = "response_time_hrs"
                col_label = "response time (hrs)"

            if target_col:
                where_str = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                col_alias = "avg_rating" if target_col == "customer_rating" and func == "AVG" else (f"avg_{target_col}" if func == "AVG" else f"{func.lower()}_{target_col}")
                if cat_match and target_col == "customer_rating":
                    sql = f"SELECT '{cat_match}' as category, ROUND({func}({target_col}), 2) as {col_alias}, COUNT(*) as ticket_count FROM tickets{where_str}"
                else:
                    sql = f"SELECT ROUND({func}({target_col}), 2) as {col_alias} FROM tickets{where_str}"
                return sql, f"Calculating {func.lower()} {col_label} with applied filters."

        # 6. Breakdown / Distribution Queries
        if any(w in q for w in ["breakdown", "distribution", "by category", "by priority", "by status"]):
            if "category" in q:
                sql = "SELECT category, COUNT(*) as ticket_count, ROUND(AVG(customer_rating), 2) as avg_csat FROM tickets GROUP BY category ORDER BY ticket_count DESC"
                return sql, "Aggregating ticket count and customer satisfaction by category."
            if "priority" in q:
                sql = "SELECT priority, COUNT(*) as ticket_count FROM tickets GROUP BY priority ORDER BY ticket_count DESC"
                return sql, "Aggregating ticket count by priority level."
            if "status" in q:
                sql = "SELECT status, COUNT(*) as ticket_count FROM tickets GROUP BY status ORDER BY ticket_count DESC"
                return sql, "Aggregating ticket count by status."

        # 7. Topic & Issue Keyword Search (e.g. dark mode, login, invoice, refund, timeout, webhook)
        known_topics = [
            "dark mode", "login failure", "login", "password", "api timeout", "timeout",
            "database connection", "database", "invoice", "refund", "subscription",
            "payment failed", "payment", "webhook", "onboarding", "account deletion",
            "notification preferences", "data sync", "team members", "export data", "checkout"
        ]
        matched_topic = None
        for topic in known_topics:
            if topic in q:
                matched_topic = topic
                break

        if matched_topic:
            where_clauses.append(f"issue_summary LIKE '%{matched_topic}%'")
            if any(w in q for w in ["how many", "count", "number of"]):
                where_str = f" WHERE {' AND '.join(where_clauses)}"
                return f"SELECT COUNT(*) as count FROM tickets{where_str}", f"Counting tickets regarding '{matched_topic}'."
            else:
                where_str = f" WHERE {' AND '.join(where_clauses)}"
                return f"SELECT ticket_id, created_at, category, priority, status, agent_id, issue_summary FROM tickets{where_str} ORDER BY created_at DESC LIMIT 15", f"Finding tickets related to '{matched_topic}'."

        # 8. Explicit Count Queries (with any combination of filters)
        is_count_query = (
            any(w in q for w in ["how many", "count of", "number of"]) or 
            q in ["count", "total records", "total tickets", "record count", "total", "totla records", "all records"]
        )
        if is_count_query:
            where_str = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            sql = f"SELECT COUNT(*) as count FROM tickets{where_str}"
            desc = f"Counting tickets matching: {', '.join(where_clauses)}" if where_clauses else "Counting total records in the tickets database."
            return sql, desc

        # 9. List / Show Queries with filters
        if where_clauses:
            where_str = f" WHERE {' AND '.join(where_clauses)}"
            sql = f"SELECT ticket_id, created_at, category, priority, status, agent_id, issue_summary FROM tickets{where_str} ORDER BY created_at DESC LIMIT 20"
            return sql, f"Filtering tickets where: {', '.join(where_clauses)}."

        # 10. Free-form Keyword Search in issue summary
        clean_words = [w for w in re.findall(r'\b\w+\b', q) if len(w) > 3 and w not in ['what', 'which', 'show', 'tell', 'find', 'tickets', 'ticket', 'there', 'with', 'about', 'some', 'give', 'list', 'please', 'display', 'mentioning']]
        if clean_words:
            keyword = clean_words[0]
            sql = f"SELECT ticket_id, created_at, category, priority, status, agent_id, issue_summary FROM tickets WHERE issue_summary LIKE '%{keyword}%' ORDER BY created_at DESC LIMIT 15"
            return sql, f"Searching for tickets mentioning '{keyword}'."

        # 11. Universal fallback
        sql = "SELECT ticket_id, created_at, category, priority, status, agent_id, issue_summary FROM tickets ORDER BY created_at DESC LIMIT 10"
        return sql, "Displaying recent tickets."

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Main query processing pipeline:
        1. Checks for anomaly queries (e.g. 'Are there any anomalies in resolution times this week?')
        2. Routes to LLM or built-in semantic Text-to-SQL
        3. Executes SQL safely
        4. Synthesizes human-readable natural language answer
        """
        start_time = datetime.now()
        q_lower = user_query.lower()

        # Direct anomaly check
        if "anomal" in q_lower or "outlier" in q_lower:
            days = 7 if "week" in q_lower else (30 if "month" in q_lower else None)
            category = "Technical" if "technical" in q_lower else ("Billing" if "billing" in q_lower else ("General" if "general" in q_lower else None))
            
            anomaly_data = self.detector.detect_all_anomalies(category=category, days_window=days)
            exec_time_ms = round((datetime.now() - start_time).total_seconds() * 1000, 2)
            
            summary_text = (
                f"Yes, detected {anomaly_data['total_anomalies']} anomalies "
                f"({anomaly_data['resolution_time_outliers_count']} resolution time outliers and "
                f"{anomaly_data['unresolved_sla_breaches_count']} unresolved SLA breaches > 24h)."
            )
            
            return {
                "query": user_query,
                "answer": summary_text,
                "sql": "-- Anomaly detection statistical pipeline executed",
                "explanation": "Evaluated Interquartile Range (IQR) outliers and high-priority unresolved tickets over 24 hours.",
                "data": anomaly_data["anomalies"][:15],
                "total_records": anomaly_data["total_anomalies"],
                "execution_time_ms": exec_time_ms,
                "provider": "Statistical Anomaly Engine"
            }

        # Attempt LLM if configured
        generated_sql = None
        explanation = None
        provider = self.get_active_provider()

        if self.groq_api_key:
            groq_resp = self.call_groq_llm(user_query)
            if groq_resp:
                try:
                    parsed = json.loads(groq_resp)
                    generated_sql = parsed.get("sql")
                    explanation = parsed.get("explanation", "Generated by Groq LLaMA-3.3-70B.")
                except Exception:
                    pass

        if not generated_sql and self.ollama_base_url:
            ollama_resp = self.call_ollama_llm(user_query)
            if ollama_resp:
                try:
                    parsed = json.loads(ollama_resp)
                    generated_sql = parsed.get("sql")
                    explanation = "Generated by Ollama local model."
                except Exception:
                    pass

        # Built-in semantic engine if no LLM output
        if not generated_sql:
            generated_sql, explanation = self.semantic_text_to_sql(user_query)
            provider = "Built-in Zero-Cost Semantic NLP Engine"

        # Execute safe SQL
        try:
            records = self.db.execute_safe_query(generated_sql)
            exec_time_ms = round((datetime.now() - start_time).total_seconds() * 1000, 2)
            
            # Synthesize natural language answer
            answer = self.synthesize_answer(user_query, records, explanation)

            return {
                "query": user_query,
                "answer": answer,
                "sql": generated_sql,
                "explanation": explanation,
                "data": records,
                "total_records": len(records),
                "execution_time_ms": exec_time_ms,
                "provider": provider
            }
        except Exception as err:
            exec_time_ms = round((datetime.now() - start_time).total_seconds() * 1000, 2)
            return {
                "query": user_query,
                "answer": f"Error executing query: {str(err)}",
                "sql": generated_sql,
                "explanation": explanation,
                "data": [],
                "total_records": 0,
                "execution_time_ms": exec_time_ms,
                "provider": provider,
                "error": str(err)
            }

    def synthesize_answer(self, query: str, records: List[Dict[str, Any]], explanation: str) -> str:
        """Formats query results into an informative natural language summary."""
        if not records:
            return "No matching tickets or records found."

        first = records[0]

        # Single scalar aggregation (count / avg / max / min)
        if len(records) == 1:
            if "agent_id" in first and "resolved_count" in first:
                return f"Agent {first['agent_id']} resolved the most tickets ({first['resolved_count']} tickets)."
            if "agent_id" in first and "avg_rating" in first:
                return f"Agent {first['agent_id']} has an average customer rating of {first['avg_rating']} / 5.0 (across {first.get('rated_tickets', 'all')} evaluated tickets)."
            if "count" in first:
                if any(w in query.lower() for w in ["total", "totla", "all", "dataset"]):
                    return f"There are a total of {first['count']} records (support tickets) in the system."
                return f"There are currently {first['count']} matching tickets."
            if "avg_response_time_hrs" in first:
                return f"The average response time is {first['avg_response_time_hrs']} hours."
            if "max_response_time_hrs" in first:
                return f"The longest response time is {first['max_response_time_hrs']} hours."
            if "min_response_time_hrs" in first:
                return f"The fastest response time is {first['min_response_time_hrs']} hours."
            if "avg_resolution_time_hrs" in first or "avg_resolution_hrs" in first:
                val = first.get("avg_resolution_time_hrs", first.get("avg_resolution_hrs"))
                return f"The average resolution time is {val} hours."
            if "max_resolution_time_hrs" in first:
                return f"The longest resolution time is {first['max_resolution_time_hrs']} hours."
            if "min_resolution_time_hrs" in first:
                return f"The fastest resolution time is {first['min_resolution_time_hrs']} hours."
            if "avg_customer_rating" in first or "avg_rating" in first:
                val = first.get("avg_customer_rating", first.get("avg_rating"))
                cat = first.get("category", "")
                cat_str = f" for {cat}" if cat else ""
                return f"The average customer satisfaction rating{cat_str} is {val} / 5.0."

        # Multi-record table
        if len(records) > 1:
            if "agent_id" in first and "total_handled" in first:
                return f"Here is the performance summary across all {len(records)} support agents."
            if "category" in first and "ticket_count" in first:
                breakdown = ", ".join([f"{r['category']}: {r['ticket_count']}" for r in records])
                return f"Ticket distribution across categories: {breakdown}."
            if "priority" in first and "ticket_count" in first:
                breakdown = ", ".join([f"{r['priority']}: {r['ticket_count']}" for r in records])
                return f"Ticket distribution by priority: {breakdown}."
            if "status" in first and "ticket_count" in first:
                breakdown = ", ".join([f"{r['status']}: {r['ticket_count']}" for r in records])
                return f"Ticket distribution by status: {breakdown}."
            if "ticket_id" in first:
                return f"Found {len(records)} matching tickets. Showing top results including ticket {first['ticket_id']} ({first.get('priority', '')} priority, status: {first.get('status', '')})."

        return f"Query returned {len(records)} records."

# Global query engine instance
query_engine = NLQueryEngine()
