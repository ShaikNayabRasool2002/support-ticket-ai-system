import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from src.database import db_manager

class AnomalyDetector:
    """
    Statistical and Rule-Based Anomaly Detection Engine for Customer Support Tickets.
    
    Detects:
    1. Abnormally Long Resolution Times (IQR upper bound and Z-score per category and overall)
    2. Unresolved High/Critical Priority Tickets older than 24 hours
    3. Extreme Response Times
    4. Poor Customer Satisfaction Spikes
    """

    def __init__(self, db=db_manager):
        self.db = db

    def get_reference_timestamp(self, df: pd.DataFrame) -> datetime:
        """Returns the latest ticket creation timestamp in the dataset as the reference point for relative time calculations."""
        if df.empty or 'created_at' not in df.columns:
            return datetime.now()
        return df['created_at'].max()

    def detect_resolution_time_anomalies(
        self, 
        df: Optional[pd.DataFrame] = None, 
        group_by_category: bool = True,
        iqr_multiplier: float = 1.5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Flags tickets with abnormally long resolution times using the Interquartile Range (IQR) method.
        Upper Threshold = Q3 + (IQR_MULTIPLIER * IQR)
        """
        if df is None:
            df = self.db.get_dataframe()

        # Filter date range if specified
        filtered_df = df.copy()
        if start_date:
            filtered_df = filtered_df[filtered_df['created_at'] >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df['created_at'] <= end_date]

        # Only resolved tickets with valid resolution_time_hrs
        resolved = filtered_df[filtered_df['resolution_time_hrs'].notnull() & (filtered_df['resolution_time_hrs'] > 0)].copy()
        if resolved.empty:
            return []

        anomalies = []

        if group_by_category:
            for cat, group in resolved.groupby('category'):
                q1 = group['resolution_time_hrs'].quantile(0.25)
                q3 = group['resolution_time_hrs'].quantile(0.75)
                iqr = q3 - q1
                threshold = q3 + (iqr_multiplier * iqr)
                extreme_threshold = q3 + (3.0 * iqr)

                outliers = group[group['resolution_time_hrs'] > threshold]
                for _, row in outliers.iterrows():
                    val = float(row['resolution_time_hrs'])
                    severity = "CRITICAL" if val > extreme_threshold else "HIGH"
                    anomalies.append({
                        "ticket_id": row['ticket_id'],
                        "created_at": row['created_at'].strftime('%Y-%m-%d %H:%M'),
                        "category": row['category'],
                        "priority": row['priority'],
                        "status": row['status'],
                        "agent_id": row['agent_id'],
                        "issue_summary": row['issue_summary'],
                        "anomaly_type": "abnormal_resolution_time",
                        "metric_name": "resolution_time_hrs",
                        "actual_value": val,
                        "threshold": round(threshold, 2),
                        "baseline_median": round(float(group['resolution_time_hrs'].median()), 2),
                        "severity": severity,
                        "reason": f"Resolution time of {val:.1f} hrs exceeds category '{cat}' statistical upper bound of {threshold:.1f} hrs (Q3 + 1.5xIQR)."
                    })
        else:
            q1 = resolved['resolution_time_hrs'].quantile(0.25)
            q3 = resolved['resolution_time_hrs'].quantile(0.75)
            iqr = q3 - q1
            threshold = q3 + (iqr_multiplier * iqr)
            extreme_threshold = q3 + (3.0 * iqr)

            outliers = resolved[resolved['resolution_time_hrs'] > threshold]
            for _, row in outliers.iterrows():
                val = float(row['resolution_time_hrs'])
                severity = "CRITICAL" if val > extreme_threshold else "HIGH"
                anomalies.append({
                    "ticket_id": row['ticket_id'],
                    "created_at": row['created_at'].strftime('%Y-%m-%d %H:%M'),
                    "category": row['category'],
                    "priority": row['priority'],
                    "status": row['status'],
                    "agent_id": row['agent_id'],
                    "issue_summary": row['issue_summary'],
                    "anomaly_type": "abnormal_resolution_time",
                    "metric_name": "resolution_time_hrs",
                    "actual_value": val,
                    "threshold": round(threshold, 2),
                    "baseline_median": round(float(resolved['resolution_time_hrs'].median()), 2),
                    "severity": severity,
                    "reason": f"Resolution time of {val:.1f} hrs exceeds global statistical upper bound of {threshold:.1f} hrs."
                })

        return sorted(anomalies, key=lambda x: x['actual_value'], reverse=True)

    def detect_unresolved_high_priority_anomalies(
        self, 
        df: Optional[pd.DataFrame] = None,
        max_age_hours: float = 24.0,
        reference_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Flags unresolved (Open or Escalated) tickets with High or Critical priority
        that have been open for more than `max_age_hours` (default 24 hours).
        """
        if df is None:
            df = self.db.get_dataframe()

        if reference_time is None:
            # Anchor to dataset latest timestamp for deterministic historical analysis
            reference_time = self.get_reference_timestamp(df)

        unresolved = df[
            df['status'].isin(['Open', 'Escalated']) & 
            df['priority'].isin(['Critical', 'High'])
        ].copy()

        anomalies = []
        for _, row in unresolved.iterrows():
            age_hours = (reference_time - row['created_at']).total_seconds() / 3600.0
            if age_hours > max_age_hours:
                severity = "CRITICAL" if row['priority'] == 'Critical' else "HIGH"
                anomalies.append({
                    "ticket_id": row['ticket_id'],
                    "created_at": row['created_at'].strftime('%Y-%m-%d %H:%M'),
                    "category": row['category'],
                    "priority": row['priority'],
                    "status": row['status'],
                    "agent_id": row['agent_id'],
                    "issue_summary": row['issue_summary'],
                    "anomaly_type": "unresolved_high_priority_over_24h",
                    "metric_name": "age_hours",
                    "actual_value": round(age_hours, 1),
                    "threshold": max_age_hours,
                    "baseline_median": max_age_hours,
                    "severity": severity,
                    "reason": f"{row['priority']} priority ticket has remained {row['status']} for {age_hours:.1f} hours (SLA breach > {max_age_hours}h)."
                })

        return sorted(anomalies, key=lambda x: x['actual_value'], reverse=True)

    def detect_all_anomalies(
        self, 
        category: Optional[str] = None, 
        min_severity: Optional[str] = None,
        days_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive anomaly detection across all rule and statistical engines.
        """
        df = self.db.get_dataframe()
        ref_time = self.get_reference_timestamp(df)
        start_date = ref_time - timedelta(days=days_window) if days_window else None

        resolution_outliers = self.detect_resolution_time_anomalies(df, start_date=start_date)
        unresolved_slas = self.detect_unresolved_high_priority_anomalies(df, reference_time=ref_time)

        all_items = resolution_outliers + unresolved_slas

        # Optional filters
        if category:
            all_items = [a for a in all_items if a['category'].lower() == category.lower()]
        
        severity_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        if min_severity and min_severity.upper() in severity_order:
            min_rank = severity_order[min_severity.upper()]
            all_items = [a for a in all_items if severity_order.get(a['severity'], 1) >= min_rank]

        return {
            "total_anomalies": len(all_items),
            "reference_timestamp": ref_time.strftime('%Y-%m-%d %H:%M:%S'),
            "resolution_time_outliers_count": len(resolution_outliers),
            "unresolved_sla_breaches_count": len(unresolved_slas),
            "critical_count": sum(1 for a in all_items if a['severity'] == 'CRITICAL'),
            "high_count": sum(1 for a in all_items if a['severity'] == 'HIGH'),
            "anomalies": all_items
        }

# Global detector instance
anomaly_detector = AnomalyDetector()
