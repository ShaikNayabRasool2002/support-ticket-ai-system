import uvicorn
import os
import sys

# Ensure src is discoverable
sys.path.insert(0, os.path.dirname(__file__))

from src.database import db_manager
from src.api import app

def startup_banner():
    print("=" * 70)
    print(" 🚀 Support Ticket Intelligence & Anomaly Detection System")
    print(" DOTMappers IT Pvt. Ltd. | AI Engineer Role Assessment")
    print("=" * 70)
    # Ensure database is loaded
    count = db_manager.init_db()
    print(f" [+] Ingested Tickets Count: {count}")
    print(f" [+] REST API Documentation: http://localhost:8000/docs")
    print(f" [+] Interactive Web Dashboard: http://localhost:8000/")
    print("=" * 70)

if __name__ == "__main__":
    startup_banner()
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
