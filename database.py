"""
Database module for storing prediction history and analytics
"""
import sqlite3
from datetime import datetime
from pathlib import Path
import json
from typing import List, Dict, Optional
import pandas as pd

# Database path
DB_DIR = Path("data")
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "predictions.db"


class PredictionDatabase:
    """Handle all database operations for predictions"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cluster_id INTEGER NOT NULL,
                confidence REAL,
                age INTEGER,
                education INTEGER,
                marital_status INTEGER,
                parental_status INTEGER,
                children INTEGER,
                income REAL,
                total_spending REAL,
                days_as_customer INTEGER,
                recency INTEGER,
                wines REAL,
                fruits REAL,
                meat REAL,
                fish REAL,
                sweets REAL,
                gold REAL,
                web_purchases INTEGER,
                catalog_purchases INTEGER,
                store_purchases INTEGER,
                discount_purchases INTEGER,
                total_promo INTEGER,
                web_visits INTEGER,
                user_id INTEGER DEFAULT 1
            )
        ''')
        
        # Model training history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT,
                optimal_clusters INTEGER,
                silhouette_score REAL,
                davies_bouldin_score REAL,
                calinski_harabasz_score REAL,
                training_samples INTEGER,
                features_used INTEGER
            )
        ''')
        
        # Users table (for future authentication)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME,
                role TEXT DEFAULT 'user'
            )
        ''')
        
        # Insert default user
        cursor.execute('''
            INSERT OR IGNORE INTO users (id, username, email, role) 
            VALUES (1, 'admin', 'admin@example.com', 'admin')
        ''')
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, prediction_data: Dict) -> int:
        """Save a prediction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                cluster_id, confidence, age, education, marital_status,
                parental_status, children, income, total_spending,
                days_as_customer, recency, wines, fruits, meat, fish,
                sweets, gold, web_purchases, catalog_purchases,
                store_purchases, discount_purchases, total_promo, web_visits
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_data['cluster_id'],
            prediction_data.get('confidence', 0),
            prediction_data['age'],
            prediction_data['education'],
            prediction_data['marital_status'],
            prediction_data['parental_status'],
            prediction_data['children'],
            prediction_data['income'],
            prediction_data['total_spending'],
            prediction_data['days_as_customer'],
            prediction_data['recency'],
            prediction_data['wines'],
            prediction_data['fruits'],
            prediction_data['meat'],
            prediction_data['fish'],
            prediction_data['sweets'],
            prediction_data['gold'],
            prediction_data['web_purchases'],
            prediction_data['catalog_purchases'],
            prediction_data['store_purchases'],
            prediction_data['discount_purchases'],
            prediction_data['total_promo'],
            prediction_data['web_visits']
        ))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def get_prediction_history(self, limit: int = 100) -> List[Dict]:
        """Get recent predictions"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_cluster_statistics(self) -> Dict:
        """Get statistics about predictions by cluster"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                cluster_id,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                AVG(income) as avg_income,
                AVG(total_spending) as avg_spending
            FROM predictions
            GROUP BY cluster_id
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        stats = {}
        for row in rows:
            stats[row[0]] = {
                'count': row[1],
                'avg_confidence': round(row[2], 2) if row[2] else 0,
                'avg_income': round(row[3], 2) if row[3] else 0,
                'avg_spending': round(row[4], 2) if row[4] else 0
            }
        
        return stats
    
    def export_to_csv(self, filepath: str = None) -> str:
        """Export all predictions to CSV"""
        if filepath is None:
            filepath = f"predictions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()
        
        df.to_csv(filepath, index=False)
        return filepath
    
    def save_model_training(self, metrics: Dict):
        """Save model training metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_history (
                model_version, optimal_clusters, silhouette_score,
                davies_bouldin_score, calinski_harabasz_score,
                training_samples, features_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            metrics.get('optimal_clusters', 0),
            metrics.get('silhouette_score', 0),
            metrics.get('davies_bouldin_score', 0),
            metrics.get('calinski_harabasz_score', 0),
            metrics.get('training_samples', 0),
            metrics.get('features_used', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def get_total_predictions(self) -> int:
        """Get total number of predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_predictions_by_date(self, days: int = 7) -> List[Dict]:
        """Get predictions from last N days"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
            ORDER BY timestamp DESC
        ''', (days,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]


# Global database instance
db = PredictionDatabase()
