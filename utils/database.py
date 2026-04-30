"""
SQLite Database Manager for IDS
Статистикаларды және болжам тарихын сақтау
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_PATH = os.getenv('DATABASE_PATH', './data/ids_database.db')


class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or DATABASE_PATH
        # Дерекқор папкасын жасау
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def get_connection(self):
        """Дерекқор байланысын алу"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Кестелерді инициализациялау"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Болжамдар тарихы кестесі
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL NOT NULL,
                confidence REAL NOT NULL,
                features TEXT,
                feature_importance TEXT,
                is_attack INTEGER DEFAULT 0,
                alert_sent INTEGER DEFAULT 0
            )
        ''')

        # Backward-compatible schema migration for attack-type analytics.
        cursor.execute("PRAGMA table_info(predictions)")
        prediction_columns = {row[1] for row in cursor.fetchall()}
        if 'attack_type' not in prediction_columns:
            cursor.execute("ALTER TABLE predictions ADD COLUMN attack_type TEXT")
        if 'attack_type_confidence' not in prediction_columns:
            cursor.execute("ALTER TABLE predictions ADD COLUMN attack_type_confidence REAL")
        
        # Модель статистикалары кестесі
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                total_predictions INTEGER DEFAULT 0,
                attack_predictions INTEGER DEFAULT 0,
                normal_predictions INTEGER DEFAULT 0
            )
        ''')
        
        # Алерттер тарихы кестесі
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                timestamp TEXT NOT NULL,
                message TEXT,
                status TEXT DEFAULT 'sent',
                FOREIGN KEY (prediction_id) REFERENCES predictions (id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, model_name, prediction, probability, confidence, 
                       features=None, feature_importance=None, alert_sent=False,
                       attack_type=None, attack_type_confidence=None):
        """Болжамды дерекқорға сақтау"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        is_attack = 1 if prediction == "Attack" else 0
        
        features_json = json.dumps(features) if features else None
        importance_json = json.dumps(feature_importance) if feature_importance else None
        
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, model_name, prediction, probability, confidence, 
             features, feature_importance, is_attack, alert_sent, attack_type, attack_type_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, model_name, prediction, probability, confidence,
              features_json, importance_json, is_attack, 1 if alert_sent else 0,
              attack_type, attack_type_confidence))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def save_alert(self, prediction_id, message, status='sent'):
        """Алертті сақтау"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO alerts (prediction_id, timestamp, message, status)
            VALUES (?, ?, ?, ?)
        ''', (prediction_id, timestamp, message, status))
        
        conn.commit()
        conn.close()
    
    def get_prediction_history(self, limit=100):
        """Болжам тарихын алу"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_statistics(self):
        """Жалпы статистика"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Жалпы статистика
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(is_attack) as attacks,
                SUM(CASE WHEN is_attack = 0 THEN 1 ELSE 0 END) as normal,
                SUM(alert_sent) as alerts_sent
            FROM predictions
        ''')
        
        stats_row = cursor.fetchone()
        
        # Модельдер бойынша статистика
        cursor.execute('''
            SELECT 
                model_name,
                COUNT(*) as count,
                AVG(probability) as avg_probability,
                AVG(confidence) as avg_confidence
            FROM predictions
            GROUP BY model_name
        ''')
        
        model_stats = cursor.fetchall()
        
        # Соңғы 24 сағаттағы статистика
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(is_attack) as attacks
            FROM predictions
            WHERE timestamp >= datetime('now', '-1 day')
        ''')
        
        daily_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            'total': stats_row['total'] or 0,
            'attacks': stats_row['attacks'] or 0,
            'normal': stats_row['normal'] or 0,
            'alerts_sent': stats_row['alerts_sent'] or 0,
            'models': [dict(row) for row in model_stats],
            'last_24h': dict(daily_stats) if daily_stats else {'total': 0, 'attacks': 0}
        }
    
    def update_model_stats(self, model_name, metrics):
        """Модель статистикасын жаңарту"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO model_stats 
            (model_name, timestamp, accuracy, precision, recall, f1_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_name, timestamp, metrics.get('accuracy'), 
              metrics.get('precision'), metrics.get('recall'), metrics.get('f1_score')))
        
        conn.commit()
        conn.close()
    
    def get_recent_attacks(self, limit=10):
        """Соңғы шабуылдарды алу"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE is_attack = 1
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_attack_type_stats(self):
        """Get attack type aggregates and timeline for analytics dashboards."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                COALESCE(attack_type, 'Unknown') AS attack_type,
                COUNT(*) AS count,
                ROUND(AVG(probability) * 100, 1) AS avg_probability
            FROM predictions
            WHERE is_attack = 1
            GROUP BY COALESCE(attack_type, 'Unknown')
            ORDER BY count DESC
        ''')
        by_type_rows = [dict(row) for row in cursor.fetchall()]

        cursor.execute('''
            SELECT
                strftime('%Y-%m-%d %H:00', timestamp) AS hour,
                COALESCE(attack_type, 'Unknown') AS attack_type,
                COUNT(*) AS count
            FROM predictions
            WHERE is_attack = 1
              AND timestamp >= datetime('now', '-7 days')
            GROUP BY strftime('%Y-%m-%d %H:00', timestamp), COALESCE(attack_type, 'Unknown')
            ORDER BY hour ASC
        ''')
        timeline_rows = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {
            'by_type': by_type_rows,
            'timeline': timeline_rows,
        }

    def create_user(self, username, password_hash):
        """Create new user"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO users (username, password_hash, created_at)
                VALUES (?, ?, ?)
            ''', (username, password_hash, datetime.now().isoformat()))
            user_id = cursor.lastrowid
            conn.commit()
            return {'success': True, 'user_id': user_id}
        except sqlite3.IntegrityError:
            return {'success': False, 'error': 'Username already exists'}
        finally:
            conn.close()

    def get_user_by_username(self, username):
        """Fetch user by username"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, username, password_hash, created_at
            FROM users
            WHERE username = ?
            LIMIT 1
        ''', (username,))

        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def clear_old_records(self, days=30):
        """Ескі жазбаларды өшіру"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM predictions 
            WHERE timestamp < datetime('now', '-? days')
        ''', (days,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted
    
    def clear_all_history(self):
        """Барлық тарихты өшіру"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Болжамдарды өшіру
        cursor.execute('DELETE FROM predictions')
        predictions_deleted = cursor.rowcount
        
        # Алерттерді өшіру
        cursor.execute('DELETE FROM alerts')
        alerts_deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return {'predictions_deleted': predictions_deleted, 'alerts_deleted': alerts_deleted}


# Глобалды дерекқор менеджері
db_manager = DatabaseManager()
