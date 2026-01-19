"""数据库模块"""
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager


class Database:
    """SQLite 数据库管理"""

    def __init__(self, db_path: str = "data/traffic.db"):
        """
        初始化数据库

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_tables(self):
        """初始化数据表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 车辆记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    plate_number TEXT,
                    vehicle_type TEXT,
                    color TEXT,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    avg_speed REAL,
                    direction TEXT
                )
            ''')

            # 违规记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    violation_type TEXT,
                    timestamp TIMESTAMP,
                    location_x INTEGER,
                    location_y INTEGER,
                    speed REAL,
                    plate_number TEXT,
                    snapshot_path TEXT
                )
            ''')

            # 交通流量统计表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS traffic_flow (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    vehicle_count INTEGER,
                    avg_speed REAL,
                    direction TEXT
                )
            ''')

            # 创建索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_vehicles_plate
                ON vehicles(plate_number)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_violations_type
                ON violations(violation_type)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_violations_time
                ON violations(timestamp)
            ''')

    def add_vehicle(self, track_id: int, plate_number: Optional[str],
                    vehicle_type: str, color: str, speed: float,
                    direction: str) -> int:
        """添加车辆记录"""
        now = datetime.now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO vehicles
                (track_id, plate_number, vehicle_type, color, first_seen,
                 last_seen, avg_speed, direction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (track_id, plate_number, vehicle_type, color, now, now,
                  speed, direction))
            return cursor.lastrowid

    def update_vehicle(self, track_id: int, speed: float, direction: str):
        """更新车辆记录"""
        now = datetime.now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE vehicles
                SET last_seen = ?, avg_speed = ?, direction = ?
                WHERE track_id = ?
            ''', (now, speed, direction, track_id))

    def add_violation(self, track_id: int, violation_type: str,
                      location: tuple, speed: Optional[float] = None,
                      plate_number: Optional[str] = None,
                      snapshot_path: Optional[str] = None) -> int:
        """添加违规记录"""
        now = datetime.now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO violations
                (track_id, violation_type, timestamp, location_x, location_y,
                 speed, plate_number, snapshot_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (track_id, violation_type, now, location[0], location[1],
                  speed, plate_number, snapshot_path))
            return cursor.lastrowid

    def add_traffic_flow(self, vehicle_count: int, avg_speed: float,
                         direction: str):
        """添加交通流量记录"""
        now = datetime.now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO traffic_flow
                (timestamp, vehicle_count, avg_speed, direction)
                VALUES (?, ?, ?, ?)
            ''', (now, vehicle_count, avg_speed, direction))

    def get_violations(self, violation_type: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """查询违规记录"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT * FROM violations WHERE 1=1'
            params = []

            if violation_type:
                query += ' AND violation_type = ?'
                params.append(violation_type)
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)

            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_traffic_stats(self, start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """获取交通统计"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT
                    COUNT(*) as total_vehicles,
                    AVG(avg_speed) as avg_speed,
                    COUNT(DISTINCT direction) as direction_count
                FROM traffic_flow WHERE 1=1
            '''
            params = []

            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)

            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else {}

    def get_violation_stats(self) -> Dict[str, int]:
        """获取违规统计"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT violation_type, COUNT(*) as count
                FROM violations
                GROUP BY violation_type
            ''')
            return {row['violation_type']: row['count'] for row in cursor.fetchall()}

    def search_by_plate(self, plate_number: str) -> List[Dict[str, Any]]:
        """按车牌号搜索"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM vehicles
                WHERE plate_number LIKE ?
                ORDER BY last_seen DESC
            ''', (f'%{plate_number}%',))
            return [dict(row) for row in cursor.fetchall()]

    def clear_old_records(self, days: int = 30):
        """清理旧记录"""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM vehicles WHERE last_seen < ?', (cutoff,))
            cursor.execute('DELETE FROM violations WHERE timestamp < ?', (cutoff,))
            cursor.execute('DELETE FROM traffic_flow WHERE timestamp < ?', (cutoff,))
