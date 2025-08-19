#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全地向 llm_usage 表添加列 model_assign_name 及其索引
支持 MySQL / MariaDB（PyMySQL） 和 SQLite（sqlite3）
"""

from __future__ import annotations
import sqlite3
import pymysql
from contextlib import closing

# ===== 按需修改 =====
# 1) MySQL 示例
# CONNECTION = dict(
#     host="127.0.0.1",
#     port=3306,
#     user="root",
#     password="123456",
#     database="test_db",
#     charset="utf8mb4",
#     driver="mysql"      # 关键标识
# )

# 2) SQLite 示例
CONNECTION = dict(
    database="test.db",   # 数据库文件路径
    driver="sqlite"
)
# =====================

COLUMN_NAME = "model_assign_name"
COLUMN_TYPE = "VARCHAR(100)"
INDEX_NAME  = f"idx_llmusage_{COLUMN_NAME}"
TABLE_NAME  = "llm_usage"


# --------------------------------------------------
# MySQL 实现
# --------------------------------------------------
def _mysql_ensure(cfg: dict) -> None:
    with closing(pymysql.connect(**{k: v for k, v in cfg.items() if k != "driver"})) as conn, closing(conn.cursor()) as cur:
        # 1. 列是否存在
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.COLUMNS
            WHERE table_schema = %s
              AND table_name   = %s
              AND column_name  = %s
        """, (cfg["database"], TABLE_NAME, COLUMN_NAME))
        col_exists = cur.fetchone()[0] > 0

        # 2. 索引是否存在
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.STATISTICS
            WHERE table_schema = %s
              AND table_name   = %s
              AND index_name   = %s
        """, (cfg["database"], TABLE_NAME, INDEX_NAME))
        idx_exists = cur.fetchone()[0] > 0

        if not col_exists:
            sql = f"ALTER TABLE {TABLE_NAME} ADD COLUMN {COLUMN_NAME} {COLUMN_TYPE}"
            print(f"[MySQL DDL] {sql}")
            cur.execute(sql)

        if not idx_exists:
            sql = f"CREATE INDEX {INDEX_NAME} ON {TABLE_NAME} ({COLUMN_NAME})"
            print(f"[MySQL DDL] {sql}")
            cur.execute(sql)

        conn.commit()


# --------------------------------------------------
# SQLite 实现
# --------------------------------------------------
def _sqlite_ensure(cfg: dict) -> None:
    db = cfg["database"]
    with closing(sqlite3.connect(db)) as conn, closing(conn.cursor()) as cur:
        # 1. 列是否存在
        cur.execute("PRAGMA table_info({})".format(TABLE_NAME))
        cols = {row[1] for row in cur.fetchall()}
        col_exists = COLUMN_NAME in cols

        # 2. 索引是否存在
        cur.execute("PRAGMA index_list({})".format(TABLE_NAME))
        idxs = {row[1] for row in cur.fetchall()}
        idx_exists = INDEX_NAME in idxs

        # SQLite ≥3.35 支持 ALTER TABLE ADD COLUMN IF NOT EXISTS
        if not col_exists:
            sql = f"ALTER TABLE {TABLE_NAME} ADD COLUMN {COLUMN_NAME} {COLUMN_TYPE}"
            print(f"[SQLite DDL] {sql}")
            cur.execute(sql)

        if not idx_exists:
            sql = f"CREATE INDEX {INDEX_NAME} ON {TABLE_NAME} ({COLUMN_NAME})"
            print(f"[SQLite DDL] {sql}")
            cur.execute(sql)

        conn.commit()


# --------------------------------------------------
# 调度器
# --------------------------------------------------
def ensure_column_and_index(cfg: dict) -> None:
    driver = cfg.get("driver", "").lower()
    if driver == "mysql":
        _mysql_ensure(cfg)
    elif driver == "sqlite":
        _sqlite_ensure(cfg)
    else:
        raise ValueError("connection.driver 必须是 'mysql' 或 'sqlite'")
    print("✅ 完成。")


if __name__ == "__main__":
    ensure_column_and_index(CONNECTION)