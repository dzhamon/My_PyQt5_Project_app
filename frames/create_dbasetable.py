import sqlite3

from utils.config import SQL_PATH

conn = sqlite3.connect(SQL_PATH)
cur = conn.cursor()

param_kp = """CREATE TABLE IF NOT EXISTS data_kp ( lot_number TEXT, lot_status TEXT,
								discipline TEXT, project_name TEXT, open_date INTEGER,
								close_date INTEGER, actor_name TEXT, good_name TEXT,
								good_count REAL, unit TEXT, supplier_qty REAL,
								supplier_unit TEXT, winner_name TEXT, unit_price REAL,
								total_price REAL, currency TEXT )"""
cur.execute(param_kp)

param_db = """CREATE TABLE IF NOT EXISTS data_tmp (lot_number INTEGER, lot_status TEXT, discipline TEXT,
                                        project_name TEXT, open_date INTEGER, close_date INTEGER,
                                        actor_name TEXT, good_name TEXT, good_count REAL, unit TEXT,
                                        supplier_qty REAL, supplier_unit TEXT, winner_name TEXT,
                                        unit_price REAL, total_price REAL, currency TEXT )"""
cur.execute(param_db)

param_contract = """CREATE TABLE IF NOT EXISTS data_contract (lot_number TEXT, close_date INTEGER,
							contract_number TEXT, contract_date INTEGER, contract_maker TEXT,
							contract_keeper TEXT, good_name TEXT, supplier_unit TEXT, good_count REAL,
							unit TEXT, unit_price REAL, total_price REAL, add_expenses REAL,
							lottotal_price REAL, currency TEXT )"""
cur.execute(param_contract)

param_contr = """CREATE TABLE IF NOT EXISTS data_contr_tmp (lot_number TEXT, close_date INTEGER,
							contract_number TEXT, contract_date INTEGER, contract_maker TEXT,
							contract_keeper TEXT, good_name TEXT, supplier_unit TEXT, good_count REAL,
							unit TEXT, unit_price REAL, total_price REAL, add_expenses REAL,
							lottotal_price REAL, currency TEXT )"""
cur.execute(param_contr)
conn.close()