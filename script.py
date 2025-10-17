import os
import re
import requests
import psycopg2
import time
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# --- Configuration ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
APP_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL")
DB_CONNECTION_STRING = os.getenv("DATABASE_URI")


# ---------- HELPER FUNCTIONS ----------

# FIX: Defined standard date formats for consistency
DATE_FORMATS = ["%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d"]
DATETIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M",
]

def _parse_datetime(value: str, formats: list):
    """Helper to attempt parsing a string using a list of format codes."""
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
    return None

# FIX: Rewritten detection function to be more robust and use the helper
def detect_date_or_timestamp(value: str):
    """Detect if a string is a timestamp or a date, checking timestamp first for specificity."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None

    # Check for more specific timestamp formats first
    if _parse_datetime(s, DATETIME_FORMATS):
        return "TIMESTAMP"
    # Then check for simpler date formats
    if _parse_datetime(s, DATE_FORMATS):
        return "DATE"
    
    return None

# FIX: Rewritten normalization function to correctly convert dates
def normalize_datetime_value(value: str, col_type: str):
    """Normalize a date/timestamp string to the required ISO 8601 format for PostgreSQL."""
    if not isinstance(value, str):
        return value
    
    s = value.strip()
    dt_object = None

    if col_type == "TIMESTAMP":
        dt_object = _parse_datetime(s, DATETIME_FORMATS)
    elif col_type == "DATE":
        dt_object = _parse_datetime(s, DATE_FORMATS)
    
    if dt_object:
        if col_type == "DATE":
            # Format as YYYY-MM-DD, which PostgreSQL understands
            return dt_object.strftime("%Y-%m-%d")
        else:
            # Format as full ISO string for timestamps
            return dt_object.isoformat()
            
    # Fallback if parsing fails (should not happen if detection worked)
    return value

def get_value_type_level(value):
    """Determines the type level: 0=BIGINT, 1=FLOAT8, 2=DATE, 3=TIMESTAMP, 4=TEXT"""
    if value is None:
        return -1
    s = str(value).strip()
    if s == "":
        return -1

    if re.fullmatch(r"[-+]?\d+", s):
        return 0
    if re.fullmatch(r"[-+]?\d*\.\d+|\d+\.\d*", s):
        return 1

    detected = detect_date_or_timestamp(s)
    if detected == "DATE":
        return 2
    elif detected == "TIMESTAMP":
        return 3

    return 4

def get_column_types(rows):
    """Scan rows and determine final SQL type per column, safely handling mixed types."""
    if not rows:
        return {}
    
    headers = list(rows[0].keys())
    type_hierarchy = ["BIGINT", "FLOAT8", "DATE", "TIMESTAMP", "TEXT"]
    column_levels = {h: -1 for h in headers}

    for row in rows:
        for col_name, value in row.items():
            if col_name not in column_levels:
                continue

            value_level = get_value_type_level(value)
            if value_level == -1:
                continue

            current_level = column_levels[col_name]
            if current_level == -1:
                column_levels[col_name] = value_level
                continue

            if current_level == 4:
                continue
            
            if value_level == 4:
                column_levels[col_name] = 4
                continue

            is_current_numeric = current_level in (0, 1)
            is_value_numeric = value_level in (0, 1)
            is_current_datetime = current_level in (2, 3)
            is_value_datetime = value_level in (2, 3)

            if (is_current_numeric and is_value_datetime) or \
               (is_current_datetime and is_value_numeric):
                column_levels[col_name] = 4
            else:
                column_levels[col_name] = max(current_level, value_level)

    final_types = {}
    for h, level in column_levels.items():
        final_types[h] = type_hierarchy[level] if level != -1 else "TEXT"
        
    return final_types


# ---------- MAIN SYNC LOGIC (No changes needed here) ----------

def sync_to_db():
    """Fetch data from Apps Script and sync to PostgreSQL (Supabase)."""
    if not all([SUPABASE_URL, SUPABASE_KEY, APP_SCRIPT_URL, DB_CONNECTION_STRING]):
        print("❌ Error: Missing environment variables.")
        return

    try:
        print("🚀 Fetching data from Google Apps Script...")
        response = requests.get(APP_SCRIPT_URL)
        response.raise_for_status()
        sheets_data = response.json()
        print(f"✅ Fetched data for {len(sheets_data)} sheet(s).")

        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        conn = psycopg2.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return

    for table_name, rows in sheets_data.items():
        sanitized_table_name = "".join(c if c.isalnum() else '_' for c in table_name)
        print(f"\n--- Processing table: {sanitized_table_name} ---")

        if not rows:
            print("- Skipping empty sheet.")
            continue
            
        sanitized_rows = []
        for row in rows:
            sanitized_rows.append({
                "".join(c if c.isalnum() else '_' for c in key): val
                for key, val in row.items() if key
            })
            
        if not sanitized_rows:
            print("- Skipping sheet with no valid data rows.")
            continue

        column_definitions = get_column_types(sanitized_rows)

        try:
            cursor.execute(f'DROP TABLE IF EXISTS "{sanitized_table_name}";')
            
            cols_sql = ", ".join(
                f'"{col}" {sql_type}' for col, sql_type in column_definitions.items()
            )
            
            create_table_sql = f'CREATE TABLE "{sanitized_table_name}" (id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, {cols_sql});'
            cursor.execute(create_table_sql)
            conn.commit()
            cursor.execute("NOTIFY pgrst, 'reload schema';")
            
            print(f"✅ Table '{sanitized_table_name}' created.")
            
            time.sleep(2)

        except Exception as e:
            print(f"❌ Schema error for '{sanitized_table_name}': {e}")
            conn.rollback()
            continue

        try:
            clean_rows = []
            for row in sanitized_rows:
                clean_row = {}
                for key, val in row.items():
                    sql_type = column_definitions.get(key)
                    if val is None or str(val).strip() == "":
                        clean_row[key] = None
                    elif sql_type in ("DATE", "TIMESTAMP"):
                        clean_row[key] = normalize_datetime_value(str(val), sql_type)
                    else:
                        clean_row[key] = val
                clean_rows.append(clean_row)

            print(f"- Inserting {len(clean_rows)} rows...")
            supabase.table(sanitized_table_name).upsert(clean_rows).execute()
            print(f"✅ Inserted data for '{sanitized_table_name}'.")

        except Exception as e:
            print(f"❌ Insert error for '{sanitized_table_name}': {e}")
            continue

    cursor.close()
    conn.close()
    print("\n🎉 Sync complete!")


if __name__ == "__main__":
    sync_to_db()