import pandas as pd
import sqlite3
import os
# --- Configuration ---
CSV_PATH = 'music.csv'  # <-- Your new custom CSV
DB_PATH = 'music.db'
TABLE_NAME = 'songs'

def convert_custom_csv_to_sqlite(csv_file, db_file, table_name):
    """
    Reads a simple custom CSV with columns: name, emotion, filepath
    and creates the SQLite database.
    """
    print(f"Reading custom CSV from: {csv_file}")
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: CSV file not found at {csv_file}")
        return

    try:
        # 1. Read the CSV
        df = pd.read_csv(csv_file)
        
        # 2. Rename columns to match what the app expects
        df.rename(columns={
            'emotion': 'emotion_category',
            'filepath': 'link'
        }, inplace=True)

        print(f"Read {len(df)} songs.")
        print(df.head())

        # 3. Write to SQLite
        conn = sqlite3.connect(db_file)
        print(f"Writing to database: {db_file}...")
        
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Create index for speed
        cursor = conn.cursor()
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_emotion ON {table_name} (emotion_category);")
        conn.commit()
        conn.close()
        
        print("✅ Database created successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    convert_custom_csv_to_sqlite(CSV_PATH, DB_PATH, TABLE_NAME)