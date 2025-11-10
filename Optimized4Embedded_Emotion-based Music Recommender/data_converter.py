import pandas as pd
import sqlite3
import os
import numpy as np

# --- Configuration ---
CSV_PATH = 'muse_v3.csv'
DB_PATH = 'music.db'
TABLE_NAME = 'songs'

# --- Conversion Function ---
def convert_csv_to_sqlite(csv_file, db_file, table_name):
    """
    Reads relevant columns from a CSV, sorts them, adds emotion categories,
    and writes the data to an SQLite database table.
    """
    print(f"Reading CSV from: {csv_file}")
    if not os.path.exists(csv_file):
        print(f"ERROR: CSV file not found at {csv_file}")
        return

    try:
        # 1. Read and select columns using Pandas
        df = pd.read_csv(csv_file)
        df_selected = df[['track', 'lastfm_url', 'artist', 'number_of_emotion_tags', 'valence_tags']].copy()
        df_selected.rename(columns={
            'track': 'name',
            'lastfm_url': 'link',
            'number_of_emotion_tags': 'emotional',
            'valence_tags': 'pleasant'
        }, inplace=True)
        print(f"Read {len(df_selected)} rows from CSV.")

        # 2. Sort data (same as your app.py)
        df_sorted = df_selected.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)
        print("Data sorted.")

        # 3. Add an 'emotion' category column based on row index (approximating your slices)
        #    This is simpler than creating separate tables and efficient for querying.
        conditions = [
            (df_sorted.index < 18000),
            (df_sorted.index >= 18000) & (df_sorted.index < 36000),
            (df_sorted.index >= 36000) & (df_sorted.index < 54000),
            (df_sorted.index >= 54000) & (df_sorted.index < 72000),
            (df_sorted.index >= 72000)
        ]
        # Match emotion names used in your app.py's fun()
        emotions = ['Sad', 'Fearful', 'Angry', 'Neutral', 'Happy'] # Adjusted names
        df_sorted['emotion_category'] = np.select(conditions, emotions, default='Unknown')
        print("Emotion categories assigned.")

        # Keep only necessary columns for the database
        df_final = df_sorted[['name', 'link', 'artist', 'emotion_category']]

        # 4. Write to SQLite database
        print(f"Connecting to SQLite database: {db_file}")
        conn = sqlite3.connect(db_file)
        print(f"Writing data to table: '{table_name}'...")

        # Use 'replace' to overwrite the table if the script is run again
        df_final.to_sql(table_name, conn, if_exists='replace', index=False)

        # Optional: Create an index for faster querying by emotion
        print(f"Creating index on 'emotion_category' column...")
        cursor = conn.cursor()
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_emotion ON {table_name} (emotion_category);")
        conn.commit()

        conn.close()
        print(f"✅ Data successfully written to {db_file}")
        print(f"Database size: {os.path.getsize(db_file) / (1024*1024):.2f} MB")

    except Exception as e:
        print(f"❌ An error occurred during conversion: {e}")
        if 'conn' in locals() and conn:
            conn.close()

# --- Main execution ---
if __name__ == "__main__":
    convert_csv_to_sqlite(CSV_PATH, DB_PATH, TABLE_NAME)