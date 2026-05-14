import sqlite3
import json
import pandas as pd
import os
import sqlite3
import json
import sys

try:
    with open("config.json") as f:
        config = json.load(f)
    database_path = config["database_path"]

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    print(f"Connected to database at {database_path}")

except sqlite3.Error as db_err:
    print(f"Database connection error: {db_err}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

data_folder = config.get("data_folder_path", "")


for subject_id in range(1, 26):
    subject_folder = os.path.join(data_folder, f"{subject_id:03d}")

    if not os.path.exists(subject_folder):
        print(f"Warning: {subject_folder} does not exist")
        continue

    table_name = f"subject_{subject_id}"

    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    connection.commit()

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        sign_name TEXT,
        timestamp REAL,
        user_id INTEGER,
        flex_1 REAL,
        flex_2 REAL,
        flex_3 REAL,
        flex_4 REAL,
        flex_5 REAL,
        Qw REAL,
        Qx REAL,
        Qy REAL,
        Qz REAL,
        GYRx REAL,
        GYRy REAL,
        GYRz REAL,
        ACCx REAL,
        ACCy REAL,
        ACCz REAL,
        ACCx_body REAL,
        ACCy_body REAL,
        ACCz_body REAL,
        ACCx_world REAL,
        ACCy_world REAL,
        ACCz_world REAL
    )
    """
    cursor.execute(create_table_query)

    # so that data doesn't load twice
    cursor.execute(f"DELETE FROM {table_name}")
    connection.commit()

    for sign_csv in os.listdir(subject_folder):

        sign_name = os.path.splitext(sign_csv)[0]
        csv_file = subject_folder+ "/"+str(sign_csv)
        df = pd.read_csv(csv_file)

        for _,row in df.iterrows():
            cursor.execute(f"""
                INSERT INTO {table_name} (
                    sign_name, timestamp, user_id, flex_1, flex_2, flex_3, flex_4, flex_5,
                    Qw, Qx, Qy, Qz,
                    GYRx, GYRy, GYRz,
                    ACCx, ACCy, ACCz,
                    ACCx_body, ACCy_body, ACCz_body,
                    ACCx_world, ACCy_world, ACCz_world
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)
            """, tuple([sign_name] + [row[col] for col in df.columns]))

        connection.commit()
        print(f"Data for subject {subject_id}, sign {sign_name} was loaded")


cursor.close()
connection.close()