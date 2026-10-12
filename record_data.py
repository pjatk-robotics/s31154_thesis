import serial
import time
import sqlite3
import uuid
import os
import threading
import sensor_parsers
from dotenv import load_dotenv

load_dotenv()

FLEX_PORT = "COM3"
IMU_PORT = "COM7"

BAUD= 115200
RECORDING_SECONDS= 2
COUNTDOWN_SECONDS= 0

DB_PATH= os.getenv("DB_PATH")

#allows a mismatch of +- 5 samples
MAX_SENSOR_AGE= 0.05

latest_flex= None
latest_flex_time= None

latest_imu= None
latest_imu_time= None

data_lock= threading.Lock()

recording_active= False

def create_database():
    db_directory= os.path.dirname(DB_PATH)

    if db_directory:
        os.makedirs(db_directory, exist_ok=True)

    conn= sqlite3.connect(DB_PATH)
    cursor= conn.cursor()

    cursor.execute("""CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id TEXT NOT NULL,
            sign_name TEXT NOT NULL,
            timestamp REAL NOT NULL,
            flex_1 REAL NOT NULL,
            flex_2 REAL NOT NULL,
            flex_3 REAL NOT NULL,
            flex_4 REAL NOT NULL,
            flex_5 REAL NOT NULL,
            GYRx REAL NOT NULL,
            GYRy REAL NOT NULL,
            GYRz REAL NOT NULL,
            ACCx_body REAL NOT NULL,
            ACCy_body REAL NOT NULL,
            ACCz_body REAL NOT NULL)""")

    conn.commit()
    conn.close()

def read_flex(ser):
    global latest_flex
    global latest_flex_time

    while True:
        try:
            line= (ser.readline().decode(errors="ignore").strip())
            flex= sensor_parsers.parse_flex(line)

            if flex is None:
                continue

            current_time = time.perf_counter()

            with data_lock:
                latest_flex= flex
                latest_flex_time= current_time

        except Exception as e:
            print("Flex serial error:", e)
            break

def read_imu(ser):
    global latest_imu
    global latest_imu_time

    while True:
        try:
            line= (ser.readline().decode(errors="ignore").strip())

            imu= sensor_parsers.parse_imu(line)

            if imu is None:
                continue

            current_time= time.perf_counter()

            with data_lock:
                latest_imu= imu
                latest_imu_time= current_time

        except Exception as e:
            print("IMU serial error:", e)
            break

def get_combined_sample():
    with data_lock:
        if latest_flex is None:
            return None
        if latest_imu is None:
            return None
        if latest_flex_time is None:
            return None
        if latest_imu_time is None:
            return None

        age= latest_imu_time- latest_flex_time

        #If flex is too old, don't create a sample!
        if abs(age) > MAX_SENSOR_AGE:
            return None

        return {
            "flex": latest_flex.copy(),
            "gyro": latest_imu["gyro"].copy(),
            "acc_body": latest_imu["acc_body"].copy()
        }

def save_row(cursor, recording_id, sign_name, timestamp, data):
    flex= data["flex"]
    gyro= data["gyro"]
    acc= data["acc_body"]

    cursor.execute("""INSERT INTO sensor_data (
            recording_id,
            sign_name,
            timestamp,

            flex_1,
            flex_2,
            flex_3,
            flex_4,
            flex_5,

            GYRx,
            GYRy,
            GYRz,

            ACCx_body,
            ACCy_body,
            ACCz_body
        )

        VALUES (
            ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?
        )""", (

        recording_id,
        sign_name,
        timestamp,

        flex[0],
        flex[1],
        flex[2],
        flex[3],
        flex[4],

        gyro[0],
        gyro[1],
        gyro[2],

        acc[0],
        acc[1],
        acc[2]
    ))

def countdown():
    print()
    print("RECORDING")
    print()

def record_gesture(flex_ser, imu_ser, sign_name):
    global recording_active
    recording_id = str(uuid.uuid4())

    print()
    print(f"Sign: {sign_name}")
    print(f"Duration: {RECORDING_SECONDS} seconds")
    print("RECORD NOW")
    print()

    #Clear old sensor values before recording
    global latest_flex
    global latest_flex_time
    global latest_imu
    global latest_imu_time

    with data_lock:
        latest_flex= None
        latest_flex_time= None

        latest_imu= None
        latest_imu_time= None

    time.sleep(0.1)
    conn= sqlite3.connect(DB_PATH)
    cursor= conn.cursor()
    recording_active= True
    start_time= time.perf_counter()
    sample_count= 0
    skipped_count= 0
    last_imu_time= None

    while True:
        current_time= time.perf_counter()
        elapsed= current_time- start_time
        if elapsed >= RECORDING_SECONDS:
            break

        with data_lock:
            imu_time= latest_imu_time

        #We only create new database row when a NEW IMU reading has arrived.
        if imu_time is None:
            time.sleep(0.001)
            continue

        if imu_time == last_imu_time:
            time.sleep(0.001)
            continue

        last_imu_time = imu_time
        data = get_combined_sample()

        if data is None:
            skipped_count += 1
            continue

        timestamp= current_time- start_time

        save_row(cursor, recording_id, sign_name, timestamp, data)
        sample_count += 1

    recording_active= False

    conn.commit()
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM sensor_data
        WHERE recording_id = ?
        """,
        (recording_id,)
    )

    saved_count= cursor.fetchone()[0]

    print()
    print(f"Rows actually in database: {saved_count}")
    print(f"Database file: {DB_PATH}")

    conn.close()
    actual_duration= time.perf_counter()- start_time

    print()
    print("RECORDING COMPLETE")
    print()
    print(f"Sign: {sign_name}")
    print(f"Recording: {recording_id}")
    print(f"Samples saved: {sample_count}")
    print(f"Samples skipped: {skipped_count}")
    print(f"Duration: {actual_duration:.2f} seconds")
    print()

    return recording_id

def main():
    print("DATA COLLECTOR")
    create_database()

    print()
    print(f"Database: {DB_PATH}")
    print()

    print(f"Connecting to Arduino Nano ({FLEX_PORT})...")

    try:
        flex_ser= serial.Serial(FLEX_PORT, BAUD, timeout=1)

    except serial.SerialException as e:
        print()
        print("Could not connect to Arduino Nano.")
        print(e)
        return

    print(f"Connecting to Arduino UNO ({IMU_PORT})...")

    try:
        imu_ser= serial.Serial(IMU_PORT, BAUD, timeout=1)

    except serial.SerialException as e:
        print()
        print("Could not connect to IMU Arduino.")
        print(e)
        flex_ser.close()
        return

    time.sleep(2)

    print()
    print("Both Arduinos connected!")
    print(f"Nano: {FLEX_PORT}")
    print(f"Uno:  {IMU_PORT}")
    print()

    threading.Thread(target=read_flex,  args=(flex_ser,), daemon=True).start()

    threading.Thread(target=read_imu, args=(imu_ser,), daemon=True).start()

    print("Sensor readers started.")
    print()

    try:
        while True:
            sign_name= input("Enter sign name (or 'q' to quit): ").strip()

            if sign_name.lower() == "q":
                break

            if not sign_name:
                print("Sign name cannot be empty.")
                continue

            print()
            print(f"CURRENT SIGN: {sign_name}")
            print()
            print("Press ENTER to record a 2-second gesture.")
            print()

            while True:
                command= input(f"[{sign_name}] ENTER = record, Q = change sign: ").strip().lower()

                if command == "q":
                    print()
                    print(f"Finished recordings for '{sign_name}'.")
                    break

                if command != "":
                    print("Please press ENTER to record or Q to change sign.")
                    continue

                record_gesture(flex_ser, imu_ser, sign_name)

                print()
                print(f"Ready for another '{sign_name}' recording.")
                print()

    except KeyboardInterrupt:
        print()
        print("Stopping...")

    finally:
        flex_ser.close()
        imu_ser.close()

        print()
        print("Arduinos disconnected.")
        print("Data collection finished.")

if __name__ == "__main__":
    main()