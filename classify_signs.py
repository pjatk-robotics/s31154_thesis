import os
import serial
import time
import threading
import numpy as np
from collections import deque
import sensor_parsers
from tensorflow.keras.models import load_model
import joblib
from dotenv import load_dotenv

load_dotenv()

FLEX_PORT= "COM3"
IMU_PORT= "COM7"
BAUD= 115200
WINDOW_SIZE= 160
FEATURE_COLS= ["flex_1", "flex_2", "flex_3", "flex_4", "flex_5", "GYRx", "GYRy", "GYRz", "ACCx_body", "ACCy_body", "ACCz_body"]
MODEL_PATH= os.getenv("MODEL_PATH")
SCALER_PATH= os.getenv("SCALER_PATH")
LABEL_ENCODER_PATH= os.getenv("LABEL_ENCODER_PATH")

model= load_model(MODEL_PATH)
scaler= joblib.load(SCALER_PATH)
le= joblib.load(LABEL_ENCODER_PATH)
buffer= deque(maxlen=WINDOW_SIZE)
latest_flex= None
data_lock= threading.Lock()

def create_combined_sample(imu_values):
    global latest_flex
    with data_lock:
        if latest_flex is None:
            return None

        combined= (latest_flex + imu_values)
        return np.array(combined, dtype=float)

def process_sample(features):
    buffer.append(features)
    if len(buffer) < WINDOW_SIZE:
        return

    X_window= np.array(buffer, dtype=float)

    #scale using scaler from training
    X_window_scaled= scaler.transform(X_window)

    #Add batch dimension!
    # (150, 11) --> (1, 150, 11)
    X_window_seq= X_window_scaled.reshape(1, WINDOW_SIZE, X_window_scaled.shape[1])

    y_pred= model.predict(X_window_seq, verbose=0)

    predicted_index= y_pred.argmax(axis=1)[0]
    predicted_label= (le.inverse_transform([predicted_index])[0])

    confidence= (y_pred[0][predicted_index]* 100)

    print(f"Predicted sign: {predicted_label} ({confidence:.1f}%)")
    buffer.clear()

def read_flex(ser):
    global latest_flex
    while True:
        try:
            line= (ser.readline().decode(errors="ignore").strip())
            flex= sensor_parsers.parse_flex(line)

            if flex is None:
                continue

            with data_lock:
                latest_flex= flex

        except Exception as e:
            print("Flex serial error:", e)
            break

def read_imu(ser):
    while True:
        try:
            line= (ser.readline().decode(errors="ignore").strip())
            imu= sensor_parsers.parse_imu(line)

            if imu is None:
                continue

            imu_values= imu["gyro"] + imu["acc_body"]
            features= create_combined_sample(imu_values)
            if features is None:
                continue
            process_sample(features)

        except Exception as e:
            print("IMU serial error:", e)
            break

def write_commands(ser):
    while True:
        try:
            cmd = input("")
            ser.write((cmd + "\n").encode())

        except Exception as e:
            print("Command error:", e)
            break


def main():
    print("Connecting to Arduino Nano...")

    flex_ser= serial.Serial(FLEX_PORT, BAUD, timeout=1)
    print("Connecting to IMU Arduino...")

    imu_ser= serial.Serial(IMU_PORT, BAUD, timeout=1)
    time.sleep(2)

    print()
    print("Both Arduinos connected!")
    print(f"Nano: {FLEX_PORT}")
    print(f"Uno: {IMU_PORT}")
    print()

    print("Available signs:")
    for i, sign in enumerate(le.classes_):
        print(f"{i}: {sign}")

    print()
    print(f"Waiting for {WINDOW_SIZE} samples before first prediction")
    print()

    #Flex thread
    threading.Thread(
        target=read_flex,
        args=(flex_ser,),
        daemon=True
    ).start()

    #IMU Thread
    threading.Thread(
        target=read_imu,
        args=(imu_ser,),
        daemon=True
    ).start()

    write_commands(flex_ser)

if __name__ == "__main__":
    main()