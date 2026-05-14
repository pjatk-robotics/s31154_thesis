import serial
import time
import re
import threading
import numpy as np
from collections import deque
from calibration import align_to_dataset, flip_flex_values
from tensorflow.keras.models import load_model
import joblib

np.set_printoptions(suppress=True, precision=6)

PORT = "COM7"
BAUD = 115200

quat_regex = re.compile(r"Qw=([-0-9.]+)\s+Qx=([-0-9.]+)\s+Qy=([-0-9.]+)\s+Qz=([-0-9.]+)")
gyro_regex = re.compile(r"GYRx=([-0-9.]+)\s+GYRy=([-0-9.]+)\s+GYRz=([-0-9.]+)")
acc_regex = re.compile(r"ACCx=([-0-9.]+)\s+ACCy=([-0-9.]+)\s+ACCz=([-0-9.]+)")
acc_body_regex = re.compile(r"ACCx_body=([-0-9.]+)\s+ACCy_body=([-0-9.]+)\s+ACCz_body=([-0-9.]+)")
acc_world_regex = re.compile(r"ACCx_world=([-0-9.]+)\s+ACCy_world=([-0-9.]+)\s+ACCz_world=([-0-9.]+)")
flex_regex = re.compile(r"F1=(\d+)\s+F2=(\d+)\s+F3=(\d+)\s+F4=(\d+)\s+F5=(\d+)")

FEATURE_COLS = [
    "flex_1", "flex_2", "flex_3", "flex_4", "flex_5",
    "GYRx", "GYRy", "GYRz",
    "ACCx_body", "ACCy_body", "ACCz_body"
]

model = load_model("cnn_lstm_model.h5")
scaler = joblib.load("../scaler.pkl")
le = joblib.load("../label_encoder.pkl")

WINDOW_SIZE = 150
buffer = deque(maxlen=WINDOW_SIZE)

def parse_line(line: str):
    q = quat_regex.search(line)
    gyro = gyro_regex.search(line)
    acc = acc_regex.search(line)
    accb = acc_body_regex.search(line)
    accw = acc_world_regex.search(line)
    flex = flex_regex.search(line)

    if not (q and gyro and acc and accb and accw and flex):
        return None

    return {
        "Q": list(map(float, q.groups())),
        "GYR": list(map(float, gyro.groups())),
        "ACC": list(map(float, acc.groups())),
        "ACC_body": list(map(float, accb.groups())),
        "ACC_world": list(map(float, accw.groups())),
        "FLEX": flip_flex_values(list(map(int, flex.groups())))
    }

def extract_features(parsed):
    fv = align_to_dataset(parsed)
    return np.array(fv, dtype=float)

def read_serial(ser):
    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            parsed = parse_line(line)
            if parsed:
                #print(parsed)
                fv = extract_features(parsed)
                buffer.append(fv)

                if len(buffer) == WINDOW_SIZE:
                    X_window = np.array(buffer)
                    X_window_scaled = scaler.transform(X_window)
                    X_window_seq = X_window_scaled.reshape(1, WINDOW_SIZE, X_window_scaled.shape[1])
                    y_pred = model.predict(X_window_seq, verbose=0)
                    pred_label = le.inverse_transform([y_pred.argmax()])[0]
                    print(f"Predicted sign: {pred_label}")
                    buffer.clear()  # start new window
            else:
                print("Skipping line:", line)
        except Exception as e:
            print("Serial read error:", e)
            break

def write_commands(ser):
    while True:
        cmd = input("")
        ser.write((cmd + "\n").encode())

def main():
    ser = serial.Serial(PORT, BAUD)
    time.sleep(2)
    print("Connected")

    threading.Thread(target=read_serial, args=(ser,), daemon=True).start()
    write_commands(ser)

if __name__ == "__main__":
    main()