import os
import sqlite3
import numpy as np
import joblib
import pandas as pd
from dotenv import load_dotenv

from scipy.signal import medfilt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, LSTM, Dense, Input)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

load_dotenv()

DB_PATH= os.getenv("DB_PATH")
WINDOW_SIZE= 160
MODEL_PATH= os.getenv("MODEL_PATH")
SCALER_PATH= os.getenv("SCALER_PATH")
LABEL_ENCODER_PATH= os.getenv("LABEL_ENCODER_PATH")

FEATURE_COLS= ["flex_1", "flex_2", "flex_3", "flex_4", "flex_5", "GYRx", "GYRy", "GYRz", "ACCx_body", "ACCy_body", "ACCz_body"]

#Median filter removes isolated sensor spikes
#3 means that every value is compared with its two neighbouring values
MEDIAN_FILTER_SIZE= 3

#Set to true for additional smoothing if raw data is noisy, for now I kept as false.
USE_MOVING_AVERAGE= False
MOVING_AVERAGE_WINDOW= 3

def load_recordings():
    conn= sqlite3.connect(DB_PATH)

    query = f"""
        SELECT
            recording_id,
            sign_name,
            timestamp,
            {', '.join(FEATURE_COLS)}
        FROM sensor_data
        ORDER BY
            recording_id,
            timestamp
    """

    rows= conn.execute(query).fetchall()
    conn.close()
    recordings = {}

    for row in rows:
        recording_id= row[0]
        sign_name= row[1]
        features= np.array(row[3:], dtype=float)
        if recording_id not in recordings:
            recordings[recording_id] = {
                "label": sign_name,
                "data": []
            }
        recordings[recording_id]["data"].append(features)
    return recordings

def preprocess_recording(data, recording_id=None):
    """
    1. Convert sensor values to numeric
    2. Replace infinity with NaN
    3. Interpolate missing values
    4. Remove exact duplicate rows (commented out for now because a lot of rows were deleted)
    5. Remove isolated sensor spikes using median filter
    6. Apply moving average (optionally)
    7. Final NaN protection
    There is no Min-Max scaling here, scaling is performed later after the train/validation/test split so that validation and test data cannot influence the scaler.
    """

    df= pd.DataFrame(data, columns=FEATURE_COLS)
    original_length= len(df)

    for col in FEATURE_COLS:

        df[col]= pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    missing_before= int(df[FEATURE_COLS].isna().sum().sum())

    if missing_before > 0:
        df[FEATURE_COLS]= (df[FEATURE_COLS].interpolate(method="linear", limit_direction="both"))

    duplicate_count= (len(df)- len(df.drop_duplicates(subset=FEATURE_COLS)))

    # df.drop_duplicates(
    #     subset=FEATURE_COLS,
    #     inplace=True
    # )

    #Median filter: this is safer for gesture recognition than replacing all Z-score outliers with the mean
    for col in FEATURE_COLS:
        values= df[col].to_numpy(dtype=float)
        #scipy median filter
        filtered_values= medfilt(values, kernel_size=MEDIAN_FILTER_SIZE)

        df[col]= filtered_values

    if USE_MOVING_AVERAGE:
        for col in FEATURE_COLS:
            df[col]= (df[col].rolling(window=MOVING_AVERAGE_WINDOW, center=True, min_periods=1).mean())

    df[FEATURE_COLS]= (df[FEATURE_COLS].ffill().bfill())

    remaining_missing = int(df[FEATURE_COLS].isna().sum().sum())

    if remaining_missing > 0:
        print(f"WARNING: recording {recording_id} still contains {remaining_missing} missing values.")

    return df[FEATURE_COLS].to_numpy(dtype=float)

def preprocess_all_recordings(recordings):
    print()
    print("PREPROCESSING")

    total_before= 0
    total_after= 0

    for recording_id in recordings:
        original_data= np.array(recordings[recording_id]["data"], dtype=float)

        total_before+= len(original_data)
        cleaned_data= preprocess_recording(original_data, recording_id)

        total_after+= len(cleaned_data)
        recordings[recording_id]["data"]= (cleaned_data.tolist())

    return recordings

def print_dataset_info(recordings):
    print()
    print("DATASET")

    labels= {}
    recording_lengths= []

    for recording in recordings.values():
        label= recording["label"]

        if label not in labels:
            labels[label]= 0

        labels[label]+= 1

        recording_lengths.append(len(recording["data"]))

    print()
    for label in sorted(labels):
        print(f"{label:<20} {labels[label]} recordings")

    print()
    print(f"Total recordings: {len(recordings)}")

    if recording_lengths:
        print(f"Average samples/recording: {np.mean(recording_lengths):.1f}")
        print(f"Minimum samples/recording: {np.min(recording_lengths)}")
        print(f"Maximum samples/recording: {np.max(recording_lengths)}")

def split_recordings(recordings):
    recording_ids= np.array(list(recordings.keys()))

    labels= np.array([
        recordings[rid]["label"]
        for rid in recording_ids
    ])

    #70% train
    #30% temporary
    train_ids, temp_ids= train_test_split(recording_ids, test_size=0.30, random_state=42, stratify=labels)

    temp_labels = np.array([
        recordings[rid]["label"]
        for rid in temp_ids
    ])

    #temp -> 15% validation + 15% test
    val_ids, test_ids= train_test_split(temp_ids, test_size=0.50, random_state=42, stratify=temp_labels)

    return train_ids, val_ids, test_ids

def create_windows(recordings, recording_ids, window_size):
    X= []
    y= []

    for recording_id in recording_ids:
        recording= recordings[recording_id]
        data= np.array(recording["data"], dtype=float)
        label= recording["label"]

        #Ignore recordings that are too short
        if len(data) < window_size:
            print(f"Skipping {recording_id}: {len(data)} samples")
            continue

        #Create non-overlapping windows
        num_windows= (len(data)// window_size)

        for i in range(num_windows):
            start= i * window_size
            end= start + window_size
            window= data[start:end]

            if len(window) == window_size:
                X.append(window)
                y.append(label)

    return (np.array(X, dtype=float), np.array(y))

def print_window_distribution(y, title):
    print()
    print(title)
    if len(y) == 0:
        print("No windows.")
        return

    unique, counts= np.unique(y, return_counts=True)

    for label, count in zip(unique, counts
    ):
        print(f"{label:<20} {count} windows")

print()
print("CNN + LSTM TRAINING")
print()
print("Loading database...")

recordings= load_recordings()
if len(recordings) == 0:
    raise RuntimeError("No recordings found in database.")

print_dataset_info(recordings)

recordings= preprocess_all_recordings(recordings)

print_dataset_info(recordings)

train_ids, val_ids, test_ids= split_recordings(recordings)

print()
print("RECORDING SPLIT")

print(f"Training recordings: {len(train_ids)}")
print(f"Validation recordings: {len(val_ids)}")
print(f"Test recordings: {len(test_ids)}")

print()
print("Creating training sequences...")

X_train, y_train = create_windows(recordings, train_ids, WINDOW_SIZE)

print("Creating validation sequences...")

X_val, y_val= create_windows(recordings, val_ids, WINDOW_SIZE)

print("Creating test sequences...")

X_test, y_test = create_windows(recordings, test_ids, WINDOW_SIZE)

if len(X_train) == 0:
    raise RuntimeError("No training windows were created.")

if len(X_val) == 0:
    raise RuntimeError("No validation windows were created.")

if len(X_test) == 0:
    raise RuntimeError("No test windows were created.")

print()
print("Encoding labels...")

le= LabelEncoder()

all_labels= np.concatenate([y_train, y_val, y_test])

le.fit(all_labels)

y_train_encoded= le.transform(y_train)
y_val_encoded= le.transform(y_val)
y_test_encoded= le.transform(y_test)

print()
print("Signs:")

for i, sign in enumerate(le.classes_):
    print(f"{i}: {sign}")

print()
print("NORMALIZATION")

scaler= MinMaxScaler()
num_features= X_train.shape[2]

#Flatten time dimension
X_train_flat= X_train.reshape(-1, num_features)

X_val_flat= X_val.reshape(-1, num_features)
X_test_flat= X_test.reshape(-1, num_features)

scaler.fit(X_train_flat)

X_train= scaler.transform(X_train_flat).reshape(X_train.shape)
X_val= scaler.transform(X_val_flat).reshape(X_val.shape)
X_test= scaler.transform(X_test_flat).reshape(X_test.shape)

print("Normalization complete.")

#ONE-HOT ENCODING
num_signs= len(le.classes_)

y_train_onehot= to_categorical(y_train_encoded, num_classes=num_signs)
y_val_onehot= to_categorical(y_val_encoded, num_classes=num_signs)
y_test_onehot= to_categorical(y_test_encoded, num_classes=num_signs)

#BUILD CNN + LSTM
print()
print("BUILDING CNN + LSTM")

model= Sequential([
    Input(
        shape=(
            WINDOW_SIZE,
            num_features
        )
    ),
    Conv1D(
        filters=64,
        kernel_size=5,
        activation="relu"
    ),
    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    ),
    LSTM(
        256
    ),
    Dense(
        64,
        activation="relu"
    ),
    Dense(
        num_signs,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print()
model.summary()

#train
print()
print("TRAINING")

early_stopping= EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history= model.fit(
    X_train,
    y_train_onehot,
    validation_data=(
        X_val,
        y_val_onehot
    ),
    epochs=30,
    batch_size=32,
    shuffle=True,
    callbacks=[
        early_stopping
    ]
)

print()
print("FINAL TEST RESULTS")

y_pred= model.predict(X_test, verbose=0)

y_pred_labels= y_pred.argmax(axis=1)

print()
print("Classification Report:")
print()

print(
    classification_report(
        y_test_encoded,
        y_pred_labels,
        labels=np.arange(
            num_signs
        ),
        target_names=le.classes_,
        zero_division=0
    )
)

print("Confusion Matrix:")
print()

print(
    confusion_matrix(
        y_test_encoded,
        y_pred_labels,
        labels=np.arange(
            num_signs
        )
    )
)

model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(le, LABEL_ENCODER_PATH)