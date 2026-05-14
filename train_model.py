import sqlite3
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

DB_PATH = "../data/my_database.db"
TRAIN_TABLES = [f"subject_{i}" for i in range(1, 25)]
TEST_TABLE = "subject_25"

LABEL_COL = "sign_name"


FEATURE_COLS = [
    "flex_1", "flex_2", "flex_3", "flex_4", "flex_5",
    "GYRx", "GYRy", "GYRz",
    "ACCx_body", "ACCy_body", "ACCz_body"
]

def load_data(tables):
    X_list, y_list = [], []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for table in tables:
        print(f"Loading {table}...")
        query = f"""
        SELECT {LABEL_COL}, {', '.join(FEATURE_COLS)}
        FROM {table}
        WHERE {LABEL_COL} IS NOT NULL
        """
        rows = cursor.execute(query).fetchall()

        for row in rows:
            label = row[0]
            #if label in EXCLUDED_SIGNS:
             #   continue
            X_list.append(row[1:])
            y_list.append(label)
    conn.close()
    return np.array(X_list, dtype=float), np.array(y_list)

X_train, y_train = load_data(TRAIN_TABLES)
X_test, y_test = load_data([TEST_TABLE])

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def reshape_sequences(X, y, window_size=150):
    num_samples = len(X) // window_size
    X_seq = X[:num_samples * window_size].reshape(num_samples, window_size, X.shape[1])
    y_seq = y[:num_samples * window_size].reshape(num_samples, window_size)
    # assign the label of the majority row in each sequence
    y_labels = np.array([np.bincount([int(x) for x in seq]).argmax() for seq in y_seq])
    return X_seq, y_labels

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

X_train_seq, y_train_seq = reshape_sequences(X_train, y_train_encoded)
X_test_seq, y_test_seq = reshape_sequences(X_test, y_test_encoded)

num_signs = len(le.classes_)
y_train_onehot = to_categorical(y_train_seq, num_classes=num_signs)
y_test_onehot = to_categorical(y_test_seq, num_classes=num_signs)

print("X_train_seq shape:", X_train_seq.shape)
print("y_train_onehot shape:", y_train_onehot.shape)

model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(150, X_train_seq.shape[2])),
    Conv1D(128, kernel_size=5, activation='relu'),
    LSTM(256),
    Dense(64, activation='relu'),
    Dense(num_signs, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(
    X_train_seq, y_train_onehot,
    validation_data=(X_test_seq, y_test_onehot),
    epochs=60,
    batch_size=32
)

y_pred_onehot = model.predict(X_test_seq)
y_pred_labels = y_pred_onehot.argmax(axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test_seq, y_pred_labels, target_names=le.classes_))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test_seq, y_pred_labels))

model.save("cnn_lstm_model.h5")
joblib.dump(scaler, "../scaler.pkl")
joblib.dump(le, "../label_encoder.pkl")

print("\nCNN+LSTM model, scaler, and label encoder saved")