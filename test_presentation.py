from pybaseball import statcast_batter, playerid_lookup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import random

print("Gathering Player Data...")

# Step 1: Get Shohei Ohtani's MLBAM ID
player_info = playerid_lookup('Judge', 'Aaron')
player_id = player_info['key_mlbam'].values[0]

# Step 2: Get statcast data for 2023 season
data = statcast_batter('2023-03-30', '2023-10-01', player_id=player_id)

# Step 3: Add 'opponent' column based on home/away
data['opponent'] = data.apply(
    lambda row: row['away_team'] if row['home_team'] == 'LAA' else row['home_team'], axis=1
)

# Step 4: Filter to only games against the Padres
data = data[data['opponent'] == 'BOS']

# Step 5: Map statcast 'events' to simplified outcome classes
def map_event(e):
    if e in ['single', 'double', 'triple']:
        return 'base_hit'
    elif e == 'home_run':
        return 'home_run'
    elif e in ['walk', 'intent_walk', 'hit_by_pitch']:
        return 'walk'
    else:
        return 'out'

data['outcome'] = data['events'].apply(map_event)

# Step 6: Select features and drop rows with missing values
features = ['launch_speed', 'launch_angle', 'plate_x', 'plate_z']
data = data.dropna(subset=features + ['outcome'])
X = data[features]
y = data['outcome']

# Debug: Show label breakdown
print("Unique outcomes in filtered dataset:", y.value_counts())

# Exit early if no data
if data.empty:
    print("No data found for this player vs. the specified team. Try a different matchup.")
    exit()

# Step 7: Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # int labels
y_categorical = to_categorical(y_encoded)   # one-hot

# Step 8: Train/test split + scale
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Determine number of classes dynamically
num_classes = y_categorical.shape[1]

# Step 10: Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')  # now matches actual class count
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 11: Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Step 12: Evaluate model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Accuracy (vs Padres): {accuracy:.4f}")

# Step 13: Predict on a random test example
sample_idx = random.randint(0, len(X_test_scaled) - 1)
sample_input = X_test_scaled[sample_idx:sample_idx+1]
prediction = model.predict(sample_input)[0]

# Step 14: Show prediction breakdown
print("\nPredicted outcome probabilities:")
for i, prob in enumerate(prediction):
    label = label_encoder.inverse_transform([i])[0]
    print(f"{label}: {prob:.2%}")
