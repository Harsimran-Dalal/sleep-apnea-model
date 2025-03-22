import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("respiration_data.csv")  # Replace with actual file name

# Handle missing values
data.interpolate(method='linear', inplace=True)

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Segment data into windows
def segment_data(data, window_size, step_size):
    segments = []
    for i in range(0, len(data) - window_size, step_size):
        segments.append(data[i:i+window_size])
    return segments

window_size = 100
step_size = 50
segments = segment_data(data_scaled, window_size, step_size)

# Save processed data
pd.DataFrame(segments).to_csv("processed_data.csv", index=False)
