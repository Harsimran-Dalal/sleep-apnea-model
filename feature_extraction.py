import tsfel
import pandas as pd

# Load segmented data
data = pd.read_csv("processed_data.csv")

# Load TSFEL configuration
cfg = tsfel.get_features_by_domain()

# Extract features
features = tsfel.time_series_features_extractor(cfg, data)

# Save extracted features
features.to_csv("extracted_features.csv", index=False)
