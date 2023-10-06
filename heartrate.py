import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Generate random heart rate data (for demonstration purposes)
np.random.seed(42)  # for reproducibility
num_samples = 1000
normal_heart_rate = np.random.normal(loc=70, scale=10, size=num_samples)  # normal heart rate data
anomaly_heart_rate = np.random.normal(loc=120, scale=15, size=20)  # anomalous heart rate data

# Combine normal and anomaly data
heart_rate_data = np.concatenate((normal_heart_rate, anomaly_heart_rate))

# Reshape the data for Isolation Forest input (column vector)
heart_rate_data = heart_rate_data.reshape(-1, 1)

# Create a DataFrame
data = pd.DataFrame(heart_rate_data, columns=['HeartRate'])

# Visualize the generated data
plt.figure(figsize=(8, 6))
plt.hist(data['HeartRate'], bins=50, density=True, alpha=0.6, color='g', label='Heart Rate Data')
plt.xlabel('Heart Rate')
plt.ylabel('Density')
plt.legend()
plt.title('Generated Heart Rate Data')
plt.show()

# Apply Isolation Forest algorithm for anomaly detection
outliers_fraction = 0.02  # assuming 2% of the data are anomalies (adjust as needed)
model = IsolationForest(contamination=outliers_fraction, random_state=42)
model.fit(data[['HeartRate']])

# Predict outliers
data['Anomaly'] = model.predict(data[['HeartRate']])
anomalies = data[data['Anomaly'] == -1]

# Visualize anomalies
plt.figure(figsize=(8, 6))
plt.hist(data['HeartRate'], bins=50, density=True, alpha=0.6, color='g', label='Heart Rate Data')
plt.scatter(anomalies['HeartRate'], np.zeros_like(anomalies['HeartRate']), color='r', label='Anomalies')
plt.xlabel('Heart Rate')
plt.ylabel('Density')
plt.legend()
plt.title('Detected Anomalies in Heart Rate Data')
plt.show()

# Display the detected anomalies
print("Detected Anomalies:")
print(anomalies)