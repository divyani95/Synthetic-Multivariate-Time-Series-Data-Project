import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

def generate_synthetic_data(duration_minutes=1440, sampling_rate_hz=10, target_size_gb=1):
    """
    Generate a 1 GB synthetic multivariate time series dataset
    simulating hardware monitoring metrics.
    """
    np.random.seed(42)

    # Calculate the number of samples needed for a 1 GB dataset
    bytes_per_sample = 64  # Assuming 8 columns x 8 bytes per float
    target_num_samples = int(target_size_gb * 1024 ** 3 / bytes_per_sample)
    sampling_frequency = duration_minutes * 60 * sampling_rate_hz
    scale_factor = target_num_samples / sampling_frequency

    # Generate base data with more complex periodic and random variations
    timestamps = pd.date_range(start='now', periods=target_num_samples, freq=f'{1000/sampling_rate_hz}ms')
    
    cpu_temperature = (50 + 15 * np.sin(np.linspace(0, 240, target_num_samples) * scale_factor)) + np.random.normal(0, 7, target_num_samples)
    cpu_usage = (30 + 25 * np.sin(np.linspace(0, 240, target_num_samples) * scale_factor)) + np.random.normal(0, 12, target_num_samples)
    cpu_load = (2 + 1.5 * np.sin(np.linspace(0, 240, target_num_samples) * scale_factor)) + np.random.normal(0, 1.2, target_num_samples)
    memory_usage = (50 + 25 * np.sin(np.linspace(0, 240, target_num_samples) * scale_factor)) + np.random.normal(0, 12, target_num_samples)
    battery_level = (80 + 15 * np.sin(np.linspace(0, 240, target_num_samples) * scale_factor)) + np.random.normal(0, 7, target_num_samples)
    cpu_power = (40 + 20 * np.sin(np.linspace(0, 240, target_num_samples) * scale_factor)) + np.random.normal(0, 10, target_num_samples)

    # Introduce artificial anomalies
    anomaly_indices = np.random.choice(target_num_samples, size=int(target_num_samples * 0.05), replace=False)
    
    cpu_temperature[anomaly_indices[:len(anomaly_indices)//4]] += 55  # High temperature spikes
    cpu_usage[anomaly_indices[len(anomaly_indices)//4:2*len(anomaly_indices)//4]] += 75  # Extreme usage
    memory_usage[anomaly_indices[2*len(anomaly_indices)//4:3*len(anomaly_indices)//4]] += 50  # High memory consumption
    battery_level[anomaly_indices[3*len(anomaly_indices)//4:]] -= 45  # Low battery level

    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_temperature': cpu_temperature,
        'cpu_usage': cpu_usage,
        'cpu_load': cpu_load,
        'memory_usage': memory_usage,
        'battery_level': battery_level,
        'cpu_power': cpu_power
    })

    return df

def detect_anomalies(df):
    """
    Apply multiple anomaly detection techniques
    """
    features = ['cpu_temperature', 'cpu_usage', 'cpu_load', 
                'memory_usage', 'battery_level', 'cpu_power']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Z-Score Method
    z_scores = np.abs((X - X.mean()) / X.std())
    z_score_anomalies = (z_scores > 3).any(axis=1)

    # 2. Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest_labels = iso_forest.fit_predict(X_scaled)
    iso_forest_anomalies = iso_forest_labels == -1

    # 3. Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_labels = lof.fit_predict(X_scaled)
    lof_anomalies = lof_labels == -1

    df['z_score_anomaly'] = z_score_anomalies
    df['isolation_forest_anomaly'] = iso_forest_anomalies
    df['lof_anomaly'] = lof_anomalies

    return df

def visualize_anomalies(df):
    """
    Create well-organized visualizations of detected anomalies
    """
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    features = ['cpu_temperature', 'cpu_usage', 'cpu_load', 
                'memory_usage', 'battery_level', 'cpu_power']
    
    for i, feature in enumerate(features):
        row = i // 2
        col = i % 2
        ax = axs[row, col]
        ax.plot(df['timestamp'], df[feature], label='Normal Data', alpha=0.7)
        for method, color in [
            ('z_score_anomaly', 'red'), 
            ('isolation_forest_anomaly', 'green'), 
            ('lof_anomaly', 'purple')
        ]:
            anomalies = df[df[method]]
            ax.scatter(
                anomalies['timestamp'], 
                anomalies[feature], 
                color=color, 
                label=f'{method.replace("_", " ").title()}', 
                alpha=0.7
            )
        ax.set_title(f'{feature.replace("_", " ").title()} Anomalies')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Value')
        ax.legend()
    plt.tight_layout()
    plt.savefig('Figure1.png')
    print('Visualization saved as Figure1.png')
def main():
    # Generating data more than 1 GB
    synthetic_data = generate_synthetic_data()
    anomaly_data = detect_anomalies(synthetic_data)
    anomaly_data.to_csv('synthetic_hardware_monitoring_data.csv', index=False)
    print("Dataset saved successfully")
    visualize_anomalies(anomaly_data)
if __name__ == "__main__":
    main()