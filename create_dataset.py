
import numpy as np
import pandas as pd

np.random.seed(42)

# Generate synthetic network traffic data
def generate_traffic(n, attack_type):

    if attack_type == "Normal":
        return pd.DataFrame({
            "duration": np.random.uniform(0, 5, n),
            "src_bytes": np.random.randint(100, 5000, n),
            "dst_bytes": np.random.randint(500, 10000, n),
            "count": np.random.randint(1, 20, n),
            "srv_count": np.random.randint(1, 15, n),
            "sload": np.random.uniform(100, 1000, n),
            "dload": np.random.uniform(100, 2000, n),
            "rate": np.random.uniform(1, 20, n),
            "protocol_type": np.random.choice(["TCP", "UDP"], n),
            "service": np.random.choice(["HTTP", "DNS", "FTP"], n),
            "label": "Normal"
        })

    elif attack_type == "DoS":
        return pd.DataFrame({
            "duration": np.random.uniform(0, 2, n),
            "src_bytes": np.random.randint(1000, 50000, n),
            "dst_bytes": np.random.randint(0, 1000, n),
            "count": np.random.randint(100, 1000, n),
            "srv_count": np.random.randint(100, 1000, n),
            "sload": np.random.uniform(5000, 50000, n),
            "dload": np.random.uniform(0, 500, n),
            "rate": np.random.uniform(100, 1000, n),
            "protocol_type": np.random.choice(["TCP", "UDP"], n),
            "service": np.random.choice(["HTTP", "DNS"], n),
            "label": "DoS"
        })

    else:
        return pd.DataFrame({
            "duration": np.random.uniform(1, 20, n),
            "src_bytes": np.random.randint(50, 2000, n),
            "dst_bytes": np.random.randint(50, 3000, n),
            "count": np.random.randint(20, 150, n),
            "srv_count": np.random.randint(10, 100, n),
            "sload": np.random.uniform(100, 3000, n),
            "dload": np.random.uniform(100, 4000, n),
            "rate": np.random.uniform(20, 150, n),
            "protocol_type": np.random.choice(["TCP", "ICMP"], n),
            "service": np.random.choice(["DNS", "FTP", "SSH"], n),
            "label": "Probe"
        })


# Generate 300 samples per class
normal = generate_traffic(300, "Normal")
dos = generate_traffic(300, "DoS")
probe = generate_traffic(300, "Probe")

# Combine datasets
df = pd.concat(
    [normal, dos, probe],
    ignore_index=True
)

# Shuffle records
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save CSV
df.to_csv("your_dataset.csv", index=False)

print("Dataset created successfully!")
print("Filename: your_dataset.csv")
print("Total records:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())

print("\nDataset preview:")
print(df.head())
