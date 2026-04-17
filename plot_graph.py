import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("traffic_data.csv")

plt.figure(figsize=(10, 5))
plt.plot(data["Time(sec)"], data["Vehicles Passed"], label="Vehicles Passed")
plt.plot(data["Time(sec)"], data["Vehicles Per Minute"], label="Vehicles Per Minute")
plt.xlabel("Time (seconds)")
plt.ylabel("Count")
plt.title("Traffic Flow Over Time")
plt.legend()
plt.tight_layout()
plt.show()
