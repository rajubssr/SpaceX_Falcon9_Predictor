import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("spacex_launches.csv")

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nClass distribution:\n", df["landing_success"].value_counts())

# Plot 1: Landing success rate
plt.figure(figsize=(6, 4))
df["landing_success"].value_counts().plot(kind="bar", color=["#2ecc71", "#e74c3c"])
plt.title("Landing Success vs Failure")
plt.xlabel("Landing Success")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("landing_success_dist.png")
plt.close()

# Plot 2: Success rate by reuse
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="reused", hue="landing_success", palette="Set2")
plt.title("Reused Core vs Landing Success")
plt.tight_layout()
plt.savefig("reused_vs_success.png")
plt.close()

print("EDA plots saved.")
