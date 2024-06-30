import pandas as pd
import matplotlib.pyplot as plt

df =pd.read_csv(r"D:\machine_learning_AI_Builders\บท4\Object_Detection\Animal\training_log.csv")
df.columns = df.columns.str.strip()

plt.figure(figsize=(6,6))

plt.plot(df["epoch"],df["train_loss"],label="train_loss",color = "blue",marker="x")
plt.plot(df["epoch"],df["valid_loss"],label="val_loss",color = "red",marker="x")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title('Train and Validation Loss by Epoch')
plt.show()

plt.figure(figsize=(6,6))
plt.plot(df.epoch,df.COCOMetric,label = "mAP", color = "blue",marker="x")
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title('Validation mAP by Epoch')
plt.show()

