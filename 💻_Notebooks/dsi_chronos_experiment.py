
import torch
import pandas as pd
import matplotlib.pyplot as plt
from chronos import ChronosPipeline

# Load Model
pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cuda", torch_dtype=torch.bfloat16)

# Load Data
url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
df = pd.read_csv(url)
data = torch.tensor(df['OT'].values).to(torch.float32)

# Experiment
context = data[-608:-96]
standard_forecast = pipeline.predict(context, 96)
baseline_mean = torch.mean(standard_forecast[0], dim=0)

# DSI Loop
all_preds = []
for i in range(64):
    shaken = context + (torch.randn_like(context) * 0.01)
    all_preds.append(pipeline.predict(shaken, 96)[0])

dsi_final = torch.median(torch.stack(all_preds).view(-1, 96), dim=0).values
print("DSI experiment code executed successfully.")
