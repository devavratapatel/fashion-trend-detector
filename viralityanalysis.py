from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch
from PIL import Image
import os
from texttrends import df
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

features = []

for img_path in df['image']:
    if pd.notna(img_path):
        img = cv2.imread(img_path)
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            embedding = model(img).squeeze().numpy()
            features.append(embedding)

print(features)

df['date'] = (pd.to_datetime(df['created'])).dt.date

df_keywords = df.explode('keywords')
df_keywords['keywords'] = df_keywords['keywords'].apply(lambda x: x[0] if isinstance(x, tuple) else x)

top_keywords = df_keywords['keywords'].value_counts().head(10).index.tolist()
df_filtered = df_keywords[df_keywords['keywords'].isin(top_keywords)]
df_grouped = df_filtered.groupby(['date','keywords']).size().unstack(fill_value=0)

df_grouped.plot(kind="line", figsize=(12,6),title="Trend of top Fashion Keywords Over Time")

print("Total image embeddings:", len(features))
print("Shape of one embedding:", features[0].shape if features else "No embeddings extracted")


plt.figure(figsize=(14, 7))
sns.lineplot(data=df_grouped)
plt.title("Fashion Keyword Trends Over Time", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Post Count")
plt.xticks(rotation=45)
plt.legend(title="Keyword", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("trend_over_time.png")
plt.show()