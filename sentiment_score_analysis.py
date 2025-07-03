import open_clip
import torch
import pandas as pd
import cv2
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np

nltk.download('vader_lexicon')

from texttrends import df

model, _, preprocess = open_clip.create_model_and_transforms(
    "hf-hub:Marqo/marqo-fashionSigLIP"
)
tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

classes = ["t-shirt", "jeans", "jacket", "hoodie", "dress", "skirt", "shoe", "hat"]

text_tokens = tokenizer([f"a photo of a {c}" for c in classes]).to(device)
with torch.no_grad():
    text_emb = model.encode_text(text_tokens)
    text_emb /= text_emb.norm(dim=-1, keepdim=True)

def classify_clip(img_path):
    try:
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            return "error"
        img = cv2.imread(img_path)
        if img is None:
            return "error"
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        img_input = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            img_emb = model.encode_image(img_input)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
            sims = (img_emb @ text_emb.T).cpu().numpy()[0]
            best = sims.argmax()
            return classes[best]
    except Exception as e:
        print("CLIP classify error:", e)
        return "error"

df["clothing_type"] = df["image"].apply(classify_clip)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sid = SentimentIntensityAnalyzer()
df["combined"] = df["title"].fillna("") + " " + df["text"].fillna("")
df["sentiment"] = df["combined"].apply(lambda x: sid.polarity_scores(x)["compound"])
df["upvote_score"] = MinMaxScaler().fit_transform(df[["upvotes"]])
df['created'] = pd.to_datetime(df['created'])
df['age_hrs'] = (pd.Timestamp.now() - df['created']).dt.total_seconds() / 3600
df["popularity_score"] = 0.5 * np.log1p(df["upvote_score"]) + 0.2 * sigmoid(df['sentiment']) + (0.3 * (df['sentiment']  / (df['age_hrs'] + 2)**1.5))
df["date"] = pd.to_datetime(df["created"]).dt.date

df2 = df[df["clothing_type"].isin(classes)]

grouped = df2.groupby(["date", "clothing_type"])["popularity_score"].mean().unstack(fill_value=0)

print(df2['clothing_type'].value_counts())
print("Grouped data shape:", grouped.shape)
print("\n[Debug] Clothing type value counts:")
print(df['clothing_type'].value_counts())
print("\n[Debug] Popularity score sample:")
print(df[['clothing_type', 'popularity_score']].head())
print("\n[Debug] Grouped dataframe shape:")
print(grouped.shape)
print("\n[Debug] Grouped data preview:")
print(grouped.tail())

plt.figure(figsize=(14, 7))
sns.lineplot(data=grouped)
plt.title("Clothing Popularity Trends with FashionCLIP", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Avg Popularity Score")
plt.xticks(rotation=45)
plt.legend(title="Clothing Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("fashionclip_trends.png")
plt.show()
