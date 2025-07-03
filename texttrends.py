from keybert import KeyBERT
import pandas as pd

df = pd.read_csv("labeled.csv")
kw_model = KeyBERT("all-MiniLM-L6-v2")

df["combined"] = df['title'].fillna('') + " " + df["text"].fillna('')

df['keywords'] = df['combined'].apply(lambda x: kw_model.extract_keywords(x, top_n=3, stop_words='english'))

print(df.iloc[0]['keywords'])