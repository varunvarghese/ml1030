import pandas as pd
import numpy as np
import spacy
nlp = spacy.load('en')

df = pd.read_csv("reviews_cleaned.csv")
df.head()
df["comments_cleaned"].astype(str)
df["coments_cleaned"] = df["comments_cleaned"].astype(str)
npa = np.array(df["comments_cleaned"])

