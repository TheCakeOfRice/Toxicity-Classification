import numpy as np
import sqlite3
import pandas as pd
import pickle
from nltk.corpus import stopwords
from string import ascii_lowercase, whitespace, printable
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from time import time


start = time()
print("== Running cleaning.py ==")


# Grabbing data from SQL DB
con = sqlite3.connect("data/ToxicityDB.db")
df = pd.io.sql.read_sql_query(
	"""
	SELECT id, target, comment_text, severe_toxicity, obscene,
		identity_attack, insult, threat, funny, wow, sad, likes,
		disagree, sexual_explicit
	FROM train
	""",
	con,
	chunksize=None
)


# The following columns feel like they are dependent on the rest of the data,
# contain NULL values, and may introduce bias if used for prediction.
# The misclassifications mentioned on the Kaggle page I think are better
# solved using data from distributions with less toxicity towards identities.
"""
asian, atheist, bisexual, black, buddhist, christian, female, heterosexual,
hindu, homosexual_gay_or_lesbian, intellectual_or_learning_disability, jewish,
latino, male, muslim, other_disability, other_gender, other_race_or_ethnicity,
other_religion, other_sexual_orientation, physical_disability,
psychiatric_or_mental_illness, transgender, white
"""


# Text cleanup
# -- Removing non-ASCII characters
print("Removing non-ASCII characters...")
df["comment_text"] = df["comment_text"].str.replace(
	r"[^\x00-\x7f]",
	"",
	regex=True
)

# -- Removing urls
print("Removing urls...")
df["comment_text"] = df["comment_text"].str.replace(
	r"\bhttp\S+",
	"",
	regex=True
).replace(
	r"\bwww\S+",
	"",
	regex=True
).replace(
	r".com\b",
	"",
	regex=True
)

# -- Removing all remaining characters that aren't letters or whitespace
print("Removing all remaining characters that aren't letters or whitespace...")
kept_chars = ascii_lowercase + whitespace
removed_chars = "".join(c for c in printable if c not in kept_chars)
trans = {ord(c): None for c in removed_chars}
df["comment_text"] = df["comment_text"].str.lower()
df["comment_text"] = df["comment_text"].str.translate(trans)

# -- Replacing characters that are repeated 3 or more times with just 2
# 	of that character
print("Replacing repeated characters...")
df["comment_text"] = df["comment_text"].str.replace(
	r"(.)(\1{1})(\1+)",
	r"\1\2",
	regex=True
)


# Fitting a bag-of-words model to the text data, allowing unigrams & bigrams
print("Fitting a bag-of-words vectorizer and transforming the data...")
BOW_vect = CountVectorizer(
	stop_words=set(stopwords.words("english")),
	lowercase=True,
	min_df=20,			# Only count grams which occur at least 20 times,
	max_df=0.7,			# and in at most 70% of the comments.
	max_features=10000,
	ngram_range=(1, 2)
)
BOW = BOW_vect.fit_transform(df.comment_text)
vocab = BOW_vect.get_feature_names()


# Saving bag-of-words results and targets
with open("variables/BOW_vect.pickle", "wb") as file:
	pickle.dump(BOW_vect, file, protocol=pickle.HIGHEST_PROTOCOL)
sparse.save_npz("variables/BOW.npz", BOW)
df.to_csv(
	"variables/target.csv",
	columns=["id", "target"],
	index=False
)
pd.DataFrame(vocab).to_csv(
	"variables/vocab.csv",
	index=False
)


# Auxiliary data organization
print("Collecting auxiliary data and organizing...")

# -- Text and target data no longer needed
del df["comment_text"]
del df["target"]

# -- Saving the rest
df.to_csv(
	"variables/other_cols.csv",
	index=False
)


end = time()
minutes = (end - start) // 60
seconds = (end - start) % 60
print(
	"Finished in " +
	str(minutes) +
	" minutes and " +
	str(seconds) +
	" seconds."
)
