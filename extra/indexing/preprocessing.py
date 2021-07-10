#IMPORTS
from transformers import RobertaTokenizerFast, RobertaModel
import torch
from tqdm import tqdm
# import pickle
# import spacy

#PARAMETERS
MIN_REVIEW_LENGTH = 128
MAX_REVIEW_LENGTH = 256

# Initializing Dataframes

reviews_df: pd.DataFrame = pd.read_csv("/kaggle/input/myanimelist-dataset-animes-profiles-reviews/reviews.csv")
reviews_df = reviews_df.rename(columns={'score':'review_score', 'scores':'specific_scores'})
reviews_df1: pd.DataFrame = pd.read_csv("/kaggle/input/mal-reviews-dataset/reviews.csv")
reviews_df1 = reviews_df1.rename(columns={'score':'review_score', 'scores':'specific_scores'})

animes_df: pd.DataFrame = pd.read_csv("/kaggle/input/myanimelist-dataset-animes-profiles-reviews/animes.csv")
animes_df = animes_df.rename(columns={'uid':'anime_uid', 'score':'anime_score'})
animes_df1: pd.DataFrame = pd.read_csv("/kaggle/input/mal-reviews-dataset/animes.csv")
animes_df1 = animes_df1.rename(columns={'uid':'anime_uid', 'score':'anime_score'})

reviews_df = reviews_df.append(reviews_df1)
animes_df = animes_df.append(animes_df1)
del reviews_df1,animes_df1


#HELPER FUNCTIONS
def format_review(x: str):
  """
  Preprocess the review to remove text elements.
  """
  x = re.sub("\r|\n", "", x)
  x = re.sub(" +", " ", x)
  if x[:10] == " more pics": x = x[11:]
  if x[-8:] == "Helpful ": x = x[:-8]
  x = re.sub(r"^Overall \d+ Story \d+ Animation \d+ Sound \d+ Character \d+ Enjoyment \d+ ", "", x)
  return x[:-1]

def get_review_category(x):
  if x >= 9:
    return 1
  elif x >= 7:
    return 2
  elif x == 6:
    return 3
  elif x <= 5:
    return 4

def greedy_sentence_filling(review):
  doc = nlp(review)
  curr_sent = ""

  for sent in doc.sents:
    if len(curr_sent.split()) >= MIN_REVIEW_LENGTH:
      yield curr_sent
      curr_sent = sent
    else:
      curr_sent += " " + sent
  if curr_sent: yield curr_sent

#PREPROCESSING DATA
df = pd.merge(reviews_df[["uid","anime_uid",'text','review_score', 'specific_scores']],
    animes_df[["anime_uid",'genre','anime_score']],on='anime_uid')
del animes_df, reviews_df

df = df.drop_duplicates(subset='uid', keep="last")
del df["uid"], df["specific_scores"], df["anime_score"]

df["text"] = df['text'].apply(format_review)
df = df[df["text"].map(lambda x: len(x.split()) >= MIN_REVIEW_LENGTH)]
df["review_category"] = df["review_score"].apply(get_review_category)
df = df.sample(frac=1)
df["sents"] = np.empty((len(df), 0)).tolist()
gc.collect()

def main():
  num_rows = df.shape[0]

  for idx in tqdm(range(num_rows)):
    review = df.iloc[idx, 0]
    if len(review.split()) >= MAX_REVIEW_LENGTH:
      df.iloc[idx, -1] = list(greedy_sentence_filling(review))
  df.to_csv("/kaggle/working/reviews_features.csv", index=False)

## SUMMARIZATION

#imports
from transformers import pipeline
import torch


MAX_SUMMARIZATION_LENGTH: int = 128
TOP_K: int = 72
TOP_P: int = 0.95
SAMPLE: int = True
NUM_BEAMS: int = 4
BATCH_SIZE: int = 24
EARLY_STOPPING: bool = True
TRUNCATION: bool = True
MODEL:str = "facebook/bart-large-cnn"

summarizer = pipeline("summarization", model=MODEL, device=0)


def main():
  df = pd.read_csv("/kaggle/input/review-embeddings/reviews_features.csv")
  num_rows = df.shape[0]

  for idx in tqdm(range(num_rows)):
    review_sents = df.iloc[idx,-1]
    b_sents = []
    b_lengths = [0]
    b_idxs = []
    if len(review_sents):
      if len(review_sents[-1].split()) < MIN_REVIEW_LENGTH:
        review_sents[-2] += " " + review_sents[-1]
        review_sents.pop()

        b_sents.extend(review_sents)
        b_lengths.append(b_idxs[-1] + len(review_sents))
        b_idxs.append(idx)

      if len(b_sents) >= BATCH_SIZE:
        review_summary = summarizer(b_sents, max_length=MAX_SUMMARIZATION_LENGTH,
                          do_sample=SAMPLE, top_k=TOP_K, top_p=TOP_P, num_beams=NUM_BEAMS,
                          early_stopping=EARLY_STOPPING, truncation=TRUNCATION)
        review_summary = [i["summary_text"] for i in review_summary]

        for start, end, idx in zip(b_lengths, b_lengths[1:], b_idxs):
          df.iloc[idx,1] = " ".join(review_summary[start:end])
        b_sents = []
        b_lengths = [0]
        b_idxs = []

  del df["sents"]
  df.to_csv("/kaggle/working/reviews_features.csv", index=False)
