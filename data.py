import re
import spacy
import gc
from tqdm import tqdm


#PARAMETERS
MAX_SUMMARIZATION_LENGTH = 256
MIN_SUMMARIZATION_LENGTH = 128

#HELPER functions

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

def greedy_sentence_filling(review):
  doc = nlp(review)
  all_sents = []

  curr_sent = ""
  for sent in doc.sents:
    if len(curr_sent.split()) >= max_summarization_length:
      all_sents.append(curr_sent)
      curr_sent = sent.text
    else:
      curr_sent += " " + sent.text

  if len(curr_sent.split()) >= min_summarization_length or (not all_sents):
    all_sents.append(curr_sent)
  else:
    all_sents[-1] += " " + curr_sent
  return all_sents

def main():
  df = pd.read_csv("/kaggle/input/mal-reviews-dataset/reviews_real.csv")
  df = df[df["text"].notna()]

  del df["profile"], df["link"]
  df["text"] = df["text"].apply(format_review)

  df["sents"] = np.empty((len(df), 0)).tolist()

  for idx in tqdm(range(df.shape[0])):
    review = df.iloc[idx, 2]
    if len(review.split()) >= MIN_SUMMARIZATION_LENGTH:
      df.iloc[idx, -1].extend(greedy_sentence_filling(review))

  df.to_csv("review_features.csv")
