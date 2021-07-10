import gc
import random
import ast
import torch
import spacy
from keybert import KeyBERT
from transformers import RobertaModel, RobertaTokenizerFast

#PARAMS
MAX_QUERY_REVIEW_LEN = 128
MIN_ANS_REVIEW_LEN = 128
QUERY_LENS = [8,12,16]
PROB_QUERY_LENS = [0.5,0.25,0.25]
MIN_REVIEW_SCORE = 7
KEY_PHRASE_LEN = 2
NUM_KEY_PHRASE = 5
# NLP = spacy.load("en_core_web_trf")
REMOVE_DIGITS = str.maketrans('', '', digits)
ANIME_INFO = []

def load_df():
  reviews_df = pd.read_csv("/kaggle/input/review-embeddings/reviews_features.csv")
  reviews_df = reviews_df[reviews_df["review_score"] >= MIN_REVIEW_SCORE]

  anime_df = pd.read_csv("/kaggle/input/get-data/anime_df_with_tags.csv")
  del anime_df["Unnamed: 0"]
  anime_df["tags"] = anime_df["tags"].apply(ast.literal_eval)
  anime_df = anime_df[anime_df["tags"].map(lambda x: len(x) != 0)]

  anime_uids = list(set(reviews_df["anime_uid"]).intersection(anime_df["uid"].unique()))

  reviews_df = reviews_df[reviews_df["anime_uid"].isin(anime_uids)]
  anime_df = anime_df[anime_df["uid"].isin(anime_uids)]

  query_df = reviews_df[reviews_df["text"].apply(lambda x: len(x.split()) <= MAX_QUERY_REVIEW_LEN)]
  query_df = query_df[query_df["review_score"] >= 8]

  ans_df = reviews_df[reviews_df["text"].apply(lambda x: len(x.split()) >= MIN_ANS_REVIEW_LEN)]

  anime_uids = list(set(query_df["anime_uid"].unique()) - set(ans_df["anime_uid"].unique()))

  tags_df = pd.read_csv("/kaggle/input/get-data/tags_df.csv")
  del tags_df["Unnamed: 0"]

  return anime_df, anime_uids, tags_df, query_df, ans_df

def create_queries(review):
  phrases = KW_MODEL.extract_keywords(review, keyphrase_ngram_range=(1, KEY_PHRASE_LEN), top_n=NUM_KEY_PHRASE)
  phrases = [i[0].split() for i in phrases]
  doc = NLP(review)
  sents = [i.text for i in doc.sents if len(i.text.split())]

  for idx, sent in enumerate(sents):
    if len(sent.split()) <= 4 and idx != 0:
      sents[idx-1] += " " + sent
      sents.pop(idx)
  sents = [i.split() for i in sents]
  queries = []

  for sent in sents:
    for phrase in phrases:
      lower_sent = [word.lower() for word in sent]
      query = sent
      if set(phrase).issubset(lower_sent):
        query_len = random.choices(QUERY_LENS, weights=PROB_QUERY_LENS)[0]
        while set(phrase).issubset(lower_sent[1:-1]):
          if len(lower_sent) >= query_len:
            lower_sent = lower_sent[1:-1]
            query = query[1:-1]
          else:
            break
        queries.append(" ".join(query))
  return list(set(queries))

def split_review(review):
  doc = NLP(review)
  review_sents = []

  x = ""
  for sent in doc.sents:
    if len(sent.text.split()) > 1:
      if len(x.split()) + len(sent.text.split()) <= MAX_REVIEW_LEN:
        x += " " + sent.text
      else:
        review_sents.append(x)
        x = sent.text

  if len(x.split()) < MIN_REVIEW_LEN and len(review_sents):
    review_sents[-1] += " " + x
  else:
    review_sents.append(x)
  return review_sents

queries_dict = {uid: [] for uid in query_df["anime_uid"].unique()}
for idx in tqdm(range(query_df.shape[0])):
  uid = query_df.iloc[idx,0]
  text = query_df.iloc[idx,1]
  queries_dict[uid].extend(create_queries(text))


neg_classes_dict = {uid: [] for uid in anime_df["uid"].unique()}
for obj in ANIME_INFO:
  neg_classes = [uid for uid in obj._similar_anime_uid
      if obj._similarity_dict[uid]["score"] == 0]
  neg_classes_dict[obj.uid].extend(neg_classes)

answers_dict = {uid: [] for uid in ans_df["anime_uid"].unique()}
for idx in tqdm(range(ans_df.shape[0])):
  uid = ans_df.iloc[idx, 0]
  split_ans = split_review(ans_df.iloc[idx,1])
  answers_dict[uid].append(split_ans)



