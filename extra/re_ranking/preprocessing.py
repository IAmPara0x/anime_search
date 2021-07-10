import ast
import random
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizerFast

#Params
MIN_ANS_LEN = 256
QUERY_LENS = [32, 64, 96]
MAX_SEQ_LENGTH = 4096

def load_data():
  # preprocessing reviews
  reviews_df = pd.read_csv("/kaggle/input/review-embeddings/reviews_features.csv")
  reviews_num = dict(reviews_df["anime_uid"].value_counts())
  anime_uids = [uid for uid, count in reviews_num.items() if count >= 5]
  reviews_df = reviews_df[reviews_df["anime_uid"].map(lambda x: x in anime_uids)]
  reviews_df = reviews_df[['anime_uid', 'text']]
  reviews_df["id"] = np.arange(reviews_df.shape[0])
  #preprocessing animes
  anime_df = pd.read_csv("/kaggle/input/get-data/anime_df_with_tags.csv")
  anime_df = anime_df[["uid", "genre"]]
  anime_df["genre"] = anime_df["genre"].apply(ast.literal_eval)

  return reviews_df, anime_df, anime_uids

def create_anime_clusters(reviews_df, anime_df, anime_uids):
  anime_clusters = {}
  for uid in tqdm(anime_uids):
    uid_genre = anime_df[anime_df["uid"]==uid].iloc[0,-1]
    anime_clusters[uid] = list(anime_df[
                               anime_df['genre'].apply(
                               lambda x: set(uid_genre).isdisjoint(x))
                               ]["uid"])
  return anime_clusters


def create_data(reviews_df, anime_clusters):
  answers_id = list(reviews_df[reviews_df["text"].map(lambda x: len(x.split()) >= 256)]["id"])
  anime_reviews = list(reviews_df.groupby(["anime_uid"]))

  for i in tqdm(anime_reviews):
    uid = i[0]
    df = i[1]
    pos_reviews_id = df[df["review_score"] >= 7]["id"]
    neg_reviews_id = df[df["review_score"] < 7]["id"]

    pos_queries_id = list(set(pos_reviews_id) - set(answers_id))
    pos_answers_id = list(set(pos_reviews_id).instersection(answers_id))
    neg_queries_id = list(set(neg_reviews_id) - set(answers_id))
    neg_answers_id = list(set(neg_reviews_id).instersection(answers_id))
    data[uid] = {"pos_queries_id": pos_queries_id, "pos_answers_id": pos_answers_id,
        "neg_queries_id": neg_queries_id, "neg_answers_id": neg_answers_id, "neg_anime_uids": anime_clusters[uid]}
  return data

reviews_df["tokenized_text"] = reviews_df["tokenized_text"].apply(tokenizer, max_length=MAX_SEQ_LENGTH, truncation=True, padding=True)
