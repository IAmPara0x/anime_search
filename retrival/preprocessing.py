import pickle
import gc
import random
import torch
from tqdm import tqdm
import ast
import json
from transformers import RobertaTokenizerFast

# Params
MAX_SEQ_LEN = 512


def load_data():
    # preprocessing reviews
    reviews_df = pd.read_csv(
        "/kaggle/input/review-embeddings/reviews_features.csv")
    reviews_num = dict(reviews_df["anime_uid"].value_counts())
    anime_uids = [uid for uid, count in reviews_num.items() if count >= 5]
    reviews_df = reviews_df[reviews_df["anime_uid"].map(
        lambda x: x in anime_uids)]
    reviews_df = reviews_df[['anime_uid', 'text']]
    # preprocessing animes
    anime_df = pd.read_csv(
        "/kaggle/input/myanimelist-dataset-animes-profiles-reviews/animes.csv")
    anime_df1 = pd.read_csv("/kaggle/input/mal-reviews-dataset/animes.csv")
    anime_df2 = pd.read_csv(
        "/kaggle/input/mal-reviews-dataset/animes_real.csv")

    anime_df = anime_df.append(anime_df1)
    anime_df = anime_df.append(anime_df2)
    anime_df = anime_df[["uid", "genre"]]
    anime_df = anime_df.drop_duplicates()
    anime_df = anime_df[anime_df["uid"]
                        .map(lambda x: x in anime_uids)]
    anime_df["genre"] = anime_df["genre"].apply(ast.literal_eval)

    return reviews_df, anime_df, anime_uids


def create_anime_clusters(reviews_df, anime_df, anime_uids):
    def similarity_f(x, y): return len(list(set(y).intersection(x))) <= 1
    anime_clusters = {}
    for uid in tqdm(anime_uids):
        uid_genre = anime_df[anime_df["uid"] == uid].iloc[0, -1]
        anime_clusters[uid] = list(anime_df[anime_df['genre'].
                                            apply(lambda x: set(
                                                uid_genre).isdisjoint(x))
                                            ]["uid"])
    return anime_clusters


def truncate_text(text, max_len=MAX_SEQ_LEN):
    num_words = len(text.split())
    truncate_first = (num_words - max_len) // 2
    truncate_last = (num_words - max_len) // 2 + 1
    return text[truncate_first:truncate_last]


def create_data(tokenizer, reviews_df, anime_clusters):
    reviews_df["tokenized_text"] = tokenizer(list(reviews_df["text"]),
                                             max_length=MAX_SEQ_LEN, padding=True, truncation=True)["input_ids"]
    del reviews_df["text"]
    gc.collect()
    anime_reviews = list(reviews_df.groupby(["anime_uid"]))
    anime_reviews = {i[0]: {
        "reviews": np.array(list(i[1].to_numpy()[:, 1]), dtype=np.uint16),
        "neg_classes": anime_clusters[i[0]]}
        for i in anime_reviews}
    return anime_reviews
