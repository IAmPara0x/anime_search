import requests
import json
from tqdm import tqdm

def get_anime_df():
  # preprocessing reviews
  reviews_df = pd.read_csv("/kaggle/input/review-features-batch2/reviews_features.csv")
  reviews_num = dict(reviews_df["anime_uid"].value_counts())
  anime_uids = [uid for uid, count in reviews_num.items() if count >= 5]
  reviews_df = reviews_df[reviews_df["anime_uid"].map(lambda x: x in anime_uids)]
  reviews_df = reviews_df[['anime_uid', 'text', 'review_score', 'genre']]

  #preprocessing animes
  anime_df = pd.read_csv("/kaggle/input/myanimelist-dataset-animes-profiles-reviews/animes.csv")
  anime_df1 = pd.read_csv("/kaggle/input/mal-reviews-dataset/animes.csv")
  anime_df2 = pd.read_csv("/kaggle/input/mal-reviews-dataset/animes_real.csv")

  anime_df = anime_df.append(anime_df1)
  anime_df = anime_df.append(anime_df2)
  anime_df = anime_df[["uid", "genre"]]
  anime_df = anime_df.drop_duplicates()
  anime_df = anime_df[anime_df["uid"]
      .map(lambda x: x in anime_uids)]
  anime_df["genre"] = anime_df["genre"].apply(ast.literal_eval)

  ### Removing duplicates
  z = anime_uids.copy()
  def helper(x):
    if x in z:
      z.remove(x)
      return True
    return False
  anime_df = anime_df[anime_df["uid"].map(helper)]
  ###

  anime_df["tags"] = [[] for _ in range(len(anime_df))]
  return anime_df, list(anime_df["uid"])

def get_anime_tags(mal_id, tags_list, anime_df, anime_uids):
  query = '''
  query ($id: Int) {
    Media (idMal: $id, type: ANIME) {
      id
      idMal
      title {
        romaji
      }
      tags{
        id
        name
        description
        category
        rank
      }
    }
  }
  '''

  variables = {
      'id': mal_id
  }

  url = 'https://graphql.anilist.co'
  response = requests.post(url, json={'query': query, 'variables': variables})

  data = json.loads(reponse.text)["data"]["Media"]
  anime_tags = [(tag["id"], tag["rank"]) for tag in data["tags"]]
  idx = anime_uids.index(mal_id)
  anime_df.iloc[idx, -1].extend(anime_tags)
  tags_list.extend([tags.pop("rank") for tag in data["tags"]])

  return tags_list, anime_uids

