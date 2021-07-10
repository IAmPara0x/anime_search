import requests
import json
from tqdm import tqdm

query = '''
query ($id: Int, $limit: Int, $page: Int, $perPage: Int) {
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
    reviews(limit: $limit, sort: ID, page: $page, perPage: $perPage){
      nodes{
        body
        rating
        ratingAmount
        userRating
        score
      }
    }
  }
}
'''

variables = {
    'id': 14813,
    'limit':1000,
    'page': 1,
    'perPage':1000,
}

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

  return anime_df, anime_uids

url = 'https://graphql.anilist.co'

# Make the HTTP Api request
response = requests.post(url, json={'query': query, 'variables': variables})
