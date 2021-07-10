
#IMPORTS
import re
import pickle
import gc
import spacy
from tqdm import tqdm


is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
  torch.set_default_tensor_type("torch.cuda.FloatTensor")
nlp = spacy.load("en_core_web_trf")

#PARAMETERS
MAX_TEXT_LENGTH:int = 128
MIN_TEXT_LENGTH:int = 70
STRIDE:int = 48
MIN_SENTENCE_RATIO:int = 0.6
SAVE_DATA_ITERATION:int = 35_000

EXCLUDED_GENRES:list = ["Samurai", "Parody", "Cars", "Yaoi", "Yuri"]
HIGH_AVAIL_GENRES:list = ["Action", "Comedy"]
ALL_GENRES:list = []
ALL_GENRES_DICT:dict = {}

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

# Helper functions

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

def one_hot_enc(anime_genres):
  ohe_genre = np.zeros(TOTAL_GENRES)
  for genre in ast.literal_eval(anime_genres):
    if genre not in EXCLUDED_GENRES:
      ohe_genre[ALL_GENRES.index(genre)] = 1
  return list(ohe_genre)

def ohe_to_genre(ohe_vec: list) -> list:
  idxs = list(np.where(ohe_vec)[0])
  genres = ALL_GENRES[idxs]
  return genres

def greedy_sentence_filling(review, label):
  doc_len = len(review.split(" "))
  genres = ohe_to_genre(label)
  stride = STRIDE if set(genres).isdisjoint(HIGH_AVAIL_GENRES) else doc_len
  doc = nlp(review)
  total_sents = 0
  should_quit = False
  print(genres)
  print(stride)

  for start_idx in range(0, doc_len, stride):
    curr_total_sents, word_count, g_sent, g_sents = 0, 0, "", []
    if start_idx: ALL_GENRES_DICT.update({x: ALL_GENRES_DICT[x]+1 for x in genres})
    for sent in doc.sents:
      curr_word_count = len(sent.text.split())
      word_count += curr_word_count
      if word_count >= start_idx:
        if len(g_sent.split()) < MAX_TEXT_LENGTH:
          g_sent = " ".join((g_sent, sent.text))
        else:
          g_sents.append(g_sent)
          g_sent = sent.text
          if not start_idx: total_sents += 1
          curr_total_sents += 1
    if len(g_sent.split()) > MIN_TEXT_LENGTH:
      if not start_idx: total_sents += 1
      curr_total_sents += 1
      if (curr_total_sents/total_sents) < MIN_SENTENCE_RATIO:
        should_quit = True
      g_sents.append(g_sent)
    else:
      if not start_idx: total_sents += 1
      g_sents[-1] = " ".join((g_sents[-1], g_sent))
      if (curr_total_sents/total_sents) < MIN_SENTENCE_RATIO:
        should_quit = True
    yield g_sents
    if should_quit:
      return

# Merging Dataframes
df = pd.merge(reviews_df[["uid","anime_uid",'text','review_score', 'specific_scores']],
    animes_df[["anime_uid",'genre','anime_score']],on='anime_uid')
del animes_df, reviews_df

df = df.drop_duplicates(subset='uid', keep="last")
del df["uid"]

# Preprocessing data
df["text"] = df['text'].apply(format_review)

df["genre"].apply(lambda x: [ALL_GENRES.append(y) for y in ast.literal_eval(x)
  if y not in ALL_GENRES and y not in EXCLUDED_GENRES])

ALL_GENRES_DICT = {x: 0 for x in ALL_GENRES}
df["genre"].apply(lambda x: [ALL_GENRES_DICT.update({i: ALL_GENRES_DICT[i]+1}) for i in ast.literal_eval(x)
  if i not in EXCLUDED_GENRES])
TOTAL_GENRES:int = len(ALL_GENRES)
df["genre"] = df["genre"].apply(one_hot_enc)
ALL_GENRES = np.array(ALL_GENRES)


df = df[df["genre"].map(np.count_nonzero) != 0]
del df["anime_uid"], df["review_score"], df["specific_scores"], df["anime_score"] #NOTE: Can utilize these features for training.

df = df.sample(frac=1)
gc.collect()

input_data = list(df["text"])
labels = list(df["genre"])
del df


def main():
  params_dict = {}
  params_dict["MAX_TEXT_LENGTH"] = MAX_TEXT_LENGTH
  params_dict["MIN_TEXT_LENGTH"] = MIN_TEXT_LENGTH
  params_dict["STRIDE"] = STRIDE
  params_dict["MIN_SENTENCE_RATIO"] = MIN_SENTENCE_RATIO
  params_dict["EXCLUDED_GENRES"] = EXCLUDED_GENRES
  params_dict["HIGH_AVAIL_GENRES"] = HIGH_AVAIL_GENRES
  params_dict["ALL_GENRES"] = ALL_GENRES
  params_dict["GENRES_FREQUENCY_DICT"] = ALL_GENRES_DICT

  with open("/kaggle/working/params.pkl", "wb") as f:
    pickle.dump(params_dict, f)

  b_reviews = []
  b_labels = []
  batch_no = 0

  for review, label in tqdm(zip(input_data, labels)):
    s_review = list(greedy_sentence_filling(review, label))
    s_label = [label]*len(s_review)

    b_reviews.extend(s_review)
    b_labels.extend(s_label)

    if len(b_review) >= SAVE_DATA_ITERATION:
      with open(f"/kaggle/working/data_batch{batch_no}.pkl", "wb") as f:
        data = {}
        data["input_data"] = b_reviews
        data["labels"] = b_labels
        pickle.dump(data, f)
        del data

        b_reviews = []
        b_labels = []
        batch_no += 1
