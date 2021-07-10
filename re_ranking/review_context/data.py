#PARAMS
MAX_LEN = 512
NLP = spacy.load("en_core_web_sm")

def load_dfs():
  filterd_reviews_df = pd.read_csv("/kaggle/input/review-context/filterd_reviews.csv")
  filterd_reviews_df = filterd_reviews_df.dropna()

  _anime_df = pd.read_csv("/kaggle/input/get-data/anime_df_with_tags.csv")
  del _anime_df["Unnamed: 0"]
  _anime_df["tags"] = _anime_df["tags"].apply(ast.literal_eval)
  _anime_df["genre"] = _anime_df["genre"].apply(ast.literal_eval)
  _anime_df = _anime_df[_anime_df["tags"].apply(lambda x: len(x) != 0)]
  filterd_reviews_df = filterd_reviews_df[filterd_reviews_df["anime_uid"].isin(_anime_df["uid"].unique())]


  anime_df = pd.read_csv("/kaggle/input/mal-reviews-dataset/animes_real.csv")
  tags_df = pd.read_csv("/kaggle/input/get-data/tags_df.csv")
  tags_df.iloc[252,-1] = "Others"
  del tags_df["Unnamed: 0"]
  return filterd_reviews_df, anime_df, tags_df


def greedy_sentence_filling(doc, docs):
  assert len(docs) != 0
  if len(doc.split()) >= MAX_LEN:
    return review
  while True:
    s_doc = random.choice(docs)
    s_doc_sents = [i.text for i in NLP(s_doc).sents]
    start_idx = round(len(s_doc_sents)*0.2)

    for sent in s_doc_sents[start_idx:]:
      if len(sent.split()) + len(doc.split()) < MAX_LEN:
        doc += " . " + sent
      else:
        return doc

def create_data():
  filterd_reviews_df = pd.read_csv("/kaggle/input/review-context/filterd_reviews.csv")
  score_category = {11:1,10:1, 9:1, 8:1, 7:1, 6:2, 5:2}
  filterd_reviews_df["review_category"] = filterd_reviews_df["review_score"].apply(lambda x: score_category[x])
  filterd_reviews_df["queries"] = filterd_reviews_df["queries"].apply(ast.literal_eval)
  anime_uids = list(set(filterd_reviews_df["anime_uid"].unique()).intersection(ANIME_INFO.keys()))
  filterd_reviews_df = filterd_reviews_df[filterd_reviews_df["anime_uid"].isin(anime_uids)]
  filterd_reviews_df = filterd_reviews_df.dropna()
  print(filterd_reviews_df.shape)
  filterd_reviews_df = filterd_reviews_df.tail(500)
  filterd_reviews_np = filterd_reviews_df.to_numpy()

  for idx in tqdm(range(filterd_reviews_np.shape[0])):
    x_category = filterd_reviews_np[idx,-1]
    anime_uid = filterd_reviews_np[idx,0]
    x_df = filterd_reviews_df[filterd_reviews_df["anime_uid"]==anime_uid]
    x_df = x_df[x_df["review_category"]==x_category]
    filterd_reviews_df.iloc[idx,-3] = greedy_sentence_filling(filterd_reviews_np[idx,-3], list(x_df["filtered_review"]))

  anime_uids = list(filterd_reviews_df["anime_uid"].unique())
  data = {uid:{"queries":[], "answers": [], "neg_cls_uids": get_neg_classes(uid)} for uid in tqdm(anime_uids)}
  filterd_reviews_df = filterd_reviews_df.to_numpy()

  for idx in tqdm(range(filterd_reviews_df.shape[0])):
    anime_uid = filterd_reviews_df[idx,0]
    data[anime_uid]["answers"].append(filterd_reviews_df[idx,-3])
    data[anime_uid]["queries"].extend(filterd_reviews_df[idx,-2])
  return data

def get_neg_classes(a_uid):
  x = ANIME_INFO[a_uid]
  s_scores = {}
  for uid in ANIME_INFO.keys():
    s_scores[uid], _, _ = x.similarity_info(ANIME_INFO[uid]._tags_score)

  neg_cls_uids = [uid for uid, score in s_scores.items() if score == 0]
  return neg_cls_uids
