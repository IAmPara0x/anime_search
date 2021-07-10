import pandas as pd

#PARAMS
ANIME_DF = pd.read_csv("/kaggle/input/mal-reviews-dataset/animes_real.csv")
TAGS_DF = pd.read_csv("/kaggle/input/get-data/tags_df.csv")
TAGS_DF.iloc[252,-1] = "Others"
del TAGS_DF["Unnamed: 0"]
del ANIME_DF["Unnamed: 0"]
ANIME_DF["tags"] = ANIME_DF["tags"].apply(ast.literal_eval)
with open("/kaggle/input/review-indexing/data.pkl", "rb") as f:
    DATA = pickle.load(f)
TOP_K = 128
INDEX = faiss.IndexFlatL2(EMBEDDING_DIM)
INDEX.add(DATA["embeddings"])
TOKENIZER = RobertaTokenizerFast.from_pretrained("roberta-base")

class AnimeTag:
  anime_df = ANIME_DF
  tags_info = TAGS_INFO
  tags_df = TAGS_DF
  def __init__(self, uid, tags):
    self.anime_info = self.anime_df[self.anime_df["uid"]==uid][["uid", "link", "score", "members", "genre"]].to_numpy()
    self._uid = uid
    self._tags_score = self._get_tags_score(tags)
    _, self._similarity_vec, _ = self.similarity_info(self._tags_score, True)

  def __repr__(self):
    return self.anime_info[0,1].split("/")[-1]

  def __getitem__(self, category_name):
    return {tag.name: score
            for tag, score in zip(self.tags_info[category_name], self._tags_score[category_name])}

  def __eq__(self, other):
    return other._uid == self._uid

  @property
  def similarity_vec(self):
    return [(category_name, score) for category_name, score in
              zip(list(self.tags_info.keys()), list(self._similarity_vec))]

  def _get_tags_score(self,tags):
    tags_score = {}
    anime_tags_uid = [i[0] for i in tags]
    anime_tags_dict = {i[0]:i[1] for i in tags}

    for category_name in self.tags_info.keys():
      score = np.zeros(len(self.tags_info[category_name]))
      tags_uid = [tag.uid for tag in self.tags_info[category_name]]
      common_tags = list(set(anime_tags_uid).intersection(tags_uid))
      tags_idx = [tags_uid.index(uid) for uid in common_tags]
      score[tags_idx] = [anime_tags_dict[tag]/100 for tag in common_tags]
      tags_score[category_name] = score
    return tags_score

  @staticmethod
  def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

  def similarity_info(self, tags_score, same=False):
    similarity_scores = []
    scores_dict = {}
    for category_name in tags_score.keys():
      vec1 = self._tags_score[category_name]
      vec2= tags_score[category_name]
      category_score = np.dot(vec1,vec2)
      similarity_scores.append(category_score)
      scores_dict[category_name] = category_score
    if same:
      score = np.dot(similarity_scores, similarity_scores)
    else:
      score = np.dot(similarity_scores, self._similarity_vec)
    return score, similarity_scores, scores_dict


class Query:
  data = DATA
  anime_info = ANIME_INFO
  tags_info = TAGS_INFO
  def __init__(self, text):
    self._text = text
    self.embedding = Query.get_embedding(text)
    self._similar_text_id, self._similar_anime = self.search(np.expand_dims(self.embedding,0))
    self.tags_similarity()
    self.re_index()

  def get_similar_text(self, idx):
    return TOKENIZER.decode(self.data["tokenized_sents"][self._similar_text_id[idx]])

  def top_k_anime(self, k):
    return self._similar_anime[:k]

  @staticmethod
  def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

  def tags_similarity(self):
    tags_score = {}
    tags_similarity = {}
    for category_name in self.tags_info.keys():
      category_score = []
      for tag in self.tags_info[category_name]:
        category_score.append(Query.cosine_similarity(tag.embedding, self.embedding))
      tags_score[category_name] = [sum(category_score)]*len(category_score)
      tags_similarity[category_name] = category_score

    self.tags_score = tags_score
    self.tags_similarity = tags_similarity

  def re_index(self):
    s_anime = []
    s_anime_uid = {}
    for anime in self._similar_anime:
      if anime.uid in s_anime_uid.keys():
        s_anime_uid[anime.uid] += 1
      else:
        s_anime_uid[anime.uid] = 1
        s_anime.append(anime)

    anime_s_scores = []
    for anime in s_anime:
      score, _, _ = anime.similarity_info(self.tags_score)
      anime_s_scores.append(score*s_anime_uid[anime.uid])

    self._similar_anime = [x for _,x in sorted(zip(anime_s_scores, s_anime))]
    self._similar_anime = list(reversed(self._similar_anime))

  def search(self, embd):
    _, neighbours_id = INDEX.search(embd, TOP_K)

    neighbours_id = neighbours_id.squeeze()
    similar_anime = []
    for i in neighbours_id:
      anime_uid = self.data["labels"][i]
      try:
        similar_anime.append(self.anime_info[anime_uid])
      except KeyError:
        similar_anime.append(self.anime_info[self.data["labels"][0]])
    return neighbours_id, similar_anime

  @staticmethod
  def get_embedding(text):
    with torch.no_grad():
      input = TOKENIZER(text, return_tensors="pt")["input_ids"]
      embd = MODEL(True, x=input.to(DEVICE)).cpu().numpy().squeeze()
    return embd
