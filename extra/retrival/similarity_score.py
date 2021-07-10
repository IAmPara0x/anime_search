import ast
from tqdm import tqdm

def load_dfs():
  tags_df = pd.read_csv("/kaggle/input/get-data/tags_df.csv")
  del tags_df["Unnamed: 0"]

  anime_df = pd.read_csv("/kaggle/input/get-data/anime_df_with_tags.csv")
  del anime_df["Unnamed: 0"]

  anime_df["tags"] = anime_df["tags"].apply(ast.literal_eval)
  anime_df = anime_df[anime_df["tags"].map(lambda x: len(x) != 0)]

  return tags_df, anime_df


class AnimeTags:
  tags_category = {category: list(tags_df[tags_df["category"]==category]["id"])
                  for category in tags_df["category"].unique()}
  id_category = {id: category_name for id, category_name in
                 zip(list(tags_df["id"]), list(tags_df["category"]))
                }
  tags_name = [category for category in tags_category.keys()]
  max_category_tags = max([len(i) for i in tags_category.values()])
  def __init__(self, uid, tags):
    self.uid = uid
    self.tags = tags
    self.tags_mat = self.get_tags_vector(self.tags)
    self._similarity_dict = {}
    self._similar_anime_uid = []
    self.get_similarity_vector(self.uid, self.tags_mat, same=True)

  def __getitem__(self, uid):
    if uid not in self._similarity_dict:
      raise ValueError(f"Similarity vector of {uid} has not been calculated")
    return self._similarity_dict[uid]

  def top_k(self, k):
    return self._similar_anime_uid[:k]

  @classmethod
  def get_tags_vector(cls, tags):
    tags_mat = np.zeros((len(cls.tags_category), cls.max_category_tags))

    for id, rank in tags:
      category = cls.id_category[id]
      i = cls.tags_name.index(category)
      j = cls.tags_category[category].index(id)
      tags_mat[i,j] = rank/100
    return tags_mat

  def get_similarity_vector(self, uid, other_tags_mat, same=False):
    if uid in self._similarity_dict:
      return self._similarity_dict[uid]["vector"]
    s_vector = np.diag(np.dot(self.tags_mat, other_tags_mat.T))

    if same:
      self._similarity_dict[uid] = {"vector": s_vector,
          "score": np.dot(s_vector, s_vector)}
    else:
      self._similarity_dict[uid] = {"vector": s_vector,
          "score": np.dot(self._similarity_dict[self.uid]["vector"], s_vector)}
    self._insort(uid)
    return s_vector

  def _insort(self,uid):
    lo = 0
    hi = len(self._similar_anime_uid) 
    get_score = lambda _uid: self._similarity_dict[_uid]["score"]
    while lo < hi:
      mid = (lo + hi) // 2
      if get_score(uid) < get_score(self._similar_anime_uid[mid]):
        hi = mid
      else:
        lo = mid + 1

    self._similar_anime_uid.insert(lo, uid)
