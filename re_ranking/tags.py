class Tag:
  tags_df = TAGS_DF
  def __init__(self, uid):
    tag_info = self.tags_df[self.tags_df["id"]==uid].to_numpy()
    self._uid = uid
    self._name = tag_info[0,1]
    self._description = tag_info[0,2]
    self._category = tag_info[0,3]
    self._embedding = Tag.get_embedding(self._name+" . "+self._description)

  @property
  def uid(self):
    return self._uid
  @property
  def name(self):
    return self._name

  @property
  def description(self):
    return self._description

  @property
  def category(self):
    return self._category

  @property
  def embedding(self):
    return self._embedding

  @staticmethod
  def get_embedding(text):
    with torch.no_grad():
      input = TOKENIZER(text, return_tensors="pt")["input_ids"]
      embd = MODEL(True, x=input.to(DEVICE)).cpu().numpy().squeeze()
    return embd

def create_taginfo():
  TAGS_INFO = {category_name:[] for category_name in TAGS_DF["category"].unique()}
  for idx in tqdm(range(TAGS_DF.shape[0])):
    tag_category = TAGS_DF.iloc[idx, 3]
    tag_uid = TAGS_DF.iloc[idx,0]
    TAGS_INFO[tag_category].append(Tag(tag_uid))
  return TAGS_INFO
