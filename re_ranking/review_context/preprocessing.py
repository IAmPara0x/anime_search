
#PARAMS
TOP_K_RATIO = 0.75
MAX_SEQ_LEN = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NLP = spacy.load("en_core_web_sm")
TOKENIZER = RobertaTokenizerFast.from_pretrained("roberta-base")

ANIME_DF = pd.read_csv("/kaggle/input/mal-reviews-dataset/animes_real.csv")
TAGS_DF = pd.read_csv("/kaggle/input/get-data/tags_df.csv")
TAGS_DF.iloc[252,-1] = "Others"
del TAGS_DF["Unnamed: 0"]


class Doc:
  model = MODEL
  tokenizer = TOKENIZER
  nlp = NLP
  device = DEVICE
  max_len = MAX_SEQ_LEN
  top_k = TOP_K_RATIO
  def __init__(self, text, idx):
    self._text = self.format_text(text)
    self._s_doc = self.nlp(text)
    self._sents = [i.text for i in self._s_doc.sents]
    self._reindex_sentsidx = None
    self.idx = idx

#     self._reindex()

  @property
  def sents(self):
    return self._sents

  def reindex_sents(self, k=None):
    if not k:
      return list(map(self._sents.__getitem__, self._reindex_sentsidx))
    else:
      return list(map(self._sents.__getitem__, self._reindex_sentsidx[:k]))

  @property
  def filtered_text(self):
    return self._filtered_text

  def _cosine_similarity(self, a,b):
    return (torch.dot(a, b)/(torch.norm(a)*torch.linalg.norm(b))).cpu().item()

  def _reindex(self):
    text = self._split_text()
    text_len = len(text)
    all_text = text+self._sents

    tokenized_text = self.tokenizer(all_text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    with torch.no_grad():
      embeddings = self.model(True, x=tokenized_text.to(self.device))
      text_embd, sents_embd = torch.mean(embeddings[:text_len], 0), embeddings[text_len:]
      similarity_scores = [self._cosine_similarity(text_embd, s_embd) for s_embd in sents_embd]

    self._reindex_sentsidx = [i for _,i in sorted(zip(similarity_scores, range(len(self.sents))), reverse=True)]
    top_k_sents = round(self.top_k*len(self.sents))
    self._filtered_text = " ".join([self._sents[i] for i in sorted(self._reindex_sentsidx[:top_k_sents])])
    self._filtered_text = self.format_text(self._filtered_text)

  def reindex_embds(self, embds):
    embds = torch.stack(embds, 0)
#     print(embds.shape)
    text = self._split_text()
    text_len = len(text)
    with torch.no_grad():
      text_embd, sents_embd = torch.mean(embds[:text_len], 0), embds[text_len:]
      similarity_scores = [self._cosine_similarity(text_embd, s_embd) for s_embd in sents_embd]
    self._reindex_sentsidx = [i for _,i in sorted(zip(similarity_scores, range(len(self.sents))), reverse=True)]
    top_k_sents = round(self.top_k*len(self.sents))
    self._filtered_text = " ".join([self._sents[i] for i in sorted(self._reindex_sentsidx[:top_k_sents])])
    self._filtered_text = self.format_text(self._filtered_text)

  @property
  def get_all_text(self):
    text = self._split_text()
    all_text = ["a "*self.max_len]+text+self._sents
    tokenized_text = self.tokenizer(all_text,max_length=self.max_len,padding=True,truncation=True, return_tensors="pt")["input_ids"]
    tokenized_text = list(tokenized_text[1:])
    return tokenized_text

  def _split_text(self):
    text_paras = []
    if len(self._text.split()) > self.max_len:
      text_para = ""
      for sent in self.sents:
        if len(text_para.split()) + len(sent.split()) >= self.max_len:
          text_paras.append(text_para)
          text_para = sent
        else:
          text_para += " " + sent
      text_paras.append(text_para)
      return text_paras
    else:
      return [self._text]

  @staticmethod
  def format_text(text):
    text = re.sub(r" +", " ", text)
    text = re.sub(r"[.](?=\w+\b)", " . ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"[.]{2,}", "", text)
    return text

def get_filtered_reviews():
  docs = []
  review_sents = []
  n_sents = 0

  for idx in range(reviews_df.shape[0]):
    doc = Doc(reviews_df.iloc[idx,1])
    doc_text = doc.get_all_text()
    review_sents.append(doc_text)
    n_sents += len(doc_text)
    docs.append(doc)
    if n_sents >= BATCH_SIZE:
      tokenized_sents = tokenizer(review_sents, padding=True, truncation=True,
                        max_length=MAX_SEQ_LEN, return_tensors="pt")["input_ids"]
      review_embd = embeddings_lists(tokenized_sents)
      pass


def embeddings_lists(l_xs, func=MODEL, batch_size=BATCH_SIZE):
  xs = []
  _ = [xs.extend(x) for x in l_xs]
  idxs = [0]
  for x in l_xs:
    idxs.append(idxs[-1] + len(x))

  y_ = []
  for idx in range(0, len(xs), BATCH_SIZE):
    with torch.no_grad():
      input_data = xs[idx:idx+BATCH_SIZE].to(DEVICE))
      y_.extend(list(func(True, x=input_data))
  y = [y_[i1:i2] for i1, i2 in zip(idxs, idxs[1:])]
  return y
