import pickle
import spacy
import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaModel

#PARAMS
TOP_K_RATIO = 0.6
MAX_SEQ_LEN = 384
MIN_REVIEW_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NLP = spacy.load("en_core_web_sm")
TOKENIZER = RobertaTokenizerFast.from_pretrained("roberta-base")
df = pd.read_csv("/kaggle/input/review-embeddings/reviews_features.csv")


class Doc:
  model = MODEL
  tokenizer = TOKENIZER
  nlp = NLP
  device = DEVICE
  max_len = MAX_SEQ_LEN
  top_k = TOP_K_RATIO

  def __init__(self, text):
    self._text = self.format_text(text)
    self._s_doc = self.nlp(text)
    self._sents = [i.text for i in self._s_doc.sents]
    self._reindex_sentsidx = None

    self._reindex()

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
      return text_paras
    else:
      return [self._text]

  @staticmethod
  def format_text(text):
    text = re.sub(" +", " ", text)
    text = re.sub(r"[.](?=\w+\b)", " . ", text)
    text = re.sub(" +", " ", text)
    text = re.sub("[.]{2,}", "", text)
    return text
