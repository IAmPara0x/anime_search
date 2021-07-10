
import gc
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaModel

#PARAMS
MAX_SEQ_LEN = 128
TRUCATION = True
PADDING = True
MODEL = "roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL)


class DanEncoder(nn.Module):
  def __init__(self, input_dim:int, embedding_dim:int, n_hidden_layers:int, n_hidden_units:[int], dropout_prob:int):
    super(DanEncoder, self).__init__()

    assert n_hidden_layers != 0
    assert len(n_hidden_units) + 1 == n_hidden_layers

    encoder_layers = []
    for i in range(n_hidden_layers):
      if i == n_hidden_layers - 1:
        out_dim = embedding_dim
        encoder_layers.extend(
          [
            nn.Linear(input_dim, out_dim),
          ])
        continue
      else:
        out_dim = n_hidden_units[i]

      encoder_layers.extend(
        [
          nn.Linear(input_dim, out_dim),
          nn.Tanh(),
          nn.Dropout(dropout_prob, inplace=False),
        ]
      )
      input_dim = out_dim
    self.encoder = nn.Sequential(*encoder_layers)

  def forward(self, x_array):
      return self.encoder(x_array)



#MODEL
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.roberta = RobertaModel.from_pretrained("/kaggle/input/review-emddings-pt2/roberta_base_anime_finetuned.h5", output_hidden_states=True)
    self.hid_mix = 5
    self.feats = self.roberta.pooler.dense.out_features
    self.embedding_layers = DanEncoder(self.feats, 1280, 4, [896, 1024, 1152], 0.1)
    self.tanh = nn.Tanh()

  def forward(self, calc_triplets=False, **kwargs):
    if calc_triplets:
        x = kwargs["x"]
        return self.forward_once(x)
    x1 = kwargs["x1"]
    x2 = kwargs["x2"]
    x3 = kwargs["x3"]
    return self.forward_once(x1), self.forward_once(x2), self.forward_once(x3)

  def forward_once(self, x):
    outputs = self.roberta(x)
    hidden_states = outputs[2]
    hmix = []
    for i in range(1, self.hid_mix+1):
      hmix.append(hidden_states[-i][:,0].reshape((-1,1,self.feats)))

    hmix_tensor = torch.cat(hmix,1)
    pool_tensor = torch.mean(hmix_tensor,1)
    pool_tensor = self.tanh(pool_tensor)
    embeddings = self.embedding_layers(pool_tensor)
    return embeddings

def load_data():
  with open("/kaggle/input/review-re-ranking/queries_dict.pkl", "rb") as f:
      queries_dict = pickle.load(f)
  with open("/kaggle/input/review-re-ranking/answers_dict.pkl", "rb") as f:
      answers_dict = pickle.load(f)
  with open("/kaggle/input/review-re-ranking/neg_classes_dict.pkl", "rb") as f:
      neg_classes_dict = pickle.load(f)
  return queries_dict, answers_dict, neg_classes_dict


queries_dict = {uid: {"sents":sents} for uid, sents in queries_dict.items()}
answers_dict = {uid: {"sents":sents} for uid, sents in answers_dict.items()}

def process_list(l_xs, func):
  xs = []
  _ = [xs.extend(x) for x in l_xs]
  idxs = [0]
  for x in l_xs:
    idxs.append(idxs[-1] + len(x))
  xs = func(xs)["input_ids"]
  y = [xs[i1:i2] for i1, i2 in zip(idxs, idxs[1:])]
  return y

for uid in answers_dict.keys():
  answers_dict[uid] = {"sents": answers_dict[uid], "tokenized_sents": process_list(answers_dict[uid])}


def embeddings_lists(l_xs, func, batch_size=BATCH_SIZE):
  xs = []
  _ = [xs.extend(x) for x in l_xs]
  idxs = [0]
  for x in l_xs:
    idxs.append(idxs[-1] + len(x))

  y_ = []
  for idx in range(0, len(xs), BATCH_SIZE):
    with torch.no_grad():
      y_.extend(list(func(xs[idx:idx+BATCH_SIZE].to(DEVICE)).cpu().numpy()))
  y = [y_[i1:i2] for i1, i2 in zip(idxs, idxs[1:])]
  return y
