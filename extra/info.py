import json
import gc
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tansfomers import RobertaModel, RobertaTokenizerFast

#PARAMS
MAX_SEQ_LENGTH = 256
FEATURE_SPACE = 768
MODEL = "roberta-base"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MINI_BATCH_SIZE = 128

#INTIALIZATION
reviews_df = pd.read_csv("/kaggle/input/review-ranking/review_features_v2.csv")

animes_df = pd.read_csv("/kaggle/input/mal-reviews-dataset/animes_real.csv")
animes_uids = reviews_df["anime_uid"].unique()
animes_df = animes_df[animes_df['uid'].apply(lambda x: x in anime_uids)]
animes_df["genre"] = animes_df["genre"].apply(json.loads)

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL)


#data creation

def create_anime_clusters():
  anime_clusters = {}

  for uid in tqdm(anime_uids):
    uid_genre = animes_df[animes_df["uid"]==uid].iloc[0,-3]
    anime_clusters[uid] = list(anime_df[anime_df['genre'].
                         apply(lambda x: set(x).isdisjoint(uid_genre))]["uid"])
  return anime_clusters

def create_data(anime_clusters):
  reviews_df["tokenized_text"] = tokenizer(list(reviews_df["text"]),
      max_length=MAX_SEQ_LENGTH, padding=True, truncation=True)
  anime_reviews = list(reviews_df.groupby(["anime_uid"]))
  anime_reviews = { i[0]: i[1].sort_values(["helpful"], ascending=False).to_numpy() for i in anime_reviews }
  data = []

  for uid,reviews in tqdm(anime_reviews.items()):
    neg_animes = anime_clusters[uid].copy()
    i = 0
    for idx in range(reviews.shape[0]):
      for idx2 in range(idx+1, reviews.shape[0]):
        neg_anime_uid = neg_animes.pop()
        try:
          neg_sent = anime_reviews[neg_anime_uid][i,-1]
        except IndexError:
          neg_sent = anime_reviews[neg_anime_uid][0,-1]
        data.append([
          reviews[idx,-1], reviews[idx2,-1],neg_sent
          ])
        if not len(neg_animes):
          neg_animes = anime_clusters[uid].copy()
          i += 1
  return torch.tensor(data)

anime_clusters = create_anime_clusters()

#Creating model
class Model(nn.Module):
  def __init__(self):
    self.roberta = RobertaModel.from_pretrained(MODEL, output_hidden_states=True)
    self.hid_mix = 5
    self.feats = self.roberta.pooler.dense.out_features
    self.embedding_layer = nn.Linear(self.feats, self.feats)
    self.tanh = nn.Tanh()

  def forward(self, x1,x2,x3):
    return self.forward_once(x1), self.forward_once(x2), self.forward_once(x3)

  def forward_once(self, x):
    output = self.roberta(x)
    hidden_states = outputs[2]
    hmix = []
    for i in range(1, self.hid_mix+1):
      hmix.append(hidden_states[-i][:,0].reshape((-1,1,self.feats)))

    hmix_tensor = torch.cat(hmix,1)
    pool_tensor = torch.mean(hmix_tensor,1)
    pool_tensor = self.tanh(pool_tensor)
    embeddings = self.embedding_layer(pool_tensor)
    return embeddings

def dot_prod_loss(embd1, embd2, embd3, margin=5):
  sim1 = torch.diag(
      torch.matmul(embd1,embd2.T) * torch.eye(BATCH_SIZE).to(DEVICE), 0)
  sim2 = torch.diag(
      torch.matmul(embd1,embd3.T) * torch.eye(BATCH_SIZE).to(DEVICE),0)
  loss = torch.mean((sim1 + sim2))
  return loss


def train_loop(data_loader, model, optimizer, device, scheduler=None):
  model.train()
  avg_loss = []
  tbar = tqdm(enumerate(data_loader))
  for idx, batch in tbar:
    x1 = batch[:,0,:].to(device)
    x2 = batch[:,1,:].to(device)
    x3 = batch[:,2,:].to(device)

    embd1,embd2,embd3 = model(x1,x2,x3)
    loss = criterion(embd1,embd2,embd3)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if scheduler is not None:
      scheduler.step()
    avg_loss.append(loss.item())
    tbar.set_desription(f"loss: {loss.item()}, avg_loss: {np.average(avg_loss)}")

