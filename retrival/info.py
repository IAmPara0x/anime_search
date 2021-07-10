import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transfomers import RobertaModel

# PARAMS
MAX_SEQ_LENGTH: int = 384
MODEL = "roberta-base"
BATCH_SIZE = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MINI_BATCH_SIZE = 128
SAMPLE_CLASS = 16
SAMPLE_DATA = 4
EMBEDDING_DIM = 768
LR = 2e-6
ACCUMULATION_STEPS = 2


class SampleClass():
  data = anime_reviews

  def __init__(self, uid, sample_neg_class=True, p=SAMPLE_CLASS, k=SAMPLE_DATA,
               max_seq_length=MAX_SEQ_LENGTH, embedding_dim=EMBEDDING_DIM, device=DEVICE):
    self.uid = uid
    self.k = k
    self.p = p
    self.max_seq_length = max_seq_length
    self.device = device
    self.embedding_dim = embedding_dim
    self.tokenized_data = None
    self.sample_data_idx = random.sample(
        range(anime_reviews[self.uid]["reviews"].shape[0]), self.k)
    if sample_neg_class:
        self.neg_classes = [SampleClass(neg_uid, sample_neg_class=False)
                            for neg_uid in random.sample(anime_reviews[self.uid]["neg_classes"], self.p)]
    self.triplets = []

  def __getitem__(self, idx):
    # _idx = self.sample_data.index(idx)
    return self.tokenized_data[idx]

  @property
  def data(self):
    if self.tokenized_data is None:
      sampled_reviews = anime_reviews[self.uid]["reviews"][self.sample_data_idx].astype(
          np.float)
      self.tokenized_data = torch.tensor(
          sampled_reviews, dtype=torch.long).to(self.device)
    return self.tokenized_data

  def pairwise_distance(self, x, is_squared=False):
    if isinstance(x, tuple):
      m1, m2 = x
      dot1 = torch.matmul(m1, m1.T)
      dot2 = torch.matmul(m2, m1.T)
      dot3 = torch.matmul(m2, m2.T)
      squared_norm1 = torch.diag(dot1)
      squared_norm2 = torch.diag(dot3)
      distances = (squared_norm1.unsqueeze(0) - 2.0 *
                   dot2 + squared_norm2.unsqueeze(1)).T
    else:
      dot1 = torch.matmul(x, x.T)
      squared_norm1 = torch.diag(dot1)
      distances = torch.unsqueeze(
          squared_norm1, 0) - 2.0 * dot1 + torch.unsqueeze(squared_norm1, 1)
  distances = torch.maximum(distances, torch.zeros(
      distances.shape).to(self.device))
    if not is_squared:
      mask = (distances == torch.zeros(
          distances.shape).to(self.device)).float()
      distances = distances + mask * 1e-16
      distances = torch.sqrt(distances)
      distances = distances * (1.0 - mask)
    return distances

  def hard_triplets(self, model):
    anchors = self.data
    neg_data = torch.cat(
      [neg_class.data for neg_class in self.neg_classes], 0)

    mini_batch_data = torch.cat((anchors, neg_data), 0)

    with torch.no_grad():
      embeddings = model(calc_triplets=True, x=mini_batch_data)

    anchors = embeddings[:self.k]
    neg_embeddings = embeddings[self.k:]
    pos_distances = self.pairwise_distance(anchors)
    hard_positives = torch.max(pos_distances, 1)

    anchors = torch.cat((anchors, torch.zeros(
      neg_embeddings.shape[0]-anchors.shape[0], EMBEDDING_DIM).to(self.device)))
    neg_distances = self.pairwise_distance((anchors, neg_embeddings))
    neg_distances = neg_distances[:self.k]
    hard_negatives = torch.min(neg_distances, 1)

    for i in range(self.k):
      anchor = self.__getitem__(i)
      positive = self.__getitem__(hard_positives[1][i])
      negative = self.neg_classes[hard_negatives[1]
                                  [i]//self.k][hard_negatives[1][i] % self.k]
      self.triplets.append((anchor, positive, negative))
    return self.triplets


def sample_data(model):
  anime_uids = list(anime_reviews.keys())
  sampled_uids = random.sample(anime_uids, 2)
  data = []
  for uid in sampled_uids:
    x = SampleClass(uid)
    data.extend(x.hard_triplets(model))
  random.shuffle(data)
  anchors = torch.cat([i[0].unsqueeze(0) for i in data], 0).to(DEVICE)
  positives = torch.cat([i[1].unsqueeze(0) for i in data], 0).to(DEVICE)
  negatives = torch.cat([i[2].unsqueeze(0) for i in data], 0).to(DEVICE)
  return (anchors, positives, negatives)


def train(model, criterion, optim, device=DEVICE):
  model.train()
  avg_loss = []
  acc_loss = []
  i = 1
  step = 0
  tbar = tqdm(total=2_500)

  while True:
    if step == 2_500:
      tbar.close()
      break
    data = sample_data(model)
    anchors = data[0]
    positives = data[1]
    negatives = data[2]

    ebd1, ebd2, ebd3 = model(x1=anchors, x2=positives, x3=negatives)
    loss = criterion(ebd1, ebd2, ebd3)
    tbar.set_description(
      f"AVG_LOSS: {np.average(avg_loss):.5f}, LOSS:{loss.item():.5f}, STEP: {step}")
    loss = loss / ACCUMULATION_STEPS
    loss.backward()
    acc_loss.append(loss.item())
    i += 1
    if i % ACCUMULATION_STEPS == 0:
      step += 1
      optim.step()
      model.zero_grad()
      avg_loss.append(sum(acc_loss))
      tbar.update(1)
      acc_loss = []
  torch.save(model, "/kaggle/working/embedding_model_weights.h5")
