
from transformers import RobertaModel
import torch
import gc

#PARAMS
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Data():
  def __init__(self):
    self.data = np.load("/kaggle/input/review-emddings-pt2/review_embeddings_arrays.npy", allow_pickle=True)
    self.data = self.data[:,1:]
    gc.collect()

  def __getitem__(self, idx: int):
    return (torch.tensor(self.data[idx,0]),
        torch.tensor(self.data[idx,1]),
        torch.tensor(self.data[idx,2]/5))

  def __len__(self) -> int:
    return len(self.data)


class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.roberta = RobertaModel.from_pretrained("/kaggle/input/review-emddings-pt2/roberta_base_anime_finetuned.h5",
        output_hidden_states=True)
    self.hid_mix = 3
    self.feats = self.roberta.pooler.dense.out_features
    self.embedding_layer = nn.Linear(self.feats, self.feats)

  def forward(self, x):
    outputs = self.roberta(x)
    hidden_states = outputs[2]

    hmix = []
    for i in range(1, self.hid_mix+1):
      hmix.append(hidden_states[-i][:,0].reshape(1,self.feats))

    hmix_tensor = torch.cat(hmix,0)
    pool_tensor = torch.mean(hmix_tensor,0)

    embeddings = self.embedding_layer(pool_tensor)
    return embeddings

def cosine_similarity_loss(emb1, emb2, labels):
  output = torch.cosine_similarity(emb1, emb2)
  loss = F.mse_loss(output, labels.view(-1))
  return loss


def train():
  tbar = tqdm(data_loader)
  avg_loss = []
  for batch in tbar:
    optimizer.zero_grad()
    x1 = batch[0].to(DEVICE)
    x2 = batch[1].to(DEVICE)
    y = batch[2].to(DEVICE)

    x_ = torch.cat((x1,x2),0)

    output = model(x_)
    embeddings1 = output[:BATCH_SIZE]
    embeddings2 = output[BATCH_SIZE:]

    loss = cosine_similarity_loss(embeddings1, embeddings2, labels)
    loss.backwards()
    optimizer.step()
    avg_loss.append(loss.item())
    tbar.set_description(f"LOSS: {loss.item()} AVG_LOSS: {np.average(avg_loss)}")

