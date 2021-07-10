
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#PARAMS
QUERY_LENS = [8,16]
MAX_SEQ_LENGTH=2048
LR=2e-6
BATCH_SIZE = 3
SAMPLE_CLASS = 12
ACCUMULATION_STEPS= SAMPLE_CLASS // BATCH_SIZE
TOTAL_STEPS = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SampleClass:
  reviews_df = reviews_df.to_numpy()
  pad_token_id = tokenizer.pad_token_id
  data_dict = data


class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.long_former = LongformerModel.from_pretrained("allenai/longformer-base-4096", output_hidden_states=True)
    self.hid_mix = 5
    self.feats = self.long_former.pooler.dense.out_features
    self.dropout = nn.Dropout(0.2)
    self.output_layer = nn.Linear(self.feats, 1)

  def forward(self,x):
    outputs = self.long_former(x)
    hidden_states = outputs[2]
    hmix = []
    for i in range(1, self.hid_mix+1):
      hmix.append(hidden_states[-i][:,0].reshape((-1,1,self.feats)))

    hmix_tensor = torch.cat(hmix,1)
    pool_tensor = torch.mean(hmix_tensor,1)
    logits = self.dropout(pool_tensor)
    return self.output_layer(logits)

def sample_data(data, sample_class=BATCH_SIZE):
  sampled_objs = random.sample(data, sample_class)
  sample_type = [0,1,2]
  batch_data = []
  batch_labels = []

  for sampled_obj in sampled_objs:
    c = random.choices(sample_type, weights=[0.25,0.25,0.5])[0]
    if c == 0:
      _x,_y = sampled_obj.sample_positive
    elif c == 1:
      _x,_y = sampled_obj.sample_negative
    else:
      _x,_y = sampled_obj.sample_false
    batch_data.append(_x)
    batch_labels.append(_y)
  return torch.tensor(batch_data).to(DEVICE), torch.tensor(batch_labels).to(DEVICE)


def train(model, optimizer, criterion, anime_objs):
  model.train()
  avg_loss = []
  acc_loss = []
  avg_preds = []
  steps = 0
  i = 1

  tbar = tqdm(total=TOTAL_STEPS)

  while True:
    if steps == TOTAL_STEPS:
      tbar.close()
      break
    x, y = sample_data(anime_objs, BATCH_SIZE)
    y = y.unsqueeze(1).float()
    output = model(x)
    loss = criterion(output,y)
    tbar.set_description(f"AVG_LOSS: {np.average(avg_loss):.5f}, LOSS:{loss.item():.5f},\
                            AVG_PREDS: {avg_preds.count(True)/len(avg_preds)}, STEP: {step}")
    loss = loss / ACCUMULATION_STEPS

    loss.backward()
    acc_loss.append(loss.item())
    i += 1
    if i % ACCUMULATION_STEPS == 0:
      step += 1
      optimizer.step()
      optimizer.zero_grad()
      avg_loss.append(sum(acc_loss))
      tbar.update(1)
      acc_loss = []
    with torch.no_grad():
      output_prob = F.softmax(output)
      output_prob = (output_prob > 0.5).float()
      preds = (output_prob == y).squeeze().tolist()
      avg_preds.append(preds)
