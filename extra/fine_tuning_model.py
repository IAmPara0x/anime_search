from transformers import RobertaForMaskedLM, RobertaTokenizerFast
import random
import logging
from tqdm import tqdm
import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#Parameters
MAX_SEQ_LENGTH = 256
MODEL = "roberta-base"
PROB_MERGE_SENTS = 0.25
BATCH_SIZE = 28
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
BETAS =(0.9, 0.98)

model = RobertaForMaskedLM.from_pretrained(MODEL)
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL)
VOCAB_DICT = tokenizer.get_vocab()
VOCAB_LIST = list(VOCAB_DICT)
LEN_VOCAB_LIST = len(VOCAB_LIST)

def random_word(tokens, label):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    PAD_TOKEN = 0
    for i, token in enumerate(tokens):
        if token == PAD_TOKEN:
            label[i:] = -100
            break
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = 103

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.randrange(LEN_VOCAB_LIST)
            else:
                label[i] = tokens[i].item()
        else:
            label[i] = -100
    return tokens, label

def create_training_data(batch_num_start, batch_num_end):
  b_sents = []
  for i in range(batch_num_start,batch_num_end):
      with open(f"/kaggle/input/review-ranking-pt2/data_batch{i}.pkl", "rb") as f:
        data = pickle.load(f)
        b_sents.extend(data["input_data"])
        del data
        gc.collect()

  random.shuffle(b_sents)
  input_sents = []

  for sents in tqdm(b_sents):
    for sents_x, sents_y in zip(sents, sents[1:]):
      prob = random.random()
      if prob < PROB_MERGE_SENTS:
        input_sents.append((sents_x + " " + sents_y).lower())
      else:
        input_sents.append(sents_x.lower())
  input_sents = tokenizer(input_sents, return_tensors="pt", max_length=MAX_SEQ_LENGTH,
      truncation=True, padding=True)["input_ids"]
  labels = input_sents.clone().detach()

  for idx, tokens in enumerate(tqdm(input_sents)):
    tokens, label = random_word(tokens[1:-1], labels[idx,1:-1])
    input_sents[idx, 1:-1] = tokens
    labels[idx,1:-1] = label
    labels[idx,[0,-1]] = -100
  return input_sents, torch.tensor(labels)

def train_model(input_sents, labels):
  model.train()
  avg_loss = []
  tbar = tqdm(range(0, len(input_sents), BATCH_SIZE))
  for i in tbar:
      optim.zero_grad()
      x = input_sents[i:i+BATCH_SIZE].to(DEVICE)
      y = labels[i:i+BATCH_SIZE].to(DEVICE)
      outputs = model(x)
      loss = F.cross_entropy(outputs.logits.view(-1, tokenizer.vocab_size), y.view(-1))
      loss.backward()
      optim.step()
      avg_loss.append(loss.item())
      tbar.set_description(f"loss:{loss.item()} avg_loss: {np.average(avg_loss)}")

def eval_model(input_sents, labels):
  print("ENTERING EVAL MODE")
  model.eval()
  avg_loss = []
  tbar = tqdm(range(0, len(input_sents), BATCH_SIZE*2))
  for i in tbar:
      x = input_sents[i:i+BATCH_SIZE*2].to(DEVICE)
      y = labels[i:i+BATCH_SIZE*2].to(DEVICE)
      with torch.no_grad():
          outputs = model(x)
          loss = F.cross_entropy(outputs.logits.view(-1, tokenizer.vocab_size), y.view(-1))
          avg_loss.append(loss.item())
      tbar.set_description(f"eval loss:{loss.item()} avg_eval_loss: {np.average(avg_loss)}")

def main():

  model = model.to(DEVICE)
  optim = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS)

  input_sents, labels = create_training_data(0, 1)
  gc.collect()
  train_model()

  del input_sents, labels
  gc.collect()
  input_sents, labels = create_training_data(16,17)
  eval_model(input_sents, labels)

  ##SAVE MODEL
  model = model.cpu()
  torch.save(model.state_dict(), "/kaggle/working/roberta_base_anime_finetuned.h5")
