import random
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from transformers import LongformerTokenizerFast, LongformerForMaskedLM, AdamW


#Params
SEQ_LENGTHS = [256,512,1024]
MAX_SEQ_LENGTH = max(SEQ_LENGTHS)
MODEL = "allenai/longformer-base-4096"
PROB_SEQ_LENGTHS = [0.25,0.5,0.25]
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-6
BETAS = (0.9, 0.98)

#Initialization
model = LongformerForMaskedLM.from_pretrained(MODEL)
tokenizer = LongformerTokenizerFast.from_pretrained(MODEL)
VOCAB_DICT = tokenizer.get_vocab()
VOCAB_LIST = list(VOCAB_DICT)
LEN_VOCAB_LIST = len(VOCAB_LIST)
PAD_TOKEN_ID = tokenizer.pad_token_id
MASK_TOKEN_ID = tokenizer.mask_token_id


def mask_random_word(tokens, label):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
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
                tokens[i] = MASK_TOKEN_ID

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.randrange(LEN_VOCAB_LIST)
            else:
                label[i] = tokens[i].item()
        else:
            label[i] = -100
    return tokens, label

def load_sents(batch_num_start, batch_num_end):
  b_sents = []
  for i in range(batch_num_start,batch_num_end):
    with open(f"/kaggle/input/review-ranking-pt2/data_batch{i}.pkl", "rb") as f:
      data = pickle.load(f)
      b_sents.extend(data["input_data"])
      del data
      gc.collect()
  random.shuffle(b_sents)
  return b_sents

def mask_sents(b_sents: List[List[str]]):
  input_sents = []
  for sents in tqdm(b_sents):
    sent_len = random.choices(SEQ_LENGTHS, weights=PROB_SEQ_LENGTHS)[0]
    x = ""
    for sent in sents:
      if len(x.split()) >= sent_len:
        input_sents.append(x)
        sent_len = random.choices(SEQ_LENGTHS, weights=PROB_SEQ_LENGTHS)[0]
        x = sent
      else:
        x += " " + sent
  input_sents = tokenizer(input_sents, return_tensors="pt", max_length=MAX_SEQ_LENGTH, trucation=True, padding=True)["input_ids"]
  labels = input_sents.detach().clone()

  for idx, tokens in enumerate(tqdm(input_sents)):
    tokens, labels = random_word(tokens[1:-1], labels[idx,1:-1])
    input_sents[idx, 1:-1] = tokens
    labels[idx,1:-1] = label
    labels[idx,[0,-1]] = -100
  return input_sents, torch.tensor(labels)


