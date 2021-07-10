import spacy
import gc
from transfomers import RobertaTokenizerFast

#PARAMS
MAX_REVIEW_LEN = 128
MIN_REVIEW_LEN = 64
MODEL = "roberta-base"

nlp = spacy.load("en_core_web_trf")
df = pd.read_csv("/kaggle/input/review-embeddings/reviews_features.csv")
df = df[df["review_score"] >= 6]
df = df[["anime_uid", "text"]]

def split_review(review):
  doc = nlp(review)
  review_sents = []

  x = ""
  for sent in doc.sents:
    if len(sent.text.split()) > 1:
      if len(x.split()) + len(sent.text.split()) <= MAX_REVIEW_LEN:
        x += " " + sent.text
      else:
        review_sents.append(x)
        x = sent.text

  if len(x.split()) < MIN_REVIEW_LEN and len(review_sents):
    review_sents[-1] += " " + x
  else:
    review_sents.append(x)
  return review_sents

def create_sents():
  all_sents = []
  tokenized_sents = []
  labels = []
  df = df.to_numpy()
  for idx in tqdm(range(df.shape[0])):
    res = split_review(df[idx, 1])
    all_sents.extend(res)
    labels.extend([df[idx,0]]*len(res))

  print(len(all_sents))
  for idx in range(0, len(all_sents), 10_000):
    x = tokenizer(all_sents[idx,idx+10_000], truncation=True, max_length=MAX_REVIEW_LEN,
        padding=True, return_tensors="pt")["input_ids"]
    tokenized_sents.append(x)
    gc.collect()
  tokenized_sents = torch.cat(x, 0)
  return tokenized_sents, labels


###Creating Embeddings
sents_embeddings = []


with torch.no_grad():
  for idx in range(tokenized_sents.shape[0], 384):
    b_e = model(True, x=tokenized_sents[idx:idx+384].to("cuda")).cpu().numpy()
    sents_embeddings.append(b_e)
