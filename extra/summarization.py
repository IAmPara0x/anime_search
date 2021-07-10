
import pickle
import gc
import random
from sklearn.utils import shuffle
from transformers import pipeline
import torch

#Parameters
MAX_SUMMARIZATION_LENGTH: int = 192
TOP_K: int = 72
TOP_P: int = 0.95
SAMPLE: int = True
NUM_BEAMS: int = 4
BATCH_SIZE: int = 24
EARLY_STOPPING: bool = True
TRUNCATION: bool = True
MODEL:str = "sshleifer/distilbart-cnn-12-6"

#Model
summarizer = pipeline("summarization", model=MODEL, device=0)

def main(batch_num):
  with open(f"/kaggle/input/review-ranking-pt2/data_batch{batch_num}.pkl", "rb") as f:
    data = pickle.load(f)
    input_data = data["input_data"]
    labels = data["labels"]
    input_data,labels = shuffle(input_data,labels)
    del data
    gc.collect()

  summ_input_data = []

  tbar = tqdm(input_data)
  b_review_sents = []
  b_review_len = [0]
  for review_sents in tbar:
    b_review_sents.extend(review_sents)
    b_review_len.append(b_review_len[-1] + len(review_sents))
    if len(b_review_sents) >= BATCH_SIZE:
      tbar.set_description(f"LEN B_DATA: {len(b_review_sents)}, LEN B_DATA_LEN: {len(b_review_len)}")
      b_review_summ = summarizer(b_review_sents, max_length=MAX_SUMMARIZATION_LENGTH, do_sample=SAMPLE,
                    top_k=TOP_K, top_p=TOP_P, num_beams=NUM_BEAMS,
                    early_stopping=EARLY_STOPPING, truncation=TRUNCATION)
      b_review_summ = [review_summ["summary_text"] for review_summ in b_review_summ]
      for start, end in zip(b_review_len, b_review_len[1:]):
        summ_input_data.append(" ".join(b_review_summ[start:end]))
        b_review_sents = []
        b_review_len = [0]

  with open(f"/kaggle/working/data_batch{batch_num}.pkl", "wb") as f:
    data = {}
    data["input_data"] = summ_input_data
    data["labels"] = labels
    pickle.dump(data, f)



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM1 = 1280
EMBEDDING_DIM2 = 768

                # make faiss available
index1 = faiss.IndexFlatL2(EMBEDDING_DIM1)   # build the index
print(index1.is_trained)
index1.add(data1["embeddings"])


index2 = faiss.IndexFlatL2(EMBEDDING_DIM2)   # build the index
print(index2.is_trained)
index2.add(data2["embeddings"])

while True:
  count = 0
  for i1, i2 in zip(I1,I2):
    count += 1
    text1 = tokenizer.decode(data1["tokenized_sents"][i1], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    text2 = tokenizer.decode(data2["tokenized_sents"][i2], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(f"Neighborh: {count}")
    print(f"Text1: {text1} \n
            Anime: {data1["labels"][i1]}")

    print(f"Text2: {text2} \n
            Anime: {data2["labels"][i2]}")
    br = input()
    if br:
      break
