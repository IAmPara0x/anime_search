import json
import gc
import torch
from transformers import pipeline
from tqdm import tqdm

# Parameters
MAX_SUMMARIZATION_LENGTH: int = 192
TOP_K: int = 72
TOP_P: int = 0.95
SAMPLE: int = True
NUM_BEAMS: int = 4
BATCH_SIZE: int = 24
EARLY_STOPPING: bool = True
TRUNCATION: bool = True
MODEL: str = "sshleifer/distilbart-cnn-12-6"


def main():
    df = pd.read_csv("/kaggle/input/review-ranking/review_features.csv")
    del df["Unnamed: 0"]
    df = df.sample(frac=1)

    b_review_sents = []
    b_review_idxs = []
    b_review_len = [0]

    for idx in tqdm(range(df.shape[0])):
        review_sents = json.loads(df.iloc[idx, -1])
        if len(review_sents):
            b_review_sents.extend(review_sents)
            b_review_len.append(b_review_len[-1] + len(review_sents))
            b_review_idxs.append(idx)
        if len(b_review_sents) >= BATCH_SIZE:
            b_review_summ = summarizer(b_review_sents, max_length=MAX_SUMMARIZATION_LENGTH, do_sample=SAMPLE,
                                       top_k=TOP_K, top_p=TOP_P, num_beams=NUM_BEAMS,
                                       early_stopping=EARLY_STOPPING, truncation=TRUNCATION)
            b_review_summ = [review_summ["summary_text"]
                             for review_summ in b_review_summ]

            for start, end, idx in zip(b_review_len, b_review_len[1:]):
                df.iloc[idx, 2] = " ".join(b_review_summ[start:end])
                b_review_sents = []
                b_review_len = [0]
