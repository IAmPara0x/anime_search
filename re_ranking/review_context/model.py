#PARAMS
LR = 1e-5
SAMPLE_SIZE = 3
ANS_MAX_LEN = 512
QUERY_MAX_LEN = 64
MARGIN = 1

with open("/kaggle/input/k/iamparadox/k/iamparadox/review-context/data.pkl", "rb") as f:
    DATA = pickle.load(f)
    DATA = {uid: d for uid,d in DATA.items() if len(d["answers"]) >= SAMPLE_SIZE}
ANIME_UIDS = list(DATA.keys())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = RobertaTokenizerFast.from_pretrained("roberta-base")

class SampleClass:
  data = DATA
  sample_size = SAMPLE_SIZE
  tokenizer = TOKENIZER
  ans_len = ANS_MAX_LEN
  query_len = QUERY_MAX_LEN

  def __init__(self,uid, main=False):
    self.sampled_answers = self.tokenizer(random.sample(self.data[uid]["answers"], self.sample_size),
                            padding="max_length", truncation=True, max_length=self.ans_len, return_tensors="pt")["input_ids"]
    if main:
      self.sampled_queries = self.tokenizer(random.sample(self.data[uid]["queries"], self.sample_size),
                            padding="max_length", truncation=True, max_length=self.query_len, return_tensors="pt")["input_ids"]
      self.sampled_neg_cls = [SampleClass(sampled_uid)
                              for sampled_uid in random.sample(self.data[uid]["neg_cls_uids"], self.sample_size)]

  def sample_hard(self, model):
    pass


  @property
  def random_sample(self):
    return (self.sampled_queries[0].unsqueeze(0),
            self.sampled_answers[0].unsqueeze(0),
            self.sampled_answers[1].unsqueeze(0),
            self.sampled_neg_cls[0].sampled_answers[0].unsqueeze(0))


class QueryModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.roberta = RobertaModel.from_pretrained("/kaggle/input/review-emddings-pt2/roberta_base_anime_finetuned.h5",
        output_hidden_states=True)
    self.hid_mix = 5
    self.feats = self.roberta.pooler.dense.out_features
    self.embedding_layer = nn.Linear(self.feats, self.feats)
    self.tanh = nn.Tanh()

  def forward(self,x):
    outputs = self.roberta(x)
    hidden_states = outputs[2]
    hmix = []
    for i in range(1, self.hid_mix+1):
      hmix.append(hidden_states[-i][:,0].reshape((-1,1,self.feats)))

    hmix_tensor = torch.cat(hmix,1)
    pool_tensor = torch.mean(hmix_tensor,1)
    logits = self.tanh(pool_tensor)
    return self.embedding_layer(logits)

class StoryModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.roberta = RobertaModel.from_pretrained("/kaggle/input/review-emddings-pt2/roberta_base_anime_finetuned.h5",
        output_hidden_states=True)
    self.hid_mix = 5
    self.feats = self.roberta.pooler.dense.out_features
    self.dropout = nn.Dropout(0.1)
    self.embedding_layer = nn.Linear(self.feats, self.feats)
    self.tanh = nn.Tanh()

  def forward(self,x):
    outputs = self.roberta(x)
    hidden_states = outputs[2]
    hmix = []
    for i in range(1, self.hid_mix+1):
      hmix.append(hidden_states[-i][:,0].reshape((-1,1,self.feats)))

    hmix_tensor = torch.cat(hmix,1)
    pool_tensor = torch.mean(hmix_tensor,1)
    pool_tensor = self.tanh(pool_tensor)
    logits = self.dropout(pool_tensor)
    return self.embedding_layer(logits)


class AttentionLayer(nn.Module):
  def __init__(self, embed_dim, attn_h, dropout_prob):
    super().__init__()
    self.attn_h = attn_h
    self.embed_dim = embed_dim
    self.d_k = self.embed_dim // self.attn_h

    self.q_v1 = nn.Linear(self.embed_dim, self.embed_dim)
    self.a_v1 = nn.Linear(self.embed_dim, self.embed_dim)

    self.a_v2 = nn.Linear(self.embed_dim, self.embed_dim)

    self.mlp_q = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Tanh(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Tanh(),
                nn.Linear(self.embed_dim, self.embed_dim),
        )
    self.mlp_a = nn.Sequential(
                nn.Linear(embed_dim, embed_dim*2),
                nn.Tanh(),
                nn.Linear(embed_dim*2, embed_dim*2),
                nn.Tanh(),
                nn.Linear(embed_dim*2, embed_dim),
        )

  def forward(self, b_q, b_a):
    bs = b_q.size(0)
    bq_p1 = self.q_v1(b_q).view(bs, self.attn_h, self.d_k)
    ba_p1 = self.a_v1(b_a).view(bs, self.attn_h, self.d_k)
    similarity_scores = torch.nn.functional.cosine_similarity(
                          bq_p1.transpose(-2,-1), ba_p1.transpose(-2,-1))

    ba_props = self.a_v2(b_a).view(bs, self.attn_h, self.d_k)
    ba_props = (ba_props.transpose(-2,-1) * similarity_scores[:,None]).transpose(-2,-1)

    bq_mlp = self.mlp_q(b_q)
    ba_mlp = self.mlp_a(b_a)
    return ba_props, bq_mlp, ba_mlp


class Model(nn.Module):
  def __init__(self, embed_dim, attn_l, attn_h, dropout_prob):
    super().__init__()
    self.attn_h = attn_h
    self.attn_l = attn_l
    self.embed_dim = embed_dim
    self.d_k = self.embed_dim // self.attn_h
    self.query_model = QueryModel()
    self.story_model = StoryModel()
    self.attention_layers = nn.ModuleList([AttentionLayer(embed_dim, attn_h, dropout_prob) for _ in range(attn_l)])

  def forward(self, b_q, b_a, b_p, b_n):
    b_qa = self.query_model(b_q)

    bs = b_a.size(0)
    answers_seq   = torch.vstack((b_a, b_p, b_n))
    answers_vec   = self.story_model(answers_seq)
    questions_vec = torch.vstack((b_qa, b_qa, b_qa))

    answers_props = []
    for layer in self.attention_layers:
      ans_props, questions_vec, answers_vec = layer(questions_vec, answers_vec)
      answers_props.append(ans_props)
    answers_props = torch.cat(answers_props, 1)

    answers_props = answers_props.view(3, bs, self.attn_h*self.attn_l, self.d_k)
    a_props, p_props, n_props = answers_props
    return a_props, p_props, n_props



def triplet_loss(anc,pos,neg, margin=1):
  pos_d = torch.nn.functional.pairwise_distance(anc.transpose(-2, -1), pos.transpose(-2, -1))
  neg_d = torch.nn.functional.pairwise_distance(anc.transpose(-2, -1), neg.transpose(-2, -1))

  diff_dist = (pos_d - neg_d) + margin
  diff_dist = diff_dist.masked_fill(diff_dist.le(0), 0)
  loss = torch.mean(torch.sum(diff_dist, 1))
  return loss

