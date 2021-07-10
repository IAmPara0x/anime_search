import torch.nn as nn

#PARAMS
SAMPLE_SIZE = 4
LR = 1e-5
EMBEDDING_DIM = 1280
MAX_SEQ_LEN = 16

class SampleData:
  queries_dict = queries_dict
  answers_dict = answers_dict
  neg_classes_dict = neg_classes_dict

  def __init__(self, uid, main=False):
    self.uid = uid
    self.sample_size = SAMPLE_SIZE
    self.embed_dim = EMBEDDING_DIM
    self.max_seq_len = MAX_SEQ_LEN
    self._sampled_queries = torch.tensor(random.sample(queries_dict[uid]["embeddings"], self.sample_size))
    self._sampled_answers = random.sample(answers_dict[uid]["embeddings"], self.sample_size)
    self._sampled_answers = [torch.tensor(ans) for ans in self._sampled_answers]
    if main:
      self._neg_classes = [SampleData(uid) for uid in random.sample(neg_classes_dict[uid], self.sample_size)]

  @property
  def queries(self):
    return self._sampled_queries

  @property
  def answers(self):
    return self._sampled_answers

  def hard_negatives(self, model):
    queries = self.queries
    pos_data = self._get_data(self.answers)
    neg_data = []
    for neg_class in self._neg_classes:
      neg_data.append(self._get_data(neg_class.answers))
    neg_data = torch.cat(neg_data, 0)

  def _get_data(self, answers):
    ans_comb = list(itertools.combinations(list(range(self.sample_size)), 2))

    input_seq = []

    for comb in ans_comb:
      b_ans = itemgetter(*list(comb))(answers)
      b_ans = [ans for ans in b_ans]
      b_ans = torch.cat(b_ans, 0)
      b_seq = torch.cat((queries, b_ans.view(1,-1).repeat(self.sample_size, 1)), 1).reshape(
                            self.sample_size,-1,self.embed_dim)
      b_seq = b_seq[:, :self.max_seq_len, :]
      seq_len = b_seq.shape[1]
      if seq_len != self.max_seq_len:
        padd = torch.zeros(self.sample_size, self.max_seq_len-seq_len, self.embed_dim)
        b_seq = torch.cat((b_seq, padd), 1)
        assert b_seq.shape == (self.sample_size, self.max_seq_len, self.embed_dim)
      input_seq.append(seq_len)
    return input_seq

  def random_sample(self):
    pos_data = self._get_data(self.answers)
    for neg_class in self._neg_classes:
      neg_data.append(self._get_data(neg_class.answers))
    neg_data = torch.cat(neg_data, 0)
    print(pos_data.shape)
    print(neg_data.shape)





class AttentionLayer(nn.Module):
  def __init__(self, embed_dim, num_heads, dropout_p):
    super(AttentionLayer, self).__init__()
    self.multiheadattention = nn.MultiheadAttention(embed_dim, num_heads)

    self.q_proj = nn.Linear(embed_dim, embed_dim) #(B, S, E1) -> (B, S, E2)
    self.k_proj = nn.Linear(embed_dim, embed_dim) #(B, S, E1) -> (B, S, E2)
    self.v_proj = nn.Linear(embed_dim, embed_dim) #(B, S, E1) -> (B, S, E2)

    self.outputLayer = nn.Sequential(
                        nn.Linear(embed_dim, embed_dim),
                        nn.GELU(),
                        nn.LayerNorm(embed_dim, elementwise_affine=True),
                        nn.Dropout(p=dropout_p),
                        nn.Linear(embed_dim, int(embed_dim*1.5)),
                        nn.GELU(),
                        nn.LayerNorm(int(embed_dim*1.5), elementwise_affine=True),
                        nn.Dropout(p=dropout_p),
                        nn.Linear(int(embed_dim*1.5), embed_dim),
                        )

  def forward(self, x):
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    x_,_ = self.multiheadattention(q,k,v)
    return self.outputLayer(x_)


class Model(nn.Module):
  def __init__(self, layers=4, embed_dim=1280, num_heads=4, dropout_p=0.1):
    super(Model, self).__init__()
    self.attention_layers = nn.ModuleList([AttentionLayer(embed_dim, num_heads, dropout_p)])
    self.mlp = nn.Sequential(
               nn.Linear(self.embed_dim, 768),
               nn.Tanh(),
               nn.Linear(768, 512),
               nn.Tanh(),
               nn.Linear(512, 1)
        )
    self.embed_dim = embed_dim
    self.tanh = nn.Tanh()

  def forward(x):
    cls_tensors = []
    for attention_layer in self.attention_layers:
      x = attention_layer(x)
      cls_tensors.append(x[:,0]).reshape(-1, 1, self.embed_dim)

    mean_cls_tensors = torch.cat(cls_tensors, dim=1)
    pool_tensors = torch.cat(mean_cls_tensors, dim=1)
    pool_tensors = self.tanh(pool_tensors)
    embeddings = self.mlp(pool_tensors)
    return embeddings
