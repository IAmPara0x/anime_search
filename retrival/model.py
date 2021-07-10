
# MODEL
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.roberta = RobertaModel.from_pretrained(
        "/kaggle/input/review-emddings-pt2/roberta_base_anime_finetuned.h5", output_hidden_states=True)
    self.hid_mix = 5
    self.feats = self.roberta.pooler.dense.out_features
    self.embedding_layer = DanEncoder(self.feats, EMBEDDING_DIM, 3, [1024, 1280], 0.1)

  def forward(self, calc_triplets=False, **kwargs):
    if calc_triplets:
      x = kwargs["x"]
      return self.forward_once(x)
    x1 = kwargs["x1"]
    x2 = kwargs["x2"]
    x3 = kwargs["x3"]
    return self.forward_once(x1), self.forward_once(x2), self.forward_once(x3)

  def forward_once(self, x):
    outputs = self.roberta(x)
    hidden_states = outputs[2]
    hmix = []
    for i in range(1, self.hid_mix+1):
      hmix.append(hidden_states[-i][:, 0].reshape((-1, 1, self.feats)))

    hmix_tensor = torch.cat(hmix, 1)
    pool_tensor = torch.mean(hmix_tensor, 1)
    pool_tensor = self.tanh(pool_tensor)
    embeddings = self.embedding_layer(pool_tensor)
    return embeddings

class DanEncoder(nn.Module):
  def __init__(self, input_dim:int, embedding_dim:int, n_hidden_layers:int, n_hidden_units:[int], dropout_prob:int):
    super(DanEncoder, self).__init__()

    assert n_hidden_layers != 0
    assert len(n_hidden_units) + 1 == n_hidden_layers

    encoder_layers = []
    for i in range(n_hidden_layers):
      if i == n_hidden_layers - 1:
        out_dim = embedding_dim
        encoder_layers.extend(
          [
            nn.Linear(input_dim, out_dim),
          ])
        continue
      else:
        out_dim = n_hidden_units[i]

      encoder_layers.extend(
        [
          nn.Linear(input_dim, out_dim),
          nn.Tanh(),
          nn.Dropout(dropout_prob, inplace=False),
        ]
      )
      input_dim = out_dim
    self.encoder = nn.Sequential(*encoder_layers)

  def forward(self, x_array):
      return self.encoder(x_array)
