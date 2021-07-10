
#PARAMTERS
PROB_SENTENCE_PAIR = {"SAME_ANIME":0.75, "DIFFERENT_ANIME":0.01, "SAME_GENRE":0.75}

#INITIALIZING
df = pd.read_csv("/kaggle/input/review-embeddings/reviews_features.csv")

#HELPER FUNCTIONS

def create_data():
  num_rows = df.shape[0]
  data = []

  def helper(idx1,idx2, type):
    prob = random.random()
    if PROB_SENTENCE_PAIR[type] >= prob:
      if type == "SAME_ANIME":
        label = abs(df[i,-1] - df[j,-1])
      elif type == "DIFFERENT_ANIME":
        label = 5
      else:
        label = 4
      data.append({"sentences": [df[i,1],df[j,1]],
        "label": label})

  for i in tqdm(range(num_rows)):
    for j in range(i+1, num_rows):
      if df[i,0] == df[j,0]:
        helper(i,j,"SAME_ANIME")
      elif df[i,3] == df[j,3]:
        helper(i,j,"SAME_GENRE")
      elif set(df[i,3]).isdisjoint(df[j,3]):
        helper(i,j,"DIFFERENT_ANIME")
  return data
