from config import Config


class Preprocess(Config):

  @staticmethod
  def format_review(x: str):
    """
    Preprocess the review to remove text elements.
    """
    x = re.sub("\r|\n", "", x)
    x = re.sub(" +", " ", x)
    if x[:10] == " more pics": x = x[11:]
    if x[-8:] == "Helpful ": x = x[:-8]
    x = re.sub(r"^Overall \d+ Story \d+ Animation \d+ Sound \d+ Character \d+ Enjoyment \d+ ", "", x)
    return x[:-1]

  # def


