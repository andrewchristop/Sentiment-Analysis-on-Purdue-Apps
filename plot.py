import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
def word_cloud(text):
  stopwords = set(STOPWORDS)
  wordcloud = WordCloud(width = 800, height = 800, background_color = 'white', stopwords = stopwords, min_font_size = 10).generate(text)

  plt.figure(figsize = (8,8), facecolor = None)
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout(pad = 0)
  plt.show()
