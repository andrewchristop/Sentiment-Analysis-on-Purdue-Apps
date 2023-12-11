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


def bar_pos(gui, app, ath, rec, mob):
  fig, ax = plt.subplots()
  apps = ['Purdue Guide', 'Purdue App', 'Purdue Athletics', 'Purdue RecWell', 'Purdue Mobile App']
  freq = [gui, app, ath, rec, mob]

  ax.bar(apps, freq)
  ax.set_ylabel('Frequency')
  ax.set_title('Purdue Apps with Positive Reviews')

  plt.show()

def bar_neg(gui, app, ath, rec, mob):
  fig, ax = plt.subplots()
  apps = ['Purdue Guide', 'Purdue App', 'Purdue Athletics', 'Purdue RecWell', 'Purdue Mobile App']
  freq = [gui, app, ath, rec, mob]

  ax.bar(apps, freq)
  ax.set_ylabel('Frequency')
  ax.set_title('Purdue Apps with Negative Reviews')

  plt.show()

