import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Restaurant.csv",delimiter="\t",quoting=2
import re
import nlt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

corpus=[]

dataset.iloc[01,01]

for i in range(0,1000):
rev=re.sub("[^a-zA-Z]"," ",dataset["Review"][i])

rev=rev.lower()
rev=rev.split()

rev=[word for word in rev if not word in stopwords.words("english")]

rev=[ps.stem(word) for word in rev if not word in set(stopwords.words("english"))]

rev=" ".join(rev)
corpus.append(rev)

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=1400)

x1=cv.fit_transform(corpus).toarray()
y1=pd.DataFrame(dataset["Liked"])

from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x1,y1)
y1 = column_or_1d(y1, warn=True)

GaussianNB(priors=None, var_smoothing=1e-09)

Y1_pred=nb.predict(x)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y1,y1_pred)

cm

array([[409,  97],
[  01, 501
]])


sentiment analysis

from textblob import TextBlob

analysis = TextBlob("I am whatever")
if analysis.sentiment.polarity > 0:
print("positive")
elif analysis.sentiment.polarity == 0:
print("Neutral")
else:
print("negative")
Neutral

dataset=pd.read_csv("Restaurant.csv",delimiter="\t",quoting=3)

dataset.head(2)

	Review	Liked
0	Wow... Its nice	1
1	Crust is not good.	0

X1=pd.DataFrame(dataset["Review"])
y1=pd.DataFrame(dataset["Liked"])

l1=len(x)

li=[]
for i in range(0,l1,1):
analysis = TextBlob(x.iloc[i,0])
if analysis.sentiment.polarity > 0:
li.append(1)
elif analysis.sentiment.polarity == 0:
li.append(2)
else:
li.append(0)

y1_pred=pd.DataFrame(li)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y1["Liked"],y1_pred)

cm

array([[222, 120, 158],
[ 28, 394,  78],
[  0,   0,   0]])

X1["liked"]=y1_pred

X1.head()

	Review	liked
0	Wow... its nice	1
1	Crust is not good.	0
2	Not tasty and the texture was just nasty.	0
3	Stopped by during the late May bank holiday of...	1
4	The selection on the menu was great 	1

pos,neg,neu=y_pred[0].value_counts()

#donut chart
labels = ["Positive","Negative","neutral"]
sizes = [pos,neg,neu]
colors = ['#f9999','#66b3ff','#99ff99']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=50)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax1.axis('equal')  
plt.tight_layout()
plt.show()
