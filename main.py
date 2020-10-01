# Importing the dataset
import pandas as pd
dataset = pd.read_csv('Tweets10.txt', delimiter = '\t', quoting = 3, encoding = "ISO-8859-1")

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 12894):
    tweet = re.sub(r"http\S+", '', dataset['tweet'][i])
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3450)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression()
classifierLR.fit(X_train, y_train)


# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(criterion='entropy')
classifierDT.fit(X_train, y_train)

# Predicting the test set
y_predNB = classifierNB.predict(X_test)
y_predLR = classifierLR.predict(X_test)

y_predDT = classifierDT.predict(X_test)

# Confusion matrix, accuracy , precision and recall
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cmNB = confusion_matrix(y_test, y_predNB)
aNB = accuracy_score(y_predNB, y_test)
rNB = classification_report(y_predNB, y_test)
print("Accuracy of Naive Bayes: ",aNB*100)

cmLR = confusion_matrix(y_test, y_predLR)
aLR = accuracy_score(y_predLR, y_test)
rLR = classification_report(y_predLR, y_test)
print("Accuracy of Logistic regression: ",aLR*100)





cmDT = confusion_matrix(y_test, y_predDT)
aDT = accuracy_score(y_predDT, y_test)
rDT = classification_report(y_predDT, y_test)
print("Accuracy of Decision Tree: ",aDT*100)


#Best Accuracy
print("\n")
if aNB>aLR and aNB>aDT:
    print("Naive Bayes has the best accuracy with ",aNB*100)
elif aLR>aDT:
    print("Logistic Regression has the best accuracy with ",aLR*100)
else:
    print("Decision Tree has the best accuracy with ",aDT*100)                