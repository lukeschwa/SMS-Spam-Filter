import string

import matplotlib.pyplot as plt
import pandas as pd
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')

# Reading File and sotring stopwords and punctuation
file = pd.read_csv('spam.csv')
stopwords = nltk.corpus.stopwords.words('english')
punc = string.punctuation


# preprocessing of text
def preprocess_text(text):
    del_punct = "".join([word.lower() for word in text if word not in punc])
    tokenize = nltk.tokenize.word_tokenize(del_punct)
    del_stopwords = [word for word in tokenize if word not in stopwords]
    return del_stopwords


file['processed'] = file['Message'].apply(lambda x: preprocess_text(x))


# Sorts words into either ham words or spam words.
def sort_words():
    spam = []
    ham = []

    for message in file['processed'][file['Category'] == 'spam']:
        for word in message:
            spam.append(word)

    for message in file['processed'][file['Category'] == 'ham']:
        for word in message:
            ham.append(word)

    return spam, ham


spam_words, ham_words = sort_words()


# Predicts whether input text is spam or not
def predict(input):
    spam_count = 0
    ham_count = 0

    for word in input:
        spam_count += spam_words.count(word)
        ham_count += ham_words.count(word)

    if ham_count > spam_count:
        accuracy = (ham_count / (ham_count + spam_count)) * 100
        print('Message is not spam')
        print('Accuracy: {}%'.format(accuracy))
    elif spam_count > ham_count:
        accuracy = (spam_count / (ham_count + spam_count)) * 100
        print('Message is spam')
        print('Accuracy: {}%'.format(accuracy))
    else:
        print('Message might be spam')


# Multinomial Naive Bayes Model
bow = CountVectorizer(analyzer=preprocess_text).fit_transform(file['Message'])
x_train, x_test, y_train, y_test = train_test_split(bow, file['Category'], test_size=0.20, random_state=0)
classifier = MultinomialNB().fit(x_train, y_train)
prediction = classifier.predict(x_train)
print('Multinomial NB Accuracy: {}%'.format(accuracy_score(y_train, prediction)))


# shows wordcloud of spam and ham words
def show_wordcloud(category):
    words = ''
    for message in file[file['Category'] == category]['Message']:
        message = message.lower()
        words += message + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# Shows wordclouds to user, and lets them see what words are common spam/ham words
show_wordcloud('spam')
show_wordcloud('ham')
user_input = input("Please type a message to be evaluated as either spam or ham(not spam)\n")
processed_user_input = preprocess_text(user_input)
predict(processed_user_input)
