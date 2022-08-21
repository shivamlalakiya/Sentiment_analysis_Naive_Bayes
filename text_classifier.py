import string
import nltk
import sys
import json
import re
nltk.download('omw-1.4')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')


def clean_review(review):
    lower_text = review.lower()
    punctuationfree ="".join([i for i in lower_text if i not in string.punctuation])
    words = re.split(' ', punctuationfree)
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    review_cleaned = ' '.join(words)


    return review_cleaned

def naive_bayes_predict(review, logprior, loglikelihood):

    word_l = clean_review(review).split(' ')

    words = review.split(' ')
    for word in words:
        if word in set(stopwords.words('english')) or word not in loglikelihood:
            print('probability of '+ word +' word is '+ '0')
        else:
            print('probability of '+ word +' word is '+ str(loglikelihood[word]))


    # initialize probability to zero
    total_prob = 0

    # add the logprior
    total_prob += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            total_prob += loglikelihood[word]

    if total_prob > 0:
        return 1
    else:
        return 0


def main(args = []):
    file = open("data.json", 'r')
    json_data = json.load(file)

    while 1:
        review = input("enter review: ")

        if review == "x" or review == "X":
            quit()
        predict = naive_bayes_predict(review, 0.0, json_data)
        if predict == 0:
            print("The prediction is " + str(predict) +' (positive review)')
        else:
            print("The prediction is " + str(predict) + ' (negative review)')


if __name__ == "__main__":
    main(sys.argv[1:])