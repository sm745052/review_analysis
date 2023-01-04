import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
sid_obj = SentimentIntensityAnalyzer()
from nltk.stem import WordNetLemmatizer
  
def logical_xor(a, b):
    if bool(a) == bool(b):
        return False
    else:
        return a or b

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')



lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

dftr = pd.read_csv('./data_senti.csv')

st = [('. '.join(dftr[(dftr['senti']==0) & (dftr['Product  ']==lp) & (dftr['Job Role']==lr)]['Verbatim Feedback '])) for lp in dftr['Product  '].unique() for lr in dftr[dftr['Product  ']==lp]['Job Role'].unique()]
prod_jobs = [(lp, lr) for lp in dftr['Product  '].unique() for lr in dftr[dftr['Product  ']==lp]['Job Role'].unique()]

prod_jobs_dict = {}

for jnd, txt in enumerate(st):
    #lemmatize the txt
    tokenized = sent_tokenize(txt)
    ls = []
    for i in tokenized:
        
        # Word tokenizers is used to find the words
        # and punctuation in a string
        wordsList = nltk.word_tokenize(i)

        #  Using a Tagger. Which is part-of-speech
        # tagger or POS-tagger.
        tagged = nltk.pos_tag(wordsList)

        ls.append(tagged)
    kk=[]
    txt = ' '.join([w for w in txt.split() if not w in stop_words])
    lem_txt = ' '.join([lemmatizer.lemmatize(w) for w in txt.split()])
    lem_dist = nltk.FreqDist(lem_txt.split())
    lem_words_len = len(set(lem_txt.split()))
    for s in ls:
        for ind, i in enumerate(s):
            if(i[1]=='JJ'):
                for j in s[ind-1:ind+2]:
                    if(logical_xor(sid_obj.polarity_scores(i[0])['neg']>0.01, sid_obj.polarity_scores(j[0])['neg']>0.01)):
                        if('NN' in j[1]):
                            # print((lem_dist.items()))
                            if(lem_dist[j[0]]/lem_words_len>=0):
                                kk.append([i[0], j[0]])
    if(len(kk)>0):
        for ind in range(len(kk)):
            kk[ind] = ' '.join(kk[ind])
        prod_jobs_dict[prod_jobs[jnd]] = kk
    else:
        pass
        # prod_jobs_dict[prod_jobs[jnd]] = ['Not Enough Data']

for i in prod_jobs_dict:
    print("Product:", i[0])
    print("Job Role:", i[1])
    print("Improvement Suggestions:", end=" ")
    for j in prod_jobs_dict[i]:
        print(j, end="")
        if(j==prod_jobs_dict[i][-1]):
            pass
        else:
            print(", ", end="")
    print('\n-------------------------------------')
