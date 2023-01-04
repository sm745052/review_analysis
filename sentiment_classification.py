print("importing libraries")
import pandas as pd
from happytransformer import HappyTextClassification
import warnings
warnings.filterwarnings("ignore")

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
print("reading data.xlsx")
dftr = pd.read_excel('data.xlsx')
print("building model")
happy_tc = HappyTextClassification(model_type="DISTILBERT", model_name="distilbert-base-uncased-finetuned-sst-2-english", num_labels=2)
print("classifying text")
dftr['senti'] = dftr['Verbatim Feedback '].apply(happy_tc.classify_text)
dftr['senti'] = dftr['senti'].apply(lambda x:1 if x.label[0]=='P' else 0)
print("saving file to data_senti.csv")
dftr.to_csv('data_senti.csv', index=False)
# print(classification_report(dftr['Sentiment (1=Positive & 0= Negative)'], dftr['prd']))
