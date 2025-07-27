#Kütüphane ekleme
import nltk as nltk
import pandas as pd
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
def fileRead(file_path): #Dosya Okuma Fonksiyonu Tanımlandı
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
        df = pd.json_normalize(raw_data, record_path=['messages'], meta=['conversation_id'])
        return df
    except FileNotFoundError:
        print(f"Hata! {file_path} dosyası bulunamadı")
    except json.JSONDecodeError:
        print(f"Hata! {file_path} geçersiz JSON formatında!")
    return None

df = fileRead('data.json') #Okunan dosyayı dataframe'e aktardık

if df is not None:
    user_messages = df[df['sender_id'].isna() | df['sender_id'].str.contains('user_id_pattern')]

    def is_unanswered(row, df):
        next_idx = df.index[df.index > row.name].min()
        if pd.isna(next_idx):
            return True
        next_msg = df.loc[next_idx]
        return next_msg['sender_id'] != 'bf17272dc3f0'

    user_messages.loc[:,'unanswered'] = user_messages.apply(lambda x: is_unanswered(x, df), axis=1)
    unanswered_questions = user_messages[user_messages['unanswered']]


    def classifyIntent(text):
        if pd.isna(text) or not isinstance(text, str):
            return "Diğer"
        text = text.lower()
        if "mekan" in text or "yer" in text:
            return "Mekan Arayışı"
        elif "fiyat" in text or "bütçe" in text:
            return "Bütçe Sorusu"
        elif "ürün" in text or "abiye" in text or "damatlık" in text:
            return "Ürün Arayışı"
        else:
            return "Diğer"

    def classifyCategory(text):
        if pd.isna(text) or not isinstance(text, str):
            return "Diğer"
        text = text.lower()
        if "düğün" in text:
            return "Düğün"
        elif "kına" in text:
            return "Kına"
        elif "nişan" in text:
            return "nişan"
        else:
            return "Diğer"

sent_analysis = SentimentIntensityAnalyzer()
def get_sentiment(text):
    if pd.isna(text) or not isinstance(text, str):
            return "Nötr"
    sentiment_score = sent_analysis.polarity_scores(text)
    compound_score = sentiment_score['compound']
    if compound_score >= 0.05:
        return "Pozitif"
    elif compound_score <= -0.05:
        return "Negatif"
    else:
        return "Nötr"
    
unanswered_questions.loc[:,'intent'] = unanswered_questions['content.text'].apply(classifyIntent)
unanswered_questions.loc[:,'category'] = unanswered_questions['content.text'].apply(classifyCategory)
unanswered_questions.loc[:, 'sentiment'] = unanswered_questions['content.text'].apply(get_sentiment)

print(unanswered_questions[['content.text', 'intent', 'category', 'sentiment','created_at']])

unanswered_questions.to_csv('classified_questions_with_sentiment.csv', index=False, encoding='utf-8')
