#Kütüphane ekleme
import nltk as nltk
import pandas as pd
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
def fileRead(file_path): #Dosya Okuma Fonksiyonu Tanımlandı
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)  # Tek JSON nesnesi olarak oku
        # Tüm mesajları normalize et
        all_messages = []
        for conversation in raw_data:
            messages = pd.json_normalize(conversation['messages'])
            messages['conversation_id'] = conversation['conversation_id']
            all_messages.append(messages)
        df = pd.concat(all_messages, ignore_index=True)
        print("Sütun isimleri:", df.columns)
        print("Toplam mesaj sayısı:", len(df))
        return df
    except FileNotFoundError:
        print(f"Hata! {file_path} dosyası bulunamadı")
    except json.JSONDecodeError:
        print(f"Hata! {file_path} geçersiz JSON formatında!")
    return None

df = fileRead('data.json') #Okunan dosyayı dataframe'e aktardık

if df is not None:
    all_messages = df.copy()
    print("İşlenen mesaj sayısı: ", len(all_messages))

    def is_answered(row, df):
        next_idx = df.index[df.index > row.name].min()
        if pd.isna(next_idx):  # Son mesajsa
            return "Hayır"  # Yanıtlanmamış kabul et
        next_msg = df.loc[next_idx]
        return "Evet" if next_msg['sender_id'] == 'bf17272dc3f0' else "Hayır"


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
    
all_messages.loc[:, 'is_answered'] = all_messages.apply(lambda x: is_answered(x, df), axis=1)
all_messages.loc[:,'intent'] = all_messages['content.text'].apply(classifyIntent)
all_messages.loc[:,'category'] = all_messages['content.text'].apply(classifyCategory)
all_messages.loc[:, 'sentiment'] = all_messages['content.text'].apply(get_sentiment)

print(all_messages[['content.text', 'is_answered', 'intent', 'category', 'sentiment','created_at']])

all_messages.to_csv('classified_questions_with_sentiment.csv', index=False, encoding='utf-8')
