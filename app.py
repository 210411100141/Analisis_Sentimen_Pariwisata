import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from nltk.corpus import stopwords
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

import re
import string

# Inisialisasi NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon') 

# Fungsi untuk membersihkan teks
def clean_text(text):
    return re.sub('[^a-zA-Z]', ' ', text).lower()

# Fungsi untuk menghitung jumlah tanda baca
def count_punct(review):
    count = sum([1 for char in review if char in string.punctuation])
    return round(count/(len(review) - review.count(" ")), 3)*100

# Fungsi untuk tokenisasi teks
def tokenize_text(text):
    tokenized_text = text.split()
    return tokenized_text

# Fungsi untuk lemmatize teks dan menghapus stopwords
def lemmatize_text(token_list, lemmatizer, stop_words):
    return " ".join([lemmatizer.lemmatize(token) for token in token_list if not token in stop_words])

# Fungsi utama aplikasi
def main():
    st.title("Aplikasi Analisis Sentimen NLP")
    st.subheader("Review Pantai Bali")

    # Form untuk input teks
    with st.form(key='nlpForm'):
        raw_text = st.text_area("Masukkan Teks di Sini")
        submit_button = st.form_submit_button(label='Analisis')

    # Load data dan preprocessing
    if submit_button:
        df = pd.read_csv("reviewPantaiBali_fix.csv", encoding="latin-1")
        df.drop(columns=['pantai_name', 'name'], inplace=True)

        # Proses data
        df['cleaned_text'] = df['review'].apply(lambda x: clean_text(x))
        df['label'] = df['rating'].map({1.0:0, 2.0:0, 3.0:0, 4.0:1, 5.0:1})
        df['review_len'] = df['review'].apply(lambda x: len(str(x)) - str(x).count(" "))
        df['punct'] = df['review'].apply(lambda x: count_punct(str(x)))
        df['tokens'] = df['cleaned_text'].apply(lambda x: tokenize_text(x))

        stop_words = stopwords.words('english')
        stop_words.remove('not')
        
        lemmatizer = nltk.stem.WordNetLemmatizer()
        df['lemmatized_review'] = df['tokens'].apply(lambda x: lemmatize_text(x, lemmatizer, stop_words))

        # Ekstraksi fitur dengan TF-IDF
        X = df[['lemmatized_review', 'review_len', 'punct']]
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        tfidf = TfidfVectorizer(max_df=0.5, min_df=2)
        tfidf_train = tfidf.fit_transform(X_train['lemmatized_review'])
        tfidf_test = tfidf.transform(X_test['lemmatized_review'])

        # Prediksi dengan model SVM
        classifier = SVC(kernel='linear', random_state=10)
        classifier.fit(tfidf_train, y_train)

        # Prediksi untuk teks input pengguna
        data = [raw_text]
        vect = tfidf.transform(data).toarray()
        prediction = classifier.predict(vect)

        # Tampilkan hasil prediksi
        st.write("**Hasil Prediksi:**")
        if prediction == 1:
            st.markdown("Sentimen: Positif :smiley:")
        else:
            st.markdown("Sentimen: Negatif :angry:")

        # Analisis sentimen token
        def analyze_token_sentiment(text):
            analyzer = SentimentIntensityAnalyzer()
            pos_list = []
            neg_list = []
            neu_list = []
            for i in text.split():
                res = analyzer.polarity_scores(i)['compound']
                if res > 0.1:
                    pos_list.append((i, res))
                elif res <= -0.1:
                    neg_list.append((i, res))
                else:
                    neu_list.append(i)
            result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
            return result

        token_sentiments = analyze_token_sentiment(raw_text)
        st.write("**Analisis Sentimen Token:**")
        st.write(token_sentiments)

    else:
        st.subheader("About")

if __name__ == '__main__':
    main()
