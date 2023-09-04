"""
This module provides functions for sentiment analysis and topic modeling
on user comments from a dataset.
"""
import sys
import pandas as pd
from nltk import WordNetLemmatizer, word_tokenize, pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import words, stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from multiprocessing import Pool, cpu_count

english_vocab = set(w.lower() for w in words.words())
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def extract_opinions(text):
    """Extract adjectives (opinions) from the provided text."""
    tokenized = word_tokenize(text)
    opinions = [word for word, tag in pos_tag(tokenized) if tag in ['JJ', 'JJR', 'JJS']]
    return ', '.join(opinions)

def is_english(text):
    """Check if the given text is predominantly in English."""
    if not isinstance(text, str):
        return False
    words_in_text = text.lower().split()
    english_count = sum(1 for word in words_in_text if word in english_vocab)
    return english_count / len(words_in_text) > 0.7

def get_sentiment(text):
    """Analyze the sentiment of the provided text."""
    simple_responses = ['no', 'yes']
    if text.strip().lower() in simple_responses:
        return 'neutral'
    score = sia.polarity_scores(text)['compound']
    return 'positive' if score > 0.05 else 'negative' if score < -0.05 else 'neutral'

def preprocess_text(text):
    """Preprocess the text by lemmatizing and removing stopwords."""
    tokens = word_tokenize(text.lower())
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words])

def get_dominant_topic(doc, model_vectorizer, lda_model):
    """Identify the dominant topic for the provided document."""
    topic_probabilities = lda_model.transform(model_vectorizer.transform([doc]))
    return topic_probabilities.argmax()

def remove_adjectives_from_topic_keywords(topic_keywords_string):
    """Remove adjectives from a list of keywords."""
    tokenized = word_tokenize(topic_keywords_string)
    non_adjectives = [
        word for word, tag in pos_tag(tokenized)
        if tag not in ['JJ', 'JJR', 'JJS'] and word != ','
    ]
    return ", ".join(non_adjectives)

if __name__ == "__main__":
    df = pd.read_excel(sys.argv[1])
    filtered_df = df[df['QuestionType'] == 'User Comment']
    filtered_df = filtered_df[
        filtered_df['ParticipantResponse'].notna() & filtered_df['ParticipantResponse'].apply(is_english)
    ]

    # Multiprocessing for sentiment analysis
    with Pool(cpu_count() - 1) as pool:
        filtered_df['Sentiment'] = pool.map(get_sentiment, filtered_df['ParticipantResponse'])
        filtered_df['Opinions'] = pool.map(extract_opinions, filtered_df['ParticipantResponse'])
        filtered_df['ProcessedText'] = pool.map(preprocess_text, filtered_df['ParticipantResponse'])

    vectorizer = CountVectorizer(max_df=0.9, min_df=5, token_pattern=r'\w+|\$[\d\.]+|\S+')
    dtm = vectorizer.fit_transform(filtered_df['ProcessedText'])
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)

    # Multiprocessing for dominant topic assignment
    with Pool(cpu_count() - 1) as pool:
        func_args = [(doc, vectorizer, lda) for doc in filtered_df['ProcessedText']]
        filtered_df['DominantTopic'] = pool.starmap(get_dominant_topic, func_args)

    topic_keywords = {}
    for index, topic in enumerate(lda.components_):
        keywords = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        topic_keywords[index] = ", ".join(keywords)

    filtered_df['TopicKeywords'] = filtered_df['DominantTopic'].apply(
        lambda x: remove_adjectives_from_topic_keywords(topic_keywords[x])
    )
    filtered_df.drop(columns=['ProcessedText'], inplace=True)
    filtered_df.to_excel('./output/analysis_output.xlsx', index=False)