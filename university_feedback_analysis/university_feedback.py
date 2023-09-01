import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import words, stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load your list of English words
english_vocab = set(w.lower() for w in words.words())

sia = SentimentIntensityAnalyzer()

def is_english(text):
    """Checks if the text is primarily in English."""
    if not isinstance(text, str):
        return False
    words_in_text = text.lower().split()
    total_words = len(words_in_text)
    english_count = sum(1 for word in words_in_text if word in english_vocab)

    if total_words == 0:
        return False

    english_ratio = english_count / total_words

    # Checking if more than 70% of the words are English
    return english_ratio > 0.7

def get_sentiment(text):
    """Get sentiment using SentimentIntensityAnalyzer."""
    simple_responses = ['no', 'yes']
    if text.strip().lower() in simple_responses:
        return 'neutral'
    if not isinstance(text, str):
        return None
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        return 'positive'
    elif score < -0.05:
        return 'negative'
    else:
        return 'neutral'

def preprocess_text(text):
    # Handle simple responses
    simple_responses = ['no', 'yes']
    if text.strip().lower() in simple_responses:
        return text.strip().lower()

    # Convert to lowercase
    text = text.lower()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove stopwords and lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if
              token.isalpha() and token not in stopwords.words('english')]

    return ' '.join(tokens)

if __name__ == "__main__":
    df = pd.read_excel('AI_Engineer_Dataset_Task_1.xlsx')

    # Filter the dataframe to only include rows where QuestionType is "User Comment"
    # and ParticipantResponse is in English
    filtered_df = df[df['QuestionType'] == 'User Comment']
    filtered_df = filtered_df[filtered_df['ParticipantResponse'].notna()]
    filtered_df = filtered_df[filtered_df['ParticipantResponse'].apply(is_english)]

    # Analyze sentiment for the ParticipantResponse
    filtered_df['Sentiment'] = filtered_df['ParticipantResponse'].apply(get_sentiment)

    # Applying preprocessing on the ParticipantResponse
    filtered_df['ProcessedText'] = filtered_df['ParticipantResponse'].apply(preprocess_text)

    # Building LDA Model
    vectorizer = CountVectorizer(max_df=0.9, min_df=5, token_pattern='\w+|\$[\d\.]+|\S+')
    dtm = vectorizer.fit_transform(filtered_df['ProcessedText'])

    lda = LatentDirichletAllocation(n_components=5, random_state=42)  # Assuming 5 topics
    lda.fit(dtm)

    # Assign Topics
    def get_dominant_topic(doc):
        topic_probabilities = lda.transform(vectorizer.transform([doc]))
        dominant_topic = topic_probabilities.argmax()
        return dominant_topic

    filtered_df['DominantTopic'] = filtered_df['ProcessedText'].apply(get_dominant_topic)

    # Getting representative keywords for each topic
    topic_keywords = {}
    for index, topic in enumerate(lda.components_):
        keywords = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        topic_keywords[index] = ", ".join(keywords)

    filtered_df['TopicKeywords'] = filtered_df['DominantTopic'].apply(lambda x: topic_keywords[x])

    # Export the dataframe to an Excel file
    filtered_df.to_excel('output_filename_with_topics.xlsx', index=False)