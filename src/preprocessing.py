import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = text.replace('-', ' ')
    text = re.sub(r'[^a-z0-9.\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w.strip('.') for w in tokens]
    tokens = [w for w in tokens if w]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


