import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Snowball Stemmer
stemmer = SnowballStemmer(language='english')

# Function to tokenize and stem text
def tokenize(text):
    """
    Tokenizes and stems the input text.
    
    Args:
    text (str): The input text to tokenize and stem.
    
    Returns:
    list: A list of stemmed tokens.
    """
    return [stemmer.stem(token) for token in word_tokenize(text)]

# Get English stopwords to reduce noise in data
english_stopwords = stopwords.words('english')

# Function to create a TfidfVectorizer
def create_vectorizer():
    """
    Creates and returns a TfidfVectorizer with custom tokenizer and stopwords.
    
    Returns:
    TfidfVectorizer: The configured TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        stop_words=english_stopwords,
        token_pattern= None,
        max_features=1080
    )
    return vectorizer