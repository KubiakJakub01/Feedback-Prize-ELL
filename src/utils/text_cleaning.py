"""
Text cleaning pipeline
"""
# Import libraries for text processing
import re
import string
import nltk
from nltk.corpus import stopwords

# I have already downloaded all-nltk
# nltk.download()

placeholders_replacements = {
    'Generic_School': '[GENERIC_SCHOOL]',
    'Generic_school': '[GENERIC_SCHOOL]',
    'SCHOOL_NAME': '[SCHOOL_NAME]',
    'STUDENT_NAME': '[STUDENT_NAME]',
    'Generic_Name': '[GENERIC_NAME]',
    'Genric_Name': '[GENERIC_NAME]',
    'Generic_City': '[GENERIC_CITY]',
    'LOCATION_NAME': '[LOCATION_NAME]',
    'HOTEL_NAME': '[HOTEL_NAME]',
    'LANGUAGE_NAME': '[LANGUAGE_NAME]',
    'PROPER_NAME': '[PROPER_NAME]',
    'OTHER_NAME': '[OTHER_NAME]',
    'PROEPR_NAME': '[PROPER_NAME]',
    'RESTAURANT_NAME': '[RESTAURANT_NAME]',
    'STORE_NAME': '[STORE_NAME]',
    'TEACHER_NAME': '[TEACHER_NAME]',
}

def text_cleaning(x, stop_words=stopwords.words("english")):
    """Clean the text.
    Args:
        x (str): Text to clean.
        stop_words (list): List of stopwords to remove.
    Returns:
        x (str): Cleaned text."""
    x = x.lower()
    x = " ".join([word for word in x.split(" ") if word not in stop_words])
    x = x.encode("ascii", "ignore").decode()
    x = re.sub(r"https*\S+", " ", x)
    x = re.sub(r'\n', " ", x)
    x = re.sub(r"@\S+", " ", x)
    x = re.sub(r"#\S+", " ", x)
    x = re.sub(r"\'\w+", "", x)
    x = re.sub("[%s]" % re.escape(string.punctuation), " ", x)
    x = re.sub(r"\w*\d+\w*", "", x)
    x = re.sub(r"\s{2,}", " ", x)
    return x


def replace_placeholders(text: str) -> str:
    """Replace placeholders.
    Args:
        text (str): Text to replace placeholders.
    Returns:
        text (str): Text with placeholders replaced."""
    for placeholder, replacement in placeholders_replacements.items():
        text = text.replace(placeholder, replacement)
    return text


def text_cleaning_pipeline(df, text_column):
    """Clean the text.
    Args:
        df (pandas.DataFrame): Dataframe containing the text to clean.
        text_column (str): Name of the column containing the text to clean.
    Returns:
        df (pandas.DataFrame): Dataframe with the cleaned text."""
    df[text_column] = df[text_column].apply(lambda x: text_cleaning(x))
    df[text_column] = df[text_column].apply(lambda x: define_encodings(x))
    df[text_column] = df[text_column].apply(lambda x: replace_placeholders(x))
    return df

def define_encodings(text: str) -> str:
    """Define encodings."""
    return text.encode("utf-8", "ignore").decode("ascii", "ignore")


def lematize_text(text: str) -> str:
    """Lematize text."""
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split(" ")])


def stem_text(text: str) -> str:
    """Stem text."""
    stemmer = nltk.stem.PorterStemmer()
    return " ".join([stemmer.stem(word) for word in text.split(" ")])
