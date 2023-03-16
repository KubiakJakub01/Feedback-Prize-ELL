"""
Text cleaning pipeline
"""
# Import libraries for text processing
import re
import string
# import nltk
from nltk.corpus import stopwords

# I have already downloaded all-nltk
# nltk.download()

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


def text_cleaning_pipeline(df, text_column):
    """Clean the text.
    Args:
        df (pandas.DataFrame): Dataframe containing the text to clean.
        text_column (str): Name of the column containing the text to clean.
    Returns:
        df (pandas.DataFrame): Dataframe with the cleaned text."""
    df[text_column] = df[text_column].apply(lambda x: text_cleaning(x))
    return df
