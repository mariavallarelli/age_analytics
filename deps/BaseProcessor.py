import re
import nltk
import emoji
from sklearn.base import BaseEstimator, TransformerMixin


class BaseProcessor(BaseEstimator, TransformerMixin):
    """class wrapping a sentence processing function so that it can be used in a sklearn.pipeline.Pipeline"""

    def fit(self, x, y=None):
        """ This method actually does nothing. It will be overridden by the child classes.
        all setup activities required before calling either the method transform()
        will be performed in its children implementations
        """
        # does nothing
        return self

    def manageSentence(self, sentence):  # This method is usually overridden in children classes
        """Called by transform(). The sentence is expected to be a either a string or a list of words,
        this method can return either a string or a list of words"""
        return sentence

    def transform(self,  listOfSentences):
        """ listOfSentences: list of sentences.
        Every sentence can be either a string or a list of words
        Return a list of lists. Each sentence is preprocessed using the manageSentence() method.
        Each child class can override the manageSentence() method to implement a specific preprocessing behavior.
        The list of preprocessed documents is returned."""
        toReturn = []
        for sentence in listOfSentences:
            #print(sentence)
            if sentence:
                processedSentence = self.manageSentence(sentence)
                toReturn.append(processedSentence)
        return toReturn


class TextProcessor(BaseProcessor):
    def manageSentence(self, text):
        return processor(text)


def processor(text):
    """ This method process text of tweet to clean it. """
    text = remove_mentions(text)
    text = remove_url(text)
    text = html_accents_replacer(text)
    text = strip_chinese(text)
    text = remove_emoji(text)
    text = remove_numbers(text)
    text = to_lower(text)
    text = remove_punctuation(text)
    text = stopwordsremover(text)
    text = filter_by_len(text)
    text = tokenizer(text)
    return text


def remove_emoji(text):
    """Sentence is expected to be a list of words
    this method returns the input list where the emoji are removed. """
    # emoticons
    # symbols & pictographs
    # transport & map symbols
    # flags (iOS)
    # Unicode Block Miscellaneous Symbols
    for l in emoji.UNICODE_EMOJI:
        # print(s)
        text = text.replace(l, u" ")
    return text


def remove_numbers(text):
    """Sentence is expected to be a list of words,
    this method returns the input list where the numbers and birth date numbers are removed. """
    text = re.sub(r'\d{2}[th|rd|st]*', '', text)
    return re.sub(r'\d+', '', text)


def remove_punctuation(text):
    """Sentence is expected to be a list of words,
    this method returns the input list where the punctuations are removed """
    # punctuation = u'!{}[]?"",;.:-<>|/\\*=+-_% \n\t\r()' + u"'" + u'\u2019' + u'\u2018'  # \r and \n can be used as "new line"
    return re.sub(r'[^\w\s]', '', text)


def stemmer(list_sentence):
    """sentence is expected to be a list of words (a list where each item is a string containing a single word),
    this method returns a list of stemmed words"""
    st = nltk.stem.SnowballStemmer("english")
    return [st.stem(w) for w in list_sentence]


def to_lower(text):
    """sentence is expected to be a list of words (each item is a string),
    this method returns a list of strings whereas each string is the lower case version of the original word"""
    return text.lower()


def html_accents_replacer(text):
    """Replace html representations of special letters with the corresponding unicode character.
    E.g.  &agrave with à.
    Args:
       * s(string): the string where the html codes should be replaced  """
    assert type(text) == type('') or type(text) == type(
        u''), "HTMLAccentsReplacer Assertion Error"

    replacemap = {u'&Ecirc;': u'\xca', u'&raquo;': u'\xbb', u'&eth;': u'\xf0', u'&divide;': u'\xf7',
                  u'&atilde;': u'\xe3', u'&Aelig;': u'\xc6', u'&frac34;': u'\xbe', u'&nbsp;': u' ',
                  u'&Aumbl;': u'\xc4', u'&Ouml;': u'\xd6', u'&Egrave;': u'\xc8', u'&Icirc;': u'\xce',
                  u'&deg;': u'\xb0', u'&ocirc;': u'\xf4', u'&Ugrave;': u'\xd9', u'&ndash;': u'\u2013',
                  u'&gt;': u'>', u'&Thorn;': u'\xde', u'&aring;': u'\xe5', u'&frac12;': u'\xbd',
                  u'&frac14;': u'\xbc', u'&Aacute;': u'\xc1', u'&szlig;': u'\xdf', u'&trade;': u'\u2122',
                  u'&igrave;': u'\xec', u'&aelig;': u'\xe6', u'&times;': u'\xd7', u'&egrave;': u'\xe8',
                  u'&Atilde;': u'\xc3', u'&Igrave;': u'\xcc', u'&Eth;': u'\xd0', u'&ucirc;': u'\xfb',
                  u'&lsquo;': u'\u2018', u'&agrave;': u'\xe0', u'&thorn;': u'\xfe', u'&Ucirc;': u'\xdb',
                  u'&amp;': u'&', u'&uuml;': u'\xfc', u'&yuml;': u'', u'&ecirc;': u'\xea', u'&laquo;': u'\xab',
                  u'&infin;': u'\u221e', u'&Ograve;': u'\xd2', u'&oslash;': u'\xf8', u'&yacute;': u'\xfd',
                  u'&plusmn;': u'\xb1', u'&icirc;': u'\xee', u'&auml;': u'\xe4', u'&ouml;': u'\xf6',
                  u'&Ccedil;': u'\xc7', u'&euml;': u'\xeb', u'&lt;': u'<', u'&eacute;': u'\xe9',
                  u'&ntilde;': u'\xf1', u'&pound;': u'\xa3', u'&Iuml;': u'\xcf', u'&Eacute;': u'\xc9',
                  u'&Ntilde;': u'\xd1', u'&rsquo;': u'\u2019', u'&euro;': u'\u20ac', u'&rdquo;': u'\u201d',
                  u'&Acirc;': u'\xc2', u'&ccedil;': u'\xe7', u'&Iacute;': u'\xcd', u'&quot;': u'"',
                  u'&Aring;': u'\xc5', u'&Oslash;': u'\xd8', u'&Otilde;': u'\xd5', u'&Uacute;': u'\xda',
                  u'&reg;': u'\xae', u'&Yacute;': u'\xdd', u'&iuml;': u'\xef', u'&ugrave;': u'\xf9',
                  u'&alpha;': u'\u03b1', u'&copy;': u'\xa9', u'&ldquo;': u'\u201c', u'&oacute;': u'\xf3',
                  u'&Euml;': u'\xcb', u'&uacute;': u'\xfa', u'&ograve;': u'\xf2', u'&acirc;': u'\xe2',
                  u'&aacute;': u'\xe1', u'&Agrave;': u'\xc0', u'&Oacute;': u'\xd3', u'&Uuml;': u'\xdc',
                  u'&iacute;': u'\xed', u'&cent;': u'\xa2', u'&Ocirc;': u'\xd4', u'&mdash;': u'\u2014',
                  u'&otilde;': u'\xf5', u'&beta;': u'\u03b2'}
    for before in replacemap:
        after = replacemap[before]  # getting the string to be replaced
        text = text.replace(before, after)
    return text


def getStopWords():
    """This method returns a list of English stop words. Stop words can be added to the list"""
    return [u'birthday', u'happy', u'good', u'go', u'get', u'got', u'thats', u'back', u'ive', u'always',
             u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'yes', u'given',
             u'you', u'your', u'yours', u'yourself', u'yourselves', u'it s', u'i m', u'ever', u'even',
             u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'going',
             u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves',
             u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'much',
             u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'thank',
             u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'automatically',
             u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while',
             u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through',
             u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out',
             u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when',
             u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other',
             u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very',
             u's', u't', u'can', u'will', u'just', u'don', u'should', u'now', u'd', u'll', u'm', u'o', u're',
             u've', u'y', u'ain', u'aren', u'couldn', u'didn', u'doesn', u'hadn', u'hasn', u'haven', u'isn',
             u'ma', u'mightn', u'mustn', u'needn', u'shan', u'shouldn', u'wasn', u'weren', u'won', u'wouldn']


def stopwordsremover(text):
    """this method returns the input list where the stop words are removed"""
    stopWords = getStopWords()
    words = [w for w in text.split() if w not in stopWords]
    return " ".join(words)


def filter_by_len(words):
    words_split = str(words).split()
    filtered = [word for word in words_split if len(word) >= 3]
    return " ".join(filtered)


def remove_mentions(text):
    """sentence is expected to be a list of words,
    this method returns a list of strings whereas each mention to twitter user is removed """
    return re.sub(r'(@\w+)', ' ', text)


def strip_chinese(text):
    """sentence is expected to be a list of words,
    this method returns a list of strings whereas each chinese symbols are removed """
    chinese_list = re.findall(u'[^\u4E00-\u9FA5]', text)
    for c in text:
        if c not in chinese_list:
            text = text.replace(c, '')
    return text


def remove_url(text):
    """sentence is expected to be a list of words,
    this method returns a list of strings whereas each url is removed """
    return re.sub(r'https?://\S+', ' ', text)


def unityFunction(x):
  """This function returns the same object received as input."""
  return x


def tokenizer(sentence):
    """This method turn a single document (i.e., a string) into a list of single words (i.e., tokens).
    Then the string is splitted in substring using the spaces as split markers"""
    return sentence
    if sentence == None:
        return []
    while sentence.find(u"  ") != -1:
        sentence = sentence.replace(u"  ", u" ")  # replacing double spaces with a single one
    return sentence.split(u' ')  # e.g., "a b c d".split(' ')  returns ['a','b','c','d']









