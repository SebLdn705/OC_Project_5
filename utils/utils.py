import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import regexp_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk import pos_tag

def text_preprocess(text: str) -> str:
    clean = CleanText()
    token = Tokenisation()
    normalise = Normalise(add_stop_word=['gt', 'lt'])

    pipeline = [clean.multi_clean,
                token.nltk_reg_ex,
                normalise.stopwords,
                normalise.pos_filter,
                normalise.lemmatizer
                ]

    cleaned_text = pipe_exec(text, pipeline)

    return cleaned_text


def pipe_exec(text: object, pipeline: object) -> str:
    token = text
    for transform in pipeline:
        token = transform(token)

    return token


class CleanText:
    REG_EX = {'HTML': (r'<[^<>]*>', ''),
              'CONTRACTION': (r'([A-Za-z]+)[\'`]([A-Za-z]+)', r'\1'),
              'STANDALONE': (r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' '),
              'PUNCTUATION': (r'[,!?:]+', ''),
              'SYMBOLS': (r'[@~%]', ''),
              'EMOJI': (r'[:;<]\-?[\)\(3]', ''),
              'SPACE': (' +', ' '),
              'QUOTATION': (r'[\'"]', '')
              }

    def multi_clean(self, text: str,
                    lower_cap: bool = True) -> str:

        text_output = text

        if lower_cap:
            text_output = text_output.lower()

        for k in self.REG_EX.keys():
            text_output = self.process(text_output, k, lower_cap=False)

        return text_output

    def process(self, text: str,
                reg_ex: str,
                lower_cap: bool = True) -> str:

        if lower_cap:
            text.lower()

        if reg_ex not in self.REG_EX.keys():
            raise KeyError('key not in the REG_EX dictionary')

        sequence, replace = self.REG_EX[reg_ex]
        re_chain = re.compile(sequence)

        return re_chain.sub(replace, text)


class Normalise:

    DEFAULT_POS_EXCLUSION = ['PRP', 'PRP$', 'CC', 'IN', 'DT']

    def __init__(self,
                 add_stop_word: list = []):

        self.stop_word = add_stop_word

    def stopwords(self,
                  text: list):

        if self.stop_word:
            exclude_stopword = stopwords.words('english')
            exclude_stopword += self.stop_word
            return [w for w in text if w not in exclude_stopword]
        else:
            exclude_stopword = set(stopwords.words('english'))
            return [w for w in text if w not in exclude_stopword]

    def stemming(self,
                 text: list):
        stemmer = SnowballStemmer('english')
        return [stemmer.stem(w) for w in text]

    def lemmatizer(self,
                   text: list):
        lemma = WordNetLemmatizer()
        return [lemma.lemmatize(w) for w in text]

    def pos_filter(self,
                   text: list,
                   word_exclusion: str = None):

        if word_exclusion is None:
            word_excl = word_exclusion

        output = []

        pos_tagged = pos_tag(text)

        for t in pos_tagged:
            word, pos = t
            if pos not in word_excl:
                output.append(word)

        return output


class Tokenisation:

    def nltk_reg_ex(self, text: str):
        pattern = re.compile(r"""(\w[#\+]+|[#]?[@\w\'\"\.\-\:]*\w)""", re.VERBOSE)

        return regexp_tokenize(text, pattern=pattern)

    def nltk_word_token(self, text: str):
        return word_tokenize(text)

