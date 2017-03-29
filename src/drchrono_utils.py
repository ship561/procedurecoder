from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import pandas as pd

tokenizer = RegexpTokenizer(r'\w+')

en_stop = set(stopwords.words('english'))  # english stop words database

en_stop.update(('pt', 'patient', 'male', 'female'))

p_stemmer = PorterStemmer()              # stemmer
wordnet_lemmatizer = WordNetLemmatizer()


def is_ascii(s):
    try:
        s.decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def partition_all(n, V):
    return [V[i:i+n] for i in range(0, len(V), n)]


def flatten(V):
    return [item for sublist in V for item in sublist]


def process_chunks(cur, n=100):
    rows = cur.fetchmany(n)
    for row in rows:
        yield row


def cpt_code_to_list(cpt_string):
    return map(str, cpt_string.split(','))


def pre_process_document(doc):
    doc_cleaned_tokens = []
    doc_tokens = filter(lambda x: x.isalpha(), nltk.word_tokenize(doc))
    for w in nltk.pos_tag(doc_tokens):
        if w[1] != 'NNP':
            w = w[0].lower()
            if w not in en_stop and len(w) > 2:
                word = wordnet_lemmatizer.lemmatize(w)
                doc_cleaned_tokens.append(word)

    cleaned_text = " ".join(doc_cleaned_tokens)
    return cleaned_text


def pre_process_corpus(df):
    corpus_doc_notes = []
    corpus_doc_notes_cpt = []
    corpus_doc_notes_id = []

    for index, d in df.iterrows():
        if is_ascii(d['note']):
            new_data = pre_process_document(d['note'])
            corpus_doc_notes.append(new_data)
            cpts = set(cpt_code_to_list(d['cpt']))
            corpus_doc_notes_cpt.append(cpts)
            corpus_doc_notes_id.append(d['id'])

    return pd.DataFrame({'notes': corpus_doc_notes,
                         'cpt': corpus_doc_notes_cpt,
                         'id': corpus_doc_notes_id})


def separate_class(corpus, labs, ids):
    df = zip(corpus, labs, ids)
    zero = filter(lambda x: x[1] == 0, df)
    ones = filter(lambda x: x[1] == 1, df)
    return [zero, ones]
