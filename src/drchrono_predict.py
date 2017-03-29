import pandas as pd  # for making easy-to-use data structures
import numpy as np
from collections import Counter
import pickle
from sklearn.metrics.pairwise import linear_kernel
import drchrono_utils as du


## example corpus for testing
ex_corpus = ['here is a sentence.',
             'my lower back is hurt. consider chiropractic adjustment on spine',
             'i hate black cats',
             'similarly, for text processing we also have another improtant concept of n-grams. n-gram is a sequence of n consecutive words.']

## load in models
doc_notes_vectorizer = pickle.load(open("models/doc_notes_vectorizer_save.p", "rb"))
cpt_tfidf_vectorizer = pickle.load(open("models/cpt_vectorizer_save.p", "rb"))
x_tfidf = pickle.load(open("models/x_tfidf_save.p", "rb"))
outpatient_bayes_mod = pickle.load(open("models/outpatient_bayes_mod_save.p", "rb"))
outpatient_bg_probs = pickle.load(open("models/outpatient_bg_probs_save.p", "rb"))
chiropractic_bayes_mod = pickle.load(open("models/chiropractic_bayes_mod_save.p", "rb"))
chiropractic_bg_probs = pickle.load(open("models/chiropractic_bg_probs_save.p", "rb"))

## cpt description corpus' from data bases
cpt_descriptions = pickle.load(open("models/cpt_descript_save.p", "rb"))


def rank_to_score(norm_rank):
    """Adjusts the score so that high ranks get high scores"""

    score = 1-norm_rank
    return score


def classifier_combiner(k, rank_cpt_knn, rank_cpt_bayes):
    """This function combines the keyword (TF-idf) and NB models using the
       relative ranks and belief in keywords vs. model probability"""

    score_cpt_knn = rank_to_score(rank_cpt_knn)
    score_cpt_bayes = rank_to_score(rank_cpt_bayes)  # rank_to_score(rank_cpt_bayes)

    score_cpt = k*score_cpt_knn + (1-k)*score_cpt_bayes
    return score_cpt


def normalize_ranks(raw_ranks_dict):
    """Takes a numerical (integer) ranks (array/list) and normalizes them
    to interval [0, 1]"""
    
    N = float(max(raw_ranks_dict.values()))
    norm_ranks = dict(map(lambda (k, v): (k, v/N), raw_ranks_dict.iteritems()))
    return norm_ranks


def calc_knn_document(d):
    """Ranks document (d) according the CPT code descriptions using cosine similarly."""
    
    new_x = cpt_tfidf_vectorizer.transform([d])
    sims = linear_kernel(new_x, x_tfidf).flatten()
    related_docs_indices = sims.argsort()[:-20:-1]
    return [cpt_descriptions.iloc[related_docs_indices].set_index('code').T.to_dict('list'),
            cpt_descriptions.code.iloc[related_docs_indices].tolist()]


def calc_bayes_sentence(s, bayes_mod):
    """Gets the NB model probability for a given document (d) and a model (bayes_mod).  """

    test_new_ex_vec = doc_notes_vectorizer.transform([s])
    bayes_mod_probs = bayes_mod.predict_proba(test_new_ex_vec.toarray())
    return bayes_mod_probs


def bayes_rank(prob_sentence, prob_background_list):
    """Uses the probability predicted from NB model and ranks it relative
    to the background. This function ranks the prob relative to random
    simulated notes.

    """
    bg_prob_greater = filter(lambda p: p > prob_sentence, prob_background_list)
    return len(bg_prob_greater)/float(len(prob_background_list))


def calc_top_cpt_codes(d, alpha=0.5, n=5):
    """Take a document d and return the 5 highest ranked cpt codes"""

    potential_cpt_codes, knn_closest = calc_knn_document(d)
    potential_cpt_codes.update({'9920x': ['Outpatient care',
                                          'Outpatient care'],
                                '9894x': ['Chiropractic adjustment',
                                          'Chiropractic adjustment']})

    ## these classifiers should be built into a better data structure
    ## but are kept this way due to time constraints. Should move it
    ## into a dictionary where key=cpt code block and value=NB
    ## classifier model.
    bayes_9920x = calc_bayes_sentence(d, outpatient_bayes_mod)
    bayes_9894x = calc_bayes_sentence(d, chiropractic_bayes_mod)
    bayes_ranks = {'9920x': bayes_rank(bayes_9920x[0, 1], outpatient_bg_probs),
                   '9894x': bayes_rank(bayes_9894x[0, 1], chiropractic_bg_probs)}

    score_dict = {}

    knn_ranks = normalize_ranks(dict(zip(knn_closest, range(0, len(knn_closest), 1))))

    for k in knn_ranks.keys()+bayes_ranks.keys():
        score_k =  classifier_combiner(alpha, knn_ranks.get(k, 1), bayes_ranks.get(k, 1))
        score_dict.update({k: score_k})

    top_score_list = sorted(score_dict.items(), key=lambda x: -x[1])

    output = []
    for code in [i[0] for i in top_score_list]:
        output.append([code, potential_cpt_codes.get(code, ['no descript', 'no descript'])[1]])

    return pd.DataFrame.from_records(output[:n],
                                     columns=['code', 'description'])


def main(d):
    return calc_top_cpt_codes(du.pre_process_document(d))

real_doc_99214 = "Overall better goal decrease hypertonicity and increase joint mobility by increase flexibility"

print main(real_doc_99214)
