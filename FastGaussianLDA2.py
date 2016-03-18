from __future__ import division

import random
import time
from collections import defaultdict, Counter

import gensim
import numpy as np
from numpy import log, pi
from scipy import linalg
from scipy.special import gammaln

import cholesky

__author__ = "Michael Mansour, Jared Thompson, Mike Rinehart"


class Wishart(object):
    def __init__(self, word_vecs):
        self.nu = None
        self.kappa = None
        self.psi = None
        self.mu = None
        self.set_params(word_vecs)

# ======================================================================================================================

    def set_params(self, word_vecs):
        word_vecs = np.vstack(word_vecs.values())  # turn dict of word vectors into a matrix
        self.nu = word_vecs.shape[1]  # dimensionality of word-vectors
        self.kappa = 0.01
        # self.psi = word_vecs.T.dot(word_vecs)#, axis=0)  # sum of squres -- from Murphy(2012)
        self.psi = np.identity(
            word_vecs.shape[1]) * 3.  # changed this to identity matrix as in paper. No intuition here
        self.mu = np.mean(word_vecs, axis=0)


class Gauss_LDA(object):
    def __init__(self, num_topics, corpus, word_vector_filepath=None, word_vector_model=None):
        self.doc_topic_CT = None
        self.word_topics = {}
        self.corpus = corpus
        self.priors = None
        self.word_vecs = {}
        self.numtopics = num_topics
        self.vocab = set([])
        self.topic_params = defaultdict(dict)
        self.wordvecFP = word_vector_filepath
        self.word_index = {}
        self.word_vec_size = None
        self.alpha = 50. / self.numtopics
        self.solver = cholesky.Helper()
        self.wvmodel = word_vector_model
        self.doc_word_counts = {}

# ======================================================================================================================

    def process_corpus(self, documents):
        """
        Tokenizes documents into dict of lists of tokens
        :param documents: expects list of strings
        :return: dict{document ID: list of tokens
        """

        temp_corpus = defaultdict(dict)
        for index, doc in enumerate(documents):
            words = doc.split()
            temp_corpus[index]['words'] = words
            temp_corpus[index]['topics'] = np.random.randint(0, self.numtopics, size=len(words))  # Random topic assign

            for word in words:
                self.vocab.add(word)
        self.corpus = temp_corpus
        print "Done processing corpus with {} documents".format(len(documents))

# ======================================================================================================================

    def process_wordvectors(self, filepath=None):
        """
        Takes a trained Word2Vec model, tests each word in vocab against it, and only keeps word vectors that
        are in your document corpus, and that are in the word2vec corpus.
        Decreases memory requirements for holding word vector info.
        :param filepath: filepath of word-vector file.  Requires 2 things at top of .txt document:
        number of tokens trained on & dimensionality of word-vectors
        :return: None - sets class-variable (self.word_vecs) to be a dict{word: word-vector}
        """
        if filepath:
            print "Processing word-vectors, this takes a moment"
            self.wvmodel = gensim.models.Word2Vec.load_word2vec_format(fname=filepath, binary=False)
            useable_vocab = 0
            unusable_vocab = 0
            self.word_vec_size = self.wvmodel.vector_size

            for word in self.vocab:
                try:
                    self.wvmodel[word]
                    self.word_vecs[word] = self.wvmodel[word]
                    useable_vocab += 1
                    self.word_vecs[word] = self.wvmodel[word]
                except KeyError:
                    unusable_vocab += 1

            print "There are {0} words that could be converted to word vectors in your corpus \n" \
                  "There are {1} words that could NOT be converted to word vectors".format(useable_vocab,
                                                                                           unusable_vocab)
            print "Word-vectors for the corpus are created"

# ======================================================================================================================

    def clean_docs(self):
        print "cleaning out docs of words not in your Word2Vec model"
        approved_words = set(self.word_vecs.vocab.keys())
        for idx, doc in self.corpus.iteritems():
            self.corpus[idx] = [word for word in doc if word in approved_words]
        print "Done cleaning out docs of bad words"

# ======================================================================================================================

    def fit(self, iterations=1, init=True):
        if init:
            self.init()
            init = False

        print "Starting fit"
        for i in xrange(iterations):
            self.sample()
            print "{0} iterations complete".format(i)
            for k in xrange(self.numtopics):
                for param in ("Topic Mean", "Lower Triangle"):
                    results_file = "/Users/michael/Documents/GaussianLDA/output/iter{0}topic{1}{2}.txt".format(i,
                                                                                                               k, param)
                    open(results_file, 'w')
                    np.savetxt(results_file, self.topic_params[k][param])

# ======================================================================================================================

    def init(self):

        self.process_corpus(self.corpus)

        self.process_wordvectors(self.wordvecFP)
        self.priors = Wishart(self.word_vecs)  # set wishhart priors
        self.doc_topic_CT = np.zeros((len(self.corpus.keys()), self.numtopics))

        for k in range(self.numtopics):
            self.topic_params[k]["Topic Sum"] = 0.0

        # get Doc-Topic Counts
        for docID in self.corpus.keys():
            for topic, word in zip(self.corpus[docID]['topics'], self.corpus[docID]['words']):
                self.doc_topic_CT[docID, topic] += 1.  # Ndk
                self.topic_params[topic]['Topic Sum'] += self.word_vecs[word]  # sum of topic vectors

        for k in range(self.numtopics):  # Init parameters for topic distributions
            self.topic_params[k]["Lower Triangle"] = linalg.cholesky(self.priors.psi, lower=True,
                                                                     check_finite=True)
            self.topic_params[k]["Topic Mean"] = self.topic_params[k]["Topic Sum"] / np.sum(self.doc_topic_CT[:, k],
                                                                                            axis=0)
            # 2 * sum_m_i(log(L_i,i))
            self.topic_params[k]["Chol Det"] = np.sum(np.log(np.diag(self.topic_params[k]["Lower Triangle"]))) * 2
            self.topic_params[k]["Topic Count"] = np.sum(self.doc_topic_CT[:, k], axis=0)

        print "Initialization complete"

# ======================================================================================================================

    def sample(self):
        """
        Collapsed Gibbs Sampler derived from Steyver's method, adapted for continuous word-vectors
        :return: None.  Readjusts topic distribution parameters and topic-counts
        """

        for docID in self.corpus.keys():
            for word, topic in zip(self.corpus[docID]['words'], self.corpus[docID]['topics']):
                self.update_document_topic_counts(word, topic, docID, "-")
                self.topic_params[topic]["Topic Sum"] -= self.word_vecs[word]

                self.recalculate_topic_params(word, topic, "-")

                posterior = []
                for k in range(self.numtopics):  # Get PDF for each possible word-topic assignment
                    log_pdf = self.draw_new_wt_assgns(word, k)
                    Nkd = self.doc_topic_CT[docID, k]  # Count of topic in doc, Ndk
                    log_posterior = log(
                        Nkd + self.alpha) + log_pdf  # actual collapsed sampler from R. Das Paper, except in log form
                    posterior.append(log_posterior)
                posterior = np.exp(posterior)
                normalized_post = posterior / np.sum(posterior)
                new_word_topic = np.random.multinomial(1, pvals=normalized_post)

                self.corpus[docID][topic] = np.argmax(new_word_topic)
                self.update_document_topic_counts(word, self.corpus[docID][topic], docID, "+")
                self.recalculate_topic_params(word, self.corpus[docID][topic], "+")
            if docID % 20 == 0:
                print "{0} docs sampled".format(int(docID))
                for k in range(self.numtopics):
                    print self.wvmodel.most_similar(positive=[self.topic_params[k]["Topic Mean"]])

# ======================================================================================================================

    def recalculate_topic_params(self, word, topic, operation, init=False):
        """
        :param topic_id: index for topic
        :param topic_counts: a copy of the doc-topic count table
        :return: None - sets internal class variables
        """

        L = self.topic_params[topic]["Lower Triangle"]

        if operation == "-":  # Cholesky Rank One Downdate
            centered = self.word_vecs[word] - self.topic_params[topic]["Topic Mean"]  # * np.sqrt((kappa_k+1)/kappa_k)
            self.topic_params[topic]["Lower Triangle"] = self.solver.chol_downdate(L, centered)

        topic_count = np.sum(self.doc_topic_CT[:, topic], axis=0)  # N_k
        kappa_k = self.priors.kappa + topic_count  # K_k
        nu_k = self.priors.nu + topic_count  # V_k
        scaled_topic_mean_K = self.get_scaled_topic_mean(topic, topic_count)  # V-Bar_k
        # psi_k = self.priors.psi + scaled_topic_cov_K + ((self.priors.kappa * topic_count) / kappa_k) * (vk_mu.T.dot(vk_mu))  # Psi_k
        topic_mean = ((self.priors.kappa * self.priors.mu) + (topic_count * scaled_topic_mean_K)) / kappa_k  # Mu_k

        if operation == "+":  # Cholesky Rank One Update
            centered = self.word_vecs[word] - topic_mean  # * np.sqrt((kappa_k+1)/kappa_k)
            self.topic_params[topic]["Lower Triangle"] = self.solver.chol_update(L, centered)

        L = self.topic_params[topic]["Lower Triangle"]
        self.topic_params[topic]["Chol Det"] = np.sum(np.log(np.diag(L))) * 2  # 2 * sum_m_i(log(L_i,i))
        self.topic_params[topic]["Topic Count"] = topic_count
        self.topic_params[topic]["Topic Kappa"] = kappa_k
        self.topic_params[topic]["Topic Nu"] = nu_k
        self.topic_params[topic]["Topic Mean"] = topic_mean

# ======================================================================================================================

    def get_scaled_topic_mean(self, topic, topic_count):
        """
        For a given topic, method calculates scaled topic Mean and Covariance (V-bar_k and C_k in R. Das Paper)
        \sum_d \sum_z=i (V_di) / N_k
        ^^ =
        wordvec_sum = array[zero] > shape(word-vec dimensionality)
        for each doc:
                for each word that has topic assignment i:
                    wordvec_sum + word
        wordvec_sum / count of topic
        N_k = count of topic occurences across all documents
        :param topic_id: The topic ID, integer
        :param topic_count: A copy of the document-topic counts table, numpy array
        :return: mean and covariance matrix.  Mean will be of shape (1 X word-vector dimension).
        Covariance will be matrix of size (word-vector dim X word-vector dim)
        """
        scaled_topic_mean = self.topic_params[topic]["Topic Sum"] / topic_count
        return scaled_topic_mean

# ======================================================================================================================

    # noinspection PyStatementEffect
    def update_document_topic_counts(self, word, topic, docID, operation):

        if operation == "-":
            self.topic_params[topic]["Topic Sum"] -= self.word_vecs[word]
            self.doc_topic_CT[docID, topic] -= 1

        if operation == "+":
            self.topic_params[topic]["Topic Sum"] += self.word_vecs[word]
            self.doc_topic_CT[docID, topic] += 1

# ======================================================================================================================

    def draw_new_wt_assgns(self, word, topic, new_doc=False, wvmodel=None):
        """
        Log of the probablity density function for the Student-T Distribution
        Provides a PDF for a word (really a word-vector) in a given topic distribution.
        :param word: string of the word to find probabilty of word-topic assignment
        :param topic: Interger, a topic id to reference a topic distribution and its params
        :param new_doc: False (default), optional.  True if predicting topics from unseen document/not currently training
        :param wvmodel: None by default.  If predicting topics from an unseen document, requires a loaded word2vec model
        from GenSim
        :type wvmodel: gensim.models.word2vec.Word2Vec
        :return: log of PDF from t-distribution for a given word.  Type: Float
        """

        cov_det = self.topic_params[topic]["Chol Det"]
        Nk = self.topic_params[topic]["Topic Count"]
        # (V_di - Mu)
        centered = self.word_vecs[word] - self.topic_params[topic]["Topic Mean"]
        # (L^-1b)^T(L^-1b)
        linalg.cho_solve((self.topic_params[topic]["Lower Triangle"], True), centered, overwrite_b=True,
                         check_finite=True)
        LLcomp = centered.T.dot(centered)  # TODO check math operations here
        d = self.word_vec_size  # dimensionality of word vector
        nu = self.priors.nu + Nk - d + 1.

        # Log PDF of multivariate student-T distribution
        log_prob = gammaln(nu + d / 2.) - \
                   (gammaln(nu / 2.) + (d / 2.) * (log(nu) + log(pi)) + 0.5 * cov_det + ((nu + d) / 2.) * log(
                       (1. + LLcomp) / nu))
        return log_prob

# ======================================================================================================================

    def bin_search(self, pdf, key, start, end):
        if start > end:
            return start
        mid = (start+end) / 2.0
        if key == pdf[mid]:
            return mid + 1
        if key < pdf[mid]:
          return self.bin_search(pdf, key, start, mid-1)
        if key > pdf[mid]:
            return self.bin_search(pdf, key, mid + 1, end)
        else:
            return None

    def extract_topics_new_doc(self, doc, wv_model):
        """
        :type wv_model: gensim.models.word2vec.Word2Vec
        :param doc: Document to extrac topics from.  should be one string
        :param wv_model: a loaded word2vec model with same dimensionality as training one.  Use GenSim Word2Vec
        :return: List of tuples (word, topic)
        Method removes words in doc that are not in the Word2Vec corpus, and extracts word-topic assignments for each
        word by drawing densities from the multivariate student-T distribution.  Uses MLE method.
        """
        assert wv_model.vector_size == self.word_vec_size, "word-vector dimensionality does not match trained topic" \
                                                           "distribution dimensions({0})".format(self.word_vec_size)
        filtered_doc = []
        nkd = defaultdict(float)
        for word in doc.split():
            try:
                wv_model[word]
                filtered_doc.append(word)  # Remove words from doc that are not in word-vec model
                nkd[self.word_topics[word]] += 1.
            except KeyError:
                continue
        print "{} words removed from doc".format(len(filtered_doc) - len(doc.split()))
        word_topics = []
        c = Counter(self.word_topics.values())
        for word in filtered_doc:
            posterior = []
            for k in range(self.numtopics):
                # print nkd[k]
                prob = self.draw_new_wt_assgns(word, k, wvmodel=wv_model, new_doc=True) * log(self.alpha + c[k])
                print "probablity of {0} for word {1} assigned to topic {2}".format(prob, word, k)
                posterior.append(prob)
            posterior /= np.sum(posterior)

            word_topics.append((word, np.argmax(posterior)))
        return word_topics


if __name__ == "__main__":
    # corpus = [
    #     "apple orange mango melon ", "canvas art mural paint painting ", "pineapple kiwi grape strawberry ",
    #     "picture frame picasso sculpture art ", "coconut guava blueberry blackberry ", "statue monument art artist "
    # ]
    # corpus = [sent * 5 for sent in corpus]*4

    f = '/Users/michael/Documents/GaussianLDA/clean20news.txt'
    with open(f, 'r') as fi:
        docs = fi.read().splitlines()  # These are all cleaned out
        fi.close()
    wordvec_fileapth = "/Users/michael/Documents/Gaussian_LDA-master/data/glove.wiki/glove.6B.50d.txt"
    start = time.time()
    g = Gauss_LDA(20, docs, word_vector_filepath=wordvec_fileapth)
    g.fit(5)
    print time.time() - start
