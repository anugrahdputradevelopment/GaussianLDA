from __future__ import division

import random
import time
from collections import defaultdict, Counter

import gensim
import numpy as np
from numpy import log, pi
from scipy import linalg
from scipy.special import gammaln
import copy
from sklearn.cluster import KMeans
from numba import jit

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
        self.kappa = 0.1
        # self.psi = word_vecs.T.dot(word_vecs)#, axis=0)  # sum of squres -- from Murphy(2012)
        # self.psi = np.identity(
        #     word_vecs.shape[1]) * 3.  # changed this to identity matrix as in paper. No intuition here
        # self.mu = np.mean(word_vecs, axis=0)

# ======================================================================================================================

class Gauss_LDA(object):
    def __init__(self, num_topics, corpus, word_vector_filepath=None,
                 word_vector_model=None, alpha=0.2, outputfile=None, preprocess=False):
        self.doc_topic_CT = None
        self.corpus = corpus
        self.priors = None
        self.word_vecs = {}
        self.numtopics = num_topics
        self.vocab = set([])
        self.topic_params = defaultdict(dict)
        self.wordvecFP = word_vector_filepath
        self.word_vec_size = None
        self.alpha = alpha
        self.solver = cholesky.Helper()
        self.wvmodel = word_vector_model
        self.test_word_topics = defaultdict(list)
        self.test_word_topic_count = defaultdict(int)
        self.word_topics = {}
        self.output_file_name = outputfile
        self.preprocess = preprocess

# ======================================================================================================================

    def process_corpus(self, documents):
        """
        Tokenizes documents into dict of lists of tokens
        :param documents: expects list of strings
        :return: dict{document ID: list of tokens
        """
        if not self.preprocess:
            temp_corpus = defaultdict(dict)
            random.shuffle(documents)  # making sure topics are formed semi-randomly
            for index, doc in enumerate(documents):
                words = doc.split()
                temp_corpus[index]['words'] = words
                temp_corpus[index]['topics'] = np.empty(len(words))  # Random topic assign

                # temp_corpus[index]['topics'] = np.random.randint(0, self.numtopics, size=len(words))  # Random topic assign

                for word in words:
                    self.vocab.add(word)
            self.corpus = temp_corpus
            print "Done processing corpus with {} documents".format(len(documents))

        else: # Docs are tokenized and such, just add it into class
            temp_corpus = defaultdict(dict)
            for idx, doc in enumerate(documents):
                temp_corpus[idx]["words"] = doc
                temp_corpus[idx]["topics"] = np.empty(len(doc))
                for word in doc:
                    self.vocab.add((word))
            self.corpus = temp_corpus


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
                    self.word_vecs[word] = self.wvmodel[word]
                    useable_vocab += 1
                except KeyError:
                    unusable_vocab += 1

            print "There are {0} words that could be converted to word vectors in your corpus \n" \
                  "There are {1} words that could NOT be converted to word vectors".format(useable_vocab,
                                                                                           unusable_vocab)
        else:
            useable_vocab = 0
            unusable_vocab = 0
            self.word_vec_size = self.wvmodel.vector_size

            for word in self.vocab:
                try:
                    self.word_vecs[word] = self.wvmodel[word]
                    useable_vocab += 1
                except KeyError:
                    unusable_vocab += 1

            print "There are {0} words that could be converted to word vectors in your corpus \n" \
                  "There are {1} words that could NOT be converted to word vectors".format(useable_vocab,
                                                                                           unusable_vocab)

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
        self.display_results()
        for i in range(iterations):

            self.sample()
            print "{0} iterations complete".format(i)
        if self.output_file_name:  #TODO: fix such that it prints regardless of outputfilename
            for k in xrange(self.numtopics):

                for param, name in zip(("mean", "cov"),
                                       (self.topic_params[k]["Topic Mean"], self.topic_params[k]["Topic Covariance"])):
                    self.output_file_name = self.output_file_name + "{}_{}"

                    results_file = self.output_file_name.format(k, param)
                    open(results_file, 'w')
                    np.savetxt(results_file, param)

# ======================================================================================================================

    def init(self):

        self.process_corpus(self.corpus)
        self.process_wordvectors(self.wordvecFP)
        self.priors = Wishart(self.word_vecs)  # set wishhart priors
        self.doc_topic_CT = np.zeros((len(self.corpus.keys()), self.numtopics))  # Init document-topic counts matrix
        centroids, km = self.smart_centroids()  # Init topic means with KMeans cluster centroids>>faster convergence

        # Prior mean
        mu_0 = np.zeros(self.word_vec_size)
        count = 0
        for docID in self.corpus.keys():  # hard setting word-topic assignments as per cluster membership to help model along
            for i, word in enumerate(self.corpus[docID]['words']):
                self.corpus[docID]['topics'][i] = self.word_topics[word]  # word_topics from KMeans cluster membership
                mu_0 += self.word_vecs[word]
                count += 1
        self.priors.mu = mu_0 / float(count)  # trying a different prior mean init

        # Prior co-variance
        self.priors.psi = .01 * np.identity(self.word_vec_size)

        # Sample means
        for k in range(self.numtopics):
            self.topic_params[k]["Topic Sum"] = np.zeros(self.word_vec_size)
            self.topic_params[k]["Topic Mean"] = centroids[k]
            self.topic_params[k]["Sample Cov"] = np.zeros((self.word_vec_size, self.word_vec_size))
        # Sample co-variances and document-topic counts
        co_variances = [np.zeros((self.word_vec_size, self.word_vec_size)) for _ in range(self.numtopics)]
        for docID in self.corpus.keys():
            for topic, word in zip(self.corpus[docID]['topics'], self.corpus[docID]['words']):
                topic = int(topic)
                wv = self.word_vecs[word]
                sample_mu = self.topic_params[topic]["Topic Mean"]
                self.doc_topic_CT[docID, topic] += 1.  # Ndk
                self.topic_params[topic]['Topic Sum'] += wv  # sum of topic vectors
                # self.topic_params[topic]["Sample Cov"] += np.outer(wv - sample_mu, wv-sample_mu)
                co_variances[topic] += np.outer(wv - sample_mu, wv - sample_mu) #+ self.priors.psi

# another attempt at doing the covariances, closer to the paper
#         co_variances = [np.zeros((self.word_vec_size, self.word_vec_size)) for _ in range(self.numtopics)]
#         for docID in self.corpus.keys():
#             for topic, word in zip(self.corpus[docID]['topics'], self.corpus[docID]['words']):
#                 topic = int(topic)
#                 sample_mu = self.topic_params[topic]["Topic Mean"]
#                 Nk = np.sum(self.doc_topic_CT[:, topic], axis=0)
#                 scale = (self.priors.kappa * Nk) / (self.priors.kappa + Nk)
#                 co_variances[topic] = scale * np.outer(sample_mu - self.priors.mu, sample_mu - self.priors.mu)
#                 co_variances[topic] += self.topic_params[topic]["Sample Cov"] + self.priors.psi

        #   Normalize the sample co-variance
        for k in range(self.numtopics):
            co_variances[k] = (co_variances[k] / (np.sum(self.doc_topic_CT[:, k]) - 1.)) + self.priors.psi
            # Possible error spot
        kappa = self.priors.kappa
        nu = self.priors.nu
        d = self.word_vec_size
        scaleT = (kappa + 1.) / (kappa * (nu - d + 1.))  # Needed to convert L => covariance

        for k in range(self.numtopics):  # Init parameters for topic distributions
            Nk = np.sum(self.doc_topic_CT[:, k], axis=0)
            self.topic_params[k]["Lower Triangle"] = linalg.cholesky(co_variances[k], lower=True,
                                                                     check_finite=True)
            self.topic_params[k]["Topic Count"] = Nk
            self.topic_params[k]["Topic Kappa"] = self.priors.kappa + Nk

            # 2 * sum_m_i(log(L_i,i)) + log(scaleT)
            self.topic_params[k]["Chol Det"] = np.sum(np.log(np.diag(self.topic_params[k]["Lower Triangle"]))) * 2 \
            + np.log(scaleT)

        print np.sum(self.doc_topic_CT, axis=0)

        print "Initialization complete"

# ======================================================================================================================

    def smart_centroids(self):
        print "getting cluster centroids"
        from sklearn.cluster import KMeans
        vecs = []
        for word in self.vocab:
            vecs.append(self.word_vecs[word])
        km = KMeans(n_clusters=self.numtopics, n_jobs=1, tol=1e-6, init='k-means++')
        km.fit(np.array(vecs))
        for idx, word in enumerate(self.vocab):
            self.word_topics[word] = km.labels_[idx]
        vec_matrix = np.array(vecs)
        for k in range(self.numtopics):
            idx = np.where(km.labels_ == k)
            # covar = np.cov(vec_matrix[idx] - km.cluster_centers_[k], rowvar=0)  # Mean centered covariance matrix
            # self.topic_params[k]['Topic Covar'] = covar
            self.topic_params[k]["Topic Mean"] = km.cluster_centers_[k]
        return km.cluster_centers_, km

# ======================================================================================================================

    def sample(self):
        """
        Collapsed Gibbs Sampler derived from Steyver's method, adapted for continuous word-vectors
        :return: None.  Readjusts topic distribution parameters and topic-counts
        """

        ASSIGN_NEW_TOPICS = True
        MULTINOMIAL_TOPIC_SELECTION = True

        for docID in self.corpus.iterkeys():
            for idx in range(len(self.corpus[docID]['words'])):
                word = self.corpus[docID]['words'][idx]
                current_topic = self.corpus[docID]['topics'][idx]

                # former_dists = copy.deepcopy(self.topic_params)

                self.recalculate_topic_params(word, current_topic, docID, "-")
                log_posterior = np.zeros(self.numtopics)
                for k in range(self.numtopics):  # Get PDF for each possible word-topic assignment
                    log_pdf = self.draw_new_wt_assgns(word, k)
                    Nkd = self.doc_topic_CT[docID, k]  # Count of topic in doc, Ndk
                    log_posterior[k] = log(Nkd + self.alpha) + log_pdf  # actual collapsed sampler from R. Das Paper, except in log form

                max_log_posterior = np.max(log_posterior)
                log_posterior -= max_log_posterior
                normalized_post = np.exp(log_posterior - np.log(np.sum(np.exp(log_posterior))))
                if MULTINOMIAL_TOPIC_SELECTION:
                    new_topic = np.argmax(np.random.multinomial(1, pvals=normalized_post))
                else:
                    new_topic = np.argmax(normalized_post)

                if not ASSIGN_NEW_TOPICS:
                    new_topic = current_topic

                self.corpus[docID]['topics'][idx] = new_topic
                self.recalculate_topic_params(word, new_topic, docID, "+")

                # last_word = word
                # last_word_current_topic = current_topic
                # last_word_new_topic = new_topic

            if docID % 20 == 0:
                print "{0} docs sampled".format(int(docID))

        self.display_results()

# ======================================================================================================================

    def recalculate_topic_params(self, word, topic, docID, operation):
        """
        :param topic_id: index for topic
        :param topic_counts: a copy of the doc-topic count table
        :return: None - sets internal class variables
        """
        # Update the topic-count table
        UPDATE_COUNT = True
        if UPDATE_COUNT:
            self.update_document_topic_counts(word, topic, docID, operation)

        # Update parameters related to the priors
        topic_count = np.sum(self.doc_topic_CT[:, topic], axis=0)  # N_k
        kappa_k = self.priors.kappa + topic_count  # K_k
        nu_k = self.priors.nu + topic_count  # V_k
        scaleT = (kappa_k + 1.) / (kappa_k * (nu_k - self.word_vec_size + 1.))  # Needed to convert L => covariance

        UPDATE_DISTS = True

        if operation == "-":  # Remove data point contribution to the topic distribution
            # Original equation is:
            #    \Sigma \leftarrow \Sigma - (k_0 + N + 1)/(k_0 + N)(X_{n} - \mu_{n-1})(X_{n} - \mu_{n-1})^T
            if UPDATE_DISTS:
                L = self.topic_params[topic]["Lower Triangle"]
                centered = self.word_vecs[word] - self.topic_params[topic]["Topic Mean"] # Get rank-1 matrix from point
                # centered = (self.topic_params[topic]["Topic Mean"] - self.word_vecs[word])  # paper says this way
                centered *= np.sqrt((kappa_k + 1.) / kappa_k)  # Scale for recursive downdate
                L = self.solver.chol_downdate(L, centered)  # Choleksy downdate
                self.topic_params[topic]["Lower Triangle"] = L

            # Correct the mean for the removed point
            sample_mean_K = self.topic_sample_mean(topic, topic_count)  # V-Bar_k
            # topic_sum = self.topic_params[topic]["Topic Sum"]
            topic_mean = ((self.priors.kappa * self.priors.mu) + (topic_count * sample_mean_K)) / kappa_k  # Mu_k

            # topic_mean = self.topic_params[topic]["Topic Mean"]
            # topic_mean *= kappa_k+1
            # topic_mean -= self.word_vecs[word]
            # topic_mean /= kappa_k

        else:  # operation == "+":  # Add data point contribution to the topic distribution
            # Correct the mean for the added point
            # Trying a new method of calculating the Mean
            # topic_mean = self.topic_params[topic]["Topic Mean"]
            # topic_mean *= kappa_k-1
            # topic_mean += self.word_vecs[word]
            # topic_mean /= kappa_k

            sample_mean_K = self.topic_sample_mean(topic, topic_count)  # V-Bar_k
            # topic_sum = self.topic_params[topic]["Topic Sum"]
            topic_mean = ((self.priors.kappa * self.priors.mu) + (topic_count * sample_mean_K)) / kappa_k  # Mu_k
            # topic_mean = ((self.priors.kappa * self.priors.mu) + (sample_mean_K)) / kappa_k  # Mu_k

            # Original equation is:
            #    \Sigma \leftarrow \Sigma + (k_0 + N + 1)/(k_0 + N)(X_{n} - \mu_{n-1})(X_{n} - \mu_{n-1})^T
            if UPDATE_DISTS:
                L = self.topic_params[topic]["Lower Triangle"]
                centered = (self.word_vecs[word] - topic_mean)
                # centered = (topic_mean - self.word_vecs[word])# Get rank-1 matrix from point
                # centered = centered.dot(centered.T)
                centered *= np.sqrt(kappa_k / (kappa_k - 1.))  # Scale for recursive update
                L = self.solver.chol_update(L, centered)  # Choleksy update
                self.topic_params[topic]["Lower Triangle"] = L

        L = self.topic_params[topic]["Lower Triangle"]
        self.topic_params[topic]["Chol Det"] = (np.sum(np.log(np.diag(L))) * 2) + np.log(scaleT) # 2 * sum_m_i(log(L_i,i))
        self.topic_params[topic]["Topic Count"] = topic_count
        self.topic_params[topic]["Topic Kappa"] = kappa_k
        self.topic_params[topic]["Topic Nu"] = nu_k
        if UPDATE_DISTS:
            self.topic_params[topic]["Topic Mean"] = topic_mean


# ======================================================================================================================

    def topic_sample_mean(self, topic, topic_count):
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
        scaled_topic_mean = self.topic_params[topic]["Topic Sum"] / \
                            float(topic_count) if topic_count > 0 else np.zeros(self.word_vec_size)
        return scaled_topic_mean

# ======================================================================================================================

    # noinspection PyStatementEffect
    def update_document_topic_counts(self, word, topic, docID, operation):

        if operation == "-":
            self.topic_params[topic]["Topic Sum"] -= self.word_vecs[word]
            self.doc_topic_CT[docID, topic] -= 1.

        if operation == "+":
            self.topic_params[topic]["Topic Sum"] += self.word_vecs[word]
            self.doc_topic_CT[docID, topic] += 1.

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
        d = self.word_vec_size  # dimensionality of word vector
        kappa_k = self.topic_params[topic]["Topic Kappa"]

        scaleT = np.sqrt((kappa_k + 1.) / kappa_k * (self.priors.nu - d + 1.))  # Covariance = chol / sqrt(scaleT)
        nu = self.priors.nu + Nk - d + 1.
        L = self.topic_params[topic]["Lower Triangle"]

        # linalg.cho_solve((L, True), centered, overwrite_b=True,
        #          check_finite=False)
        # inv = centered.T.dot(centered)  # (L^-1b)^T(L^-1b)
        #
        # # Log Multivariate T - PDF
        # return gammaln((nu + d) / 2.) - \
        #     (gammaln(nu / 2.) + (d / 2.) * (log(nu) + log(pi))
        #     + (0.5 * cov_det) + ((nu + d) / 2.) * log(1. + inv/nu))

        return self.multivariate_t_pdf(nu, cov_det, d, scaleT, centered, L)

    @jit
    def multivariate_t_pdf(self, nu, cov_det, d, scaleT, centered, L):
        L *= scaleT
        linalg.cho_solve((L, True), centered, overwrite_b=True,
                 check_finite=False)
        inv = centered.T.dot(centered)  # (L^-1b)^T(L^-1b)

        # Log Multivariate T - PDF
        return gammaln((nu + d) / 2.) - \
            (gammaln(nu / 2.) + (d / 2.) * (log(nu) + log(pi))
            + (0.5 * cov_det) + ((nu + d) / 2.) * log(1. + inv/nu))

# ======================================================================================================================

    def bin_search(self, pdf, key, start, end):  # Not using
        if start > end:
            return start
        mid = int((start + end) / 2)
        if key == pdf[mid]:
            return mid + 1
        if key < pdf[mid]:
          return self.bin_search(pdf, key, start, mid-1)
        if key > pdf[mid]:
            return self.bin_search(pdf, key, mid + 1, end)
        else:
            return None

# ======================================================================================================================

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

# ======================================================================================================================

    def display_results(self):
        print 'print topic means'
        for k in range(self.numtopics):
            print "TOPIC {0}:".format(k), \
                zip(*self.wvmodel.most_similar(positive=[self.topic_params[k]["Topic Mean"]], topn=9))[0]
            if k == max(range(self.numtopics)):
                print "\n"
        print "Document-Topic Counts:,", np.sum(self.doc_topic_CT, axis=0).astype(int)

# ======================================================================================================================

    def explore_topics(self):
        self.word_counts = {word: np.zeros(self.numtopics, dtype=int) for word in self.vocab}
        for docID in self.corpus.keys():
            for topic, word in zip(self.corpus[docID]['topics'], self.corpus[docID]['words']):
                self.word_counts[word][int(topic)] += 1
        counts = np.array(self.word_counts.values())
        ranked = np.argsort(counts, axis=0)[::-1][:20, :]
        words = np.array(self.word_counts.keys())
        for k in range(self.numtopics):
            print words[ranked[:, k]]



if __name__ == "__main__":

    f = '/Users/michael/Documents/GaussianLDA/data/cleannips.txt'
    with open(f, 'r') as fi:
        docs = fi.read().splitlines()  # These are all cleaned out
        fi.close()
    wordvec_fileapth = "/Users/michael/Documents/Gaussian_LDA-master/data/glove.wiki/glove.6B.50d.txt"
    start = time.time()
    g = Gauss_LDA(50, docs, word_vector_filepath=wordvec_fileapth, alpha=0.7)

    g.fit(3)
    print time.time() - start