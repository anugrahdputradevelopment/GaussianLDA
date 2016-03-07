from __future__ import division
import gensim
import numpy as np
from numpy import log, pi, linalg, exp
from scipy.special import gamma, gammaln
import random
from collections import defaultdict
__author__ = "Michael Mansour"


class Wishart(object):

    def __init__(self, word_vecs):
        self.nu = None
        self.kappa = None
        self.psi = None
        self.mu = None
        self.set_params(word_vecs)

    def set_params(self, word_vecs):
        # turn dict of word vecs into a matrix
        word_vecs = np.vstack(word_vecs.values())
        self.nu = word_vecs.shape[1]  # dimensionality of word-vectors
        self.kappa = 0.01
        # self.psi = word_vecs.T.dot(word_vecs)#, axis=0)
        self.psi = np.identity(word_vecs.shape[1]) * 3.  # changed this to identity matrix as in paper. No intuition here
        self.mu = np.mean(word_vecs, axis=0)

class Gauss_LDA(object):

    def __init__(self, num_topics, corpus, word_vector_filepath):
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


    def process_corpus(self, documents):

        temp_corpus = {}
        for index, doc in enumerate(documents):
            words = doc.split()
            temp_corpus[index] = words
            for word in words:
                self.vocab.add(word)
        self.corpus = temp_corpus
        print "Done processing corpus with {} documents".format(len(documents))


    def process_wordvectors(self, filepath):
        print "Processing word-vectors, this takes a moment"
        vectors = gensim.models.Word2Vec.load_word2vec_format(fname=filepath, binary=False)
        useable_vocab = 0
        unusable_vocab = 0
        self.word_vec_size = vectors.vector_size

        for word in self.vocab:
            try:
                vectors[word]
                useable_vocab += 1
            except KeyError: unusable_vocab += 1

        print "There are {0} words that could be convereted to word vectors in your corpus \n" \
              "There are {1} words that could NOT be converted to word vectors".format(useable_vocab, unusable_vocab)

        for word in self.vocab:
            try:
                self.word_vecs[word] = vectors[word]  # Dict{Word: Word-Vector}
            except KeyError: continue

        print "Word-vectors for the corpus are created"

    def fit(self, iterations=1, init=True):
        if init == True:
            self.init()
            init = False

        print "Starting fit"
        for i in xrange(iterations):
            self.sample()
            print "{0} iterations complete".format(i)

    def init(self):

        self.process_corpus(self.corpus)
        self.process_wordvectors(self.wordvecFP)
        self.priors = Wishart(self.word_vecs)  # set wishhart priors
        self.doc_topic_CT = np.zeros((len(self.corpus.keys()), self.numtopics))

        self.word_topics = {word: random.choice(range(self.numtopics)) for word in self.vocab}
        # get Doc-Topic Counts
        for docID, doc in self.corpus.iteritems():
            for word in doc:
                topicID = self.word_topics[word]
                self.doc_topic_CT[docID, topicID] += 1. # TODO: SHOULD THIS BE + INSTEAD OF +=???

        for k in range(self.numtopics):  # Init parameters for topic distributions
            self.recalculate_topic_params(k, self.doc_topic_CT)

        print "Intialization complete"

    def sample(self, init=True):

        # Randomly assign word to topics
        if init == False:
            self.word_topics = {word: random.choice(range(self.numtopics)) for word in self.vocab}
        for docID, doc in self.corpus.iteritems():
            for word in doc:
                # subtracting info about current word-topic assignment from doc-topic count table

                # self.doc_topic_CT[docID, topic_id] - float(doc.count(word)) #expirmenting with -= vs -
                # self.update_document_topic_counts(word, "-")
                # self.recalculate_topic_params(self.word_topics[word])

                posterior = []
                max = 0
                for k in range(self.numtopics):  # start getting the pdf's for each word-topic assignment
                    topic_counts = self.update_document_topic_counts(word, k, "-")
                    self.recalculate_topic_params(k, topic_counts)

                    log_prob = self.draw_new_wt_assgns(word, k)
                    # print "multivariate T PDF for wordvector", log_prob
                    Nkd = topic_counts[docID, k] # Count of topic in doc
                    # print "Nkd = {}".format(Nkd)
                    log_posterior = log(Nkd + self.alpha) * log_prob  #; print "log posteriror with doc-contribution", log_posterior
                    posterior.append(log_posterior)  # doing this for some normalization scheme
                    if log_posterior > max: max = log_posterior

                posterior.append(0)  # just a little hitch in the function that wants a zero at the end.
                post_sum = np.sum(posterior)
                normalized_post = posterior / post_sum
                new_word_topic = np.random.multinomial(1, pvals=normalized_post)  # possibly need 2 copy the util.sample from Das
                # print 'multinomial with reg-norm', new_word_topic

                self.word_topics[word] = np.argmax(new_word_topic)
                self.doc_topic_CT = self.update_document_topic_counts(word, self.word_topics[word], "+")
                self.recalculate_topic_params(self.word_topics[word], self.doc_topic_CT)

        # init = False
        return None

    def draw_new_wt_assgns(self, word, topic_id, new_doc=False, wvmodel=None):

        if new_doc == False:
            # Getting params for calculating PDF of T-Dist for a word
            inv_cov = self.topic_params[topic_id]["Inverse Covariance"]
            cov_det = self.topic_params[topic_id]["Covariance Determinant"]  #cov_det is already logged
            Nk = self.topic_params[topic_id]["Topic Count"]

            # Precalculating some terms (V_di - Mu)^T * Cov^-1 * (V_di - Mu)
            centered = self.word_vecs[word] - self.priors.mu
            LLcomp = centered.T.dot(inv_cov).dot(centered)  # for some topics, this outputs a negative value
            d = self.word_vec_size   # dimensionality of word vector
            nu = self.priors.nu + Nk - d + 1.

            # Log PDF of multivariate student-T distribution
            log_prob = gammaln(nu + d / 2.) - \
                       (gammaln(nu / 2.) + d/2. * (log(nu) + log(pi)) +0.5 * cov_det[1] + ((nu + d) / 2.) * log((1. + LLcomp ) / nu))

            return log_prob

        if new_doc == True:
            inv_cov = self.topic_params[topic_id]["Inverse Covariance"]
            cov_det = self.topic_params[topic_id]["Covariance Determinant"]
            Nk = self.topic_params[topic_id]["Topic Count"]
            centered = wvmodel[word] - self.priors.mu

            LLcomp = centered.T.dot(inv_cov).dot(centered)
            d = wvmodel.vector_size
            nu = self.priors.nu + Nk - d + 1.
            log_prob = gammaln((nu + d) / 2.) - \
                       (gammaln(nu / 2.) + d/2. * (log(nu) + log(pi)) +0.5 * cov_det[1] + ((nu + d) / 2.) * log((1. + LLcomp )/ nu))
            return log_prob


    def recalculate_topic_params(self, topic_id, topic_counts):

        topic_count = np.sum(topic_counts[:, topic_id], axis=0)  # N_k
        kappa_k = self.priors.kappa + topic_count  # K_k
        nu_k = self.priors.nu + topic_count  # V_k
        scaled_topic_mean_K, scaled_topic_cov_K  = self.get_scaled_topic_MC(topic_id, topic_counts)  # V-Bar_k and C_k
        vk_mu = scaled_topic_mean_K - self.priors.mu # V-bar_k - Mu

        psi_k = self.priors.psi + scaled_topic_cov_K + ((self.priors.kappa * topic_count) / kappa_k) * (vk_mu.T.dot(vk_mu))  # Psi_k

        topic_mean = (self.priors.kappa * self.priors.mu + topic_count * scaled_topic_mean_K) / kappa_k  # Mu_k
        topic_cov = psi_k / (nu_k - self.word_vec_size + 1.)  # Sigma_k

        self.topic_params[topic_id]["Topic Count"] = topic_count
        self.topic_params[topic_id]["Topic Kappa"] = kappa_k
        self.topic_params[topic_id]["Topic Nu"] = nu_k
        self.topic_params[topic_id]["Topic Mean"], self.topic_params[topic_id]["Topic Covariance"] = topic_mean, topic_cov
        self.topic_params[topic_id]["Inverse Covariance"] = np.linalg.inv(topic_cov)
        self.topic_params[topic_id]["Covariance Determinant"] = np.linalg.slogdet(topic_cov) # regular determinant calculator breaks with super small numbers

        return topic_mean, topic_cov

    def get_scaled_topic_MC(self, topic_id, topic_count):
        # get words assigned to topic_id

        topic_vecs = []
        for docID, doc in self.corpus.iteritems():
            for word in doc:
                # print word, topic_id, self.word_topics[word}, "word, topic id, wordtopic assignment"
                # print self.word_topics[word]
                if self.word_topics[word] == topic_id:
                    topic_vecs.append(self.word_vecs[word])

        # if len(topic_vecs) < 2: # DEBUGGING.. turned out topic-assgns were whack
        #     print 'I DONE BROKE \n HE HAW'
        #     print topic_id, "TOPIC ID"
        #     print self.word_topics
        #     print len(topic_vecs), 'LEN WORD VEC'
        #     self.word_topics = {word: random.choice(range(self.numtopics)) for word in self.vocab}
        #     self.get_scaled_topic_MC(topic_id, topic_count)

        topic_vecs = np.vstack(topic_vecs)
        mean = np.sum(topic_vecs, axis=0) / (np.sum(topic_count[:, topic_id], axis=0))

        mean_centered = topic_vecs - mean
        cov = mean_centered.T.dot(mean_centered)
        return mean, cov


    def update_document_topic_counts(self, word, topicID, operation):
        """
        :param word: a word to recalculate document x topic count table
        :param topicID: topic columm to adjust
        :param operation: '-' for subracting contribution | '+' for adding contribution
        :return: a new document-topic table (copy)
        Method only affects a copy of the ground truth
        Counts how many times each topic is assigned to a word in a document.  is a Doc X Topic array/matrix
        """
        # topicID = self.word_topics[word]
        topic_counts = np.copy(self.doc_topic_CT)
        if operation == "-":
            for docID, doc in self.corpus.iteritems():
                topic_counts[docID, topicID] - float(doc.count(word))

        if operation == "+":
            for docID, doc in self.corpus.iteritems():
                topic_counts[docID, topicID] + float(doc.count(word))
        return topic_counts

    def extract_topics_new_doc(self, doc, wv_model):
        """
        :type wv_model: gensim.models.word2vec.Word2Vec
        :param doc: Document to extrac topics from.  should be one string
        :param wv_model: a loaded word2vec model with same dimensionality as training one.  Use GenSim Word2Vec
        :return: List of tuples (word, topic)

        Method removes words in doc that are not in the Word2Vec corpus, and extracts word-topic assignments for each
        word by drawing densities from the multivariate student-T distribution.  Uses MLE method.
        """

        # clean out words in doc that are not in the word-vec space
        filtered_doc = []
        nkd = defaultdict(float)
        for word in doc.split():
            try:
                wv_model[word]
                filtered_doc.append(word)
                nkd[self.word_topics[word]] += 1
            except KeyError: continue
        print "{} words removed from doc".format(len(filtered_doc) - len(doc.split()))
        word_topics = []

        for word in filtered_doc:
            posterior = []
            for k in range(self.numtopics):
                print nkd[k]
                prob = self.draw_new_wt_assgns(word, k, wvmodel=wv_model, new_doc=True) * log(self.alpha + nkd[k])
                print "probablity of {0} for word {1} assigned to topic {2}".format(prob, word, k)
                posterior.append(prob)
            posterior /= np.sum(posterior)

            word_topics.append((word, np.argmax(posterior)))
        return word_topics


if __name__ == "__main__":
    corpus = ["apple orange mango melon", "dog cat bird rat", "pineapple kiwi grape strawberry",
              "rabbit mouse horse goat", "coconut guava blueberry blackberry", "raptor hawk shark bear",
              "lemon lime fruit pear"]

    wordvec_fileapth = "/Users/michael/Documents/Gaussian_LDA-master/data/glove.wiki/glove.6B.50d.txt"
    g = Gauss_LDA(2, corpus, wordvec_fileapth)
    g.fit(100)