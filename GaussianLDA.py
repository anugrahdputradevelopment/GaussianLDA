from __future__ import division

import random
from collections import defaultdict, Counter

import numpy as np
from gensim.models import Word2Vec
from numpy import log, pi
from scipy.special import gammaln

__author__ = "Michael Mansour, Mike Rinehart, Jared Thompson"


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
        self.psi = np.identity(word_vecs.shape[1]) * 3.  # changed this to identity matrix as in paper. No intuition here
        self.mu = np.mean(word_vecs, axis=0)

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

    def fit(self, iterations=1, init=True):
        if init:
            self.init()
            init = False

        print "Starting fit"
        for i in xrange(iterations):
            self.sample()
            print "{0} iterations complete".format(i)

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
        self.doc_topic_CT = np.zeros((len(self.corpus.keys()), self.numtopics))

        centroids, _ = self.smart_centroids()

        # Prior mean
        mu_0 = np.zeros(self.word_vec_size)
        count = 0
        for docID in self.corpus.keys():  # hard setting word-topic assignments as per cluster membership to help model along
            for i, word in enumerate(self.corpus[docID]['words']):
                self.corpus[docID]['topics'][i] = self.word_topics[word]  # word_topics from KMeans cluster membership
                mu_0 += self.word_vecs[word]
                count += 1
        self.priors.mu = mu_0 / float(count)

        # Prior co-variance
        self.priors.psi = .01 * np.identity(self.word_vec_size)

        # Sample means
        for k in range(self.numtopics):
            self.topic_params[k]["Topic Sum"] = np.zeros(self.word_vec_size)
            self.topic_params[k]["Topic Mean"] = centroids[k]
            self.topic_params[k]["Topic Cov"] = np.zeros((self.word_vec_size, self.word_vec_size))

        # Sample co-variances and document-topic counts
        co_variances = [np.zeros((self.word_vec_size, self.word_vec_size)) for _ in range(self.numtopics)]
        for docID in self.corpus.keys():
            for topic, word in zip(self.corpus[docID]['topics'], self.corpus[docID]['words']):
                topic = int(topic)
                wv = self.word_vecs[word]
                sample_mu = self.topic_params[topic]["Topic Mean"]
                self.doc_topic_CT[docID, topic] += 1.  # Init Document-Topic counts
                self.topic_params[topic]['Topic Sum'] += wv  # Sum of topic vectors
                co_variances[topic] += np.outer(wv - sample_mu, wv - sample_mu) #+ self.priors.psi

        #   Normalize the sample co-variance
        for k in range(self.numtopics):
            self.topic_params[k]["Topic Cov"] = (co_variances[k] / (np.sum(self.doc_topic_CT[:, k]) - 1.)) + self.priors.psi

        kappa = self.priors.kappa
        nu = self.priors.nu
        d = self.word_vec_size
        scaleT = (kappa + 1.) / (kappa * (nu - d + 1.))  # Needed to convert L => covariance

        for k in range(self.numtopics):  # Init parameters for topic distributions
            Nk = np.sum(self.doc_topic_CT[:, k], axis=0)

            self.topic_params[k]["Topic Count"] = Nk
            self.topic_params[k]["Topic Kappa"] = self.priors.kappa + Nk

            # 2 * sum_m_i(log(L_i,i)) + log(scaleT)
            # self.topic_params[k]["Chol Det"] = np.sum(np.log(np.diag(self.topic_params[k]["Lower Triangle"]))) * 2 \
            # + np.log(scaleT)
            self.topic_params[k]['Topic Det'] = np.linalg.det(self.topic_params[k]["Topic Cov"])
            self.topic_params[k]['Topic Inv'] = np.linalg.inv(self.topic_params[k]['Topic Cov'])
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
                wv = self.word_vecs[word]
                mu = self.topic_params[topic]["Topic Mean"]

                centered = wv - mu # Get rank-1 matrix from point
                centered *= np.sqrt((kappa_k + 1.) / kappa_k)  # Scale for recursive downdate
                self.topic_params[topic]["Topic Cov"] -= np.outer(centered, centered)

            # Correct the mean for the removed point
            sample_mean_K = self.topic_sample_mean(topic, topic_count)  # V-Bar_k
            # topic_sum = self.topic_params[topic]["Topic Sum"]
            topic_mean = ((self.priors.kappa * self.priors.mu) + (topic_count * sample_mean_K)) / kappa_k  # Mu_k


        else:  # operation == "+":  # Add data point contribution to the topic distribution
            # Correct the mean for the added point

            sample_mean_K = self.topic_sample_mean(topic, topic_count)  # V-Bar_k
            topic_mean = ((self.priors.kappa * self.priors.mu) + (topic_count * sample_mean_K)) / kappa_k  # Mu_k

            # Original equation is:
            #    \Sigma \leftarrow \Sigma + (k_0 + N + 1)/(k_0 + N)(X_{n} - \mu_{n-1})(X_{n} - \mu_{n-1})^T
            if UPDATE_DISTS:
                centered = (self.word_vecs[word] - topic_mean)
                centered *= np.sqrt(kappa_k / (kappa_k - 1.))  # Scale for recursive update
                self.topic_params[topic]["Topic Cov"] += np.outer(centered, centered)


        L = self.topic_params[topic]["Lower Triangle"]
        self.topic_params[topic]["Cov Det"] = np.linalg.det(self.topic_params[topic]["Topic Cov"]) # 2 * sum_m_i(log(L_i,i))
        self.topic_params[topic]["Topic Inv"] = np.linalg.inv(self.topic_params[topic]["Topic Cov"])
        self.topic_params[topic]["Topic Count"] = topic_count
        self.topic_params[topic]["Topic Kappa"] = kappa_k
        self.topic_params[topic]["Topic Nu"] = nu_k
        if UPDATE_DISTS:
            self.topic_params[topic]["Topic Mean"] = topic_mean


# ======================================================================================================================

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

        # Covariance Inverse
        cov_inv = self.topic_params[topic]["Topic Inv"]
        cov_inv = centered.T.dot(cov_inv).dot(centered)
        cov_inv *= scaleT

        cov_det = self.topic_params[topic]["Topic Det"]

        # Log Multivariate T - PDF
        return gammaln((nu + d) / 2.) - \
            (gammaln(nu / 2.) + (d / 2.) * (log(nu) + log(pi))
            + (0.5 * cov_det) + ((nu + d) / 2.) * log(1. + cov_inv/nu))



if __name__ == "__main__":
    corpus = ["apple orange mango melon", "dog cat bird rat", "pineapple kiwi grape strawberry",
              "rabbit mouse horse goat", "coconut guava blueberry blackberry", "raptor hawk shark bear",
              "lemon lime fruit pear"]

    f = '/Users/michael/Documents/GaussianLDA/clean20news.txt'
    with open(f, 'r') as fi:
        docs = fi.read().splitlines() #  These are all cleaned out
        fi.close()

    wordvec_fileapth = "/Users/michael/Documents/Gaussian_LDA-master/data/glove.wiki/glove.6B.50d.txt"
    g = Gauss_LDA(2, docs, wordvec_fileapth)
    g.fit(15)
# ======================================================================================================================
#
#     def sample(self):
#         """
#         Collapsed Gibbs Sampler derived from Steyver's method, adapted for continuous word-vectors
#         :return: None.  Readjusts topic distribution parameters and topic-counts
#         """
#
#         for docID, doc in self.corpus.iteritems():
#             for word in doc:
#                 # self.doc_topic_CT[docID, topic_id] - float(doc.count(word)) #expirmenting with -= vs -
#                 # self.update_document_topic_counts(word, "-")
#                 # self.recalculate_topic_params(self.word_topics[word])
#
#                 posterior = []
#                 max = 0
#                 for k in range(self.numtopics):  # start getting the pdf's for each word-topic assignment
#                     topic_counts = self.update_document_topic_counts(word, k, "-")  # subtracting info about current word-topic assignment from doc-topic count table
#                     self.recalculate_topic_params(k, topic_counts)
#
#                     log_pdf = self.draw_new_wt_assgns(word, k)
#                     Nkd = topic_counts[docID, k] # Count of topic in doc
#                     # print "Nkd = {}".format(Nkd)
#                     log_posterior = log(Nkd + self.alpha) * log_pdf  # actual collapsed sampler from R. Das Paper, except in log form
#                     posterior.append(log_posterior)  # doing this for some normalization scheme
#                     if log_posterior > max: max = log_posterior  # copied from R. Das code, not actually used here
#
#                 posterior.append(0.)  # just a little hitch in function. It wants a zero at the end, otherwise it may say sum(pvals) != 1.0.
#                 post_sum = np.sum(posterior)
#                 normalized_post = posterior / post_sum
#                 new_word_topic = np.random.multinomial(1, pvals=normalized_post)
#                 # print 'multinomial with reg-norm', new_word_topic
#
#                 self.word_topics[word] = np.argmax(new_word_topic)
#                 self.doc_topic_CT = self.update_document_topic_counts(word, self.word_topics[word], "+")
#                 self.recalculate_topic_params(self.word_topics[word], self.doc_topic_CT)
#             if docID % 1 == 0:
#                 for k in range(self.numtopics):
#                     print self.wvmodel.most_similar(positive=[self.topic_params[k]["Topic Mean"]])
#
#         return None

# ======================================================================================================================
#
#     def draw_new_wt_assgns(self, word, topic_id, new_doc=False, wvmodel=None):
#         """
#         Log of the probablity density function for the Student-T Distribution
#
#         Provides a PDF for a word (really a word-vector) in a given topic distribution.
#
#         :param word: string of the word to find probabilty of word-topic assignment
#         :param topic_id: Interger, a topic id to reference a topic distribution and its params
#         :param new_doc: False (default), optional.  True if predicting topics from unseen document/not currently training
#         :param wvmodel: None by default.  If predicting topics from an unseen document, requires a loaded word2vec model
#         from GenSim
#         :type wvmodel: gensim.models.word2vec.Word2Vec
#         :return: log of PDF from t-distribution for a given word.  Type: Float
#         """
#
#         if not new_doc:
#             # Getting params for calculating PDF of T-Dist for a word
#             inv_cov = self.topic_params[topic_id]["Inverse Covariance"]
#             cov_det = self.topic_params[topic_id]["Covariance Determinant"]  #cov_det is already logged
#             Nk = self.topic_params[topic_id]["Topic Count"]
#             mean = self.topic_params[topic_id]["Topic Mean"]
#
#             # Precalculating some terms (V_di - Mu)^T * Cov^-1 * (V_di - Mu)
#             # centered = self.word_vecs[word] - self.priors.mu  #note that this should be really the topic mean, not the prior mean!
#             centered = self.word_vecs[word] - mean
#             LLcomp = centered.T.dot(inv_cov).dot(centered)  # for some topics, this outputs a negative value // CHANGED TO BELOW
#             # SHOULD THSI BE CENTERD.DOT(INV_COV).DOT(CENTERED.T))????
#             d = self.word_vec_size   # dimensionality of word vector
#             nu = self.priors.nu + Nk - d + 1.
#
#             # Log PDF of multivariate student-T distribution
#             log_prob = gammaln(nu + d / 2.) - \
#                        (gammaln(nu / 2.) + d/2. * (log(nu) + log(pi)) +0.5 * cov_det[1] + ((nu + d) / 2.) * log((1. + LLcomp ) / nu))
#
#             return log_prob
#
#         if new_doc:
#             inv_cov = self.topic_params[topic_id]["Inverse Covariance"]
#             cov_det = self.topic_params[topic_id]["Covariance Determinant"]
#             Nk = self.topic_params[topic_id]["Topic Count"]
#             centered = wvmodel[word] - self.priors.mu
#
#             LLcomp = centered.T.dot(inv_cov).dot(centered)
#             d = wvmodel.vector_size
#             nu = self.priors.nu + Nk - d + 1.
#             log_prob = gammaln((nu + d) / 2.) - \
#                        (gammaln(nu / 2.) + d/2. * (log(nu) + log(pi)) +0.5 * cov_det[1] + ((nu + d) / 2.) * log((1. + LLcomp )/ nu))
#             return log_prob
#
# # ======================================================================================================================
#
#     def recalculate_topic_params(self, topic_id, topic_counts):
#         """
#
#         :param topic_id:
#         :param topic_counts:
#         :return:
#         """
#         topic_count = np.sum(topic_counts[:, topic_id], axis=0)  # N_k
#         kappa_k = self.priors.kappa + topic_count  # K_k
#         nu_k = self.priors.nu + topic_count  # V_k
#         scaled_topic_mean_K, scaled_topic_cov_K  = self.get_scaled_topic_MC(topic_id, topic_counts)  # V-Bar_k and C_k
#         vk_mu = scaled_topic_mean_K - self.priors.mu # V-bar_k - Mu
#
#         psi_k = self.priors.psi + scaled_topic_cov_K + ((self.priors.kappa * topic_count) / kappa_k) * (vk_mu.T.dot(vk_mu))  # Psi_k
#
#         topic_mean = ((self.priors.kappa * self.priors.mu) + (topic_count * scaled_topic_mean_K)) / kappa_k  # Mu_k
#         topic_cov = psi_k / (nu_k - self.word_vec_size + 1.)  # Sigma_k
#
#         self.topic_params[topic_id]["Topic Count"] = topic_count
#         self.topic_params[topic_id]["Topic Kappa"] = kappa_k
#         self.topic_params[topic_id]["Topic Nu"] = nu_k
#         self.topic_params[topic_id]["Topic Mean"], self.topic_params[topic_id]["Topic Covariance"] = topic_mean, topic_cov
#         self.topic_params[topic_id]["Inverse Covariance"] = np.linalg.inv(topic_cov)
#         self.topic_params[topic_id]["Covariance Determinant"] = np.linalg.slogdet(topic_cov) # regular determinant calculator breaks with super small numbers
#
#         return topic_mean, topic_cov
#
# # ======================================================================================================================
#
#     def get_scaled_topic_MC(self, topic_id, topic_count):
#         """
#         For a given topic, method calculates scaled topic Mean and Covariance (V-bar_k and C_k in R. Das Paper)
#
#         \sum_d \sum_z=i (V_di) / N_k
#         ^^ =
#         wordvec_sum = array[zero] > shape(word-vec dimensionality)
#         for each doc:
#                 for each word that has topic assignment i:
#                     wordvec_sum + word
#         wordvec_sum / count of topic
#         N_k = count of topic occurences across all documents
#
#         :param topic_id: The topic ID, integer
#         :param topic_count: A copy of the document-topic counts table, numpy array
#         :return: mean and covariance matrix.  Mean will be of shape (1 X word-vector dimension).
#         Covariance will be matrix of size (word-vector dim X word-vector dim)
#         """
#         topic_vecs = []  # creating a matrix of word-vecs assigned to topic_id
#         for docID, doc in self.corpus.iteritems():
#             for word in doc:
#                 if self.word_topics[word] == topic_id:
#                     topic_vecs.append(self.word_vecs[word])
#
#         topic_vecs = np.vstack(topic_vecs)
#         mean = np.sum(topic_vecs, axis=0) / (np.sum(topic_count[:, topic_id], axis=0))
#
#         mean_centered = topic_vecs - mean
#         cov = mean_centered.T.dot(mean_centered)  # (V_dk - Mu)^T(V_dk - Mu)
#         return mean, cov
#
# # ======================================================================================================================
#
#     def update_document_topic_counts(self, word, topicID, operation):
#         """
#         :param word: a word to recalculate document x topic count table
#         :param topicID: topic columm to adjust
#         :param operation: '-' for subracting contribution | '+' for adding contribution
#         :return: a new document-topic table (copy)
#         Method only affects a copy of the ground truth
#         Counts how many times each topic is assigned to a word in a document.  is a (Doc X Topic) array/matrix
#         """
#         # topicID = self.word_topics[word]
#         topic_counts = np.copy(self.doc_topic_CT)
#         if operation == "-":
#             for docID, doc in self.corpus.iteritems():
#                 topic_counts[docID, topicID] - float(doc.count(word))
#
#         if operation == "+":
#             for docID, doc in self.corpus.iteritems():
#                 topic_counts[docID, topicID] + float(doc.count(word))
#         return topic_counts
#
# # ======================================================================================================================
#
#     def extract_topics_new_doc(self, doc, wv_model):
#         """
#         :type wv_model: gensim.models.word2vec.Word2Vec
#         :param doc: Document to extrac topics from.  should be one string
#         :param wv_model: a loaded word2vec model with same dimensionality as training one.  Use GenSim Word2Vec
#         :return: List of tuples (word, topic)
#
#         Method removes words in doc that are not in the Word2Vec corpus, and extracts word-topic assignments for each
#         word by drawing densities from the multivariate student-T distribution.  Uses MLE method.
#         """
#         assert wv_model.vector_size == self.word_vec_size, "word-vector dimensionality does not match trained topic" \
#                                                            "distribution dimensions({0})".format(self.word_vec_size)
#         filtered_doc = []
#         nkd = defaultdict(float)
#         for word in doc.split():
#             try:
#                 wv_model[word]
#                 filtered_doc.append(word) # Remove words from doc that are not in word-vec model
#                 nkd[self.word_topics[word]] += 1.
#             except KeyError: continue
#         print "{} words removed from doc".format(len(filtered_doc) - len(doc.split()))
#         word_topics = []
#         c = Counter(self.word_topics.values())
#         for word in filtered_doc:
#             posterior = []
#             for k in range(self.numtopics):
#                 # print nkd[k]
#                 prob = self.draw_new_wt_assgns(word, k, wvmodel=wv_model, new_doc=True) * log(self.alpha + c[k])
#                 print "probablity of {0} for word {1} assigned to topic {2}".format(prob, word, k)
#                 posterior.append(prob)
#             posterior /= np.sum(posterior)
#
#             word_topics.append((word, np.argmax(posterior)))
#         return word_topics

# ======================================================================================================================

