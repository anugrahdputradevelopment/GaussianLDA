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

__author__ = "Michael Mansour"


class Wishart(object):

    def __init__(self, word_vecs):
        self.nu = None
        self.kappa = None
        self.psi = None
        self.mu = None
        self.set_params(word_vecs)

    def set_params(self, word_vecs):
        word_vecs = np.vstack(word_vecs.values())  # turn dict of word vectors into a matrix
        self.nu = word_vecs.shape[1]  # dimensionality of word-vectors
        self.kappa = 0.1
        # self.psi = word_vecs.T.dot(word_vecs)#, axis=0)  # sum of squres -- from Murphy(2012)
        self.psi = np.identity(word_vecs.shape[1]) * 3.  # changed this to identity matrix as in paper. No intuition here
        self.mu = np.mean(word_vecs, axis=0)

class Gauss_LDA(object):

    def __init__(self, num_topics, corpus, word_vector_filepath=None, word_vector_model=None, run_name=str(1)):
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
        self.alpha = 20. / self.numtopics
        self.solver = cholesky.Helper()
        self.wvmodel = word_vector_model
        self.doc_word_counts = {}
        self.run_name = run_name

    def process_corpus(self, documents):
        """
        Tokenizes documents into dict of lists of tokens
        :param documents: expects list of strings
        :return: dict{document ID: list of tokens
        """

        temp_corpus = {}
        for index, doc in enumerate(documents):
            words = doc.split()
            temp_corpus[index] = words
            for word in words:
                self.vocab.add(word)
        self.corpus = temp_corpus
        print "Done processing corpus with {} documents".format(len(documents))

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
            vectors = gensim.models.Word2Vec.load_word2vec_format(fname=filepath, binary=False)
            useable_vocab = 0
            unusable_vocab = 0
            self.word_vec_size = vectors.vector_size

            for word in self.vocab:
                try:
                    vectors[word]
                    self.word_vecs[word] = vectors[word]
                    useable_vocab += 1
                except KeyError: unusable_vocab += 1

            print "There are {0} words that could be convereted to word vectors in your corpus \n" \
                  "There are {1} words that could NOT be converted to word vectors".format(useable_vocab, unusable_vocab)
            print "Word-vectors for the corpus are created"
            self.wvmodel = vectors
        else:

            useable_vocab = 0
            unusable_vocab = 0
            self.word_vec_size = self.wvmodel.vector_size

            for word in self.vocab:
                try:
                    # noinspection PyStatementEffect
                    self.wvmodel[word]
                    self.word_vecs[word] = self.wvmodel[word]
                    useable_vocab += 1
                except KeyError: unusable_vocab += 1

            print "There are {0} words that could be convereted to word vectors in your corpus \n" \
                  "There are {1} words that could NOT be converted to word vectors".format(useable_vocab, unusable_vocab)

    def clean_docs(self):
        print "cleaning out docs of words not in your Word2Vec model"
        approved_words = set(self.word_vecs.vocab.keys())
        for idx, doc in self.corpus.iteritems():
            self.corpus[idx] = [word for word in doc if word in approved_words]
        print "Done cleaning out docs of bad words"

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
                    results_file =\
                        "/Users/michael/Documents/GaussianLDA/output/{3}iter{0}topic{1}{2}.txt".format(i, k, param, self.run_name)
                    open(results_file, 'w')
                    np.savetxt(results_file, self.topic_params[k][param])
        self.display_results()
    def init(self):

        self.process_corpus(self.corpus)
        self.process_wordvectors(self.wordvecFP)
        # self.clean_docs()
        self.priors = Wishart(self.word_vecs)  # set wishhart priors
        self.doc_topic_CT = np.zeros((len(self.corpus.keys()), self.numtopics))

        self.word_topics = {word: random.choice(range(self.numtopics)) for word in self.vocab}
        self.word_count = defaultdict(float)
        # get Doc-Topic Counts
        for docID, doc in self.corpus.iteritems():
            doc_word_count_temp = defaultdict(float)
            for word in doc:
                topicID = self.word_topics[word]
                doc_word_count_temp[word] += 1
                self.word_count[word] += 1
                self.doc_topic_CT[docID, topicID] += 1 # TODO: SHOULD THIS BE + INSTEAD OF +=???
                # print doc_word_count_temp
            self.doc_word_counts[docID] = doc_word_count_temp

        for k in range(self.numtopics):
            test = np.zeros(self.word_vec_size)
            self.topic_params[k]["Lower Triangle"] = linalg.cholesky(self.priors.psi, lower=True,
                                                                            check_finite=True)
            selected_words = set([word for word, topic in self.word_topics.iteritems() if topic == k])
            self.topic_params[k]["Topic Sum"] = np.zeros(self.word_vec_size)

            for word in selected_words:
                # print self.word_count[word]
                self.topic_params[k]["Topic Sum"] += self.word_vecs[word] * float(self.word_count[word])
                test += self.word_vecs[word]
            print "topic{0}".format(k), self.wvmodel.most_similar(positive=[test / len(selected_words)])
        for k in range(self.numtopics):  # Init parameters for topic distributions
            self.recalculate_topic_params(k, "+", init=True)


        print "Intialization complete"

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

    def sample(self):
        """
        Collapsed Gibbs Sampler derived from Steyver's method, adapted for continuous word-vectors
        :return: None.  Readjusts topic distribution parameters and topic-counts
        """

        for docID, doc in self.corpus.iteritems():
            for word in doc:
                self.update_document_topic_counts(word, self.word_topics[word], "-")
                self.recalculate_topic_params(self.word_topics[word], word, "-", init=False)

                posterior = []
                for k in range(self.numtopics):  # start getting the pdf's for each word-topic assignment
                    log_pdf = self.draw_new_wt_assgns(word, k)
                    print log_pdf
                    Nkd = self.doc_topic_CT[docID, k] # Count of topic in doc
                    print self.doc_topic_CT[docID, :]
                    log_posterior = log(Nkd + self.alpha) + log_pdf  # actual collapsed sampler from R. Das Paper, except in log form
                    posterior.append(log_posterior)  # doing this for some normalization scheme
                print posterior, np.max(posterior)
                posterior -= np.max(posterior)
                print posterior
                postb = np.exp(posterior)
                print postb
                postc = np.cumsum(postb) / np.sum(postb)
                postb /= np.sum(postb)
                print postc
                print len(postc)
                # posterior.append(0.)  # just a little hitch in function. It wants a zero at the end, otherwise it may say sum(pvals) != 1.0.
                # normalized_post = np.exp(posterior) / np.sum(np.exp(posterior))
                # print normalized_post
                new_word_topic = self.bin_search(np.cumsum(postc), np.random.uniform(0, 1), 0, len(postc)-1)
                # new_word_topic = np.random.multinomial(1, pvals=normalized_post)
                print new_word_topic
                self.word_topics[word] = np.argmax(new_word_topic) #p.argmax(new_word_topic)

                self.update_document_topic_counts(word, self.word_topics[word], "+")
                self.recalculate_topic_params(self.word_topics[word], word, "+", init=False)
                break
            if docID % 20 == 0:
                print "{0} docs sampled".format(docID)
                self.display_results()
        return None


    # noinspection PyStatementEffect
    def update_document_topic_counts(self, word, topicID, operation):
        """
        :param word: a word to recalculate document x topic count table
        :param topicID: topic columm to adjust
        :param operation: '-' for subracting contribution | '+' for adding contribution
        :return: a new document-topic table (copy)
        Method only affects a copy of the ground truth
        Counts how many times each topic is assigned to a word in a document.  is a (Doc X Topic) array/matrix
        """
        if operation == "-":
            for docID, doc in self.corpus.iteritems():
                self.doc_topic_CT[docID, topicID] -= self.doc_word_counts[docID][word]

        if operation == "+":
            for docID, doc in self.corpus.iteritems():
                self.doc_topic_CT[docID, topicID] += self.doc_word_counts[docID][word]


    def recalculate_topic_params(self, topic_id, word=None, oper=None, init=False):
        """

        :param topic_id: index for topic
        :param topic_counts: a copy of the doc-topic count table
        :return: None - sets internal class variables
        """

        if not init:
            topic_count = np.sum(self.doc_topic_CT[:, topic_id], axis=0)# N_k
            if topic_count == 0.0: print self.doc_topic_CT, topic_id
            kappa_k = self.priors.kappa + topic_count
            scaled_topic_mean_K = self.get_scaled_topic_MC(topic_id, topic_count, word, oper)

            centered = (self.word_vecs[word] - self.topic_params[topic_id]["Topic Mean"])# * np.sqrt((kappa_k+1)/kappa_k)
            L = self.topic_params[topic_id]["Lower Triangle"]
            # gettting the Cholesky fast determinant of the lower triagnle (for the det of cov)
            if oper == "-":
                self.topic_params[topic_id]["Lower Triangle"] = self.solver.chol_downdate(L, centered)
            if oper == "+":
                self.topic_params[topic_id]["Lower Triangle"] = self.solver.chol_update(L, centered)

            # print topic_id
            # print self.doc_topic_CT
            # print self.doc_topic_CT[:, topic_id]
            # print "topic count", topic_count
            kappa_k = self.priors.kappa + topic_count  # K_k
            nu_k = self.priors.nu + topic_count  # V_k
            scaled_topic_mean_K = self.get_scaled_topic_MC(topic_id, topic_count, word, oper)  # V-Bar_k
            # psi_k = self.priors.psi + scaled_topic_cov_K + ((self.priors.kappa * topic_count) / kappa_k) * (vk_mu.T.dot(vk_mu))  # Psi_k



            topic_mean = ((self.priors.kappa * self.priors.mu) + (topic_count * scaled_topic_mean_K)) / kappa_k  # Mu_k
            if np.isinf(topic_mean).any() or np.isinf(topic_mean).any():
                print 'right loop', topic_mean
            chol_det = np.sum(np.log(np.diag(self.topic_params[topic_id]["Lower Triangle"]))) #  2 * sum_m_i(log(L_i,i))

            self.topic_params[topic_id]["Chol Det"] = chol_det * 2
            self.topic_params[topic_id]["Topic Count"] = topic_count
            self.topic_params[topic_id]["Topic Kappa"] = kappa_k
            self.topic_params[topic_id]["Topic Nu"] = nu_k
            self.topic_params[topic_id]["Topic Mean"]= topic_mean

        if init:
            topic_count = np.sum(self.doc_topic_CT[:, topic_id], axis=0)  # N_k
            # print "topic count", topic_count
            kappa_k = self.priors.kappa + topic_count  # K_k
            nu_k = self.priors.nu + topic_count  # V_k
            scaled_topic_mean_K = self.topic_params[topic_id]["Topic Sum"] / float(topic_count) # V-Bar_k
            L = self.topic_params[topic_id]["Lower Triangle"]

            topic_mean = ((self.priors.kappa * self.priors.mu) + (topic_count * scaled_topic_mean_K)) / kappa_k  # Mu_k
            chol_det = np.sum(np.log(np.diag(self.topic_params[topic_id]["Lower Triangle"]))) #  2 * sum_m_i(log(L_i,i))
            if np.isinf(topic_mean).any() or np.isinf(topic_mean).any():
                print 'wrong loop', topic_mean
            self.topic_params[topic_id]["Chol Det"] = chol_det * 2
            self.topic_params[topic_id]["Topic Count"] = topic_count
            self.topic_params[topic_id]["Topic Kappa"] = kappa_k
            self.topic_params[topic_id]["Topic Nu"] = nu_k
            self.topic_params[topic_id]["Topic Mean"]= topic_mean

    def get_scaled_topic_MC(self, topic_id, topic_count, word, oper):
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
        if oper == "-":
            self.topic_params[topic_id]["Topic Sum"] -= self.word_vecs[word] #* self.word_count[word]
            return self.topic_params[topic_id]["Topic Sum"] / topic_count
        if oper == "+":
            self.topic_params[topic_id]["Topic Sum"] += self.word_vecs[word] #* self.word_count[word]
            return self.topic_params[topic_id]["Topic Sum"] / topic_count

    # def multivariate_T_pdf(self):
    #     Num = gamma(1. * (d+df)/2.)
    #     Denom = ( gamma(1.*df/2.) * pow(df*pi,1.*d/2) * pow(np.linalg.det(Sigma),1./2) * pow(1 + (1./df)*np.dot(np.dot((x - mu),np.linalg.inv(Sigma)), (x - mu)),1.* (d+df)/2))
    #     d = 1. * Num / Denom
    #     return d

    def draw_new_wt_assgns(self, word, topic_id, new_doc=False, wvmodel=None):
        """
        Log of the probablity density function for the Student-T Distribution

        Provides a PDF for a word (really a word-vector) in a given topic distribution.

        :param word: string of the word to find probabilty of word-topic assignment
        :param topic_id: Interger, a topic id to reference a topic distribution and its params
        :param new_doc: False (default), optional.  True if predicting topics from unseen document/not currently training
        :param wvmodel: None by default.  If predicting topics from an unseen document, requires a loaded word2vec model
        from GenSim
        :type wvmodel: gensim.models.word2vec.Word2Vec
        :return: log of PDF from t-distribution for a given word.  Type: Float
        """

        if not new_doc:
            # Getting params for calculating PDF of T-Dist for a word
            cov_det = self.topic_params[topic_id]["Chol Det"]
            Nk = self.topic_params[topic_id]["Topic Count"]
            # Precalculating some terms (V_di - Mu)
            centered = np.copy(self.word_vecs[word] - self.topic_params[topic_id]["Topic Mean"])
            # (L^-1b)^T(L^-1b) _
            if np.isnan(centered).any() or np.isinf(centered).any():
                print centered
                print topic_id
                print Nk
                print word
                print self.word_vecs[word]
                print self.topic_params[topic_id]["Topic Mean"]
            linalg.cho_solve((self.topic_params[topic_id]["Lower Triangle"], True), centered, overwrite_b=True,
                check_finite=True)
            LLcomp = centered.T.dot(centered)
            # SHOULD THSI BE CENTERD.DOT(INV_COV).DOT(CENTERED.T))????
            d = self.word_vec_size   # dimensionality of word vector
            nu = self.priors.nu + Nk - d + 1.

            # Log PDF of multivariate student-T distribution
            log_prob = gammaln(nu + d / 2.) - \
                       (gammaln(nu / 2.) + d/2. * (log(nu) + log(pi)) +0.5 * cov_det + ((nu + d) / 2.) * log((1. + LLcomp ) / nu))

            return log_prob

        if new_doc:
            cov_det = self.topic_params[topic_id]["Chol Det"]
            Nk = self.topic_params[topic_id]["Topic Count"]
            centered = self.word_vecs[word] - self.topic_params[topic_id]["Topic Mean"]

            cholesky_solution = linalg.cho_solve((self.topic_params[topic_id]["Lower Triangle"], True), centered)
            LLcomp = cholesky_solution.T.dot(cholesky_solution) # TODO: update to be like loop above
            d = wvmodel.vector_size
            nu = self.priors.nu + Nk - d + 1.
            log_prob = gammaln((nu + d) / 2.) - \
                       (gammaln(nu / 2.) + d/2. * (log(nu) + log(pi)) +0.5 * np.log(cov_det) + ((nu + d) / 2.) * log((1. + LLcomp )/ nu))
            return log_prob


    # def recalculate_topic_params(self, topic_id, topic_counts, word, oper, init=False):
    #     """
    #
    #     :param topic_id: index for topic
    #     :param topic_counts: a copy of the doc-topic count table
    #     :return: None - sets internal class variables
    #     """
    #
    #     topic_count = np.sum(topic_counts[:, topic_id], axis=0)  # N_k
    #     # print "topic count", topic_count
    #     kappa_k = self.priors.kappa + topic_count  # K_k
    #     nu_k = self.priors.nu + topic_count  # V_k
    #     scaled_topic_mean_K = self.get_scaled_topic_MC(topic_id, topic_count, word, oper)  # V-Bar_k
    #     # psi_k = self.priors.psi + scaled_topic_cov_K + ((self.priors.kappa * topic_count) / kappa_k) * (vk_mu.T.dot(vk_mu))  # Psi_k
    #     L = self.topic_params[topic_id]["Lower Triangle"]
    #
    #
    #     topic_mean = ((self.priors.kappa * self.priors.mu) + (topic_count * scaled_topic_mean_K)) / kappa_k  # Mu_k
    #     centered = self.word_vecs[word] - topic_mean
    #
    #     # gettting the Cholesky fast determinant of the lower triagnle (for the det of cov)
    #     if oper == "-":
    #         self.topic_params[topic_id]["Lower Triangle"] = self.solver.chol_downdate(L, centered)
    #     if oper == "+":
    #         self.topic_params[topic_id]["Lower Triangle"] = self.solver.chol_update(L, centered)
    #
    #
    #     chol_det = np.sum(np.log(np.diag(self.topic_params[topic_id]["Lower Triangle"]))) #  2 * sum_m_i(log(L_i,i))
    #
    #     self.topic_params[topic_id]["Chol Det"] = chol_det * 2
    #     self.topic_params[topic_id]["Topic Count"] = topic_count
    #     self.topic_params[topic_id]["Topic Kappa"] = kappa_k
    #     self.topic_params[topic_id]["Topic Nu"] = nu_k
    #     self.topic_params[topic_id]["Topic Mean"]= topic_mean
    #
    # def get_scaled_topic_MC(self, topic_id, topic_count, word, oper):
    #     """
    #     For a given topic, method calculates scaled topic Mean and Covariance (V-bar_k and C_k in R. Das Paper)
    #
    #     \sum_d \sum_z=i (V_di) / N_k
    #     ^^ =
    #     wordvec_sum = array[zero] > shape(word-vec dimensionality)
    #     for each doc:
    #             for each word that has topic assignment i:
    #                 wordvec_sum + word
    #     wordvec_sum / count of topic
    #     N_k = count of topic occurences across all documents
    #
    #     :param topic_id: The topic ID, integer
    #     :param topic_count: A copy of the document-topic counts table, numpy array
    #     :return: mean and covariance matrix.  Mean will be of shape (1 X word-vector dimension).
    #     Covariance will be matrix of size (word-vector dim X word-vector dim)
    #     """
    #     if oper == "-":
    #         self.topic_params[topic_id]["Topic Sum"] -= self.word_vecs[word] * self.word_count[word]
    #         return self.topic_params[topic_id]["Topic Sum"] / topic_count
    #     if oper == "+":
    #         self.topic_params[topic_id]["Topic Sum"] += self.word_vecs[word] * self.word_count[word]
    #         return self.topic_params[topic_id]["Topic Sum"] / topic_count


    # # noinspection PyStatementEffect
    # def update_document_topic_counts(self, word, topicID, operation):
    #     """
    #     :param word: a word to recalculate document x topic count table
    #     :param topicID: topic columm to adjust
    #     :param operation: '-' for subracting contribution | '+' for adding contribution
    #     :return: a new document-topic table (copy)
    #     Method only affects a copy of the ground truth
    #     Counts how many times each topic is assigned to a word in a document.  is a (Doc X Topic) array/matrix
    #     """
    #     topic_counts = np.copy(self.doc_topic_CT)
    #
    #     if operation == "-":
    #         for docID, doc in self.corpus.iteritems():
    #             topic_counts[docID, topicID] - self.doc_word_counts[docID][word]
    #
    #     if operation == "+":
    #         for docID, doc in self.corpus.iteritems():
    #             topic_counts[docID, topicID] + self.doc_word_counts[docID][word]
    #
    #     return topic_counts






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
                filtered_doc.append(word) # Remove words from doc that are not in word-vec model
                nkd[self.word_topics[word]] += 1.
            except KeyError: continue
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

    def display_results(self):
        for k in range(self.numtopics):
            print "Topic {0}: \n {1}".format(k, self.wvmodel.most_similar(positive=[self.topic_params[k]['Topic Mean']]))


if __name__ == "__main__":
    corpus = ["apple orange mango melon " *3 , "canvas art mural paint painting ", "pineapple kiwi grape strawberry ",
              "picture frame picasso sculpture art ", "coconut guava blueberry blackberry ", "statue monument art artist "]
    corpus = [sent * 5 for sent in corpus]*4

    f = '/Users/michael/Documents/GaussianLDA/clean20news.txt'
    with open(f, 'r') as fi:
        docs = fi.read().splitlines() #  These are all cleaned out
        fi.close()
    wordvec_fileapth = "/Users/michael/Documents/Gaussian_LDA-master/data/glove.wiki/glove.6B.50d.txt"
    start = time.time()
    g = Gauss_LDA(20, docs, word_vector_filepath=wordvec_fileapth, run_name='test')
    g.fit(15)
    print time.time() - start