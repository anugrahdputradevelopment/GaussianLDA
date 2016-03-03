from __future__ import division
import gensim
import numpy as np
from numpy import log, pi, linalg, exp
from scipy.special import gamma, gammaln
import random
from collections import defaultdict


class Wishart(object):

    def __init__(self, word_vecs):
        self.nu = None
        self.kappa = None
        self.psi = None

        self.set_params(word_vecs)

    def set_params(self, word_vecs):
        #turn dict of word vecs into a matrix
        word_vecs = np.vstack(word_vecs.values())

        self.nu = word_vecs.shape[1] #len of columns
        self.kappa = 0.01
        # self.psi = np.sum(word_vecs.T.dot(word_vecs), axis=0) # should this be np.sum(x.dot(x.T)))??? also changed this to x.T.dot(x)
        self.psi = np.identity(word_vecs.shape[1]) * 3 #changed this to be a simple identity matrix.. like in the paper.  No intuition here..
        self.mu = np.mean(word_vecs, axis=0)
        print "psi shape", self.psi.shape


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
        self.alpha = 50./self.numtopics


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

        index = 0
        for word in self.vocab:
            try:
                self.word_vecs[word] = vectors[word]
                index += 1
            except KeyError: continue
        print "Word-vectors for the corpus are created"


    def fit(self, iterations=1, init=True): #set hyperparams here?
        if init == True:
            self.init()
            init = False

        print "Starting fit"
        for i in xrange(iterations):
            self.sample()
            print "{} iterations complete".format(i)

    def init(self):

        self.process_corpus(self.corpus)
        self.process_wordvectors(self.wordvecFP)
        #setting wishhart priors
        self.priors = Wishart(self.word_vecs)
        self.doc_topic_CT = np.zeros((len(self.corpus.keys()), self.numtopics))

        self.word_topics = {word: random.choice(range(self.numtopics)) for word in self.vocab}
        # get Doc-Topic Counts
        for docID, doc in self.corpus.iteritems():
            for word in doc:
                topicID = self.word_topics[word]
                self.doc_topic_CT[docID, topicID] += 1

        # Init parameters for topic distributions
        for k in range(self.numtopics):
            self.recalculate_topic_params(k)

        print "Intialization complete"

    def sample(self, init=True):

        print "sampling started"
        # Randomly assign word to topics
        if init == False:
            self.word_topics = {word: random.choice(range(self.numtopics)) for word in self.vocab}

        for docID, doc in self.corpus.iteritems():
            for word in doc:
                #subtracting info about current word-topic assignment from doc-topic count table
                topic_id = self.word_topics[word]
                self.doc_topic_CT[docID, topic_id] - doc.count(word)

                # self.update_document_topic_counts(word, "-")

                self.recalculate_topic_params(topic_id)
                posterior = []
                max = 0
                for k in range(self.numtopics): #start getting the pdf's for each word-topic assignment
                    log_prob = self.draw_new_wt_assgns(word, k)
                    # print "multivariate T PDF for wordvector", log_prob

                    # Count of topic in doc
                    Nkd = self.doc_topic_CT[docID, k]
                    print "Nkd = {}".format(Nkd)
                    log_posterior = log(Nkd + self.alpha) * log_prob
                    print "log posteriror with doc-contribution", log_posterior
                    posterior.append(log_posterior)
                    #doing this for some normalization scheme
                    if log_posterior > max: max = log_posterior
                posterior.append(0) #just a little hitch in the function that wants a zero at the end.
                print 'max', max
                normalized_posterior = [exp(i-max) for i in posterior]
                post_sum = np.sum(posterior)
                print 'sum of posterior', post_sum
                # print 'exp method of normalizing', np.sum([i/np.sum(normalized_posterior) for i in posterior])
                normalized_post = posterior / post_sum
                print "normalized?", normalized_post, np.sum(normalized_post)
                new_word_topic = np.random.multinomial(1, pvals=normalized_post)
                print 'multinomial with reg-norm', new_word_topic
                # print 'multinomial with exp-norm', np.random.multinomial(1, pvals=normalized_posterior)
                ## need to copy the normalization scheme from Util.sample

                self.word_topics[word] = np.argmax(new_word_topic)
                # self.recalculate_topic_params(word, "+")
                # self.recalculate_topic_params(self.word_topics[word])


                #calculate document-topic countss

                #recalculate topic param of new assignment
        init = False
        return None

    def draw_new_wt_assgns(self, word, topic_id):

        # Getting params for calculating PDF of T-Dist for a word
        wordvec = self.word_vecs[word]
        inv_cov = self.topic_params[topic_id]["Inverse Covariance"]
        cov_det = self.topic_params[topic_id]["Covariance Determinant"]
        Nk = self.topic_params[topic_id]["Topic Count"]
        KappaK = self.topic_params[topic_id]["Topic Kappa"]
        centered = self.word_vecs[word] - self.priors.mu
        topic_cov = self.topic_params[topic_id]["Topic Covariance"]

        # print "topic cov", topic_cov


        # Precalculating some terms (V_di - Mu)^T * Cov^-1 * (V_di - Mu)
        LLcomp = centered.T.dot(inv_cov).dot(centered) #for some topics, this outputs a negative value...
        print 'll comp', LLcomp
        d = self.word_vec_size
        nu = self.priors.nu + Nk - d + 1
        print "d = {0}, nu{1}".format(d, nu)

        log_prop = gammaln(nu + d / 2) - \
                   (gammaln(nu / 2) + d/2 * (log(nu) + log(pi)) +0.5 * cov_det[1] + (nu + d) / 2 * log((1 + LLcomp) / nu)) #cov_det is already logged

        return log_prop
        # logprob = Gamma.logGamma((nu + Data.D)/2) - \
        #           (Gamma.logGamma(nu/2) + Data.D/2 * (Math.log(nu)+Math.log(Math.PI)) + 0.5 * Math.log(det) + (nu + Data.D)/2* Math.log(1+val/nu))

    def recalculate_topic_params(self, topic_id):

        topic_count = np.sum(self.doc_topic_CT[:, topic_id], axis=0) # N_k

        kappa_k = self.priors.kappa + topic_count # K_k
        nu_k = self.priors.nu + topic_count # V_k

        scaled_topic_mean_K, scaled_topic_cov_K  = self.get_scaled_topic_MC(topic_id) # V-Bar_k and C_k

        vk_mu = scaled_topic_mean_K - self.priors.mu #V-bar_k - Mu
        # print 'test 1', linalg.det(scaled_topic_cov_K)
        # print 'test 2', linalg.slogdet(scaled_topic_cov_K)
        psi_k = self.priors.psi + scaled_topic_cov_K + ((self.priors.kappa * topic_count) / kappa_k) * (vk_mu.T.dot(vk_mu)) # Psi_k
        # print 'test 3', linalg.det(psi_k)
        # print 'test 4', linalg.slogdet(psi_k)
        topic_mean = (self.priors.kappa * self.priors.mu + topic_count * scaled_topic_mean_K) / kappa_k # Mu_k
        topic_cov = psi_k / (nu_k - self.word_vec_size + 1) # Sigma_k
        # print 'test 5', linalg.det(topic_cov)
        print 'log-det of topic-covar', linalg.slogdet(topic_cov) #regular determinant calculator breaks with super small numbers..

        self.topic_params[topic_id]["Topic Count"] = topic_count
        self.topic_params[topic_id]["Topic Kappa"] = kappa_k
        self.topic_params[topic_id]["Topic Nu"] = nu_k
        self.topic_params[topic_id]["Topic Mean"], self.topic_params[topic_id]["Topic Covariance"] = topic_mean, topic_cov
        self.topic_params[topic_id]["Inverse Covariance"] = np.linalg.inv(topic_cov)
        self.topic_params[topic_id]["Covariance Determinant"] = np.linalg.slogdet(topic_cov)
        self.topic_params[topic_id]["Liklihood Componant"] = None


        return topic_mean, topic_cov

    def get_scaled_topic_MC(self, topic_id):
        # get words assigned to topic_id
        word_vecs = []
        for docID, doc in self.corpus.iteritems():
            for word in doc:
                if self.word_topics[word] == topic_id:
                    word_vecs.append(self.word_vecs[word])


        word_vecs = np.vstack(word_vecs)
        mean = np.sum(word_vecs, axis=0) / (np.sum(self.doc_topic_CT[:, topic_id], axis=0) + 2) #added a small number here to stop overflow

        # mean_centered = np.sum(word_vecs, axis=0) - mean
        mean_centered = word_vecs - mean
        # print 'doc-topic counts', self.doc_topic_CT
        # print 'mean', mean
        # print 'mean centered' , mean_centered

        cov = mean_centered.T.dot(mean_centered)
        return mean, cov


    def get_scaled_topic_mean_cov(self, topic_id):
        'mean of word vecs in a topic'
        # get words assigned to topic_id
        word_vecs = []
        for word, topic in self.word_topics.iteritems():
            if topic == topic_id:
                word_vecs.append(self.word_vecs[word])
        word_vecs = np.vstack(word_vecs)
        # print np.sum(word_vecs, axis=0)
        # print np.sum(self.doc_topic_CT[:, topic_id], axis=0)
        mean = np.sum(word_vecs, axis=0) / (np.sum(self.doc_topic_CT[:, topic_id], axis=0) + 2) #added a small number here to stop overflow

        # mean_centered = np.sum(word_vecs, axis=0) - mean
        mean_centered = word_vecs - mean
        # print 'doc-topic counts', self.doc_topic_CT
        # print 'mean', mean
        # print 'mean centered' , mean_centered

        cov = mean_centered.T.dot(mean_centered)
        return mean, cov

    def update_document_topic_counts(self, word, operation):
        topicID = self.word_topics[word]
        if operation == "-":
            for docID, doc in self.corpus.iteritems():
                self.doc_topic_CT[docID, topicID] -= doc.count(word)

        if operation == "+":
            for docID, doc in self.corpus.iteritems():
                self.doc_topic_CT[docID, topicID] += doc.count(word)


if __name__ == "__main__":
    corpus = ["apple orange mango melon", "dog cat bird rat"]
    wordvec_fileapth = "/Users/michael/Documents/Gaussian_LDA-master/data/glove.wiki/glove.6B.50d.txt"
    g = Gauss_LDA(2, corpus, wordvec_fileapth )
    g.fit(2)
    # print g.topic_params[1]["Topic Count"]