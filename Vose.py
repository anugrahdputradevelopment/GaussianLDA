# coding=utf-8
import random

import numpy as np
import queue


class VoseAlias(object):

    def __init__(self, num): ##  num is the number of topics
        self.w = None #  np.array
        self.wsum = 0
        self.n = int
        self.p = None
        self.num = num #size
        self.prob = np.zeros(num)
        self.alias = np.zeros(num)

    def generate_table(self, weights):
        self.small = queue.Queue() # stores the indices of the alias
        self.large = queue.Queue() # stores the indices of the alias
        self.w = np.asarray(weights)
        self.wsum = self.w.sum()
        self.p = w*self.n/self.wsum

        # fi
		""" 3. For each scaled probability pi:
                a. If pi<1, add i to Small.
		        b. Otherwise(pi≥1), add i to Large.
		"""
        for i in xrange(self.n):
            if self.p[i] < 1.:
                self.small.put(i)
            else:
                self.large.put(i)


        # This is now the aliasing loop
        """	4. While Small and Large are not empty : (Large might be emptied first)
				a. Remove the first element from Small; call it l.
			    b. Remove the first element from Large; call it g.
				c. Set Prob[l] = pl.
				d. Set Alias[l] = g.
				e. Set pg : = (pg + pl)−1. (This is a more numerically stable option.)
				f. If pg<1, add g to Small.
				g. Otherwise(pg≥1), add g to Large.
		"""
        while len(small) and len(large):
            l = self.small.get()
            g = self.large.get()

            self.prob[l] = 1
            self.alias[g] = 1

        def sample_vose(self):
            # sampling from the alias table:
            """
            1. generate a fiar die roll form an n-sided die: call the side i
            2. flip a biased coin (bernoulli) that comesup head with Prob[i]
            3. if the coin comes up 'heads', return i, otherwise, return Alias[i]

            """

            fair_die = random.randint(0, self.n - 1)
            biased_coin = np.random.binomial(1, p=fair_die)

            if biased_coin > self.prob[fair_die]:
                return self.alias[fair_die]
            else:
                return fair_die
