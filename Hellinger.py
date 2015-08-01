__author__ = 'zweiss'


""" Calculates the Hellinger distance for two distributions
    @param distr1 first distribution
    @param distr2 second distribution
    @return -1 if distributions don't have equal length and Hellinger distance else
"""
def hellinger(distr1, distr2):
    return euclidean_sqrt(distr1, distr2) / 2**.5


""" Calculates the Euclidean distance for two distributions, but sums over the difference of the square roots
    @param distr1 first distribution
    @param distr2 second distribution
    @return -1 if distributions don't have equal length and result else
"""
def euclidean_sqrt(distr1, distr2):
    if len(distr1) == len(distr2):
        tmp = []
        for i in len(distr1):
            tmp[i] += (distr1[i]**.5 - distr2[i]**.5)**2
        return (sum(tmp))**.5
    else:
        return -1

