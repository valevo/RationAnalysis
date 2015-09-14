__author__ = 'zweiss'


# Hellinger divergence of two distributions
# dist_a and dist_b [given as vectors of frequencies]
def hellinger(dist_a, dist_b):
    return -(1 / 2**.5 * (sum(euclideanSqrt(dist_a, dist_b)))**.5)


# Euclidean distance of twi distributions
# dist_a and dist_b [given as vectors of frequencies]
def euclideanSqrt(dist_a, dist_b):
    if len(dist_a) == len(dist_b):
        for i in range(len(dist_a)):
            yield (dist_a[i]**.5 - dist_b[i]**.5)**2


# Hellinger divergence implemented non-iteratively assuming
# a degenerated probability distribution P with probability
# 1 on a single event p_i and a uniformly distributed
# # probability distribution Q with each event q = 1/n
# num_messages = the number of messages true of a referent
# -> sum(row of correspondence matrix) or
# 1/uniform distribution
def hellinger_non_iterative_nums(num_messages):
    # dist_a = Pl(X=x) = {1/n, ...}, with len(dist_a) = n
    # dist_b = Ps(X=x) = {1, 0, 0, ...}
    #print("hellinger_non_iterative_nums(", num_messages, "): ", (1 / 2**.5) * (((1/num_messages)-1)**2 + ((num_messages-1)*(1/num_messages)**2))**.5)
    return -((1 / 2**.5) * ((((1/num_messages)**.5)-1)**2 + ((num_messages-1)*(1/num_messages)))**.5)



# Hellinger divergence implemented non-iteratively assuming
# a degenerated probability distribution P with probability
# 1 on a single event p_i and a uniformly distributed
# # probability distribution Q with each event q = 1/n
# num_messages = the number of messages true of a referent
# -> sum(row of correspondence matrix) or
# 1/uniform distribution
def hellinger_non_iterative_probs(prob_messages):
    rval = (1 / 2**.5) * (((prob_messages**.5)-1)**2 + (((1/prob_messages)-1) * prob_messages))**.5
   # print("hellinger_non_iterative_probs(", prob_messages, "): ", -rval)
    return -rval