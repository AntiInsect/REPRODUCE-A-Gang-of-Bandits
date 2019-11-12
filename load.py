from AbstractUserContextManager import AbstractUserContextManager
import numpy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD 
from collections import defaultdict
import random

class FourCliquesContextManager(AbstractUserContextManager):
    """
    For a social network with 4 cliques, each with 25 users, assigns each user within a 
    clique a random vector of size 25 representing the "ideal" context vector for that user.
    Returns random context vectors for any user when getting users and contexts. 
    Computes payoff as user_vector dot context_vector + uniform distribution within epsilon --
    Epsilon is payoff noise.
    """
    def __init__(self, epsilon=0.0):
        self.user_vectors = []
        # user_vectors will contain 100 vectors, 25 of the same vector for each clique
        self.epsilon = epsilon
        for i in range(4):
            rand_vector = numpy.random.rand(25)
            norm = numpy.linalg.norm(rand_vector)
            rand_vector = rand_vector / norm
            for j in range(25):
                self.user_vectors.append(rand_vector)
    def get_user_and_contexts(self):
        user = random.randrange(0,100)
        context_vectors = []
        for i in range(25):
            rand_vector = numpy.random.rand(25)
            norm = numpy.linalg.norm(rand_vector)
            rand_vector = rand_vector / norm
            context_vectors.append(("fakeID",rand_vector))

        return user, context_vectors
    def get_payoff(self, user, context):
        user_vector = self.user_vectors[user]
        context_vector = context[1]
        # payoff is dotted user_vector and context_vector plus a random sample bounded by epsilon 
        return numpy.dot(user_vector, context_vector) + numpy.random.uniform(-self.epsilon, self.epsilon)
    
 
        

        
        


class TaggedUserContextManager(AbstractUserContextManager):
    """
    For a social network with num_users users associated truly with contexts true_associations. 
    For get_user_and_contexts, returns a random collection of context vectors such that one is
    truly associated with the user. To compute payoff, returns 1 if the context is truly associated
    with the user and zero otherwise.
    """
    def __init__(self, num_users, true_associations, contexts):
        self.true_associations = true_associations
        self.contexts = contexts
        self.num_users = num_users
        self.context_dict = {}
        for context in self.contexts:
            self.context_dict[context[0]] = context
    def get_user_and_contexts(self):
        user = random.randrange(0, self.num_users)
        associated_contexts = self.true_associations[user]
        base_contexts = random.choices(self.contexts, k=24)
        truth_context_id = random.choice(associated_contexts)
        contexts = base_contexts + [self.context_dict[truth_context_id]]
        return user, contexts 
    def get_payoff(self, user, context):
        if context[0] in self.true_associations[user]:
            return 1
        else:
            return 0

         


def load_data(dataset_location):
    if dataset_location != "4CLIQUES":
        graph, num_users = load_graph(dataset_location)
        return TaggedUserContextManager(num_users, load_true_associations(dataset_location), load_and_generate_contexts(dataset_location)), graph
    else:
        graph = generate_cliques(.9)
        return FourCliquesContextManager(.1), graph


    
    

def generate_cliques(threshold):
    graph = numpy.zeros((100,100))
    # creates a block adjacency matrix with 4 25 x 25 blocks of ones
    # along the diagonal corresponding to each clique 
    for i in range(4):
        for j in range(25):
            for k in range(25):
                graph[j+i*25][k+i*25] = 1
    noise = numpy.random.rand(100,100)
    def check_threshold(element):
        if element > threshold:
            return 1
        else:
            return 0
    vfunc = numpy.vectorize(check_threshold)
    noisethreshold = vfunc(noise)
    result = numpy.logical_xor(graph, noisethreshold)
    vfunc = numpy.vectorize(lambda x: 1 if x else 0)
    return vfunc(result)





        
def load_graph(dataset_location):
    # graph is already represented as an adjacency matrix
    f = open("{}/graph.csv".format(dataset_location), 'r')
    rows = []
    for line in f:
        rows.append([int(s) for s in line.split(',')])
    num_users = len(rows[0])
    array = numpy.zeros((num_users, num_users))
    for i in range(num_users):
        for j in range(num_users):
            array[i][j] = rows[i][j]
    return array, num_users

def load_true_associations(dataset_location):
    # true associations are pairs of users and contexts that that user has actually interacted with
    f = open("{}/user_contexts.csv".format(dataset_location), 'r')
    user_contexts = defaultdict(list)
    for line in f:
        user_str, context = line.split(',')
        user_str = user_str.strip()
        context = context.strip()
        user = int(user_str)
        user_contexts[user].append(context)
    return user_contexts


def load_and_generate_contexts(dataset_location):
    context_idx = 0
    context_to_idx = {}
    contexts = open("{}/context_names.csv".format(dataset_location), 'r', encoding="utf-8")
    for line in contexts:
        context = line.split(',')[0]
        if context not in context_to_idx:
            context_to_idx[context] = context_idx
            context_idx += 1
   
    f = open("{}/context_tags.csv".format(dataset_location), 'r')
    tag_idx = 0
    tag_to_idx = {}
    context_to_tags = []
    # load and index contexts and tags
    for line in f:
        context, tag = line.split(',')
        if tag not in tag_to_idx:
            tag_to_idx[tag] = tag_idx
            tag_idx += 1
        if context not in context_to_idx:
            context_to_idx[context] = context_idx
            context_idx += 1
        context_to_tags.append((context_to_idx[context], tag_to_idx[tag]))
    # create matrix context_num by tag_num in size whose elements are 1
    # if the context has been associated with that tag
    array = numpy.zeros((context_idx, tag_idx))

    for context_tag_pair in context_to_tags:
        context, tag = context_tag_pair
        array[context][tag] = 1
    # perform tfidf transformation
    # value in array is decreased corresponding to the number of contexts that are tagged
    # with a given tag, making it so that rare tags count for more. TFIDF also weights by
    # the number of times that the tag appears with a given context, but since here all are 1
    # this is not meaningful
    transformer = TfidfTransformer()
    contexts_array = transformer.fit_transform(array)
    

    # use singular value decomposition to compress our high-dimensional sparse representation of each context
    # into a 25-dimensional dense representation.
    svd = TruncatedSVD(n_components=25)
    svd_contexts = svd.fit_transform(contexts_array)
    all_contexts = []
    
    for context in context_to_idx.keys():
        vector = svd_contexts[context_to_idx[context]]
        all_contexts.append((context, vector)) 
    # the format for a context is a tuple of a context_id and an associated vector
    return all_contexts




if __name__ == "__main__":

    ucm, graph = load_data('lastfm-processed')
    user, contexts = ucm.get_user_and_contexts()
    for context in contexts:
        print(ucm.get_payoff(user, context))
    

