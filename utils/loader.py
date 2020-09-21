
import numpy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict

from agents.dummy import DummyAgent
from agents.goblin import GOBLinAgent
from agents.linucb import LinUCBAgent
from agents.block import BlockAgent
from agents.macro import MacroAgent

from context.four_cliques import FourCliques
from context.tagged import Tagged


def load_agent(algorithm_name, dim_feature, alpha, graph, cluster_data):
    if algorithm_name == "dummy":
        return DummyAgent()
    elif algorithm_name == "linucb":
        return LinUCBAgent(dim_feature, alpha)
    elif algorithm_name == "linucbsin":
        return LinUCBAgent(dim_feature, alpha, True)
    elif algorithm_name == "goblin":
        return GOBLinAgent(graph, len(graph), 
                            alpha=alpha, vector_size=dim_feature)
    elif algorithm_name == "block":
        return BlockAgent(graph, len(graph), cluster_data,
                            alpha=alpha,  vector_size=dim_feature)
    elif algorithm_name == "macro":
        return MacroAgent(graph, len(graph), cluster_data,
                            alpha=alpha, vector_size=dim_feature)
    else:
        raise Exception("Algorithm not implemented!")


def load_data(dataset_location, four_cliques_graph_noise=0, four_cliques_epsilon=0.1, dim_feature=25, num_clusters=None):
    '''
    :param dataset_location: location of dataset folder, or 4cliques for builtin 4cliques dataset
    :param four_cliques_graph_noise: graph noise for 4cliques
    :param four_cliques_epsilon: reward noise for 4cliques
    :param dim_feature: number of features in vector
    :return: ContextManager, network graph (numpy 2-dimensional matrix of ones and zeroes)
    '''

    if num_clusters:
        cluster_to_idx, idx_to_cluster = load_clusters(dataset_location, num_clusters)
    else:
        cluster_to_idx, idx_to_cluster = None, None

    if dataset_location != "4cliques":
        graph, num_users = load_graph(dataset_location)
        return  Tagged(num_users,
                       load_true_associations(dataset_location),
                       load_and_generate_contexts(dataset_location, dim_feature=dim_feature)), \
                graph, cluster_to_idx, idx_to_cluster
    else:
        # there it calls the class method
        graph = FourCliques.generate_cliques(threshold=1-four_cliques_graph_noise)
        return  FourCliques(epsilon=four_cliques_epsilon,
                            dim_feature=dim_feature), \
                graph, cluster_to_idx, idx_to_cluster


def load_graph(dataset_location):
    # graph is already represented as an adjacency matrix
    f = open("{}/graph.csv".format(dataset_location), 'r')
    rows = [[int(s) for s in line.split(',')] for line in f]
    
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
        user_contexts[int(user_str.strip())].append(context.strip())
    return user_contexts


def load_and_generate_contexts(dataset_location, dim_feature=25):
    # produce context indices from context names
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
    # load associations between contexts and tags and index tags
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
    # if the context has been associated with that tag, and zero otherwise
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
    # into a num-features-dimensional dense representation.
    svd = TruncatedSVD(n_components=dim_feature)
    svd_contexts = svd.fit_transform(contexts_array)
    
    all_contexts = []
    for context_id in context_to_idx.keys():
        # extract associated vector generated from svd for each context
        vector = svd_contexts[context_to_idx[context_id]]
        all_contexts.append((context_id, vector))
        # the format for a context is a tuple of a context_id and an associated vector
    return all_contexts


def load_clusters(dataset_location, num_clusters):
    if num_clusters not in [5, 10, 20, 50, 100, 200]:
        raise Exception("Invalid cluster number!")

    filename = "{}/clustered_graph.part.{}".format(dataset_location, num_clusters)
    idx_to_cluster = {} 
    with open(filename, "r") as cluster_file:
        for i, line in enumerate(cluster_file):
            if line.strip():
                idx_to_cluster[i] = int(line.strip())
    
    cluster_to_idx = defaultdict(lambda: [])
    for idx in idx_to_cluster.keys():
        cluster = idx_to_cluster[idx]
        cluster_to_idx[cluster].append(idx)
    return cluster_to_idx, idx_to_cluster




