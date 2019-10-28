from AbstractUserContextManager import AbstractUserContextManager
import numpy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD 

class TaggedUserContextManager(AbstractUserContextManager):
    def __init__(self, true_associations, contexts):
        self.true_associations = true_associations
        self.contexts = contexts


def load_data(dataset_location):
    if dataset_location != "4CLIQUES":
        return TaggedUserContextManager(dataset_location, load_and_generate_contexts(dataset_location))


def load_and_generate_contexts(dataset_location):
    f = open("{}/context_tags.csv".format(dataset_location), 'r')
    context_idx = 0
    tag_idx = 0
    context_to_idx = {}
    tag_to_idx = {}
    context_to_tags = []
    for line in f:
        context, tag = line.split(',')
        if context not in context_to_idx:
            context_to_idx[context] = context_idx
            context_idx += 1
        if tag not in tag_to_idx:
            tag_to_idx[tag] = tag_idx
            tag_idx += 1
        context_to_tags.append((context_to_idx[context], tag_to_idx[tag]))
    array = numpy.zeros((context_idx, tag_idx))

    for context_tag_pair in context_to_tags:
        context, tag = context_tag_pair
        array[context][tag] += 1

    transformer = TfidfTransformer()

    contexts_array = transformer.fit_transform(array)
    
    svd = TruncatedSVD(n_components=25)
    
    svd_contexts = svd.fit_transform(contexts_array)
    all_contexts = []
    
    for context in context_to_idx.keys():
        vector = svd_contexts[context_to_idx[context]]
        all_contexts.append((context, vector)) 

    return all_contexts

if __name__ == "__main__":
    print(load_and_generate_contexts("lastfm-processed")[0:3])
