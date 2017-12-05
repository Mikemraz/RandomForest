import random
import math

from DecisionTree import DecisionTree

class RandomForest(object):

    """
    Class of the Random Forest
    """
    def __init__(self, tree_num):
        self.tree_num = tree_num
        self.forest = []

    def train(self, records, attributes):
        """
        This function will train the random forest, the basic idea of training a
        Random Forest is as follows:
        1. Draw n bootstrap samples using bootstrap() function
        2. For each of the bootstrap samples, grow a tree with a subset of
            original attributes, which is of size m (m << # of total attributes)
        """
        # train a bunch of trees
        for n in range(self.tree_num):
            # choose 50% of attributes(without replacement) for building this tree.
            new_attributes = random.sample(attributes, math.ceil(0.5*len(attributes)))

            # train this decision tree. And 75% of training records are sampled
            # at each node in tree_growth() method in DecisionTree class
            tree = DecisionTree()
            tree.train(records, new_attributes)
            print('a tree has been built!!!')

            # collect all the trees into the forest
            self.forest.append(tree)
        pass

    def predict(self, sample):
        """
        The predict function predicts the label for new data by aggregating the
        predictions of each tree.

        This function should return the predicted label
        """
        # collect predictions from all decision trees
        results = []
        for tree in self.forest:
            result = tree.predict(sample)
            results.append(result)
        # the final result should be the voting majority
        p_count = 0
        e_count = 0
        for result in results:
            if result == 'e':
                e_count = e_count + 1
            else:
                p_count = p_count + 1
        if e_count >= p_count:
            return 'e'
        else:
            return 'p'
        pass

    def bootstrap(self, records):
        """
        This function bootstrap will return a set of records, which has the same
        size with the original records but with replacement.
        """
        len_records = len(records)
        samples = []
        for n in range(len_records):
            sample = records[random.randint(0, len_records - 1)]
            samples.append(sample)
        return samples
