import random
import math

class TreeNode(object):
    def __init__(self, isLeaf=False):
        self.isLeaf = isLeaf
        # if a Node is leaf, then it should make a prediction(known as label);
        # if it is not leaf, then it should save the information regarding split at
        # this node.
        if self.isLeaf:
            self.label = None
        else:
            self.test_cond = {}
            self.child = {}
        # Your code here

    def predict(self, sample):
        NextNode = self
        # if the sample has not reached the leaf, then it should walk through all
        # the corresponding Nodes whose attribute's value meet its.
        while not NextNode.isLeaf:
            attribute = NextNode.test_cond['attribute']
            if sample['attributes'][attribute] == NextNode.test_cond['value']:
                NextNode = NextNode.child['LeftChild']
            else:
                NextNode = NextNode.child['RightChild']
        # output the prediction if the sample reaches the leaf.
        return NextNode.label
        """
        This function predicts the label of given sample
        """

class DecisionTree(object):
    """
    Class of the Decision Tree
    """
    def __init__(self):
        self.root = None

    def train(self, records, attributes):
        self.root = self.tree_growth(records, attributes)


        """
        This function trains the model with training records "records" and
        attribute set "attributes", the format of the data is as follows:
            records: training records, each record contains following fields:
                label - the lable of this record
                attributes - a list of attribute values
            attributes: a list of attribute indices that you can use for
                        building the tree
        Typical data will look like:
            records: [
                        {
                            "label":"p",
                            "attributes":['p','x','y',...]
                        },
                        {
                            "label":"e",
                            "attributes":['b','y','y',...]
                        },
                        ...]
            attributes: [0, 2, 5, 7,...]
        """

    def predict(self, sample):

        """
        This function predict the label for new sample by calling the predict
        function of the root node
        """
        return self.root.predict(sample)

    def stopping_cond(self, records, attributes):
        n_e = 0
        # find if all record in records has the same class label
        for record in records:
            if record['label'] == 'e':
                n_e = n_e + 1
        if n_e in [0, len(records)]:
            return True
        # if they don't share the same class label,
        # let's continue to find if all record in records has the same attribute values
        else:
            for attribute in attributes:
                for record in records:
                    if record['attributes'][attribute] != records[0]['attributes'][attribute]:
                        return False
            return True

        """
        The stopping_cond() function is used to terminate the tree-growing
        process by testing whether all the records have either the same class
        label or the same attribute values.

        This function should return True/False to indicate whether the stopping
        criterion is met
        """

    def classify(self, records):
        n_p = 0
        n_e = 0
        for record in records:
            if record['label'] == 'e':
                n_e = n_e + 1
            else:
                n_p = n_p + 1
        if n_e >= n_p:
            return 'e'
        else:
            return 'p'
        """
        This function determines the class label to be assigned to a leaf node.
        In most cases, the leaf node is assigned to the class that has the
        majority number of training records

        This function should return a label that is assigned to the node
        """

    def split(self, attribute, value, records):
        # split the records into left and right based on attribute and its value.
        left, right = list(), list()
        for record in records:
            if record['attributes'][attribute] == value:
                left.append(record)
            else:
                right.append(record)
        return left, right

    def gini_index(self, groups):
        # the method output the gini_index of a given split
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            points = 0.0
            for class_val in ['e', 'p']:
                p = [record['label'] for record in group].count(class_val)/size
                points = p*p + points
            gini = gini + (1.0 - points)*(size/n_instances)
        return gini

    def get_value_of_attribute(self, attribute, records):
        # this method return all the unique values in certain attribute.
        s = list()
        for record in records:
            s.append(record['attributes'][attribute])
        s = set(s)
        return s

    def find_best_split(self, records, attributes):
        # this method find the best split at a Node based on given records.
        best_attribute, best_value, best_points, best_groups = 999, 999, 999, None
        groups = None
        # choose the best split based on the corresponding gini_index which should be the least.
        for attribute in attributes:
            values = self.get_value_of_attribute(attribute, records)
            for value in values:
                groups = self.split(attribute, value, records)
                gini = self.gini_index(groups)
                if gini <= best_points:
                    best_attribute, best_value, best_points, best_groups = attribute, value, gini, groups
        return {'attribute': best_attribute, 'value': best_value, 'left': best_groups[0], 'right': best_groups[1]}
        """
        The find_best_split() function determines which attribute should be
        selected as the test condition for splitting the trainig records.

        This function should return multiple information:
            attribute selected for splitting
            threshhold value for splitting
            left subset
            right subset
        """

    def tree_growth(self, records, attributes, depth=1, max_depth=10):
        """
        This function grows the Decision Tree recursively until the stopping
        criterion is met. Please see textbook p164 for more details

        This function should return a TreeNode
        """
        # implement the "deep bootstrap" by sampling 75% of the given records that arrive at
        # certain Node
        records = self.bootstrap(records)
        n_of_samples = math.ceil(len(records)*3/4)
        records = records[:n_of_samples]

        # stop if it meets the stopping conditions or reaches the max_depth.
        if self.stopping_cond(records, attributes) or depth >= max_depth:
            leaf = TreeNode(isLeaf=True)
            leaf.label = self.classify(records)
            return leaf
        else:
            # keep growing the tree if it does not need to stop.
            depth = depth + 1
            max_depth = max_depth
            root = TreeNode()
            root.test_cond = self.find_best_split(records, attributes)

            # for the left child of this Node, keep growing it recursively by using tree_growth method.
            left_records = root.test_cond['left']
            LeftChild = self.tree_growth(left_records, attributes, depth=depth, max_depth=max_depth)

            # for the right child of this Node, keep growing it recursively by using tree_growth method.
            right_records = root.test_cond['right']
            RightChild = self.tree_growth(right_records, attributes, depth=depth, max_depth=max_depth)
            # record the relationship between parent Node and child Node in the Node of parent.
            root.child = {'LeftChild': LeftChild, 'RightChild': RightChild}
            return root



        # Your code here
        # Hint-1: Test whether the stopping criterion has been met by calling function stopping_cond()
        # Hint-2: If the stopping criterion is met, you may need to create a leaf node
        # Hint-3: If the stopping criterion is not met, you may need to create a
        #         TreeNode, then split the records into two parts and build a
        #         child node for each part of the subset

    def bootstrap(self, records):
        """
        This function bootstrap will return a set of records, which has the same
        size with the original records but with replacement.
        """
        # sample the records(with replacement)
        len_records = len(records)
        samples = []
        for n in range(len_records):
            sample = records[random.randint(0, len_records - 1)]
            samples.append(sample)
        return samples