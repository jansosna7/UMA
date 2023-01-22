'''
*===========================================================================*
*				Author: Marcin Latawiec                                    *
*===========================================================================*
'''

''' class for handling probabilistic approach'''
class ProbabilisticPredictor:
    def __init__(self, unique_names_probability_dict, tree):
        self.probability_dict = unique_names_probability_dict
        self.tree = tree

    def fill_prob_dictionary(self, x, tree, probability = 1):
        ''' recursive function to fill probability_dictionary '''
        '''
        :param x: 
        :param tree: 
        :param probability: 
        :return dictionary: 
        '''
        if tree.value!=None:
            self.probability_dict[tree.value] = self.probability_dict[tree.value] + probability

        else:
            feature_val = x[tree.feature_index]
            if not self.is_missing(feature_val):
                if feature_val<=tree.threshold:
                    return self.fill_prob_dictionary(x, tree.left, probability)
                else:
                    return self.fill_prob_dictionary(x, tree.right, probability)
            else:
                number_of_elements_left = len(tree.left_dataset)
                number_of_elements_right = len(tree.right_dataset)
                sum_of_elements = number_of_elements_right + number_of_elements_left
                probability_of_left_branch = probability*number_of_elements_left/(sum_of_elements)
                probability_of_right_branch = probability*number_of_elements_right/(sum_of_elements)
                self.fill_prob_dictionary(x, tree.left, probability_of_left_branch)
                self.fill_prob_dictionary(x, tree.left, probability_of_right_branch)

    def most_possible_value(self):
        '''function to find most fitting value for given element'''
        most_possible = list(self.probability_dict.keys())[0]
        for key, value in self.probability_dict.items():
            if value>self.probability_dict[most_possible]:
                most_possible = key
        return most_possible

    def is_missing(self, value):
        if str(value) == "NaN" or str(value) == "nan":
            return True
        else:
            return False