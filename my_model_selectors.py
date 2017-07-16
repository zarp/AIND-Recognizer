import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_bic_score = float("inf") # +inf because smaller bic score is better
        best_model = None

        for number_of_states in range(self.min_n_components, self.max_n_components+1): #2..15 per notebook
            try:
                
                model = self.base_model(number_of_states)
                score = model.score(self.X, self.lengths)
                
                ## formula provided by Dana S. for the number of free parameters:
                ## number of free parameters = n_components*n_components + 2*n_components*n_features - 1
                number_of_features = len(self.X[0])
                number_of_free_parameters = number_of_states**2 + 2*number_of_states*number_of_features  - 1
                
                bic_score = -2*score + number_of_free_parameters*math.log(sum(self.lengths)) #sum(self.lengths) = len(self.X)) = length/size of the observation time series
                if bic_score < best_bic_score: #smaller is better
                    best_bic_score = bic_score
                    best_model = model
            except:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_dic_score = float("-inf")
        best_model = None
        
        for number_of_states in range(self.min_n_components, self.max_n_components+1): #2..15 per notebook
            try:
                model = self.base_model(number_of_states)
                score = model.score(self.X, self.lengths) #log(P(X(i)) in the formula above

                score_without_i = 0
                M = len(self.hwords)
                for word in self.hwords:
                    wordX, wordlength = self.hwords[word]
                    score_without_i += model.score(wordX, wordlength)

                dic_score = score - score_without_i/(M-1)

                if dic_score > best_dic_score: 
                    best_dic_score = dic_score
                    best_model = model
            except:
                pass

        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float("-inf")
        number_of_splits = min(3, len(self.sequences)) # kfold default = 3 but some words in the dataset have less than 3 examples
        best_model = None
        
        #need to treat n_splits = 1 as a separate case
        #otherwise - will get a ValueError: "k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more"
        if number_of_splits > 1:   
            k_fold = KFold(n_splits = number_of_splits) 
            
            for number_of_states in range(self.min_n_components, self.max_n_components+1): #2..15 per notebook
                try:
                    for train_indices, test_indices in k_fold.split(self.sequences):
                        self.X, self.lengths = combine_sequences(train_indices, self.sequences)
                        test_x, test_length = combine_sequences(test_indices, self.sequences)
                        model = self.base_model(number_of_states)
                        score = model.score(test_x, test_length)
                        if score > best_score: 
                            best_score = score
                            best_model = model
                except:
                    pass
        else:
            pass
        
        return best_model
