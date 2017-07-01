import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    for testing_word, (X, lengths) in test_set.get_all_Xlengths().items():

        best_score, best_guess = -9999999, None
        probability_dict = {}

        for model_word, model in models.items():
        # Try the probablility of all words 
            try:
                log_probability = model.score(X, lengths)
                probability_dict[model_word] = log_probability

            except:
                probability_dict[model_word] = -9999999
        
            # If the probability is higher record as best guess
            if log_probability > best_score:
                best_score = log_probability
                best_guess = model_word 

        # At end of loop add Probability Dictionary and best guess
        probabilities.append(probability_dict)
        guesses.append(best_guess)

    return probabilities, guesses
