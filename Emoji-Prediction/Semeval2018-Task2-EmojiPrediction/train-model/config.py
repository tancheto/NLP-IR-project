# starting training data
data_limit = 11111

# Add or Skip mentions
add_mentions = False

# types of ngrams
ngrams_types = [1]  # [1, 2, 3]
# range tuple (x, y) is equvalent to ngrams_types of [x, x+1, ... , y-1, y]
ngram_range = (1, 1)
# ngram (or word in dictionary) minimum frequency
min_frequency = 2
# number of featires
max_features = 1000
# ngrams limit for each type (their sum must be <= max_features)
most_important_ngrams = [1000]  # [1000, 1000, 1000]

# generate new mask for classifier
has_to_load = True
# amount of data that the classifier uses
classifier_load = 11111
