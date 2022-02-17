'''
---- Visualisation -----
          ...
'''

import config
from libs.visualisation.detailed_results import get_results

# paths
output_dir = '../data/x-train/predictions/results-last/'

# important variables
language = {'us': "english", 'es': "spanish"}
# 20 for us, 19 for es
n_labels = {'us': 20, 'es': 19}
# max for us is 50000, max for es is 10000
total_test = {'us': 50000 if config.classifier_load > 50000 else config.classifier_load,
              'es': 10000 if config.classifier_load > 10000 else config.classifier_load}


def visualisation(lang, predictions_file_path, gold_file_path):
    get_results(language[lang], gold_file_path, predictions_file_path,
                output_dir, total_test[lang], n_labels[lang])
