'''
---- Visualisation -----
          ...
'''

from libs.visualisation.detailed_results import get_results

# paths
output_dir = '../data/x-train/predictions/results-last/'

# important variables
language = {'us': "english", 'es': "spanish"}
total_test = 11111  # max for us is 50000, max for es is 10000
n_labels = 20  # 20 for us, 19 for es


def visualisation(lang, predictions_file_path, gold_file_path):
    get_results(language[lang], gold_file_path, predictions_file_path,
                output_dir, total_test, n_labels)
