import sys
from raw_data_processing import raw_data_processing
from data_processing import data_processing

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 1:
        lang = args[0]
        # Step 1: transform raw data in proper format filter it to be different from test and trial
        raw_data_processing(lang)
        # Step 2:
        data_processing(lang)

    else:
        sys.exit('Requires language: us|es')
