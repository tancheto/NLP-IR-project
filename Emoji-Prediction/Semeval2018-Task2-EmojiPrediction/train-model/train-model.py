import sys
from raw_data_processing import raw_data_processing
from data_processing import data_processing

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 1:
        sys.exit('Requires language: (us|es)')
    elif len(args) > 2:
        sys.exit('Too many arguments...')
    else:
        lang = args[0]
        step = args[1] if len(args) == 2 else 'all'

        if step == '1' or step == 'all':  # Step 1: transform raw data in proper format filter it to be different from test and trial
            raw_data_processing(lang)
        if step == '2' or step == 'all':  # Step 2: ...
            data_processing(lang)
