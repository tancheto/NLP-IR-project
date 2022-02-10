Quick start: 
1) Download the json of the tweets with the crawler, see the README in the crawler folder. 

!!! execute python scripts from 'Emoji-Prediction/Semeval2018-Task2-EmojiPrediction/train-model' !!!

2) Prepare the dataset for the emoji prediction task using raw data processing.

3) 

python3.10 -B train-model.py us

After the execution, you will find three files in the same folder of the twitter json:
- .text file with one text per line
- .labels file with one label per line (same order as previous one)
- .ids file with one twitter id per line (to keep track of the tweet you are using)
- .key file with keywords of tweet per line (to keep track of the hashtags that are used)
