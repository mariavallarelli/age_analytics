# Tweets Classification

This project tries to classify the age of tweets in groups. The eligible age groups are the followings: '18 to 29 years', '30 to 39 years', '40 to 49 years', '50 to 59 years', and '60 to 69 years''. The main steps are:

  - load and parse all tweets. Regex are used to select tweets and create a labelled dataset for each age group. The text of each labelled tweet is parsed to extract features as number of emoticon, punctuation, url, mention, personal pronouns
the output are two files: train.csv and score.csv. The first one is used to train and validate the model and the other one to label the others tweets.
    > run python detector.py -f "nlp_challenge_tweets/" 
  - train the model: I tested Logistic Regression and Random Forest Classifier
     > python detector.py -l  "train.csv,score.csv"
  - plot the top relevant 50 keywords for age group
  
	> run script ColabNotebook_KeywordsAgeGroupsTweets.ipynb on google Colab 
