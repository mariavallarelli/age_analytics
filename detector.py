import argparse
import os
import time
from pprint import pprint

import tweepy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

from deps.BaseProcessor import TextProcessor, unityFunction
from deps.KeyWordsExtractor import get_keywords
from deps.TweetLoader import *
from deps.Selector import Selector
from deps.TweetsScraper import TweetsScraper


def load_tweets(args):
    """This method returns a dataframe of labelled tweet for age group which
    contains text features like num emoticon, num mentions, num url.
    and the text of tweet after cleaning."""
    df1 = pd.DataFrame()
    dir_listing = os.listdir(args.filepath)
    files = [fi for fi in dir_listing if fi.endswith(".json")]
    i = 1
    for filename in files:
        file = args.filepath + filename
        tempdf = load(file)
        print(i)
        i += 1
        df1 = pd.concat([df1, tempdf])
    with open("df1.csv", "w", encoding="utf-8", newline='') as reference:
        df1.to_csv(reference, sep=",", index=False, encoding="utf-8")
    # ONLY TWEET NOT RETWEET
    isTweet = df1['tweet_type'] == 'T'
    df1_tweet = df1[isTweet]
    print(len(df1_tweet))

    df2_tweet = df1_tweet.groupby(['_birthday_screenName'], as_index=False)
    df2 = df2_tweet.first()
    print(len(df2_tweet))
    # print(len(df2))
    # check sample is balanced
    print(df2.loc[:, '_range_age'].value_counts())
    # balance dataset
    df2 = under_sampleto_min(df2, '_range_age')
    print(df2.loc[:, '_range_age'].value_counts())
    df2 = df2.dropna(axis=0, how='all')
    df2.reset_index()

    # CREATED SCORE DATASET
    result = pd.merge(df1_tweet[['text', 'followers_count', 'tweet_type',
                                 '_birthday_screenName', 'num_emoticon', 'num_pronoun', 'num_mention',
                                 'num_url', 'num_punctuation', '_range_age', '_age', 'words']],
                                  df2[['_birthday_screenName']],
                                  how='left', on=['_birthday_screenName', '_birthday_screenName'])

    result.loc[result['_birthday_screenName'].isnull(), '_birthday_screenName'] = 0
    isBSNempty = result['_birthday_screenName'] == 0
    df_score = result[isBSNempty]
    print(len(df_score))
    with open("score.csv", "w", encoding="utf-8", newline='') as reference:
        df_score.to_csv(reference, sep=",", index=False, encoding="utf-8")

    df_train = pd.merge(df1_tweet[['text', 'followers_count', 'tweet_type',
                                   '_birthday_screenName', 'num_emoticon', 'num_pronoun', 'num_mention',
                                   'num_url', 'num_punctuation', '_range_age', '_age', 'words']],
                                    df2[['_birthday_screenName']],
                                    how='inner', on=['_birthday_screenName', '_birthday_screenName'])
    print(len(df_train))
    print(df_train.loc[:, '_range_age'].value_counts())
    df_train = under_sampleto_min(df_train, '_range_age')
    print(df_train.loc[:, '_range_age'].value_counts())
    with open("train.csv", "w", encoding="utf-8", newline='') as reference:
        df_train.to_csv(reference, sep=",", index=False, encoding="utf-8")
    return df_train


def scrape(args):
    """New Tweets of labelled users could be downloaded to create a bigger train dataset
    this method returns new tweets from age labelled screen names """
    df = pd.read_csv(args.screen_names, header=0, sep=',', encoding='utf-8', skip_blank_lines=True)
    # SCRAPE NEW TWEETS
    user_list = df['_birthday_screenName'].to_list()
    try:
        scraped_tweets_dict = scrape_tweets(user_list)
    except tweepy.TweepError as e:
        print(f'{len(scraped_tweets_dict)} records processed {e}')
    df_all = pd.DataFrame()
    for name, d in scraped_tweets_dict.items():
        d['_birthday_screenName'] = name
        df_all = pd.concat([df_all, d])
    result = pd.merge(df_all, df, on=['_birthday_screenName'])
    with open("new_tweets.csv", "w", encoding="utf-8") as reference:
        result.to_csv(reference, sep=",", index=False, encoding="utf-8")


def scrape_tweets(user_list):
    """This method returns dictionary with new scraped tweets """
    scraper = TweetsScraper(result_limit=1)
    scraped_tweets_dict = dict()
    for name in user_list:
        try:
            scraped_tweets = scraper.scrape_user_tweets(user=name, number_of_page=1)
            scraped_tweets_dict[name] = pd.DataFrame(scraped_tweets)
        except tweepy.TweepError as e:
            print(f'{name} could not be processed because {e}')
    return scraped_tweets_dict


def train_model(args):
    """This method trains model with labelled data set
    and runs the model on the score unlabelled dataset.
    The outpus is a full labelled dataset"""
    param_list = [item for item in args.list.split(',')]
    permitted_labels = [1, 2, 3, 4, 5]
    csv_train = param_list[0]
    csv_score = param_list[1]
    df = pd.read_csv(csv_train, header=0, sep=',', encoding='utf-8', skip_blank_lines=True)
    df1 = adjust_sample(df, permitted_labels, '_range_age')
    df1 = df.fillna('')
    x_all = df1.filter(['text', '_birthday_screenName', 'followers_count', 'num_emoticon', 'num_pronoun', 'num_mention',
                        'num_url', 'num_punctuation', 'words'], axis=1)
    y_all = df1.loc[:, '_range_age']
    x_train_vec, x_test_vec, y_train, y_test = train_test_split(
        x_all, y_all,
        test_size=0.20,
        random_state=0,
        stratify=y_all)

    x_train = x_train_vec
    x_test = x_test_vec


    cp = Pipeline([
        ('union', preproc_pipeline()),
        #('classifier', LogisticRegression()),  # LinearSVC is another classifier you can try
        ('classifier', RandomForestClassifier())
    ])

    # Setting for each parameter the value space (i.e., the set of values to evaluate)
    paramSpace = {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Values for grid search should be enclosed by []
                  'classifier__multi_class': ['ovr'],# default value, otherwise lot of warnings # one vs the rest. Strategy to turn a binary classifier into a multil-label classifier
                  'classifier__solver': ['liblinear'],
                  'classifier__class_weight': [None, 'balanced'],  # if the classes were imbalanced, we could try this approach             
                  }

    paramSpace1 = { 'classifier__n_estimators': [20, 50, 100, 200],
                    'classifier__max_samples': [2, 5, 10, 20],
                    'classifier__max_features': ['auto', 'sqrt', 'log2'],
                    'classifier__max_depth': [4, 5, 6, 7, 8],
                    'classifier__criterion': ['gini', 'entropy']}

    start_time = time.time()
    # cv=4 k-fold validation, con k=4
    gs = GridSearchCV(cp, param_grid=paramSpace1, scoring='accuracy', cv=4)
    #gs.fit(x_train, y_train)
    # ora mostro il tempo di fine
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Best parameters set found on development set:")
    print("Grid scores on development set:")
    print('Scoring result')
    #print(gs.best_score_)
    ######
    # Doing classification again, using the best parameters, as selected by Grid Search
    clsfParams = {
        'classifier__C': 100,
        'classifier__multi_class': 'ovr',
        'classifier__solver': 'liblinear'
    }
    clsfParams1 = {'classifier__n_estimators': 100,
                   'classifier__max_samples':20,
                   'classifier__max_features':'sqrt',
                   'classifier__max_depth': 6,
                   'classifier__criterion': 'entropy'}
    #cp.set_params(**clsfParams)
    #cp.set_params(**clsfParams1)

    cp.fit(x_train, y_train)
    y_pred = cp.predict(x_test)
    print('Classification report')
    print(classification_report(y_test, y_pred))
    print('Accuracy')
    print(accuracy_score(y_test, y_pred))


    df_score = pd.read_csv(csv_score, header=0, sep=',', encoding='utf-8', skip_blank_lines=True)
    df_score = df_score.fillna('')
    columns_ = ['text', '_range_age', '_birthday_screenName', 'followers_count', 'num_emoticon', 'num_pronoun',
                         'num_mention', 'num_url', 'num_punctuation', 'words']
    x = df_score.filter(columns_, axis=1)
    y_pred_score = cp.predict(x)
    df_score['_range_age'] = y_pred_score
    with open("df_score_labelled.csv", "w", encoding="utf-8", newline='') as reference:
        df_score.to_csv(reference, sep=",", index=False, encoding="utf-8")
    df_list = [df, df_score]
    df_all = pd.concat(df_list)
    df_all.reset_index()
    with open("df_all_labelled.csv", "w", encoding="utf-8", newline='') as reference:
        df_all.to_csv(reference, sep=",", index=False, encoding="utf-8")


def preproc_pipeline():
    """This method return a Feature Union of Pipeline for text feature
    and another for numeric features."""
    my_stop_words = text.ENGLISH_STOP_WORDS.union('birthday happy'.split())
    return FeatureUnion(
        transformer_list=
        [
            ('text', Pipeline(
                [
                    ('selector', Selector(key='text', dt='text')),
                    ('TextProcessor', TextProcessor()),
                    ('vectorizer', CountVectorizer(stop_words=my_stop_words, ngram_range=(1,1),
                                                   min_df=2, max_df=0.7, preprocessor=unityFunction)),
                    ('inputer', SimpleImputer(strategy='most_frequent'))

                ])
             ),
            ('followers_count', Pipeline(
                [
                    ('selector', Selector(key='followers_count', dt='num')),
                    ('inputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
             ),
            ('num_emoticon', Pipeline(
                [
                    ('selector', Selector(key='num_emoticon', dt='num')),
                    ('inputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
             )

        ]
    )
'''
These features aren't much significative.
            ('num_url', Pipeline(
                [
                    ('selector', Selector(key='num_url', dt='num')),
                    ('inputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
             ),
            ('num_pronoun', Pipeline(
                [
                    ('selector', Selector(key='num_pronoun', dt='num')),
                    ('inputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
             ),
            ('num_mention', Pipeline(
                [
                    ('selector', Selector(key='num_mention', dt='num')),
                    ('inputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
             )
'''

def adjust_sample(dfraw, permitted_labels, target_column_name):
    """This method is used to create a balanced dataset
    Unfortunately, the dataset is unbalanced."""
    # Removing missing values, if any
    print('Shape before removing missing values', dfraw.shape)
    df = dfraw.dropna(how='all')
    print('Shape after', df.shape)
    print()
    print('Removing undesired labels')
    print('df.shape', df.shape)
    df2 = df.loc[df.loc[:, target_column_name].isin(permitted_labels)]
    print('df2.shape', df2.shape)
    print()
    print('Range age sizes')
    print(df2.loc[:, target_column_name].value_counts())
    # Converting range age label to int
    df2[target_column_name] = df2[target_column_name].astype(int)
    df3 = under_sampleto_min(df2, target_column_name)
    # Let's look again to the category sizes
    print('Range sizes')
    print(df3.loc[:, target_column_name].value_counts())
    return df3


# Balancing by undersammpling
def under_sampleto_min(df, labelName):
    """ The dataset is undersampled so that all label groups will have the same size,
        corresponding to the (original) minimal label set.
        The parameter labelName is the DataFrmae column hosting the labels"""
    vc = df.loc[:, labelName].value_counts()  # Counting label frequencies
    lab2freq = dict(zip(vc.index.tolist(), vc.values.tolist()))
    # print(lab2freq) # if you want to see lab2freq, please uncomment this command
    # print(min(lab2freq.values()))
    minfreq = min(lab2freq.values())
    # print(minfreq)
    idxSample = []
    for selectedLabel, actualFreq in lab2freq.items():
        selIndexes = df.loc[df.loc[:, labelName] == selectedLabel, :].sample(n=minfreq).index.tolist()
        idxSample += selIndexes
    idxSample.sort()
    df2 = df.loc[idxSample, :]
    print(len(idxSample), df2.shape)
    pprint(df2.head(10))
    return df2


def print_top_50_keyword(args):
    df_score = pd.read_csv(args.csv, header=0, sep=',', encoding='utf-8', skip_blank_lines=True)
    for i in range(1, 6):
        is_range_of = df_score['_range_age'] == i
        df_age = df_score[is_range_of]
        df_age = df_score.fillna('')
        df = df_age[['words', '_range_age']].copy()
        get_keywords(df, i)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', help='specify files path')
    parser.add_argument('-s', '--screen_names', help='specify file csv of users for scrape new tweets')
    parser.add_argument('-l', '--list', help='specify list file train.csv and score.csv to train and score', type=str)
    parser.add_argument('-c', '--csv', help='specify file df_all_labelled.csv to print top 50 keywords for age group', type=str)
    args = parser.parse_args()
    if args.filepath:
        parser.set_defaults(func=load_tweets)
    elif args.screen_names:
        parser.set_defaults(func=scrape)
    elif args.list:
        parser.set_defaults(func=train_model)
    elif args.csv:
        parser.set_defaults(func=print_top_50_keyword)
    return parser


if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    args.func(args)
