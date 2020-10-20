import json
import pandas as pd

from deps.BaseProcessor import processor
from deps.FeaturesTweetExtractor import *


def process_tweet(tweet_obj):
    tweets_list = []
    exd_full_text = ''
    rt_text = ''
    rt_ext_text = ''
    ''' User info'''
    # Store the user screen name in 'user-screen_name'
    tweet_obj['user-screen_name'] = tweet_obj['user']['screen_name']
    tweet_obj['user_id'] = tweet_obj['user']['id_str']
    tweet_obj['followers_count'] = tweet_obj['user']['followers_count']

    ''' Text info'''
    tweet_obj['tweet_created_at'] = tweet_obj['created_at']
    tweet_obj['lang'] = tweet_obj['lang']

    # Check if this is a 140+ character tweet
    if 'extended_tweet' in tweet_obj:
        # Store the extended tweet text in 'extended_tweet-full_text'
        exd_full_text = tweet_obj['extended_tweet']['full_text']
    else:
        text = tweet_obj['text']

    if 'retweeted_status' in tweet_obj:
        # Store the retweet text in 'retweeted_status-text'
        rt_text = tweet_obj['retweeted_status']['text']

        if 'extended_tweet' in tweet_obj['retweeted_status']:
            # Store the extended retweet text in
            rt_ext_text = tweet_obj['retweeted_status']['extended_tweet']['full_text']
    if 'quoted_status' in tweet_obj:
        # Store the retweet user screen name in
        tweet_obj['quoted_status-user-screen_name'] = (
            tweet_obj['quoted_status']['user']['screen_name'])

        # Store the retweet text in 'retweeted_status-text'
        tweet_obj['quoted_status-text'] = tweet_obj['quoted_status']['text']

        if 'extended_tweet' in tweet_obj['quoted_status']:
            # Store the extended retweet text in
            tweet_obj['quoted_status-extended_tweet-full_text'] = (
                tweet_obj['quoted_status']['extended_tweet']['full_text'])
    if rt_ext_text:
        tweet_obj['text'] = rt_ext_text
        tweet_obj['tweet_type'] = 'RT_EX'
    elif rt_text:
        tweet_obj['text'] = rt_text
        tweet_obj['tweet_type'] = 'RT'
    elif exd_full_text:
        tweet_obj['text'] = exd_full_text
        tweet_obj['tweet_type'] = 'TF'
    else:
        tweet_obj['text'] = text
        tweet_obj['tweet_type'] = 'T'
    return tweet_obj


def load(file):
    columns = (['text', 'tweet_type', 'lang', 'user-screen_name', 'user_id',
                'tweet_created_at', 'followers_count', 'num_emoticon', 'num_pronoun',
                'num_mention', 'num_url', 'num_punctuation', '_owner_birthday_wishes',
                '_age', '_year_of_birth', '_birthday_screenName', '_range_age', 'words'])
    res = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            tweet = json.loads(line)
            process_tweet(tweet)
            res.append(tweet)
            # print(res)
        tempdf = pd.DataFrame(res, columns=columns)
        tempdf.loc[:, '_owner_birthday_wishes'] = tempdf['text'].apply(
            lambda x: find_happy_birthday_tome(x))
        tempdf.loc[:, '_year_of_birth'] = tempdf['user-screen_name'].apply(
            lambda x: find_year_of_birth(x))
        tempdf.loc[:, '_age'] = tempdf.apply(lambda x: get_age(x), axis=1)
        tempdf.loc[:, '_birthday_screenName'] = tempdf.apply(
            lambda x: find_birthday_screen_name(x), axis=1)
        tempdf.loc[:, '_range_age'] = tempdf['_age'].apply(lambda x: assign_range(int(x)) if x else '')
        tempdf.loc[:, '_target'] = tempdf['_age'].apply(lambda x: assign_range_binary(int(x)) if x else '')
        #tempdf.loc[:, 'n_hashtags'] = tempdf['text'].apply(lambda x: count_hash_tags(x))
        tempdf.loc[:, 'num_emoticon'] = tempdf['text'].apply(lambda x: count_emoticon(x))
        tempdf.loc[:, 'num_url'] = tempdf['text'].apply(lambda x: count_url(x))
        tempdf.loc[:, 'num_punctuation'] = tempdf['text'].apply(lambda x: count_exclamation_question_period_marks(x))
        tempdf.loc[:, 'num_pronoun'] = tempdf['text'].apply(lambda x: count_personal_pronouns(x))
        tempdf.loc[:, 'num_mention'] = tempdf['text'].apply(lambda x: count_mentions(x))
        tempdf.loc[:, 'words'] = tempdf['text'].apply(lambda x: processor(x))
    return tempdf


