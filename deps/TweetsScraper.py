import os
from pprint import pprint
import tweepy
from datetime import datetime

import pandas as pd
import twint
from dotenv import load_dotenv


class TweetsScraper(object):

    result_limit = 20
    data = list()
    api = False
    load_dotenv()
    consumer_secret = os.getenv("consumer_secret")
    consumer_key = os.getenv("consumer_key")
    access_token = os.getenv("access_token")
    access_token_secret = os.getenv("access_token_secret")

    pprint(consumer_secret)
    auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    twitter_keys = {'consumer_key': consumer_key,
                    'consumer_secret': consumer_secret,
                    'access_token_key': access_token,
                    'access_token_secret': access_token_secret}

    def __init__(self, keys_dict=twitter_keys, api=api, result_limit=20):

        self.twitter_keys = keys_dict
        auth = tweepy.OAuthHandler(keys_dict['consumer_key'],
                                   keys_dict['consumer_secret'])
        auth.set_access_token(keys_dict['access_token_key'],
                              keys_dict['access_token_secret'])
        self.api = tweepy.API(auth, wait_on_rate_limit=False,
                              wait_on_rate_limit_notify=False)
        self.twitter_keys = keys_dict
        self.result_limit = result_limit

    def search_for_keyword(self, keyword, max_tweets):
        msgs = []
        df = pd.DataFrame()
        while len(msgs) < max_tweets:
            try:
                for tweet in self.api.search(q=keyword, lang="en", rpp=1):
                    print(tweet)
                    msg = [tweet.text, tweet.user.screen_name, tweet.source_url]
                    msg = tuple(msg)
                    msgs.append(msg)
            except tweepy.TweepError as e:
                raise tweepy.TweepError
        df = pd.DataFrame(msgs)
        return df

    def scrape_user_tweets(self, user, number_of_page=1):
        twt_list = list()
        last_tweet_id = False
        page = 1
        while page <= number_of_page:
            if last_tweet_id:
                status_coll = self.api.user_timeline(screen_name=user,
                                                     count=self.result_limit,
                                                     max_id=last_tweet_id - 1,
                                                     tweet_mode='extended',
                                                     include_retweets=False)
            else:
                status_coll = self.api.user_timeline(screen_name=user,
                                                     count=self.result_limit,
                                                     tweet_mode='extended',
                                                     include_retweets=False)
            for item in status_coll:
                twt = {'tweet_id': item.id,
                         'name': item.user.name,
                         'screen_name': item.user.screen_name,
                         'text': item.full_text,
                         'hashtags': item.entities['hashtags'],
                         'status_count': item.user.statuses_count,
                         'scraped_at': datetime.now(),
                         'created_at': item.created_at,
                         'favourite_count': item.favorite_count,
                         'location': item.place,
                         'retweet_count': item.retweet_count,
                         'source_device': item.source}
                try:
                    twt['retweet_text'] = item.retweeted_status.full_text
                except:
                    twt['retweet_text'] = 'None'
                try:
                    twt['quote_text'] = item.quoted_status.full_text
                    twt['quote_screen_name'] = item.quoted_status.user.screen_name
                except:
                    twt['quote_text'] = 'None'
                    twt['quote_screen_name'] = 'None'
                last_tweet_id = item.id
                twt_list.append(twt)
            page += 1
        return twt_list

    def get_followings(username):
        pprint("sonoqui")
        followed = []
        if username:
            c = twint.Config()
            c.Limit = 100
            c.Username = username
            c.Pandas_clean = True
            c.Pandas = True
            c.Limit = 20
            try:
                twint.run.Following(c)
                followed = twint.storage.panda.Follow_df["following"].tolist()[0]
            except:
                pass
            return followed