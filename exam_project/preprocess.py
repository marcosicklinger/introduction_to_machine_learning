from pandas import DataFrame, Categorical
import pandas as pd
import datetime
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import json
import pickle
from tweepy import Cursor
import os
import random


def extractTweetFeatures(file_name, food_dict, api):
    '''extracts features of interest from file containing twitter raw data

       Parameters
       ----------
       file_name: name of file containing twitter raw data
       food_dict: food-related words for filtering tweets

       Returns
       -------
       data: dataframe with features of interest

    '''

    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from collections import Counter

    # load data
    with open(file_name, 'rb') as file:
        raw_data = file.read()
    all_tweets = pickle.loads(raw_data)

    # define needed objects
    # ---------------------
    # question-related words
    question_words = ["what", "why", "when", "where", 
                      "name", "is", "how", "do", "does", 
                      "which", "are", "could", "would", 
                      "should", "has", "have", "whom", "whose", "don't"]
    # encoder
    enc = sklearn.preprocessing.OrdinalEncoder(dtype=np.int)
    # ---------------------

    # set up dataframe object and related needed quantities
    # -----------------------------------------------------
    # define dataframe's columns
    data = DataFrame(columns=['text', 'favorite_count', 'retweet_count', 
                              # 'replies_count', 
                              'popularity',
                              'followers_count', 'friends_count', 'listed_count', 'statuses_count', 
                              'mentions_tot_followers', 'mentions_tot_friends', 'mentions_tot_statuses',
                              'day', 'time',
                              'n_sentences', 'n_words', 'n_adjectives', 'n_adverbs', 'n_nouns', 'n_pronouns', 'n_verbs', 'n_long_words',
                              'n_hashtags', 'n_user_mentions', 
                              'has_symbols', 'has_urls', 'has_photo', 'has_video', 'has_gif', 'has_questions', 'is_reply'])
    # dummy var needed to create a new row in the dataframe at each step
    n_row = 0                        
    # -----------------------------------------------------

    # loop over previously downloaded data
    # ------------------------------------
    for tweet in all_tweets:

        if (not tweet.retweeted) and ('RT @' not in tweet.full_text): # and (not tweet.in_reply_to_status_id_str != None):
            
            # get decoded tweet's text
            tweets_encoded = tweet.full_text.encode('utf-8')
            tweets_decoded = tweets_encoded.decode('utf-8')

            # preprocess and store tweet in dataframe only if contains food-related words 
            if  foodtweet_cleaner(food_dict, tweets_decoded):

                # count number of sentences in the tweet
                sent_tokens = sent_tokenize(tweets_decoded)
                n_sentences = len(sent_tokens)
 
                # detect questions in tweet
                has_questions = False
                for sentence in sent_tokens:
                    tokens = word_tokenize(sentence.lower())
                    is_question_mark = tokens[-1] == '?'
                    is_question_related = any(q_word in tokens for q_word in question_words)
                    # if question lacks of '?' character (due to mistake)
                    is_question_related_0 = any(q_word in tokens[0] for q_word in question_words)
                    if (is_question_mark and is_question_related) or is_question_related_0:
                        has_questions = True
                        break

                # count number of words in the tweet
                n_word = len(tweets_decoded.split())

                # count number of adjectives, adverbs, nouns and verbs in tweet
                text_tokens = word_tokenize(tweets_decoded.lower())
                tagged = nltk.pos_tag(nltk.Text(text_tokens))
                counts = Counter(tag for word,tag in tagged)

                n_adj = get_number_sentence_items(['JJ', 'JJR', 'JJS'], counts)
                n_adv = get_number_sentence_items(['RB', 'RBR', 'RBS'], counts)
                n_noun = get_number_sentence_items(['NN', 'NNP', 'NNS', 'NNPS'], counts)
                n_prnoun = get_number_sentence_items(['PRP', 'PRP$'], counts)
                n_verb = get_number_sentence_items(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], counts)
                # replies_count = get_number_replies(api, tweet.user.name, tweet.id)


                # Encode days of week and time zones of day in periodic fashion 
                day, time = factorizeDate(tweet.created_at)

                # detect media
                if tweet.entities.get('media') is not None:
                    has_photo = True in [medium['type'] == 'photo' for medium in tweet.entities['media']]
                    has_video = True in [medium['type'] == 'video' for medium in tweet.entities['media']]
                    has_gif = True in [medium['type'] == 'animated_gif' for medium in tweet.entities['media']]
                else:
                    has_photo, has_video, has_gif = False, False, False
                
                if api is not None:
                    # If other user are mentioned, add the metrics of the other users (as a cumulative sum)
                    mentions_tot_followers, mentions_tot_friends, mentions_tot_statuses = 0, 0, 0
                    if len(tweet.entities.get('user_mentions')):
                        for user_mentions in tweet.entities['user_mentions']:
                            try:
                                user_mentioned = api.get_user(screen_name=f"{user_mentions['screen_name']}")
                                mentions_tot_followers += user_mentioned.followers_count
                                mentions_tot_friends += user_mentioned.friends_count
                                mentions_tot_statuses += user_mentioned.statuses_count
                            except:
                                continue
                
                # detect replies
                if tweet.in_reply_to_status_id is not None:
                    # Tweet is a reply
                    is_reply = True
                else:
                    # Tweet is not a reply
                    is_reply = False

                # create row in dataframe
                data.loc[n_row] = [tweets_decoded,
                                   tweet.favorite_count, tweet.retweet_count, 
                                   # replies_count, 
                                   tweet.favorite_count + tweet.retweet_count,
                                   tweet.user.followers_count, tweet.user.friends_count, tweet.user.listed_count, tweet.user.statuses_count,
                                   mentions_tot_followers, mentions_tot_friends, mentions_tot_statuses, 
                                   day, time, 
                                   n_sentences, n_word, n_adj, n_adv, n_noun, n_prnoun, n_verb, count_long_words(tweets_decoded),
                                   len(tweet.entities['hashtags']), len(tweet.entities['user_mentions']), 
                                   len(tweet.entities['symbols']) > 0, len(tweet.entities['urls']) > 0,
                                   has_photo, has_video, has_gif, has_questions, is_reply]
                # setting new row's index for next iteration
                n_row += 1   
    # ------------------------------------

    for column in ['favorite_count', 'retweet_count', 'popularity', 'followers_count', 'friends_count', 'listed_count', 'statuses_count', 'n_pronouns', 'n_verbs', 'n_user_mentions', 'n_adverbs', 'n_hashtags', 'n_sentences', 'n_long_words', 'n_adjectives', 'n_words', 'n_nouns', 'mentions_tot_friends', 'mentions_tot_followers', 'mentions_tot_statuses']:
        data[column] = data[column].astype(float)

    return data

# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------

def factorizeDate(date): 
    ''' substitutes the date/time of creation of a tweet with a categorical variable indicate a time zone duringe the day
        Parameters
        ----------
        date: date to be factorized

        Returns
        -------
        day: number identifying a day of the week (e.g. Monday -> 1, Tuesday -> 2, etc.)
        time: time zone of the day
    '''
    # define daily time zones
    morning_time = ('06:00:00+00:00','11:59:59+00:00')
    morning_time_start_obj = datetime.datetime.strptime(morning_time[0], '%H:%M:%S%z').time()
    morning_time_end_obj = datetime.datetime.strptime(morning_time[1], '%H:%M:%S%z').time()

    afternoon_time = ('12:00:00+00:00','17:59:59+00:00')
    afternoon_time_start_obj = datetime.datetime.strptime(afternoon_time[0], '%H:%M:%S%z').time()
    afternoon_time_end_obj = datetime.datetime.strptime(afternoon_time[1], '%H:%M:%S%z').time()

    evening_time = ('18:00:00+00:00','23:59:59+00:00')
    evening_time_start_obj = datetime.datetime.strptime(evening_time[0], '%H:%M:%S%z').time()
    evening_time_end_obj = datetime.datetime.strptime(evening_time[1], '%H:%M:%S%z').time()

    night_time = ('00:00:00+00:00','05:59:59+00:00')
    night_time_start_obj = datetime.datetime.strptime(night_time[0], '%H:%M:%S%z').time()
    night_time_end_obj = datetime.datetime.strptime(night_time[1], '%H:%M:%S%z').time()

    # factorize time of tweet
    if date.time() >= morning_time_start_obj and date.time() <= morning_time_end_obj:
        time = 'morning'
    elif date.time() >= afternoon_time_start_obj and date.time() <= afternoon_time_end_obj:
        time = 'afternoon'
    elif date.time() >= evening_time_start_obj and date.time() <= evening_time_end_obj:
        time = 'evening'
    elif date.time() >= night_time_start_obj and date.time() <= night_time_end_obj:
        time = 'night'

    # factorize day of tweet
    day = pd.Timestamp(date.date()).weekday()

    # factorized_dates = Categorical(factorized_dates, categories=["morning","afternoon","evening", "night"], ordered=True)
    # day = Categorical(day, categories=["Monday","Tuesday","Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ordered=True)

    return day, time

# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------

def dictionary_downloader(link):
    '''Function which, given a link, download the whole content of an html webpage
        Parameters
        ----------
        link: the link to the webpage

        Returns
        -------
        The content of the page as a string
    '''
    import bs4
    import urllib.request

    webpage = str(urllib.request.urlopen(link).read())
    soup = bs4.BeautifulSoup(webpage, features="html.parser")

    return soup.get_text()


def api_getter(consumer_key, consumer_secret, access_token, access_token_secret):
    import tweepy as tw

    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
    return api


def page_analyzer(page, words_list):
    '''Fuction which analyse an html webpage dictionary in the format of a string, and return a list of the
    entries of the dictionary.
        Parameters
        ----------
        page: the string conteining the page
        word_list: list of the entries of the dictionary(can be an empty list)

        Returns
        -------
        List of the entries of the web dictionary
    '''

    for i in range(4, len(page)):
        word = ""
        j = 1
        if page[i] == 't' and page[i - 2] == 't' and page[i - 4] == 'n' and page[i - 6] == 'r' and page[i - 8] == 't':
            while page[i + j] != ":" and page[i + j] != "(" and (
                    page[i + j] == "\\" and page[i + j + 1] == "t") == False:
                if page[i + j] == "\\":
                    j += 1
                    continue
                word = word + page[i + j]
                j += 1
            words_list.append(word)
            i = i + j
    return words_list


def my_specific_dictionary():
    '''Function which specifically generate the dictionary of our interest
        Parameters
        ----------
        Returns
        -------
        The dictionary we want for our project
    '''
    import string
    alphabet = list(string.ascii_lowercase)
    del alphabet[0]
    food_words = []

    for k in alphabet:
        link = f'https://theodora.com/food/culinary_dictionary_food_glossary_{k}.html'
        page = dictionary_downloader(link)
        page_analyzer(page, food_words)

    link = "https://theodora.com/food/index.html"
    page = dictionary_downloader(link)
    page_analyzer(page, food_words)

    # for i in range(len(food_words)):
    #     food_words[i] = food_words[i].lower()

    return food_words


def foodtweet_cleaner(dictionar, long_string):
    '''Function which, given a sting and a set of patternes, return if any of the patterns is present in the string
        Parameters
        ----------
        dictionar: List of patterns we want to find in the string
        long_string: The string we want to find patterns into

        Returns
        -------
        True or False, whether any pattern is found in the string
    '''
    import re
    word = re.compile(r'\b|\b'.join(dictionar), re.IGNORECASE).search(long_string)
    if word:
        return True
    else:
        return False

def get_number_replies(api, user_name, tweet_id):
    '''counts the number of replies to a tweet of a given user
        Parameters
        ----------
        api: twitter API object
        user_name: name of the target account
        tweet_id: id of the tweet which replies are to be counted 

        Returns
        -------
        n_replies: number of replies to the target tweet
    '''

    t = api.search_tweets(q=f'to:{user_name}', since_id=tweet_id)

    n_replies = 0
    for i in range(len(t)):

        if t[i].in_reply_to_status_id == tweet_id:
            n_replies += 1
    
    return n_replies

def get_number_sentence_items(types, counts):
    '''get the number of a given type of word in a text, given a dictionary of tagged words
        Parameters
        ----------
        types: types of words to be counted
        counts: dictionary of tagged words

        Returns
        -------
        n_items: total number of words of a given type
    '''

    n_items = 0
    for tipe in types:
        if counts.get(tipe) is not None:
            n_items += counts.get(tipe)

    return n_items

def count_long_words(text):
    '''counts number of word in the text
        Parameters
        ----------
        text: string possibly containing long words

        Returns
        -------
        count: number of long words detected in text
    '''

    length = 10
    count = 0
    for word in text.split():
        if len(word) > length - 1:
            count += 1
    
    return count

# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------

def mostFrequentWords(data, n_most_frequent=10):
    '''computes Bag-of-Words for the dataset 'data'

       Parameters
       ----------
       data: dataframe of interest

       Returns
       -------
       data: passed dataframe extended with columns stating if corresponding 'text' contains or not
             the most frequent words in all texts

    '''

    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from collections import Counter
    
    # compute Bag-of-Words 
    words_dict = computeMostFrequentWords(data['text'], n_most_frequent)
    words_features = []
    for word in words_dict.keys():
        words_features += ['has_{}'.format(word)]

    
    # retrieve n_most_frequent features to insert into dataframe
    frequent_words_df = pd.DataFrame(columns=words_features)
    n_row = 0
    for tweet in data['text']:
        text_tokens = word_tokenize(tweet.lower())
        for i in range(len(text_tokens)):
            text_tokens[i] = text_tokens[i].replace('-', ' ').split()
        text_tokens = sum(text_tokens, [])
        frequent_words_df.loc[n_row] = [(word in text_tokens) for word in words_dict.keys()]
        n_row += 1
    frequent_words_df.reset_index(drop=True, inplace=True)
    
    data.reset_index(drop=True, inplace=True)
    data = pd.concat([data, frequent_words_df], axis=1)

    return data 

def computeMostFrequentWords(list_of_texts, n_most_frequent):
    ''' computes Bag-of-Words

        Parameters
        ----------
        list_of_texts: list of sentences/pieces of text 
        n_most_frequent: number of most frequent words to retrieve

        Returns
        -------
        largest_counts_dict: dictionary which keys are the most frequent words in dataset
                             and which values are their counts
    '''

    from nltk.corpus import stopwords
    
    count_vectorizer = CountVectorizer(ngram_range=(1,1), # to use bigrams ngram_range=(2,2)
                                       stop_words=stopwords.words('english')+['https', 'co'])

    # fit and transform on data
    count_data = count_vectorizer.fit_transform(list_of_texts)

    # create dataframe
    count_df = pd.DataFrame(count_data.toarray(), columns=count_vectorizer.get_feature_names_out())

    # get counts of words
    counts = dict.fromkeys(count_df.columns.tolist())
    for column in count_df:
        counts[column] = sum(count_df[column])
    
    # get most frequent words and create a dict with their counts
    largest_counts_idx = (-np.array(list(counts.values()))).argsort()[:n_most_frequent].tolist()
    most_frequent_words = count_df.iloc[:,largest_counts_idx].columns.tolist()
    largest_counts_dict = dict.fromkeys(most_frequent_words)
    for column in largest_counts_dict:
        largest_counts_dict[column] = counts[column]
    
    return largest_counts_dict

# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------

def getPopularityIndex(data, n_percentiles=4):
    '''returns popularity index based on which percentile of the 
       'popularity' variable's distribution the value belongs to

       Parameters
       ----------
       data: dataframe of interest
       n_percentiles: number of percentiles desired to compute the index

       Returns
       -------
       popularity_index: vector of computed indeces of popularity

    '''

    # get percentiles of variable 'popularity'
    p = get_percentiles(data['popularity'], n_percentiles)
    # compute popularity index
    popularity_index = [np.argmin(np.append(p, max(data['popularity'])) < popularity) + 1 for popularity in data['popularity']]
    # popularity_index = [np.argmax(np.append(p, max(data['popularity'])) > popularity) + 1 for popularity in data['popularity']]

    return popularity_index

def get_percentiles(points, n_percentiles):
    '''computes percentiles given a vector of points
        Parameters
        ----------
        points: vector of points over which percentiles are computed
        n_percentiles: number of percetiles point requestes

        Returns
        -------
        vector containing the percentiles
    '''

    return np.percentile(points, np.arange(0, 100, 100/n_percentiles))

# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------

def encodePandas(data, popularity_index):
    '''performs one hot encoding on categorical variables through pandas' utilities

       Parameters
       ----------
       data: dataframe of interest

       Returns
       -------
       data: dataframe with encoded categorical variables

    '''

    # detect categorical columns of the datafram
    columns = data.columns
    numerical_columns = data._get_numeric_data().columns
    other_columns = ['text']
    categorical_columns = list(set(columns) - set(numerical_columns) - set(other_columns))

    # loop over categorical columns and replace them with corresponding one hot encoding
    for column in categorical_columns:

        encoding = pd.get_dummies(data[column], prefix=column)
        data = pd.concat([data, encoding], axis=1)
        data.drop([column], axis=1, inplace=True)
    
    popularity_encoded = pd.get_dummies(popularity_index, prefix='popularity')
    
    return data, popularity_encoded

def encodeSklearn(data, y):
    '''performs one hot encoding on categorical variables through sklearn' utilities

       Parameters
       ----------
       data: dataframe of interest

       Returns
       -------
       data: one-hot-encoded dataframe

    '''

    from sklearn.preprocessing import OneHotEncoder

    data_array = np.array(data)
    y_array = np.array(y).reshape(-1,1)

    encoder = OneHotEncoder(handle_unknown='ignore', drop='first')
    
    return encoder.fit_transform(data_array).toarray(), encoder.fit_transform(y_array).toarray(), encoder

# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------

def dropZeros(data, frac_to_remove=0.8):
    '''removes frac_to_remove*100% rows with popularity equal to 0 
       
       Parameters
       ----------
       data: dataframe to "clean"
       frac_to_remove: fraction of rows with popularity = 0 to remove

       Return
       ------
       data: "cleaned" dataframe

    '''

    # find indeces of rows with popularity = 0
    where_zeros, = np.where(np.array(data['popularity']) == 0)
    where_zeros = where_zeros.tolist()

    # get frac_to_remove*100% indeces of those above
    indeces_to_remove = random.sample(where_zeros, int(frac_to_remove*len(where_zeros)))

    # remove 
    data = data.drop(indeces_to_remove)

    return data

# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------

def cleanData(data):
    '''removes from dataframe not needed columns 
       
       Parameters
       ----------
       data: dataframe of interest

       Returns
       -------
       cleaned dataframe
    '''

    columns_to_drop = ['text', 'favorite_count', 'retweet_count', 'popularity']
    data_columns = list(data.columns)

    if not any(column in columns_to_drop for column in data_columns):
        raise KeyError()('dataframe has already been cleaned')
    
    data.drop(columns_to_drop, axis=1, inplace=True)

    return data
