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

def downloadTweets(api, users, file_name = 'all_tweets.pkl', min_number_of_followers=1e6):
    ''' downloads number 'count' tweets from users listed in 'users' 

        Parameters
        ----------
        api: twitter API object for download of tweets
        users: list of target twitter accounts
        file_name: name of file on which data are saved
        min_number_of_followers: lower bound to the number of followers of the given users from which
                                 one wants to download tweets

        Returns
        -------
        
    '''

    all_tweets = []
    tot_tweets_dwlded = 0

    # download tweets of given users
    print("downloading...")
    for user in users:

        user_object = api.get_user(screen_name = user)

        if user_object.followers_count >= min_number_of_followers:
            tweets = Cursor(api.user_timeline, screen_name = user, tweet_mode="extended").items(3200)
            #tweets = api.user_timeline(screen_name = user, count=20)
            new_tweets = []
            for status in tweets:
                new_tweets += [status]
            
            all_tweets.extend(new_tweets)
            tweets_dwlded = len(new_tweets)
            tot_tweets_dwlded += tweets_dwlded
            print('\t user {} -- {} tweets downloaded'.format(user, tweets_dwlded))
            print('\t\t we have {} tweets now...'.format(tot_tweets_dwlded))

    # saving to JSON
    print("saving tweets to file...")
    with open(file_name,'wb') as file: 
        pickle.dump(all_tweets, file)

def preprocessTweets(api, file_name, food_dict, n_percentiles=4):

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
                              #'mentions_tot_followers', 'mentions_tot_friends', 'mentions_tot_statuses',
                              #'user_name',
                              'day_sin', 'day_cos', 'time_sin', 'time_cos', 
                              'n_sentences', 'n_words', 'n_adjectives', 'n_adverbs', 'n_nouns', 'n_pronouns', 'n_verbs', 'n_long_words',
                              'n_hashtags', 'n_user_mentions', 
                              'has_symbols', 'has_urls', 'has_photo', 'has_video', 'has_gif', 'has_questions'])
    # dummy var needed to create a new row in the dataframe at each step
    n_row = 0                        
    # -----------------------------------------------------

    # loop over previously downloaded data
    # ------------------------------------
    for tweet in all_tweets:

        if (not tweet.retweeted) and ('RT @' not in tweet.full_text) and (not tweet.in_reply_to_status_id_str != None):
            
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

                day_sin = np.sin(day*2*np.pi/7)
                day_cos = np.cos(day*2*np.pi/7)

                time_sin = np.sin(time*2*np.pi/4)
                time_cos = np.cos(time*2*np.pi/4)

                # detect media
                if tweet.entities.get('media') is not None:
                    has_photo = True in [medium['type'] == 'photo' for medium in tweet.entities['media']]
                    has_video = True in [medium['type'] == 'video' for medium in tweet.entities['media']]
                    has_gif = True in [medium['type'] == 'animated_gif' for medium in tweet.entities['media']]
                else:
                    has_photo, has_video, has_gif = False, False, False
                
                # If other user are mentioned, add the metrics of the other users (as a cumulative sum)
                # mentions_tot_followers, mentions_tot_friends, mentions_tot_statuses = 0, 0, 0
                # if tweet.entities.get('user_mentions') is not None:
                #     for user_mentions in tweet.entities['user_mentions']:
                #         user_mentioned = api.get_user(screen_name=f"{user_mentions['screen_name']}")
                #         mentions_tot_followers += user_mentioned.followers_count
                #         mentions_tot_friends += user_mentioned.friends_count
                #         mentions_tot_statuses += user_mentioned.statuses_count

                # create row in dataframe
                data.loc[n_row] = [tweets_decoded,
                                    tweet.favorite_count, tweet.retweet_count, 
                                    # replies_count, 
                                    tweet.favorite_count + tweet.retweet_count,
                                   tweet.user.followers_count, tweet.user.friends_count, tweet.user.listed_count, tweet.user.statuses_count,
                                   #mentions_tot_followers, mentions_tot_friends, mentions_tot_statuses, 
                                   #tweet.user.name,
                                   day_sin, day_cos, time_sin, time_cos,  
                                   n_sentences, n_word, n_adj, n_adv, n_noun, n_prnoun, n_verb, count_long_words(tweets_decoded),
                                   len(tweet.entities['hashtags']), len(tweet.entities['user_mentions']), 
                                   len(tweet.entities['symbols']) > 0, len(tweet.entities['urls']) > 0,
                                   has_photo, has_video, has_gif, has_questions]
                # setting new row's index for next iteration
                n_row += 1   
    # ------------------------------------

    # compute Bag-of-Words 
    BoW_dict = computeBoW(data['text'])
    BoW_features = []
    for word in BoW_dict.keys():
        BoW_features += ['has_{}'.format(word)]

    
    # retrieve n_most_frequent features to insert into dataframe
    frequent_words_df = pd.DataFrame(columns=BoW_features)
    n_row = 0
    for tweet in data['text']:
        text_tokens = word_tokenize(tweet.lower())
        for i in range(len(text_tokens)):
            text_tokens[i] = text_tokens[i].replace('-', ' ').split()
        text_tokens = sum(text_tokens, [])
        frequent_words_df.loc[n_row] = [(word in text_tokens) for word in BoW_dict.keys()]
        n_row += 1
    for column in frequent_words_df:
        frequent_words_df[column] = sum(enc.fit_transform(np.array(frequent_words_df[column]).reshape(-1,1)).tolist(), [])

    # encode boolean columns
    for column in ['has_symbols', 'has_urls', 'has_photo', 'has_video', 'has_gif', 'has_questions']:
        data[column] = sum(enc.fit_transform(np.array(data[column]).reshape(-1,1)).tolist(), [])

    for column in ['favorite_count', 'retweet_count', 'popularity', 'followers_count', 'friends_count', 'listed_count', 'statuses_count']:
        data[column] = data[column].astype(float)
    
    data = pd.concat([data, frequent_words_df], axis=1)

    # get percentiles of variable 'popularity'
    p = get_percentiles(data['popularity'], n_percentiles)
    # compute popularity index
    popularity_index = [np.argmin(np.append(p, max(data['popularity'])) < popularity) + 1 for popularity in data['popularity']]
    # popularity_index = [np.argmax(np.append(p, max(data['popularity'])) > popularity) + 1 for popularity in data['popularity']]

    # data = data.drop('popularity', axis=1)

    return data, popularity_index        

def preprocessTweets_OHE(api, file_name, food_dict, n_percentiles=4):

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
    enc = sklearn.preprocessing.OneHotEncoder(dtype=np.int)
    # ---------------------

    # set up dataframe object and related needed quantities
    # -----------------------------------------------------
    # define dataframe's columns
    data = DataFrame(columns=['text', 'favorite_count', 'retweet_count', 
                              # 'replies_count', 
                              'popularity',
                              'followers_count', 'friends_count', 'listed_count', 'statuses_count', 
                              #'mentions_tot_followers', 'mentions_tot_friends', 'mentions_tot_statuses',
                              #'user_name',
                              'day_sin', 'day_cos', 'time_sin', 'time_cos', 
                              'n_sentences', 'n_words', 'n_adjectives', 'n_adverbs', 'n_nouns', 'n_pronouns', 'n_verbs', 'n_long_words',
                              'n_hashtags', 'n_user_mentions', 
                              'has_symbols', 'has_urls', 'has_photo', 'has_video', 'has_gif', 'has_questions'])
    # dummy var needed to create a new row in the dataframe at each step
    n_row = 0                        
    # -----------------------------------------------------

    # loop over previously downloaded data
    # ------------------------------------
    for tweet in all_tweets:

        if (not tweet.retweeted) and ('RT @' not in tweet.full_text) and (not tweet.in_reply_to_status_id_str != None):
            
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

                day_sin = np.sin(day*2*np.pi/7)
                day_cos = np.cos(day*2*np.pi/7)

                time_sin = np.sin(time*2*np.pi/4)
                time_cos = np.cos(time*2*np.pi/4)

                # detect media
                if tweet.entities.get('media') is not None:
                    has_photo = True in [medium['type'] == 'photo' for medium in tweet.entities['media']]
                    has_video = True in [medium['type'] == 'video' for medium in tweet.entities['media']]
                    has_gif = True in [medium['type'] == 'animated_gif' for medium in tweet.entities['media']]
                else:
                    has_photo, has_video, has_gif = False, False, False
                
                # If other user are mentioned, add the metrics of the other users (as a cumulative sum)
                # mentions_tot_followers, mentions_tot_friends, mentions_tot_statuses = 0, 0, 0
                # if tweet.entities.get('user_mentions') is not None:
                #     for user_mentions in tweet.entities['user_mentions']:
                #         user_mentioned = api.get_user(screen_name=f"{user_mentions['screen_name']}")
                #         mentions_tot_followers += user_mentioned.followers_count
                #         mentions_tot_friends += user_mentioned.friends_count
                #         mentions_tot_statuses += user_mentioned.statuses_count

                # create row in dataframe
                data.loc[n_row] = [tweets_decoded,
                                    tweet.favorite_count, tweet.retweet_count, 
                                    # replies_count, 
                                    tweet.favorite_count + tweet.retweet_count,
                                   tweet.user.followers_count, tweet.user.friends_count, tweet.user.listed_count, tweet.user.statuses_count,
                                   #mentions_tot_followers, mentions_tot_friends, mentions_tot_statuses, 
                                   #tweet.user.name,
                                   day_sin, day_cos, time_sin, time_cos,  
                                   n_sentences, n_word, n_adj, n_adv, n_noun, n_prnoun, n_verb, count_long_words(tweets_decoded),
                                   len(tweet.entities['hashtags']), len(tweet.entities['user_mentions']), 
                                   len(tweet.entities['symbols']) > 0, len(tweet.entities['urls']) > 0,
                                   has_photo, has_video, has_gif, has_questions]
                # setting new row's index for next iteration
                n_row += 1   
    # ------------------------------------

    # compute Bag-of-Words 
    BoW_dict = computeBoW(data['text'])
    BoW_features = []
    for word in BoW_dict.keys():
        BoW_features += ['has_{}'.format(word)]

    
    # retrieve n_most_frequent features to insert into dataframe
    frequent_words_df = pd.DataFrame(columns=BoW_features)
    n_row = 0
    for tweet in data['text']:
        text_tokens = word_tokenize(tweet.lower())
        for i in range(len(text_tokens)):
            text_tokens[i] = text_tokens[i].replace('-', ' ').split()
        text_tokens = sum(text_tokens, [])
        frequent_words_df.loc[n_row] = [(word in text_tokens) for word in BoW_dict.keys()]
        n_row += 1
    for column in frequent_words_df:
        encoding = enc.fit_transform(np.array(frequent_words_df[column]).reshape(-1,1)).toarray()
        frequent_words_df[column] = encoding.tolist()
        frequent_words_df[column] = frequent_words_df[column].apply(lambda x: np.array(x))

    # encode boolean columns
    for column in ['has_symbols', 'has_urls', 'has_photo', 'has_video', 'has_gif', 'has_questions']:
        encoding = enc.fit_transform(np.array(data[column]).reshape(-1,1)).toarray()
        data[column] = encoding.tolist()
        data[column] = data[column].apply(lambda x: np.array(x))

    for column in ['favorite_count', 'retweet_count', 'popularity', 'followers_count', 'friends_count', 'listed_count', 'statuses_count']:
        data[column] = data[column].astype(float)
    
    data = pd.concat([data, frequent_words_df], axis=1)

    # get percentiles of variable 'popularity'
    p = get_percentiles(data['popularity'], n_percentiles)
    # compute popularity index
    popularity_index = [np.argmin(np.append(p, max(data['popularity'])) < popularity) + 1 for popularity in data['popularity']]
    # popularity_index = [np.argmax(np.append(p, max(data['popularity'])) > popularity) + 1 for popularity in data['popularity']]

    # data = data.drop('popularity', axis=1)

    return data, popularity_index        

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
        # factorized_dates += ['morning']
        time = 0
    elif date.time() >= afternoon_time_start_obj and date.time() <= afternoon_time_end_obj:
        # time += ['afternoon']
        time = 1
    elif date.time() >= evening_time_start_obj and date.time() <= evening_time_end_obj:
        # time += ['evening']
        time = 2
    elif date.time() >= night_time_start_obj and date.time() <= night_time_end_obj:
        #time += ['night']
        time = 3

    # factorize day of tweet
    day = pd.Timestamp(date.date()).dayofweek

    # factorized_dates = Categorical(factorized_dates, categories=["morning","afternoon","evening", "night"], ordered=True)
    # day = Categorical(day, categories=["Monday","Tuesday","Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ordered=True)

    return day, time

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

def computeBoW(list_of_texts, n_most_frequent=10):
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
