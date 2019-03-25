#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:23:44 2019

@author: silviu
"""
import pandas as pd
import numpy as np
import seaborn as sns
import folium
import matplotlib.pyplot as plt
import re
import nltk
#nltk.download('stopwords')
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
from nltk.util import ngrams
import re
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 


def ProcessTextData(sentence):
    text_data = sentence
    text_data = text_data.lower()
    #removes unicode strings
    text_data = re.sub(r'(\\u[0-9A-Fa-f])', r'', text_data)
    text_data = re.sub(r'[^\x00-\x7f]', r'', text_data)
    #convert any url to URL
    text_data = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text_data)
    #convert any @user to AT_USER
    text_data = re.sub('@[^\s]+', 'AT_USER', text_data)
    #remove additional white spaces
    text_data = re.sub('[\s]+', ' ', text_data)
    text_data = re.sub('[\n]+', ' ', text_data)
    #remove not alphanumeric symbols white spaces
    text_data = re.sub(r'[^\w]+', ' ', text_data)
    #remove numbers
    text_data = ''.join([i for i in text_data if not i.isdigit()])
    
    lemmatizer = WordNetLemmatizer()
    
    word_list = nltk.word_tokenize(text_data)
    
    stop_words = set(stopwords.words('english')) 
    
    filtered_sentence = [w for w in word_list if not w in stop_words]
    
# Lemmatize list of words and join
    text_data = ' '.join([lemmatizer.lemmatize(w) for w in filtered_sentence])

    
    #lemmatize
#    text_data = " ".join([Word(word).lemmatize for word in text_data.split()])
    #stemmer
#    st = PorterStemmer()
#   text_data = " ".join([st.stem(word) for word in text_data.split()])
    
    sentence = text_data
    return sentence


data_dirs = ['apr2018', 'may2018', 'jul2018', 'aug2018', 'sep2018', 'oct2018', 'nov2018', 'dec2018', 'jan2019', 'feb2019', 'mar2019']

data_frames = {}

for grp in data_dirs:
    file_name = '../data/' + grp + '/listings.csv.gz'
    data_frames[grp] = pd.read_csv(file_name, low_memory=False, compression='gzip')
    
cols_2108 = list(data_frames['apr2018'].columns)
cols_2019 = list(data_frames['feb2019'].columns)

diff_list = list(set(cols_2019) - set(cols_2108))

#print(diff_list)

#we have a difference between 2018 and 2019 data 
#drop all columns that are not contained in both sets
"""
for frame in data_frames:
    if (any(x in data_frames[frame].columns for x in diff_list)):
        data_frames[frame].drop(labels=diff_list, axis=1)
"""    
dataframes_list = []

for frame in data_frames:
    dataframes_list.append(data_frames[frame])
    
master_df = pd.concat(dataframes_list, axis=0, join='outer', join_axes=None, ignore_index=False,
          keys=None, levels=None, names=None, verify_integrity=False,
          copy=True, sort=False)

data_frames.clear()
dataframes_list.clear()

#master_df.drop(labels=diff_list, axis=1)

print(master_df.columns.values)

print()

#drop some features that don't seem interesting right now and generate nlp features vectors on others

diff_list.append('listing_url')
diff_list.append('scrape_id')
diff_list.append('last_scraped')
diff_list.append('space')
diff_list.append('experiences_offered')
diff_list.append('neighborhood_overview')
diff_list.append('notes')
diff_list.append('transit')
diff_list.append('access')
diff_list.append('interaction')
diff_list.append('house_rules')
diff_list.append('thumbnail_url')
diff_list.append('medium_url')
diff_list.append('picture_url')
diff_list.append('xl_picture_url')
diff_list.append('host_url')
diff_list.append('host_about')
diff_list.append('host_response_time')
diff_list.append('host_response_rate')
diff_list.append('host_acceptance_rate')
diff_list.append('host_thumbnail_url')
diff_list.append('host_picture_url')
diff_list.append('host_neighbourhood')
diff_list.append('host_verifications')
diff_list.append('host_has_profile_pic')
diff_list.append('market')
diff_list.append('smart_location')
diff_list.append('country_code')
diff_list.append('country')
diff_list.append('square_feet')
diff_list.append('weekly_price')
diff_list.append('monthly_price')
diff_list.append('security_deposit')
diff_list.append('cleaning_fee')
diff_list.append('minimum_nights')
diff_list.append('maximum_nights')
diff_list.append('calendar_updated')
diff_list.append('has_availability')
diff_list.append('availability_30')
diff_list.append('availability_60')
diff_list.append('availability_90')
diff_list.append('calendar_last_scraped')
diff_list.append('number_of_reviews')
diff_list.append('first_review')
diff_list.append('last_review')
diff_list.append('review_scores_rating')
diff_list.append('review_scores_accuracy')
diff_list.append( 'review_scores_cleanliness')
diff_list.append('review_scores_checkin')
diff_list.append('review_scores_communication')
diff_list.append('review_scores_location')
diff_list.append('review_scores_value')
diff_list.append('requires_license')
diff_list.append('license')
diff_list.append('jurisdiction_names')
diff_list.append('instant_bookable')
diff_list.append('is_business_travel_ready')
diff_list.append('cancellation_policy')
diff_list.append('require_guest_profile_picture')
diff_list.append('require_guest_phone_verification')
diff_list.append('calculated_host_listings_count') 
diff_list.append('reviews_per_month')
diff_list.append('neighbourhood_group_cleansed')
diff_list.append('zipcode')
diff_list.append('city')
diff_list.append('state')

master_df.drop(diff_list, axis=1, inplace=True)

#df.drop(df.ix[:,'Unnamed: 24':'Unnamed: 60'].head(0).columns, axis=1)

print(master_df.columns.values)

print()

# % of NaN values
print((len(master_df)-master_df.count())/len(master_df)*100)

master_df['price']=master_df['price'].str.replace('[$,]','',regex=True).astype(float)
master_df['extra_people']=master_df['extra_people'].str.replace('[$,]','',regex=True).astype(float)
#we know that we have a few literal strings in the price colummn - we should replace these with zeros
print(master_df.dtypes)

print(master_df.isna().sum())
print()


master_df['name']=master_df['name'].fillna(' ')
master_df['summary']=master_df['summary'].fillna(' ')
master_df['description']=master_df['description'].fillna(' ')
master_df['host_name']=master_df['host_name'].fillna(' ')
master_df['host_since']=master_df['host_since'].fillna('0')
master_df['host_location']=master_df['host_location'].fillna(' ')
# if superhost is missing (6 records) I assume that they are not + change to boolean 0,1
master_df['host_is_superhost']=master_df['host_is_superhost'].fillna('f').map({'f':0,'t':1})
master_df['host_listings_count']=master_df['host_listings_count'].fillna(0)
master_df['host_total_listings_count']=master_df['host_total_listings_count'].fillna(0)
master_df['host_identity_verified']=master_df['host_identity_verified'].fillna('f').map({'f':0,'t':1})
master_df['neighbourhood']=master_df['neighbourhood'].fillna(' ')
master_df['bathrooms']=master_df['bathrooms'].fillna(0)
master_df['bedrooms']=master_df['bedrooms'].fillna(0)
master_df['beds']=master_df['beds'].fillna(1)
master_df['is_location_exact']=master_df['is_location_exact'].fillna('f').map({'f':0,'t':1})


#now let's see how many null and na records we have and try to fill tem up with something
print(master_df.isna().sum())

#sns.boxplot(x='bathrooms', y='price', data=master_df)
#print()
#sns.boxplot(x='bedrooms', y='price', data=master_df)
#print()
#sns.boxplot(x='beds', y='price', data=master_df)

master_df['name'] = master_df['name'].apply(ProcessTextData)
master_df['summary'] = master_df['summary'].apply(ProcessTextData)
master_df['description'] = master_df['description'].apply(ProcessTextData)
master_df['amenities'] = master_df['amenities'].apply(ProcessTextData)
#DATA CLEANING DONE _ LET"S EXTRACT SOME TEXT FEATURES

master_df.to_csv('clean_data.csv')

#print(master_df.bed_type.unique())

#master_df.loc[master_df['price'].apply(lambda x: x.isnumeric()), 'price'] = 0

#index_names = np.issubdtype(master_df['price'].dtype, np.number)

#indexNames = master_df[ master_df['price'].isnumeric() ].index 

print('Done')