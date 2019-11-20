from __future__ import absolute_import
from __future__ import print_function

from rake_nltk import Rake
import pandas as pd
import rake


import six


#r.extract_keywords_from_text("The Beta Blocker Heart Attack Trial (BHAT) is a multicenter, randomized, double-blind, placebo control clinical trial sponsored by the National Heart, Lung, and Blood Institute designed to test the effectiveness of regular propranolol administration in reducing total mortality in patients who have survived a recent acute myocardial infarction., A number of other fatal and nonfatal response variables are also being monitored., Three thousand eight hundred thirty-seven patients, ages 30-69, are being followed at 31 clinical centers for a minimum of about 2 and a maximum of 4 years after the infarction., A number of design features of BHAT are discussed., These include maintenance of patient logs, guidelines for obtaining informed consent of patients, assessment of patient knowledge about BHAT, adjustment of study drug dose based on serum levels, and comparison of 1-hr and 24-hr ambulatory electrocardiogram readings., Â© 1981.")

#print(r.get_ranked_phrases()) # To get keyword phrases ranked highest to lowest.
r = Rake() 

filename = '/Users/iqra/Downloads/testing-data.csv'

stop_words_file = '/Users/iqra/Documents/Rake Examples/RAKE-tutorial/data/stoplists/SmartStoplist.txt'

rake_object = rake.Rake(stop_words_file)

data_frame = pd.read_csv(filename , encoding='latin1',engine='c')


# generate candidate keywords
stopwords = rake.load_stop_words(stop_words_file)
stopwordpattern = rake.build_stop_word_regex(stop_words_file)


def rake_implement(x,r):
	r.extract_keywords_from_text(x)
	return r.get_ranked_phrases()


data_frame['phrases'] = data_frame['abstract'].apply(lambda x:rake_implement(x,r))
print(data_frame['phrases'])
	







