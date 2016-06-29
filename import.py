
import sys # System level operations
import os # OS operations
import re # Regular expressions
import xlrd # Reading Excel documents
import glob # Filename pattern matching
import ntpath # Path and filesystem operations
import pandas as pd # Pandas data analysis library
import numpy as np # Numpy, scientific computing on python
from nltk.tokenize import sent_tokenize, word_tokenize # Natural language processing tokenizers
from nltk.stem import PorterStemmer # NLTK Porter stemmer
import sklearn # Sci-kit learn for machine learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

train_dir = sys.argv[1]
# corpus_dir = sys.argv[2]

class Score_Set(object) : 
	def __init__(self,score,text,country,year) : 
		self.country = country
		self.year = year
		self.score = score
		self.text = text

def import_text(file_dir) : 
	#print file_dir
	# Import documents and coding for machine learning purposes
	# Document list (txt) and coding results (xls) --
	# Documents single reports listed by filename with state or state abr., source, year
	# Coding results score with state or state abr., COW code, source, year

	# Documents imported into object with filename, full text, then processed
	# Code results imported line-by-line with country, year, source

	# # Method 1
	# for root,dirs,files in os.walk(file_dir) : 
	# 	# Filter file lists by file type
	# 	files = [ fi for fi in files if fi.endswith(".txt") ]
	# 	key_file = [ fi for fi in files if fi.endswith(".xls") ]
		
	# 	# Create empty Numpy array with file list length
	# 	train_set = np.empty(shape=(len(files),1),dtype="S10000")
	# 	# print train_set
	# 	index = 0 
	# 	for fname in files : 
	# 		path = os.path.join(root,fname)
	# 		with open(path) as infile : 
	# 			raw_text = unicode(infile.read(),encoding='utf-8',errors='ignore')
	# 		train_set[index] = (raw_text)
	# 		# print fname
	# 		index = index + 1
	# 	# print train_set.data

	# # return train_set

	# Method 2
	for root,dirs,files in os.walk(file_dir) : 
		# Filter file lists by file type
		reports = [ fi for fi in files if fi.endswith(".txt") ]
		key_file = [ fi for fi in files if fi.endswith(".xls") ]
		
		# Create empty Numpy array with file list length
		train_set = [] * len(reports)
		# file_set = [] * len(reports)
		# print train_set
		# index = 0 
		for fname in reports : 
			path = os.path.join(root,fname)
			with open(path) as infile : 
				raw_text = unicode(infile.read(),encoding='utf-8',errors='ignore')
			train_set.append(raw_text)
			# print fname
			# index = index + 1
		train_set.pop(0)
		# train_np = np.array(train_set)
		# print train_np
		for fname in key_file : 
			path = os.path.join(root,fname)
			workbook = xlrd.open_workbook(path, on_demand = True)
			worksheet = workbook.sheet_by_index(0)
			

	# print train_set
	return train_set
	# train_set = sklearn.datasets.load_files(file_dir)
	# return train_set
	# print train_set.data
	# count_vector = CountVectorizer()
	# train_counts = count_vector.fit_transform(train_set.data)
	# print train_counts

def extract_clean_sentences(raw_text):

        text = raw_text
        
        # Exclude non UTF chars
        exclude = re.compile('[^a-zA-Z0-9 \'-]')
        # Exclude linebreaks
        linebreaks = re.compile('\s')
        # Exclude excess spaces
        excess_space = re.compile('\s+')
        # stemmer = PorterStemmer()

        text = linebreaks.sub(' ', text)
        text = exclude.sub(' ', text)
        text = excess_space.sub(' ', text)

        # # Tokenizes sentences
        # sentences = sent_tokenize(text)
        # out = []
        # # Parses tokenized sentences
        # for sentence in sentences:
        #     # Regular expressions
        #     sentence = linebreaks.sub(' ', sentence)
        #     sentence = exclude.sub(' ', sentence)
        #     sentence = excess_space.sub(' ', sentence)
        #     # Word tokenize
        #     tokens = word_tokenize(sentence)
        #     # Stem words
        #     tokens = [stemmer.stem(t.lower()) for t in tokens]
        #     out.append(tokens)
        return text

def import_corpus(corpus_dir) : 
	corpus = sklearn.datasets.load_files(corpus_dir)
	return corpus

def feature_text(train_set) :
	# Extract counts
	# print train_set
	count_vect = CountVectorizer()
	train_counts = count_vect.fit_transform(train_set)
	# print train_counts
	# Extract term frequencies
	tf_transformer = TfidfTransformer(use_idf=False).fit(train_counts)
	train_tf = tf_transformer.transform(train_counts)
	# Extract term frequency times inverse document frequency
	tfidf_transformer = TfidfTransformer()
	train_tfidf = tfidf_transformer.fit_transform(train_counts)
	# print train_tfidf
	return train_tfidf

def train_text(train_set, train_tfidf) : 
	# Train classifier
	classifier = MultinomialNB().fit(train_tfidf, train_set.target)
	return classifier

def test_prediction(clf, corpus) : 
	predicted = clf.predict(corpus.data)
	np.mean(predicted == corpus.target)

x = import_text(train_dir)
#y = import_corpus(corpus_dir)
z = feature_text(x)
#clf = train_text(x,z)
#test_prediction(clf,y)

