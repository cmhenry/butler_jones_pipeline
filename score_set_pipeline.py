
import sys # System level operations
import os # OS operations
from sets import Set
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
import sklearn.cross_validation as cv
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import multiclass
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

train_dir = sys.argv[1]

class Score_Set(object) : 
	def __init__(self,score,text,country,year) : 
		self.country = country
		self.year = year
		self.score = score
		self.text = text

def import_text(file_dir) : 
	for root,dirs,files in os.walk(file_dir) : 
		# Filter file lists by file type
		reports = [ fi for fi in files if fi.endswith(".txt") ]
		key_file = [ fi for fi in files if fi.endswith(".xls") ]
		
		# Create empty Numpy array with file list length
		train_set = [] * len(reports)
		train_set2 = [] * len(reports)
		# print train_set
		# index = 0 
		for fname in reports : 
			path = os.path.join(root,fname)
			with open(path) as infile : 
				raw_text = unicode(infile.read(),encoding='utf-8',errors='ignore')
			fname_split = fname.split('_')
			# print fname_split
			tmp = Score_Set(None,raw_text,fname_split[0],fname_split[1])
			# print tmp.country
			train_set.append(tmp)
			# print fname
			# index = index + 1
		# train_set.pop(0)
		# train_np = np.array(train_set)
		# print train_np
		for fname in key_file : 
			path = os.path.join(root,fname)
			workbook = xlrd.open_workbook(path, on_demand = True)
			worksheet = workbook.sheet_by_index(0)
			for row in range(0, worksheet.nrows) : 
				if row == 0 : 
					continue
				else : 
					tmp = Score_Set(None,None,None,None)
					for col in range(0, worksheet.ncols) : 
						if col == 0 : 
							continue
						elif col == 1 : 
							tmp.year = worksheet.cell(row,col).value
						elif col == 2 : 
							tmp.country = worksheet.cell(row,col).value
						elif col == 3 : 
							continue
						elif col == 4 : 
							tmp.score = worksheet.cell(row,col).value
						else : 
							continue
					train_set2.append(tmp)
	for t in train_set : 
		for t2 in train_set2 : 
			if float(t.year) == float(t2.year) and t.country == t2.country :
				# print "y"
				t.score = str(t2.score)
	# print [s.score for s in train_set]
	# print len(train_set)
	return train_set
	# Combining stuff
	# dictionary = {attr1, attr2 : }


def split_corpus(train_set) :
	corpus = []
	training = []
	for t in train_set : 
		if t.score is not None : 
			training.append(t)
		else : 
			corpus.append(t)
	# print len(corpus)
	# print len(training)
	return corpus, training

def sample_training_set(train_set) : 
	training = []
	testing = []
	half = int(len(train_set) / 2)
	# print 0,half,len(train_set)
	training = train_set[0:half+1]
	testing = train_set[half:len(train_set)]
	# print len(training),len(testing)
	return training, testing

def import_corpus(corpus_dir) : 
	corpus = sklearn.datasets.load_files(corpus_dir)
	return corpus

def feature_text(train_set) :
	# Extract counts
	# print train_set
	count_vect = CountVectorizer()
	train_counts = count_vect.fit_transform([s.text for s in train_set])
	# print train_counts
	# Extract term frequencies
	# tf_transformer = TfidfTransformer(use_idf=False).fit(train_counts)
	# train_tf = tf_transformer.transform(train_counts)
	# Extract term frequency times inverse document frequency
	tfidf_transformer = TfidfTransformer()
	train_tfidf = tfidf_transformer.fit_transform(train_counts)

	# tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
	# train_vector = tfidf_vectorizer.fit_transform([s.text for s in train_set])

	# sample_counts = count_vect.transform([s.text for s in sample_set])
	# sample_tfidf = tfidf_transformer.transform(sample_counts)
	# sample_vector = tfidf_vectorizer.fit_transform([s.text for s in sample_set])
	# sample_tf = tf_transformer.transform(sample_counts)
	# print train_tfidf
	return train_tfidf
def test_text(sample_set) :
	# Extract counts
	# print train_set
	count_vect = CountVectorizer()
	sample_counts = count_vect.transform([s.text for s in sample_set])
	# print train_counts
	# Extract term frequencies
	# tf_transformer = TfidfTransformer(use_idf=False).fit(train_counts)
	# train_tf = tf_transformer.transform(train_counts)
	# Extract term frequency times inverse document frequency
	tfidf_transformer = TfidfTransformer()
	sample_tfidf = tfidf_transformer.transform(sample_counts)
	# print train_tfidf
	return sample_tfidf

def train_text(sample_set, train_tfidf) : 
	# Train classifier
	sample_set = [s.score for s in sample_set]
	sample_set = label_binarize(sample_set, classes=[0, 1, 2, 3, 4])
	# BINARIZE HERE
	# print sample_set
	# print sample_set.shape
	# print train_tfidf.shape
	# classifier = MultinomialNB().fit(train_tfidf, sample_set)
	classifier = multiclass.OneVsRestClassifier(svm.SVC(kernel='linear',probability=True)).fit(train_tfidf, sample_set)
	# RETRAIN WITH OnevsRest classifier -- MULTILABEL -- SEE bookmarks
	# classifier = MultinomialNB(fit_prior=True).fit(train_tfidf, [s.score for s in sample_set])
	print classifier
	predicted = classifier.predict(train_tfidf)
	# x = [s.score for s in sample_set]
	# print x
	# np.set_printoptions(threshold=np.inf)
	# print x
	# print np.mean(predicted == x)
	return classifier


def analyze_pipeline(model, X, y, folds=3) : 
	# X, y, X_test = load() # Load model with own load function
	# y = y # Reload as numpy
	# y = np.array([Y.score for Y in y])
	y = [Y.score for Y in y]
	y = np.array(y)
	print y.shape
	y = label_binarize(y, classes=[0, 1, 2, 3, 4])
	print y.shape
	# y = label_binarize(y, classes=[0, 1, 2, 3, 4])
	# BINARIZE HERE
	# X = np.array # Reload as numpy
	# if not model: # If no model is specified, call load_model function
 #    	model = load_model()

	# Manual x-validation to accumulate actual
	# print y.shape
	cv_skf = KFold(5, n_folds=folds, shuffle=True, random_state=123)
	print cv_skf
	# y = np.array(y)
	# Creates stratified test set from training set
	scores = [] # Actual scores
	conf_mat = np.zeros((2, 2)) # Binary classification, confusion matrix
	false_pos = Set() # False positive set
	false_neg = Set() # Falso negative set

	for train_i, val_i in cv_skf:
	    X_train, X_val = X[train_i], X[val_i]
	    y_train, y_val = y[train_i], y[val_i]

	    print "Fitting fold..."
	    model.fit(X_train, y_train)

	    print "Predicting fold..."

	    y_pprobs = model.predict_proba(X_val)       # Predicted probabilities
	    y_plabs = np.squeeze(model.predict(X_val))  # Predicted class labels
	    print y_val
	    scores.append(roc_auc_score(y_val, y_pprobs[:, 1]))
	    confusion = confusion_matrix(y_val, y_plabs)
	    conf_mat += confusion

	    # Collect indices of false positive and negatives
	    fp_i = np.where((y_plabs==1) & (y_val==0))[0]
	    fn_i = np.where((y_plabs==0) & (y_val==1))[0]
	    false_pos.update(val_i[fp_i])
	    false_neg.update(val_i[fn_i])

	    print "Fold score: ", scores[-1]
	    print "Fold CM: \n", confusion

	print "\nMean score: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores) * 2)
	conf_mat /= folds
	print "Mean CM: \n", conf_mat
	print "\nMean classification measures: \n"
	pprint(class_report(conf_mat))
	return scores, conf_mat, {'fp': sorted(false_pos), 'fn': sorted(false_neg)}

def class_report(conf_mat):
    tp, fp, fn, tn = conf_mat.flatten()
    measures = {}
    measures['accuracy'] = (tp + tn) / (tp + fp + fn + tn)
    measures['specificity'] = tn / (tn + fp)        # (true negative rate)
    measures['sensitivity'] = tp / (tp + fn)        # (recall, true positive rate)
    measures['precision'] = tp / (tp + fp)
    measures['f1score'] = 2*tp / (2*tp + fp + fn)
    return measures

# def test_prediction(clf, test_set, test_feature_set) : 
# 	predicted = clf.predict(test_feature_set)
# 	x = [str(s.score) for s in test_set]
# 	# print x
# 	# print predicted
# 	print np.mean(predicted == x)

ugly_corpus = import_text(train_dir)
# y = import_corpus(corpus_dir)
corpus, training_set = split_corpus(ugly_corpus)
#sample_set, test_set = sample_training_set(training_set)

# sample_feature_set, test_feature_set = feature_text(sample_set,test_set)
sample_feature_set = feature_text(training_set)
#test_feature_set = test_text(test_set)

# clf = 
model = train_text(training_set,sample_feature_set)

analyze_pipeline(model,sample_feature_set,training_set)

#test_prediction(clf,test_set,test_feature_set)

