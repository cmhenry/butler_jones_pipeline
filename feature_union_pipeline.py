## Colin Henry, University of New Mexico (2016)
## Pipeline classifier for human rights report text data;
## coding scheme from Butler and Jones (2016).

## Transform corpus of text data into easily readable file.
## Integrate class scores from XLS coding document.

import sys # System level operations
import os # OS operations
import xlrd # Reading Excel documents
import pandas as pd # Pandas data analysis libraries
import numpy as np # NumPy data analysis libraries
import os # OS access libraries
import sklearn.pipeline as Pipeline # sk-learn Pipeline library

args1 = sys.argv[1]

def TextConverter(file_dir) : 
	'''
	Import corpus (.txt) and one keyfile (.xls). Assumes keyfile headers. Assumes corpus file titles.
	Collapses into Panda DataFrame object. Saves as csv/tsv/something.
	''' 
	for root,dirs,files in os.walk(file_dir) : 
		corpus = [ fi for fi in files if fi.endswith(".txt") ]
		key_file = [ fi for fi in files if fi.endswith(".xls") ]

		data = {'country' : [],
				'year' : [], 
				'source' : [],
				'text' : []}
		df = pd.DataFrame(data, columns=['country','year','source','text'])

		for fname in corpus : 
			path = os.path.join(root,fname)
			with open(path) as infile : 
				raw_text = unicode(infile.read(),encoding='utf-8',errors='ignore')
			fname_split = fname.split('_')
			temp_df = pd.Series([fname_split[0],fname_split[1],fname_split[2],raw_text],
								index = ['country', 'year', 'source', 'text'])
			df = df.append(temp_df, ignore_index = True)

		# print df.loc[(df['country']=="UKR"), ['country', 'year', 'source', 'text', 'score']]

		for fname in key_file : 
			path = os.path.join(root,fname)
			key_df = pd.read_excel(path, sheetname='Sheet1', converters={'year':str})

		df = pd.merge(key_df, df, on = ['country','year'], how='left')
		print df

def TextImporter(file) : 
	'''
	Import data file from directory. Should be easily convertible to Panda DataFrame. Should include at least
	one identifying field, one feature field, and one categorization field. 
	e.g., "Afghanistan", "Observers reported instances of sexual assault by security forces.", "1". 
	'''
'''
pipeline = Pipeline([
	('import_text', TextImporter()),
	('counts', CountVectorizer()),
	('tf_idf', TfidfVectorizer()),
	('classifier', Classifier())
])
'''
TextConverter(args1)

