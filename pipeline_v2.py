## Colin Henry, University of New Mexico
## Pipeline classifier for human rights report text data;
## coding scheme from Butler and Jones (2016).

## Transform corpus of text data into easily readable file.
## Integrate class scores from XLS coding document.

class Corpus(Object) : 

	# INIT function
	def __init__(self, field1) : 
		self.field1 = field1

	# Function to read in single file and add to corpus
	def import_file(input) : 
		# Code to import single file and add to corpus

	# Function to read in multiple files and add to corpus
	def import_corpus(input) : 
		# Code to import directory of files and add to corpus

	# Function to import scored keyfile
	def import_keyfile(input) : 
		# Code to import scored keyfile and define corpus

	# Function to export corpus
	def export_corpus(format,output) : 
		# Code to export corpus and save in a variety of fileformats

	# Function to return corpus raw text for processing
	def get_corpus() : 
		# Code to return corpus raw text for processing

	# Function to return keyfile for processing
	def get_keyfile() : 
		# Code to return keyfile for processing

class Pipeline() : 

	# INIT function
	def __init__(Self, field1) : 
		self.field1 = field1

	# Vectorizer Pipeline
	def vec_pipe() : 
		# Code for creating vectorizer pipeline

	


