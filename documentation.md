## Documentation 2016-01-23

The purpose of this document is to communicate functional content of the program, as well as create roadmap for future development. It lives in the github repo and in the shared Dropbox of the program project. Intended audience includes developers Colin Henry and Christopher Butler, as well as other project partners. Note that cited code is often incomplete, in development, or contains excessive comments.

### score_set_pipeline.py

The score-set pipeline is an implementation of the NSF pilot project. It implements a `Score Set` object that contains values for `country`, `year`, `score`, and `text`. The `Score Set` object represents a single observation at the country-year-report level, where `score` is any human- or machine-generated SV score and `text` is any text-based feature extraction. This means that `text` can contain any potential feature, including a `string` of the entire report text, a `list` of sentences, or a `list` of keywords. 

The `Score Set` object:

```python
class Score_Set(object) : 
	def __init__(self,score,text,country,year) : 
		self.country = country
		self.year = year
		self.score = score
		self.text = text
```

The project accepts a sinle user input, `train_dir`, which indicates the directory containing both the corpus of reports and the human-coded training data. It also includes nine seperate functions.

`import_text`

`import_text` imports the corpus of reports and the human coded training data. It accepts a single input, `file_dir`, expected to be a string corresponding to the user input `train_dir`. 

```python
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
	```