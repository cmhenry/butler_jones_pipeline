
def analyze_pipeline(model, X, y, folds=10) : 
	# X, y, X_test = load() # Load model with own load function
	y = y.values # Reload as numpy
	X = X.values # Reload as numpy
	# if not model: # If no model is specified, call load_model function
 #    	model = load_model()

	# Manual x-validation to accumulate actual
	cv_skf = StratifiedKFold(y, n_folds=folds, shuffle=False, random_state=42)
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
