import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import time
import sys


def model_creating():

	print len(vect.get_feature_names())
	X_train_vetorised=vect.transform(X_train)

	print "starting training!!!!!"
	print "Stage 1 is completed"
	model.fit(X_train_vetorised,y_train)
	print "Stage 2 is completed"
	predictions=model.predict(vect.transform(X_test))
	print "Stage 3 is completed"
	print ("AUC:",accuracy_score(y_test,predictions))
	feature_name=np.array(vect.get_feature_names())
        sort_coeff=model.coef_[0].argsort()
        print ("small coeff : {}",format(feature_name[sort_coeff[:10]]))
        print ("large coeff : {}",format(feature_name[sort_coeff[:-11:-1]]))
	print time.ctime()
	print ("Time taken for execution is ",time.time()-start_time)

def testing():
	testing=raw_input("Enter the sentence for testing: ")
	print(model.predict(vect.transform([testing])))


if __name__ == "__main__":
	print "Please ensure that training file should be with column headers as 'sentiment' for labels and 'phrase' for text "
	#print "Pass the options for classification \t a. Bi-Classification \t b. Multi-Classification"
	global model
	global predictions	
	print time.ctime()
	start_time=time.time()
        
        df = pd.read_csv(sys.argv[-1], header=0, delimiter="\t", quoting=3)
	df.dropna(inplace=True)

	
	X_train, X_test, y_train, y_test = train_test_split(df['review'],df['sentiment'],random_state=0)
	model=LogisticRegression(multi_class='multinomial', solver='newton-cg')
	vect=CountVectorizer(min_df=5,ngram_range=(1,2)).fit(X_train)
	model_creating()
	testing()
	
		


