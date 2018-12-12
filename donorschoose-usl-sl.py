import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import seaborn as sns

#import libraries
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from kmodes import kmodes
from kmodes import kprototypes

#Donation Data
donatedata = pandas.read_csv('./Donations.csv')

#Donor Data
donordata = pandas.read_csv('./Donors.csv', low_memory=False)

#Project Data
projectdata = pandas.read_csv('./Projects.csv')
print(projectdata.info())
print(projectdata['Project Cost'].describe())
#Bins for "Project Cost" can be put at 35, 336, 516, 868, & 260,000
#based on its statistical summary of 35.3, 335.1, 515.4, 867.5, &
#2.554e+05.

null_columns = projectdata.columns[projectdata.isnull().any()]
print(projectdata[null_columns].isnull().sum())

#Schools Data
schooldata = pandas.read_csv('./Schools.csv')

#==================
#Data Preparation
#=================
#Merging relevant columns of Donations.csv, Donors.csv, Projects.csv, &
#School.csv into one data frame.
donatdonor = pandas.merge(donatedata, donordata, how='inner', on='Donor ID')
dodonslice = donatdonor.loc[:, ['Project ID', 'Donor ID', 'Donation Included Optional Donation', 'Donation Amount', 'Donor State', 'Donor Is Teacher']]

ddproj = pandas.merge(dodonslice, projectdata, how='inner', on='Project ID')
ddprojslice = ddproj.loc[:, ['Project ID', 'Donor ID', 'Donation Included Optional Donation', 'Donation Amount', 'Donor State', 'Donor Is Teacher', 'School ID', 'Project Type', 'Project Subject Category Tree', 'Project Grade Level Category', 'Project Resource Category', 'Project Cost']]

print(ddprojslice.info())
null_columns = ddprojslice.columns[ddprojslice.isnull().any()]
print(ddprojslice[null_columns].isnull().sum())

ddprsch = pandas.merge(ddprojslice, schooldata, how='inner', on='School ID')
ddprschslice = ddprsch.loc[:, ['Project ID', 'Donor ID', 'Donation Included Optional Donation', 'Donation Amount', 'Donor State', 'Donor Is Teacher', 'Project Type', 'Project Subject Category Tree', 'Project Grade Level Category', 'Project Resource Category', 'Project Cost', 'School Metro Type', 'School State']]

print(ddprschslice.info())
null_columns = ddprschslice.columns[ddprschslice.isnull().any()]
print(ddprschslice[null_columns].isnull().sum())

#Filling both data frames' NaN's with "other"
#nanrepddprsch = ddprschslice.fillna("other")
nanrepddprsch = ddprojslice.fillna("other")
print(nanrepddprsch.shape)

null_columns = nanrepddprsch.columns[nanrepddprsch.isnull().any()]
print(nanrepddprsch[null_columns].isnull().sum())

print(nanrepddprsch.info())

#Slicing relevant columns of Projects.csv to perform K-Prototype on the merged
#dataframe.
#projschool = pandas.merge(projectdata, schooldata, how='inner', on='School ID')
projslice = projectdata.loc[:, ['Project ID', 'Project Subject Category Tree', 'Project Grade Level Category', 'Project Resource Category', 'Project Cost']]
print(projslice.info())

null_columns = projslice.columns[projslice.isnull().any()]
print(projslice[null_columns].isnull().sum())

nonan_projslice = projslice.fillna("other")
print(nonan_projslice.shape)

null_columns = nonan_projslice.columns[nonan_projslice.isnull().any()]
print(nonan_projslice[null_columns].isnull().sum())

print(nonan_projslice.info())

#==========================================================================
#Unsupervised Learning - Using K-Prototypes for the mixed project data set
#==========================================================================
#Apply the clustering method for mixed data type for the project data
#set. Also, in order to test the validity of this method will be
#splitting the data into a smaller portion so that the algorithm runs
#faster and its validity can be checked.
#usl_finaldf_small = nonan_projslice.loc[0:100000]

from scipy import stats

#Standardize numeric columns according to their normal distribution
#and z-score.
clmns = ['Project Cost']
nonan_projslice[clmns] = stats.zscore(nonan_projslice[clmns])
#usl_finaldf_small[clmns] = stats.zscore(usl_finaldf_small[clmns])
print(nonan_projslice.info())
#print(usl_finaldf_small.info())
print(nonan_projslice.head())
#print(usl_finaldf_small.head())

categorical_field_names = ['Project ID', 'Project Subject Category Tree', 'Project Grade Level Category', 'Project Resource Category']
for c in categorical_field_names:
    #usl_finaldf_small[c] = usl_finaldf_small[c].astype('category')
    nonan_projslice[c] = nonan_projslice[c].astype('category')

#print(usl_finaldf_small.dtypes)
print(nonan_projslice.dtypes)
print(categorical_field_names)

categoricals_indicies = []
for col in categorical_field_names:
        categoricals_indicies.append(categorical_field_names.index(col))

print(type(categoricals_indicies))
print(categoricals_indicies)

usl_finaldf_small = nonan_projslice.loc[0:199999]
usl_small_matrix = usl_finaldf_small.values
print 'Type for usl_small_matrix'
print(type(usl_small_matrix))

DEBUG         = 2                       # set to 1 to debug, 2 for more
verbose       = 0                       # kmodes debugging
nrows         = 30                      # number of rows to read (resources)
init       = 'Huang'                    # init can be 'Cao', 'Huang' or 'random'
n_clusters = 15                         # how many clusters (hyper parameter)
max_iter   = 100                        # default 100

kproto = kprototypes.KPrototypes(n_clusters=n_clusters,init=init,verbose=verbose)
clusters = kproto.fit_predict(usl_small_matrix,categorical=categoricals_indicies)

#categorical : Index of columns that contain categorical data

print 'Centroids are:', '\n', kproto.cluster_centroids_, '\n'

print 'Cost & number of iterations for K-Prototype are:'
print(kproto.cost_)
#cost_ : float
#Clustering cost, defined as the sum distance of all points to their
#respective cluster centroids.

print(kproto.n_iter_)

proto_cluster_assignments = zip(usl_small_matrix,clusters)
#print '\nclusters:{}\nproto_cluster_assignments: {}\n'.format(clusters,proto_cluster_assignments)

cluster_df = pandas.DataFrame(columns=('Project ID', 'Project Subject Category Type','Project Grade Level Category', 'Project Resource Category', 'Project Cost', 'cluster_id'))

for array in proto_cluster_assignments:
        cluster_df = cluster_df.append({'Project ID':array[0][0], 'Project Subject Category Type':array[0][1],'Project Grade Level Category':array[0][2],'Project Resource Category':array[0][3], 'Project Cost':array[0][4], 'cluster_id':array[1]}, ignore_index=True)

c2n = []

for i in range(n_clusters):
	c2n.append([])

for x in proto_cluster_assignments:
        c2n[x[1]].append(x[0])

for index, record in enumerate(c2n):
    print 'cluster: {} ({})'.format(index,len(record))
    for i in record:
        print '\t{}_{}_{}_{}_({})'.format(i[0],i[1],i[2],i[3],i[4])

#CSV_OUT = "./proj-kprototypes-small.csv"
CSV_OUT = "./proj-kprototypes-200k-K15.csv"
cluster_df.to_csv(CSV_OUT,index=False)

#Project Cluster Data
projcluster = pandas.read_csv('./proj-kprototypes-200k-K15.csv')
print(projcluster.info())
print(projcluster.head())

#=========================================================================
#Applying Feature Engineering on the merged data frame from Donations.csv,
#Donors.csv, Projects.csv, & Schools.csv to extract donor characteristics
#and use those as input dataset for a 3-layered NN technique.
#=========================================================================
#finaldf = nanrepddprsch.drop(['Donation Included Optional Donation', 'Donor Is Teacher', 'Project Type'], axis=1)
finaldf = nanrepddprsch.drop(['Donation Included Optional Donation', 'Donor Is Teacher', 'School ID', 'Project Type'], axis=1)
print(finaldf.info())
finaldfcopy = finaldf.copy()

#DIDgroup = finaldf.groupby('Donor ID')
#type(DIDgroup)

#for name,group in DIDgroup:
#    print name
#    print group

#Create bins to separate donation amounts:
bins = (0, 15, 25, 50, 60000)
finaldfcopy['DonationBin'] = pandas.cut(finaldfcopy['Donation Amount'], bins=bins)
finaldfcopy['BinDonateProp'] = 0.0
finaldfcopy['DonateMean'] = 0.0
finaldfcopy['PCMean'] = 0.0
finaldfcopy['DonationPercent'] = 0.0
#finaldfcopy['PSCPercent'] = 0.0
#finaldfcopy['PGCPercent'] = 0.0
#finaldfcopy['PRCPercent'] = 0.0

finalsmall = finaldfcopy.sample(frac=0.2)
print(finalsmall.info())

def norm_by_BinSum(x):
    x['BinDonateProp'] = x['Donation Amount']/x['Donation Amount'].sum()
    return x

finalsmall = finalsmall.groupby('DonationBin').apply(norm_by_BinSum)
#finaldfcopy = finaldfcopy.groupby('DonationBin').apply(norm_by_BinSum)

def mean_by_donor(x):
    x['DonateMean'] = x['Donation Amount'].sum()/x['Donation Amount'].count()
    return x

finalsmall = finalsmall.groupby('Donor ID').apply(mean_by_donor)
#finaldfcopy = finaldfcopy.groupby('Donor ID').apply(mean_by_donor)

def mean_by_projcost(x):
    x['PCMean'] = x['Project Cost'].sum()/x['Project Cost'].count()
    return x

finalsmall = finalsmall.groupby('Project ID').apply(mean_by_projcost)
#finaldfcopy = finaldfcopy.groupby('Project ID').apply(mean_by_projcost)

def percent_by_donation(x):
    x['DonationPercent'] = (x['Donation Amount'] * 100)/x['Donation Amount'].sum()
    return x

finalsmall = finalsmall.groupby('Donor ID').apply(percent_by_donation)
#finaldfcopy = finaldfcopy.groupby('Donor ID').apply(percent_by_donation)

group_names = ['DG_1', 'DG_2', 'DG_3', 'DG_4']
#Bin "Donation Amount" according to above groups:
categories = pandas.cut(finalsmall['Donation Amount'], bins, labels=group_names)
#categories = pandas.cut(finaldfcopy['Donation Amount'], bins, labels=group_names)

#Assign bins to "Donation Amount" column:
finalsmall['Donation Amount'] = categories
#finaldf['Donation Amount'] = categories
print(finalsmall.head())
#print(finaldfcopy.head())

#Create bins to separate project costs:
#bins = (35, 336, 516, 868, 260000)
#finaldfcopy['PC Bin'] = pandas.cut(finaldfcopy['Project Cost'], bins=bins)
#group_names = ['PC_1', 'PC_2', 'PC_3', 'PC_4']

#Bin "Project Cost" according to the above groups:
#categories = pandas.cut(finaldfcopy['Project Cost'], bins, labels=group_names)

#Assign bins to "Project Cost" column:
#finaldfcopy['Project Cost'] = categories
#print(finaldfcopy.head())

print(finalsmall.info())
#print(finaldfcopy.info())

#The projcluster has 200,000 entries and finalsmall has 921628 entries. So
#will have to select a random sample of 200,000 entries from finalsmall.
smalldf = finalsmall.sample(frac=0.217007296)
print(smalldf.info())

#Merge the output data frame of clustering method with the feature data frame. 
uslsldf = pandas.merge(smalldf, projcluster, how='inner', on='Project ID')
print(uslsldf.info())
#Select relevant columns
uslslslice = uslsldf.loc[:, ['Donor ID', 'Donation Amount', 'Donor State', 'Project Subject Category Tree', 'Project Grade Level Category_x', 'Project Resource Category_x', 'DonationBin', 'BinDonateProp', 'DonateMean', 'PCMean', 'DonationPercent', 'cluster_id']]
print(uslslslice.info())
print(uslslslice.head())

#Look for any nan values in the data frame on which supervised ML techniques
#will be applied.
null_columns = uslslslice.columns[uslslslice.isnull().any()]
print(uslslslice[null_columns].isnull().sum())

#Select the corresponding X & Y from that prepared data frame.
X = uslslslice.drop(columns=['cluster_id'], axis=1)
Y = uslslslice['cluster_id']
print(X.info())
print(Y.size)

#===============================================================
#Split data into training and validation set using 70-30 ratio.
#Use "cluster_id" from the clustering method as the output, Y.
#===============================================================
validation_size = 0.3
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)
print(X_train.info())
print(X_validation.info())
print(Y_train.size)
print(Y_validation.size)

#Drop the Donor ID as that's not needed for training.
X_trainnoID = X_train.drop('Donor ID', axis=1)
X_valinoID = X_validation.drop('Donor ID', axis=1)
print(X_trainnoID.info())
print(X_valinoID.info())

#Make sure the size of X_train and X_validation are the same
#in order to perform fitting on the validation set later on.
X_trainnoID = pandas.get_dummies(X_trainnoID)
X_valinoID = pandas.get_dummies(X_valinoID)
print(X_trainnoID.info())
print(X_valinoID.info())

#=================================================================
#Validation Procedure
#Using 6-fold cross validation technique to estimate accuracy of
#ML algorithms. So will split our training dataset into 6 parts,
#out of which 5 will be used for training and 1 for testing and
#will be repeated for all combinations of train-test splits.
#=================================================================

scoring = 'accuracy'
#Using the metric of "accuracy" to evaluate models.

#====================
#Supervised Learning
#====================
#Will be evaluating the following 4 algorithms on the data set
#because it's a good mixture of simple linear (LR & LDA) and
#non-linear (KNN, & CART) algorithms:

#Logistic Regression (LR)
#Linear Discriminant Analysis (LDA)
#K-Nearest Neighbors (KNN)
#Classification and Regression Trees (CART)

#Build Models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
        kfold = model_selection.KFold(n_splits=6)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)      

#Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
savefig('Algo-comp-200k-usl-sl.png')

#Create a test set by randomly sampling from the main dataframe
X_testsmall = X.sample(frac=0.8)
Y_testsmall = Y.sample(frac=0.8)
print(X_testsmall.info())
print(Y_testsmall.size)

validation_size = 0.2
X_newtrain, X_test, Y_newtrain, Y_test = train_test_split(X_testsmall, Y_testsmall, test_size=validation_size)
print(X_newtrain.info())
print(X_test.info())
print(X_test.head())
print(Y_newtrain.size)
print(Y_test.size)

X_testnoID = X_test.drop('Donor ID', axis=1)
X_newtrainnoID = X_newtrain.drop('Donor ID', axis=1)

X_newtrainnoID = pandas.get_dummies(X_newtrainnoID)
X_testnoID = pandas.get_dummies(X_testnoID)
print(X_newtrainnoID.info())
print(X_testnoID.info())

#Make predictions on test set
cart = DecisionTreeClassifier()
#cart.fit(X_newtrainnoID, Y_newtrain)
#cart.fit(X_trainnoID, Y_train)
cart.fit(X_testnoID, Y_test)
predictions = cart.predict(X_testnoID)

print 'Accuracy score for the Decision Tree is:'
print(accuracy_score(Y_test, predictions))
print '\n, The corresponding confusion matrix is:'
print(confusion_matrix(Y_test, predictions))
print '\n, The final result of the prediction is:'
print(classification_report(Y_test, predictions))

output = pandas.DataFrame({ 'Donor ID' : X_test['Donor ID'], 'Cluster ID': predictions })
output.to_csv('donor-predictions-usl-sl-cart3.csv', index=False)
output
