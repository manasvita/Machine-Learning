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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from pylab import figure, axes, pie, title, show

#Donation Visualization
donatedata = pandas.read_csv('./Donations.csv')
print(donatedata.info())
print(donatedata.describe(include = 'all'))
print(donatedata['Donation Amount'].describe())
#The above shows that the Donation Amount can be binned into 5 bins
#at 0.01, 15, 25, 50, & 60,000.

maxddval = donatedata.loc[donatedata['Donation Amount'] > 50000, :]
print(maxddval['Donation Amount'])
donatedata_no_6e4 = donatedata.loc[donatedata['Donation Amount'] <= 50000, :]

#visualize the data
#univariate plot - Box plot
#donatedata.plot(kind='box', subplots=True, layout=(1,2), sharex=False, sharey=False)
#plt.show()
#Acc. to the box plot the donation amount has a huge outlier. The maximum
#amount for this case is $60,000. Should be taken into account for prediction.

#Removing that maximum amount gives us a Box Plot (BP):
#sns.boxplot(x = 'Donation Amount', data = donatedata_no_6e4, palette = 'RdBu_r')
#plt.show()

#And
#donatedata_no_6e4.plot(kind='box', subplots=True, layout=(1,2), sharex=False, sharey=False)
#plt.show()

#Histogram Plot
#donatedata.hist()
#plt.show()

#scatter-plot matrix
#scatter_matrix(donatedata)
#plt.show()
#Tough to plot because of too many data points.

#sns.countplot(x='Donation Amount', data=donatedata)
#plt.show()
#Same problem as above. The plot didn't result in anything meaningful.

#sns.countplot(x='Donation Included Optional Donation', data=donatedata)
#plt.show()
#The figure shows that lot of donors included optional donations as well.
#These donors are most likely to donate again.

#Donor Visualization
#donordata = pandas.read_csv('./Donors.csv')
donordata = pandas.read_csv('./Donors.csv', low_memory=False)
print(donordata.info())
print(donordata.describe(include = 'all'))

#Donor State Distribution
#sns.countplot(x='Donor State', data=donordata)
#plt.show()
#Maximum number of donations came from California.
#This might imply that Californians are most likely to donate again.

#Donor Is Teacher Distribution
#sns.countplot(x='Donor Is Teacher', data=donordata)
#plt.show()
#The figure shows that maximum donors are not teachers.

#What about maximum amount of donation? That came from which state?

#Project Visualization
projectdata = pandas.read_csv('./Projects.csv')
print(projectdata.info())
print(projectdata.describe(include = 'all'))

#Project Type Distribution
#sns.countplot(x='Project Type', data=projectdata)
#plt.show()
#This figure shows that Teacher-Led are maximum in number compared
#to Professional-Development or Student-Led project types.

#Project Subject Category Tree Distribution
#sns.countplot(x='Project Subject Category Tree', data=projectdata)
#plt.show()
print(projectdata['Project Subject Category Tree'].describe())
#This figure shows that Literacy & Language have the highest number
#of projects that need donations.

#Project Grade Level Distribution
#sns.countplot(x='Project Grade Level Category', data=projectdata)
#plt.show()
#This figure shows that Grades PreK-2 have the most number of
#projects that need funding.

#Project Resource Category Distribution
#sns.countplot(x='Project Resource Category', data=projectdata)
#plt.show()
#The figure shows that most projects fall into the resource category
#"Supplies"
print(projectdata['Project Resource Category'].describe())

#Project Current Status Distribution
#sns.countplot(x='Project Current Status', data=projectdata)
#plt.show()
print(projectdata['Project Current Status'].describe())
#The above shows that maximum projects are "Fully Funded", which is good.

#Project Cost Distribution
#projectdata.plot(kind='box', subplots=True, layout=(1,2), sharex=False, sharey=False)
#plt.show()
print(projectdata['Project Cost'].describe())
#Maximum project cost is for $255,737.7.

project_max_cost = projectdata.loc[projectdata['Project Cost'] > 250000]
print(project_max_cost['Project Resource Category'])
print(project_max_cost['Project Need Statement'])

#Histogram plot for the Project Cost Distribution
#projectdata.hist()
#plt.show()

print(projectdata['Project Cost'].describe())

#countplot didn't work for Project Cost.
#Trying histogram using distplot now.
#hist = sns.distplot(projectdata['Project Cost'], color='b', bins=30, kde=False)
#hist.set(xlim=(0, 260000), title = "Distribution of Project Cost")
#plt.show()
#The above figure is same as that of projectdata histogram with majority
#of the cost distributed at the lower end, so a right-skewed data.

#Schools Visualization
schooldata = pandas.read_csv('./Schools.csv')
print(schooldata.info())
print(schooldata.describe(include = 'all'))

#School State Distribution
#sns.countplot(x='School State', data=schooldata)
#plt.show()
#schooldata['School State'].describe()
#The above distribution shows that California has highest number of
#schools with probable projects and highest number of donors are
#also from California.

#School Metro Type Distribution
#sns.countplot(x='School Metro Type', data=schooldata)
#plt.show()
#schooldata['School Metro Type'].describe()
#This distribution shows that "suburban' schools have the highest count
#and are slightly more than the "urban" count.

#==================
#Data Preparation
#=================
#Merging relevant columns of Donations.csv, Donors.csv, Projects.csv, &
#School.csv into one data frame.
donatdonor = pandas.merge(donatedata, donordata, how='inner', on='Donor ID')
print(type(donatdonor))
print(donatdonor.shape)
print(donatdonor.info())

null_columns = donatdonor.columns[donatdonor.isnull().any()]
print(donatdonor[null_columns].isnull().sum())

dodonslice = donatdonor.loc[:, ['Project ID', 'Donor ID', 'Donation Included Optional Donation', 'Donation Amount', 'Donor State', 'Donor Is Teacher']]
print(dodonslice.shape)
print(dodonslice.info())
null_columns = dodonslice.columns[dodonslice.isnull().any()]
print(dodonslice[null_columns].isnull().sum())

ddproj = pandas.merge(dodonslice, projectdata, how='inner', on='Project ID')
print(ddproj.shape)
print(ddproj.info())
null_columns = ddproj.columns[ddproj.isnull().any()]
print(ddproj[null_columns].isnull().sum())

ddprojslice = ddproj.loc[:, ['Project ID', 'Donor ID', 'Donation Included Optional Donation', 'Donation Amount', 'Donor State', 'Donor Is Teacher', 'School ID', 'Project Type', 'Project Subject Category Tree', 'Project Grade Level Category', 'Project Resource Category', 'Project Cost']]
print(ddprojslice.shape)
print(ddprojslice.info())
null_columns = ddprojslice.columns[ddprojslice.isnull().any()]
print(ddprojslice[null_columns].isnull().sum())

ddprsch = pandas.merge(ddprojslice, schooldata, how='inner', on='School ID')
print(ddprsch.shape)
print(ddprsch.info())
null_columns = ddprsch.columns[ddprsch.isnull().any()]
print(ddprsch[null_columns].isnull().sum())

ddprschslice = ddprsch.loc[:, ['Donor ID', 'Donation Included Optional Donation', 'Donation Amount', 'Donor State', 'Donor Is Teacher', 'Project Type', 'Project Subject Category Tree', 'Project Grade Level Category', 'Project Resource Category', 'Project Cost', 'School Metro Type', 'School State']]

print(ddprschslice.shape)
print(ddprschslice.info())
null_columns = ddprschslice.columns[ddprschslice.isnull().any()]
print(ddprschslice[null_columns].isnull().sum())

#Trying another approach of merging various files but most likely
#will not be using it for making predictions.
#donatproj = pandas.merge(donatedata, projectdata, how='inner', on='Project ID')
#donatprojslice = donatproj.loc[:, ['Donor ID', 'Donation Included Optional Donation', 'Donation Amount', 'School ID', 'Project Type', 'Project Subject Category Tree', 'Project Grade Level Category', 'Project Resource Category', 'Project Cost']]

#print(donatprojslice.shape)
#print(donatprojslice.info())

#null_columns = donatprojslice.columns[donatprojslice.isnull().any()]
#print(donatprojslice[null_columns].isnull().sum())

#Filling both data frames' NaN's with "other"
nanrepddprsch = ddprschslice.fillna("other")
print(nanrepddprsch.shape)
#nanrepdonproj = donatprojslice.fillna("other")
#print(nanrepdonproj.shape)

null_columns = nanrepddprsch.columns[nanrepddprsch.isnull().any()]
print(nanrepddprsch[null_columns].isnull().sum())

#null_columns = nanrepdonproj.columns[nanrepdonproj.isnull().any()]
#print(nanrepdonproj[null_columns].isnull().sum())

#Selecting the data frame with columns from Donations, Donors, Projects, &
#Schools. Total number of columns are 12 with 2 integer/float type and 10
#object type.
print(nanrepddprsch.info())
print(nanrepddprsch.describe())
print(nanrepddprsch.describe(include = 'all'))

#School Metro Type distribution for this dataframe
#sns.countplot(x='School Metro Type', data=nanrepddprsch)
#plt.show()
#Now this distribution shows that "urban" school received maximum
#donations than registered "suburban" schools. This might imply that
#urban donors are most likely to donate again to "urban" schools
#most likely from their cities or states.

#Donors & School States Distribution
print(nanrepddprsch['School State'].describe())
print(nanrepddprsch['Donor State'].describe())
#California is still the state with highest number of donors and registered
#schools for projects.

#Project Grade Level Distribution for this dataframe
#sns.countplot(x='Project Grade Level Category', data=nanrepddprsch)
#plt.show()
print(nanrepddprsch['Project Grade Level Category'].describe())
#This distribution has the same shape as the one for projectdata.
#So Grades PreK-2 had maximum projects and received maximum funding as well.

#Donation Included Optional Donation Distribution
#sns.countplot(x='Donation Included Optional Donation', data=nanrepddprsch)
#plt.show()
print(nanrepddprsch['Donation Included Optional Donation'].describe())

#Donation Amount & Project Cost Distribution
#nanrepddprsch.hist()
#plt.show()
print(nanrepddprsch['Donation Amount'].describe())
print(nanrepddprsch['Project Cost'].describe())
#The distribution is same as before. So the maximum amount donated is
#$60,000 and maximum project cost $255,737.7.

#See the relationship between "Project Cost" and "Donation Amount"
#plt.plot('Project Cost', 'Donation Amount', 'bo', data=nanrepddprsch)
#plt.xlabel('Project Cost (USD)', fontsize=14)
#plt.ylabel('Donation Amount (USD)', fontsize=12)
#plt.show()

#Project Type Distribution
#sns.countplot(x='Project Type', data=nanrepddprsch)
#plt.show()
#print(nanrepddprsch['Project Type'].describe())

#sns.swarmplot(x='Project Type', y='Donation Amount', data=nanrepddprsch)

#Project Subject Category Tree Distribution
#sns.countplot(x='Project Subject Category Tree', data=nanrepddprsch)
#plt.show()
print(nanrepddprsch['Project Subject Category Tree'].describe())
#Same as above, Literacy & Language had the highest number of projects
#and donations.

#Project Resource Category
#sns.countplot(x='Project Resource Category', data=nanrepddprsch)
#plt.show()
print(nanrepddprsch['Project Resource Category'].describe())
#Same as above, Supplies is the category with highest number of projects
#and donations.

#Donation Included Optional Donation Distribution
#sns.countplot(x='Donation Included Optional Donation', data=nanrepddprsch)
#plt.show()
print(nanrepddprsch['Donation Included Optional Donation'].describe())

#Donor Is Teacher Distribution
#sns.countplot(x='Donor Is Teacher', data=nanrepddprsch)
#plt.show()
print(nanrepddprsch['Donor Is Teacher'].describe())


#=========================================================
#Feature-Engineerning and Finding Y for the input matrix#
#=========================================================

#The goal of the project is to predict which donors are likely to donate
#a second time to class room requests. That is, match prior donors to
#project requests to which they would be motivated to donate.

#=============================
#Machine Learning
#=============================
#The goal can be executed as matching donors to "Project Resource Category",
#"Project Subject Category Tree", "Project Grade Level Category",
#"Project Cost", or "School Metro Type".

#Dropping "Donation Included Optional Donation" and "Donor Is Teacher"
#column as majority of donors did include optional donation and they
#are not teachers to begin with. Also dropping "Project Cost" as there
#is no direct relationship between "Donation Amount" and "Project Cost"
#according to the scatter plot above.

#finaldf = nanrepddprsch.drop(['Donation Included Optional Donation', 'Donor Is Teacher', 'Project Cost'], axis=1)
finaldfsl = nanrepddprsch.drop(['Donation Included Optional Donation', 'Donor Is Teacher', 'Project Cost'], axis=1)
print(finaldfsl.info())

#Bin the above dataframe according to "Donation Amount" and then bin
#it further according to "Project Grade Level Category". That way, we'll
#know how much donation each grade received. This will also tell us which
#grade category does each donation/donor belong to.
#Summary statistics for this column gives these values: 0.01, 15, 25, 50,
#& 60,000. So will take these as cutoff points.

#Create bins to separate donation amounts:
bins = (0, 15, 25, 50, 60000)
group_names = ['DG_1', 'DG_2', 'DG_3', 'DG_4']

#Bin "Donation Amount" according to the above groups:
categories = pandas.cut(finaldfsl['Donation Amount'], bins, labels=group_names)

#Assign bins to "Donation Amount" column:
finaldfsl['Donation Amount'] = categories
print(finaldfsl.head())

#Drop the Donor ID column as row index will be mapped to it at
#the end
dIDdf = finaldfsl.loc[:, ['Donor ID']]
print(dIDdf.info())

#====================================================================
#Split data into training and validation set using 60-40 ratio.
#Use "Project Subject Category Tree", "Project Grade Level Category",
#and "Project Resource Category" as the output, Y, one by one and
#compare the results.
#=====================================================================
#X = finaldfsl.drop(columns=['Donor ID', 'Project Subject Category Tree'], axis=1)
X = finaldfsl.drop(columns=['Donor ID', 'Project Grade Level Category'], axis=1)
#X = finaldfsl.drop(columns=['Donor ID', 'Project Resource Category'], axis=1)
#Y = finaldfsl.loc[:, ['Project Subject Category Tree']]
#Y = finaldfsl['Project Subject Category Tree']
Y = finaldfsl['Project Grade Level Category']
#Y = finaldfsl['Project Resource Category']
print(X.info())
print(Y.size)

#finaldfsl_small = finaldfsl.loc[0:199999]
#print(finaldfsl_small.info())
#X_small = finaldfsl_small.drop(columns=['Donor ID', 'Project Subject Category Tree'], axis=1)
#Y_small = finaldfsl_small['Project Subject Category Tree']
#X_small = finaldfsl_small.drop(columns=['Donor ID', 'Project Grade Level Category'], axis=1)
#Y_small = finaldfsl_small['Project Grade Level Category']
#X_small = finaldfsl_small.drop(columns=['Donor ID', 'Project Resource Category'], axis=1)
#Y_small = finaldfsl_small['Project Resource Category']
#print(X_small.info())
#print(Y_small.size)

X_small = X.loc[0:199999]
Y_small = Y.loc[0:199999]
print(X_small.info())
print(Y_small.size)

validation_size = 0.4
X_train, X_validation, Y_train, Y_validation = train_test_split(X_small, Y_small, test_size=validation_size)
#X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)
print(X_train.info())
print(X_validation.info())
print(Y_train.size)
print(Y_validation.size)

#Make sure the size of X_train and X_validation are the same
#in order to perform fitting on the validation set later on.
X_train = pandas.get_dummies(X_train)
X_validation = pandas.get_dummies(X_validation)
print(X_train.info())
print(X_validation.info())
#Both sets now have a total of 138 columns.
#For small datasets with 10,000 rows they have 99 & 97 columns, resp.
#For dataset with 100,000 rows there are 128 & 126 columns, resp.
#The split of 100,000 rows doesn't work out for prediction making phase
#as one of the requirements is that number of features (columns) of
#model (training set) must match the input (validation/test set).
#For dataset with 200,000 rows there are 135 columns for both training
#and validation sets.

#=================================================================
#Validation Procedure
#Using 10-fold cross validation technique to estimate accuracy of
#ML algorithms. So will split our training dataset into 10 parts,
#out of which 9 will be used for training and 1 for testing and
#will be repeated for all combinations of train-test splits.
#=================================================================

scoring = 'accuracy'
#Using the metric of "accuracy" to evaluate models.


#====================
#Supervised Learning
#====================
#Will be evaluating the following 6 algorithms on the data set
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
savefig('Algo-comp-200k.png')

#Make Predictions on validation set
cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
predictions = cart.predict(X_validation)

print 'Accuracy score for the Decision Tree is:'
print(accuracy_score(Y_validation, predictions))
print '\n, The corresponding confusion matrix is:'
print(confusion_matrix(Y_validation, predictions))
print '\n, The final result of the prediction is:'
print(classification_report(Y_validation, predictions))

#Create a test set from the main dataframe
X = finaldfsl.drop(columns=['Project Grade Level Category'], axis=1)
Y = finaldfsl['Project Grade Level Category']

# Randomly sample 50% of the main dataframe
X_small = X.sample(frac=0.5)
Y_small = Y.sample(frac=0.5)
print(X_small.info())
print(Y_small.size)

validation_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X_small, Y_small, test_size=validation_size)
print(X_train.info())
print(X_test.info())
print(X_test.head())
print(Y_train.size)
print(Y_test.size)

X_testnoID = X_test.drop('Donor ID', axis=1)
X_trainnoID = X_train.drop('Donor ID', axis=1)

X_trainnoID = pandas.get_dummies(X_trainnoID)
X_testnoID = pandas.get_dummies(X_testnoID)
print(X_trainnoID.info())
print(X_testnoID.info())

#Make predictions on test set
cart.fit(X_trainnoID, Y_train)
predictions = cart.predict(X_testnoID)

print 'Accuracy score for the Decision Tree is:'
print(accuracy_score(Y_test, predictions))
print '\n, The corresponding confusion matrix is:'
print(confusion_matrix(Y_test, predictions))
print '\n, The final result of the prediction is:'
print(classification_report(Y_test, predictions))

output = pandas.DataFrame({ 'Donor ID' : X_test['Donor ID'], 'Project': predictions })
output.to_csv('donor-predictions-cart.csv', index=False)
output
