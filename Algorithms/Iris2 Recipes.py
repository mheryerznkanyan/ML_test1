#import dataset

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data         #features
y = iris.target       #lables

#spliting two parts .... training an d testin
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = .5)


#train the classifier
from sklearn import tree

my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, Y_train)

#make a predicition
predicition = my_classifier.predict(X_test)
print(predicition)

#
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predicition))