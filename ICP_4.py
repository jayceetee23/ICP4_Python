# ICP4 Implementing Naive Bayes Method

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Imports iris dataset
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

# training data
x = iris.data
# target data
y = iris.target


# ICP_4 Programming Exercise #1 #########################

# Create a Naive Bayes Gaussian Model
gau_model = GaussianNB()
# Train Naive Bayes Gaussian Model
gau_model.fit(x, y)

print("Gaussian Prediction: ", gau_model.predict([[1, 2, 3, 4], [2, 3, 4, 5]]))
print("Accuracy:", gau_model.score(x, y))

print("\n")

# Create a k-NN Model (k-nearest neighbor)
nei_model = KNeighborsClassifier(n_neighbors=10)
# Train k-NN Model
nei_model.fit(x, y)

print("k-NN Model Prediction: ", nei_model.predict([[1, 2, 3, 4], [2, 3, 4, 5]]))
print("Accuracy: ", nei_model.score(x, y))

print("\n")

# ICP_4 Programming Exercise #2 #########################

# linear SVM (Support Vector Machines)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print("Linear SVM Prediction: ", clf.predict([[1, 2, 3, 4], [2, 3, 4, 5]]))
print("Accuracy: ", clf.score(x, y))


print("\n")

# ICP_4 Programming Exercise #3 #########################

# linear SVM with RBF Kernel
RBF = svm.SVC(gamma= 'auto', kernel='rbf', C=1).fit(X_train, y_train)
print("Linear SVM with RBF Prediction: ", RBF.predict([[1, 2, 3, 4], [2, 3, 4, 5]]))
print("Accuracy: ", RBF.score(x, y))
