from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


# 8*8 bytes image
digits = load_digits()
print(digits.data.shape)
print(digits.target.shape)

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])): #first five data, first five labels
    plt.subplot(1,5, index+1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)

plt.show()

#Splitting Data into Training and Test Sets
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
#test_size: what proprtion of org data is used for test set
print(x_train.shape) #75%
print(y_train.shape) #25%

print(x_test.shape) #75%
print(y_test.shape) #25%

#Scikit-learn 4-step modeling pattern
#Step1: Import the model you want to use, In Sklearn, all machine learning models are implemented as python classes
from sklearn.linear_model import LogisticRegression

#Step2: Make an instance of the Model
logisticRegr = LogisticRegression()
print(logisticRegr)
#Step3: Training the model on the data, storing the info learned from the data
logisticRegr.fit(x_train, y_train)

#Step4: Predict the labels of new data (new images)
#Uses the info the model learned during the model training process

logisticRegr.predict(x_test[0].reshape(1,-1)) #predict one data
print(logisticRegr.predict(x_test[0].reshape(1,-1)))
logisticRegr.predict(x_test[0:10]) #preidct 10 data
print(logisticRegr.predict(x_test[0:10]))
predictions = logisticRegr.predict(x_test) #predict entire set
print(predictions)
print(predictions.shape)


#Measuring Model Performance - accuracy (fraction of correct predictions): correct predictions/ total num of data points
# how the model performs on new data (test set)

score = logisticRegr.score(x_test, y_test)
print(score)

#Confusion Matrix (Matplotlib) - visualize performance
#confusion matrix is a table that is often used to describe the performance of a classification model
# on a set of test data for which the true values are known
def plot_confusion_matrix(cm, title='Confusion matrix', cmap='Pastel1'):
    plt.figure(figsize=(9,9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size=10)
    plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size=10)
    plt.tight_layout()
    plt.ylabel('Actual label', size=15)
    plt.xlabel('Predicted label', size =15)
    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y,x), horizontalalignment='center', verticalalignment='center')

confusion = metrics.confusion_matrix(y_test, predictions)
print("confusion matrix")
print(confusion)
plt.figure()
plot_confusion_matrix(confusion)
plt.show()


#same thing with a seaborn
predictions = logisticRegr.predict(x_test)
cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
all_sample_title = 'Accuracy Score:{0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()

#Display Misclassified images with predicted labels
index = 0
misclassifiedIndex = []
for predict, actual in zip(predictions, y_test):
    if predict != actual:
        misclassifiedIndex.append(index)
    index += 1

plt.figure(figsize=(20,4))
for plotIndex, wrong in enumerate(misclassifiedIndex[10:15]):
    plt.subplot(1,5, plotIndex+1)
    plt.imshow(np.reshape(x_test[wrong], (8,8)), cmap=plt.cm.gray)
    plt.title('Predicted{}, actual:{}'.format(predictions[wrong], y_test[wrong], fontsize=20))

plt.show()