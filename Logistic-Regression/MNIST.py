#downloading the data (MNIST)

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
#there are 70000 images (28x28 images for a dimensionality of 784)
print(mnist.data.shape)
#there are the labels
print(mnist.target.shape)



#splitting data into training and test sets
from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
#test_size=1/7.0 makes the training set size 60000 and testset size of 10000


#showing the images and labels
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for index, (image,label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
    plt.subplot(1,5, index+1)
    plt.imshow(np.reshape(image, (28,28)), cmap = plt.cm.gray)
    plt.title('Training: %i\n' %label, fontsize=20)
plt.show()


#scikit-learn 4-step modeling pattern (MNIST)
#step1 import the model you want to use
from sklearn.linear_model import LogisticRegression

#step2 make an instance of the model
LogisticReg = LogisticRegression(solver='lbfgs')

#step3 training the model on the data, storing the info learned from the data
#model is learning the relationship btw x(digits) and y(labels)
LogisticReg.fit(train_img, train_lbl)

#step4 predict the labels of new data(new images)
#predict one img
LogisticReg.predict(test_img[0].reshape(1,-1))
print(LogisticReg.predict(test_img[0].reshape(1,-1)))
#predict multiple at once
LogisticReg.predict(test_img[0:10])
print(LogisticReg.predict(test_img[0:10]))
#predict entire
prediction = LogisticReg.predict(test_img)
print(LogisticReg.predict(test_img))


#Measuring Performance
score = LogisticReg.score(test_img, test_lbl)
print(score)

#One thing I briefly want to mention is that is the default optimization algorithm parameter was
# solver = liblinear and it took 2893.1 seconds to run with a accuracy of 91.45%.
# When I set solver = lbfgs , it took 52.86 seconds to run with an accuracy of 91.3%.
# Changing the solver had a minor effect on accuracy, but at least it was a lot faster.


#Display Misclassified images with Predicted Labels(MNISTS)
import numpy as np
import matplotlib.pyplot as plt

index = 0
misclassifiedIndexes = []
for label, predict in zip(test_lbl, prediction):
    if label != predict:
        misclassifiedIndexes.append(index)
    index += 1

plt.figure(figsize=(20,4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1,5, plotIndex+1)
    plt.imshow(np.reshape(test_img[badIndex], (28,28)), cmap=plt.cm.gray)
    plt.title('Predicted:{}, Actual:{}'.format(prediction[badIndex], test_lbl[badIndex], fontsize= 15))
plt.show()



#same thing with a seaborn
import seaborn as sns
from sklearn import metrics
predictions = LogisticReg.predict(test_img)
cm = metrics.confusion_matrix(test_lbl, predictions)
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
for predict, actual in zip(predictions, test_lbl):
    if predict != actual:
        misclassifiedIndex.append(index)
    index += 1

plt.figure(figsize=(20,4))
for plotIndex, wrong in enumerate(misclassifiedIndex[10:15]):
    plt.subplot(1,5, plotIndex+1)
    plt.imshow(np.reshape(test_img[wrong], (28,28)), cmap=plt.cm.gray)
    plt.title('Predicted{}, actual:{}'.format(predictions[wrong], test_lbl[wrong], fontsize=20))

plt.show()




