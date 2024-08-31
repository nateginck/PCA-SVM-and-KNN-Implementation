import os
import random
import shutil
from PIL import Image
import time
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import scipy
from weightedKNN import weightedKNN
from sklearn.neighbors import KNeighborsClassifier


# read in matlab file
data3 = scipy.io.loadmat(r'input/hw4_data3.mat')

# store data
X_train = np.array(data3['X_train'])
X_test = np.array(data3['X_test'])
y_train = np.array(data3['y_train'])
y_test = np.array(data3['y_test'])

# 1b. weighted KNN function
Sigma = 0.01, 0.07, 0.15, 1.5, 3
accuracy = np.empty(5)

for i in range(len(Sigma)):
    # Predict for Sigma[i]
    y_predict = weightedKNN(X_train, y_train.ravel(), X_test, Sigma[i])

    # compute accuracy
    accuracy[i] = accuracy_score(y_test, y_predict)

print("Sigma:    ", Sigma)
print("Accuracy: ", accuracy)

# Part 2.1: PCA Analysis

# a. Read code to training directory
source = 'input/all'
train = 'input/train'
test = 'input/test'

# empty out train folder
for item in os.listdir(train):
    item_path = os.path.join(train, item)
    if os.path.isfile(item_path) or os.path.islink(item_path):
        os.unlink(item_path)
    elif os.path.isdir(item_path):
        shutil.rmtree(item_path)

# empty out test folder
for item in os.listdir(test):
    item_path = os.path.join(test, item)
    if os.path.isfile(item_path) or os.path.islink(item_path):
        os.unlink(item_path)
    elif os.path.isdir(item_path):
        shutil.rmtree(item_path)

# iterate through every folder
for folder in range(1, 41):
    # create folder path
    name = f's{folder}'
    folder_path = f'{source}/{name}'

    files = [file for file in os.listdir(folder_path)]

    # randomly select 8 files, other 2 go to testing data
    train_files = random.sample(files, 8)
    test_files = list(set(files) - set(train_files))

    # copy files to train
    for file in train_files:
        shutil.copy(f'{folder_path}/{file}', f'{train}/{name}_{file}')

    # copt files to test
    for file in test_files:
        shutil.copy(f'{folder_path}/{file}', f'{test}/{name}_{file}')

all_train_files = [f for f in os.listdir(train)]

# 2.0, select 3 faces
selected = random.sample(all_train_files, 3)
fig, axs = plt.subplots(1, 3)
for i, img in enumerate(selected):
    path = os.path.join(train, img)
    img = Image.open(path)
    name = os.path.splitext(path)[0]
    name = name[12:]

    axs[i].imshow(img, cmap='gray')
    axs[i].set_title(name, fontsize=10)
    axs[i].axis('off')

plt.show()
plt.close()


# reshape every image as a column vector
cols = []


# for every picture, reshape vertically
for image_i in all_train_files:
    image_path = os.path.join(train, image_i)

    image = Image.open(image_path)

    image_array = np.array(image).reshape(-1, 1) # change to vertical

    cols.append(image_array)

# Save image
T = np.hstack(cols) # stack horizontally

# save to output
Timg = Image.fromarray(T)
Timg.save(r'output/ps5-1-a.png')

# Show image
plt.imshow(T, cmap='gray')
plt.axis('off')
plt.show()
plt.close()

# 2.1.b Compute Average Face
m = np.mean(T, axis=1)
m_face = np.reshape(m, (112, 92))

# display face
plt.imshow(m_face, cmap='gray')
plt.axis('off')
plt.show()

# save face
Ming = Image.fromarray(m_face.astype(np.uint8))
Ming.save(r'output/ps5-2-1-b.png')

# 2.1.c Covariance Matrix
A = T - m.reshape(10304, 1) # reshape m to vector
C = A @ A.T

# Show Covariance matrix
plt.imshow(C, cmap='gray')
plt.axis('off')
plt.savefig(r'output/ps5-2-1-c.png')
plt.show()
plt.close()

# 2.1.d.
# Compute eigenvalues of A.T @ A
eigenvalues, eigenfaces = np.linalg.eig(A.T @ A)

# sort based on eigen values
descend = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[descend]
eigenfaces = eigenfaces[:, descend]

# calculate sum of eigenvalue
total = np.sum(eigenvalues)
K = 0
rolling_sum = 0
variance = []

# calculate K needed for >= 95% variance
for i in range(len(eigenvalues)):
    rolling_sum += eigenvalues[i]
    variance.append(rolling_sum/total)
    if rolling_sum / total >= 0.95:
        K = i + 1
        break

# print K value
print("The amount of eigenvectors K to retain 95% of variance is ", K)

# plot the figure
plt.plot(variance, linestyle = '-', marker='')
plt.title('Variance Explained by K Components')
plt.xlabel('K Components')
plt.ylabel('Variance')
plt.savefig(r'output/ps5-2-1-d.png')
plt.close()

# 2.1.e. K dominant eigen vectors
eV, eF = np.linalg.eigh(C)

# sort result
descend = np.argsort(eV)[::-1]
eV = eV[descend]
eF = eF[:, descend]

# Print U matrix
U = eF[:, :K]
print("Shape of matrix U are: ", np.shape(U))

# get first 9 faces from U
for i in range(9):
    face_image = U[:, i].reshape(112, 92)

    plt.subplot(1, 9, i + 1)
    plt.imshow(face_image, cmap='gray')
    plt.axis('off')

# save plot of eigenfaces
plt.grid(False)
plt.savefig(r'output/ps5-2-1-e.png')
plt.show()
plt.close()

# 2.2.a
W = []  # result
labels = []  # label

# reshape m
m = m.reshape(10304, 1)

# training data
for image_i in os.listdir(train):
    # open up each image
    image_path = os.path.join(train, image_i)
    image_original = Image.open(image_path)

    # flatten image
    image = np.array(image_original).reshape(-1, 1)

    # compute w
    w = U.T @ (image - m)

    # truncate such that only folder number remains (for classification)
    name = os.path.splitext(image_path)[0]
    name = name[13:]
    name = name.split('_')[0]

    # save w as a row vector
    W.append(w.flatten())
    labels.append(name)

# store in np arrays the W matrix and labels
W_training = np.array(W)
y_train = np.array(labels)

# testing data
W = []  # result
labels = []  # label

for image_i in os.listdir(test):
    # open up each image
    image_path = os.path.join(test, image_i)
    image_original = Image.open(image_path)

    # flatten image
    image = np.array(image_original).reshape(-1, 1)

    w = U.T @ (image - m)

    # Take only # for person
    name = os.path.splitext(image_path)[0]
    name = name[12:]
    name = name.split('_')[0]

    W.append(w.flatten())
    labels.append(name)

# store in np arrays the W matrix and labels
W_testing = np.array(W)
y_test = np.array(labels)

# print output
print("Size of W_training: ", np.shape(W_training))
print("Size of W_testing: ", np.shape(W_testing))

# determine different values for K
k = [1, 3, 5, 7, 9, 11]
accuracy = []

# testing
y_train = np.resize(y_train, (320,))
y_test = np.resize(y_test, (80,))
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# train a KNN Classifier using W_training
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(W_training, y_train)

    prediction = knn.predict(W_testing)

    accuracy.append(accuracy_score(y_test, prediction))

print("K values: ", k)
print("Accuracy: ", accuracy)


# train SVM classifier models

# One vs One Linear
OvOL = SVC(kernel='linear', decision_function_shape='ovo')
# start time
start = time.time()
OvOL.fit(W_training, y_train)
end_OvOL_train = time.time() - start
start = time.time()
prediction = OvOL.predict(W_testing)
end_OvOL_test = time.time() - start
accuracy_OvOL = accuracy_score(y_test, prediction)

# One vs One Polynomial
OvOP = SVC(kernel='poly', decision_function_shape='ovo')
# start time
start = time.time()
OvOP.fit(W_training, y_train)
end_OvOP_train = time.time() - start
start = time.time()
prediction = OvOP.predict(W_testing)
end_OvOP_test = time.time() - start
accuracy_OvOP = accuracy_score(y_test, prediction)

# One vs One RBF
OvOR = SVC(kernel='rbf', decision_function_shape='ovo')
# start time
start = time.time()
OvOR.fit(W_training, y_train)
end_OvOR_train = time.time() - start
start = time.time()
prediction = OvOR.predict(W_testing)
end_OvOR_test = time.time() - start
accuracy_OvOR = accuracy_score(y_test, prediction)
#####

# One vs All
OvAL = SVC(kernel='linear', decision_function_shape='ovr')
# start time
start = time.time()
OvAL.fit(W_training, y_train)
end_OvAL_train = time.time() - start
start = time.time()
prediction = OvAL.predict(W_testing)
end_OvAL_test = time.time() - start
accuracy_OvAL = accuracy_score(y_test, prediction)

# One vs All Polynomial
OvAP = SVC(kernel='poly', decision_function_shape='ovr')
# start time
start = time.time()
OvAP.fit(W_training, y_train)
end_OvAP_train = time.time() - start
start = time.time()
prediction = OvAP.predict(W_testing)
end_OvAP_test = time.time() - start
accuracy_OvAP = accuracy_score(y_test, prediction)

# One vs All RBF
OvAR = SVC(kernel='rbf', decision_function_shape='ovr')
# start time
start = time.time()
OvAR.fit(W_training, y_train)
end_OvAR_train = time.time() - start
start = time.time()
prediction = OvAR.predict(W_testing)
end_OvAR_test = time.time() - start
accuracy_OvAR = accuracy_score(y_test, prediction)

# create dictionary for results
results = {}
results['One vs One Linear'] = {'Training Time': end_OvOL_train,
                                'Testing Time': end_OvOL_test,
                                'Accuracy': accuracy_OvOL}
results['One vs One Poly'] = {'Training Time': end_OvOP_train,
                                'Testing Time': end_OvOP_test,
                                'Accuracy': accuracy_OvOP}
results['One vs One RBF'] = {'Training Time': end_OvOR_train,
                                'Testing Time': end_OvOR_test,
                                'Accuracy': accuracy_OvOR}
results['One vs All Linear'] = {'Training Time': end_OvAL_train,
                                'Testing Time': end_OvAL_test,
                                'Accuracy': accuracy_OvAL}
results['One vs All Poly'] = {'Training Time': end_OvAP_train,
                                'Testing Time': end_OvAP_test,
                                'Accuracy': accuracy_OvAP}
results['One vs All RBF'] = {'Training Time': end_OvAR_train,
                                'Testing Time': end_OvAR_test,
                                'Accuracy': accuracy_OvAR}

# print training time
print("Kernel, OvO Train Time, OVA Train Time")
print(f"Linear, {results['One vs One Linear']['Training Time']:.3f}, {results['One vs All Linear']['Training Time']:.3f}")
print(f"Poly,  {results['One vs One Poly']['Training Time']:.3f}, {results['One vs All Poly']['Training Time']:.3f}")
print(f"RBF,   {results['One vs One RBF']['Training Time']:.3f}, {results['One vs All RBF']['Training Time']:.3f}")

# testing time
print("Kernel, OvO Test Time, OvA Test Time")
print(f"Linear, {results['One vs One Linear']['Testing Time']:.3f}, {results['One vs All Linear']['Testing Time']:.3f}")
print(f"Poly,   {results['One vs One Poly']['Testing Time']:.3f}, {results['One vs All Poly']['Testing Time']:.3f}")
print(f"RBF,    {results['One vs One RBF']['Testing Time']:.3f}, {results['One vs All RBF']['Testing Time']:.3f}")

# accuracy
print("Kernel, OvO Accuracy, OvA Accuracy")
print(f"Linear, {results['One vs One Linear']['Accuracy']:.4f}, {results['One vs All Linear']['Accuracy']:.4f}")
print(f"Poly,   {results['One vs One Poly']['Accuracy']:.4f}, {results['One vs All Poly']['Accuracy']:.4f}")
print(f"RBF,    {results['One vs One RBF']['Accuracy']:.4f}, {results['One vs All RBF']['Accuracy']:.4f}")
