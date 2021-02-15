import numpy as np
#import matplotlib.pyplot as plt
np.random.seed(1)
#Problem 1

# Problem 1.1
def my_euclidean_dist(X_test, X_train):
    """
    Compute the distance between each test example and each training example.

    Input:
    - X_test: A numpy array of shape (num_test, dim_feat) containing test data
    - X_train: A numpy array of shape (num_train, dim_feat) containing training data

    Output:
    - dists: A numpy array of shape (num_test, num_train) where 
            dist[i, j] is the Euclidean distance between the i-th test example 
            and the j-th training example
    """
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    # TODO:
    # Compute the L2 distance between all test and training examples.
    #
    # One most straightforward way is to use nested for loop
    # to iterate over all test and training samples.
    # Here is the pseudo-code:
    # for i = 0 ... num_test - 1
    #    a[i] = square of the norm of the i-th test example
    # for j = 0 ... num_train - 1
    #    b[j] = square of the norm of the j-th training example
    # for i = 0 ... num_test - 1
    #    for j = 0 ... num_train - 1
    #        dists[i, j] = a[i] + b[j] - 2 * np.dot(i-th test example, j-th training example)
    # return dists
    
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    a = [0 for i in range(num_test)]
    b = [0 for i in range(num_train)]
    # dists = [[0 for i in range(num_train)] for j in range(num_test)]

    for i in range(num_test):
        a[i] = sum(X_test[i] ** 2)
    for i in range(num_train):
        b[i] = sum(X_train[i] ** 2)
    for i in range(num_test):
        for j in range(num_train):
            dists[i][j] = a[i] + b[j] - 2 * np.dot(X_test[i], X_train[j])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)***** 

    return dists

def euclidean_dist(X_test, X_train):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dists = np.add(np.sum(X_test ** 2, axis=1, keepdims=True),
                   np.sum(X_train ** 2, axis=1, keepdims=True).T) - 2 * X_test @ X_train.T
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return dists


def find_k_neighbors(dists, Y_train, k):
    """
    find the labels of the top k nearest neighbors

    Inputs:
    - dists: distance matrix of shape (num_test, num_train)
    - Y_train: A numpy array of shape (num_train) containing ground true labels for training data
    - k: An integer, k nearest neighbors

    Output:
    - neighbors: A numpy array of shape (num_test, k), where each row containts the 
                labels of the k nearest neighbors for each test example
    """
    # TODO:
    # find the top k nearest neighbors for each test sample.
    # retrieve the corresponding labels of those neighbors.
    # Here is the pseudo-code:
    # for i = 0 ... num_test
    #     idx = numpy.argsort(i-th row of dists)
    #     neighbors[i] = Y_train(idx[0]), ..., Y_train(idx[k-1])
    # return neighbors
    # Advanced: You can accelerate the code by, e.g., argsort on the `dists` matrix directly

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ss = dists.shape[0]

    if not isinstance(k, int):
        k = k[0]

    neighbors = np.zeros((ss, k), dtype=int)

    for i in range(ss):
        idx = np.argsort(dists[i])
        for j in range(k):
            neighbors[i][j] = Y_train[idx[j]]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return neighbors


def knn_predict(X_test, X_train, Y_train, k):
    """
    predict labels for test data.

    Inputs:
    - X_test: A numpy array of shape (num_test, dim_feat) containing test data.
    - X_train: A numpy array of shape (num_train, dim_feat) containing training data.
    - Y_train: A numpy array of shape (num_train) containing ground true labels for training data
    - k: An integer, k nearest neighbors

    Output:
    - Y_pred: A numpy array of shape (num_test). Predicted labels for the test data.
    """
    # TODO:
    # find the labels of k nearest neighbors for each test example,
    # and then find the majority label out of the k labels
    #
    # Here is the pseudo-code:
    # dists = euclidean_dist(X_test, X_train)
    # neighbors = find_k_neighbors(dists, Y_train, k)
    # Y_pred = np.zeros(num_test, dtype=int)  # force dtype=int in case the dataset
    #                                         # stores labels as float-point numbers
    # for i = 0 ... num_test-1
    #     Y_pred[i] = # the most common/frequent label in neighbors[i], you can
    #                 # implement it by using np.unique
    # return Y_pred

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dists = euclidean_dist(X_test, X_train)
    neighbors = find_k_neighbors(dists, Y_train, k)
    num_test = X_test.shape[0]
    Y_pred = np.zeros(num_test, dtype=int)

    for i in range(num_test):
        labels, count = np.unique(neighbors[i], return_counts=True)
        l = np.argmax(count)
        Y_pred[i] = labels[l]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return Y_pred

def compute_error_rate(ypred, ytrue):
    """
    Compute error rate given the predicted results and true lable.
    Inputs:
    - ypred: array of prediction results.
    - ytrue: array of true labels.
        ypred and ytrue should be of same length.
    Output:
    - error rate: float number indicating the error in percentage
                    (i.e., a number between 0 and 100).
    """
    # Here is the pseudo-code:
    # err = 0
    # for i = 0 ... num_test - 1
    #     err = err + (ypred[i] != ytrue[i])  # generalizes to multiple classes
    # error_rate = err / num_test * 100
    # return error_rate
    #
    # Advanced (optional): 
    #   implement it in one line by using vector operation and the `mean` function

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    err = 0
    for i in range(len(ypred)):
        if ypred[i] != ytrue[i]:
            err += 1
    error_rate = (err / len(ypred)) * 100
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return error_rate


def split_nfold(num_examples, n):
    """
    Split the dataset in to training sets and validation sets.
    Inputs:
    - num_examples: Integer, the total number of examples in the dataset
    - n: number of folds
    Outputs:
    - train_sets: List of lists, where train_sets[i] (i = 0 ... n-1) contains 
                    the indices of examples for trainning
    - validation_sets: List of list, where validation_sets[i] (i = 0 ... n-1) 
                    contains the indices of examples for validation

    Example:
    When num_examples = 10 and n = 5, 
        the output train_sets should be a list of length 5, 
        and each element in this list is itself a list of length 8, 
        containing 8 indices in 0...9
    For example, 
        we can initialize by randomly permuting [0, 1, ..., 9] into, say,
        [9, 5, 3, 0, 8, 4, 2, 1, 6, 7]
        Then we can have
        train_sets[0] = [3, 0, 8, 4, 2, 1, 6, 7],  validation_sets[0] = [9, 5]
        train_sets[1] = [9, 5, 8, 4, 2, 1, 6, 7],  validation_sets[1] = [3, 0]
        train_sets[2] = [9, 5, 3, 0, 2, 1, 6, 7],  validation_sets[2] = [8, 4]
        train_sets[3] = [9, 5, 3, 0, 8, 4, 6, 7],  validation_sets[3] = [2, 1]
        train_sets[4] = [9, 5, 3, 0, 8, 4, 2, 1],  validation_sets[4] = [6, 7]
    Within train_sets[i] and validation_sets[i], the indices do not need to be sorted.
    """
    # Here is the pseudo code:
    # idx = np.random.permutation(num_examples).tolist() # generate random index list
    # fold_size = num_examples//n   # compute how many examples in one fold.
    #                               # note '//' as we want an integral result
    # train_sets = []
    # validation_sets = []
    # for i = 0 ... n-1
    #	  start = # compute the start index of the i-th fold
    #	  end = # compute the end index of the i-th fold
    #   if i == n-1
    #     end = num_examples  # handle the remainder by allocating them to the last fold
    #   For example, when num_examples = 11 and n = 5, 
    #     fold_size = 11//5 = 2
    #     i = 0: start = 0, end = 2
    #     i = 1: start = 2, end = 4
    #     i = 2: start = 4, end = 6
    #     i = 3: start = 6, end = 8
    #     i = 4: start = 8, end = 11  (take up the remainder of 11//5)
    #
    #   # Now extract training example indices from the idx list using start and end
    #   train_set = idx[`0 to num_example-1` except `start to end-1`]  
    #   train_sets.append(train_set)
    #
    #   # Extract validation example indices from the idx list using start and end
    #   val_set = idx[start to end-1] 
    #   validation_sets.append(val_set)
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    idx = np.random.permutation(num_examples).tolist()
    fold_size = num_examples // n

    train_sets = []
    validation_sets = []

    start = 0
    end = 0
    for i in range(n):
        start = end
        end = end + fold_size

        if i == n - 1:
            end = num_examples

        train_set = idx[:start] + idx[end:]
        train_sets.append(train_set)

        val_set = idx[start:end]
        validation_sets.append(val_set)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return train_sets, validation_sets



def cross_validation(classifier, X, Y, n, *args):
    """
    Perform cross validation for the given classifier, 
        and return the cross validation error rate.
    Inputs:
    - classifier: function of classification method
    - X: A 2-D numpy array of shape (num_train, dim_feat), containing the whole dataset
    - Y: A 1-D numpy array of length num_train, containing the ground-true labels
    - n: number of folds
    - *args: parameters needed by the classifier.
            In this assignment, there is only one parameter (k) for the kNN clasifier.
            For other classifiers, there may be multiple paramters. 
            To keep this function general, 
            let's use *args here for an unspecified number of paramters.
    Output:
    - error_rate: a floating-point number indicating the cross validation error rate
    """
    # Here is the pseudo code:
    # errors = []
    # size = X.shape[0] # get the number of examples
    # train_sets, val_sets = split_nfold(size, n)  # call the split_nfold function
    #
    # for i in range(n):
    #   train_index = train_sets[i]
    #   val_index = val_sets[i]
    #   # get the training and validation sets of input features from X
    # 	X_train, X_val = X[...], X[...] 
    #
    #   # get the training and validation labels from Y
    # 	y_train, y_val = Y[...], Y[...] 
    #
    #   # call the classifier to get prediction results for the current validation set
    # 	ypred = # call classifier with X_val, X_train, y_train, and *args
    #                                   
    # 	error = # call compute_error_rate to compute the error rate by comparing ypred against y_val
    # 	append error to the list `errors`
    # error_rate = mean of errors
    np.random.seed(1)
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    errors = []
    size = X.shape[0]
    train_sets, val_sets = split_nfold(size, n)

    for i in range(n):
        train_index = train_sets[i]
        val_index = val_sets[i]

        X_train = X[train_index]
        X_val = X[val_index]

        Y_train = Y[train_index]
        Y_val = Y[val_index]

        ypred = classifier(X_val, X_train, Y_train, args)

        error = compute_error_rate(ypred, Y_val)
        errors.append(error)

    error_rate = sum(errors) / n
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return error_rate 

# Problem 2

def problem2():
    """
    copy your solutions of problem 2 to this function
    DONOT copy the code for plotting 

    Outputs:
    - error_rates: A numpy array of size (len(trials_size),) which store error rates for different example size
    - cverror_rates: A numpy array of size (len(trials_fold),) which store error rates for different fold size
    """
    import os
    import gzip

    DATA_URL = 'http://yann.lecun.com/exdb/mnist/'

    # Download and import the MNIST dataset from Yann LeCun's website.
    # Each image is an array of 784 (28x28) float values  from 0 (white) to 1 (black).
    def load_data():
        x_tr = load_images('train-images-idx3-ubyte.gz')
        y_tr = load_labels('train-labels-idx1-ubyte.gz')
        x_te = load_images('t10k-images-idx3-ubyte.gz')
        y_te = load_labels('t10k-labels-idx1-ubyte.gz')

        return x_tr, y_tr, x_te, y_te

    def load_images(filename):
        maybe_download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28 * 28) / np.float32(256)

    def load_labels(filename):
        maybe_download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    # Download the file, unless it's already here.
    def maybe_download(filename):
        if not os.path.exists(filename):
            from urllib.request import urlretrieve
            print("Downloading %s" % filename)
            urlretrieve(DATA_URL + filename, filename)

    Xtrain, ytrain, Xtest, ytest = load_data()

    train_size = 10000
    test_size = 10000

    Xtrain = Xtrain[0:train_size]
    ytrain = ytrain[0:train_size]

    Xtest = Xtest[0:test_size]
    ytest = ytest[0:test_size]

    # problem 2.1
    #  nbatches must be an even divisor of test_size. Increase if you run out of memory 
    if test_size > 1000:
        nbatches = 50
    else:
        nbatches = 5

    # Let us first set up the index of each batch. 
    # After running the next line, 'batches' will be a 2D array sized nbatches-by-m,
    # where m = test_size / nbatches.
    # batches[i] stores the indices (out of 0...test_size-1) for the i-th batch
    # You can run 'print(batches[3])' etc to witness the value of 'batches'.
    batches = np.array_split(np.arange(test_size), nbatches)
    ypred = np.zeros_like(ytest)
    trial_sizes = [100, 1000, 2500, 5000, 7500, 10000]
    trials = len(trial_sizes)
    error_rates = [0]*trials
    k = 1

    # Here is the pseudo code:
    # 
    # for t = 0 ... trials-1  # loop over different number of training examples
    # 	trial_size = trial_sizes[t]
    # 	trial_X = Xtrain[...] # extract trial_size number of training examples from the whole training set
    # 	trial_Y = Ytrain[...] # extract the corresponding labels
    # 	for i = 0…nbatches—1
    # 		ypred[...] = # call knn_predict to classify the i-th batch of test examples.
    #                  # You should use 'batches' to get the indices for batch i.
    #                  # Then store the predicted labels also in the corresponding
    #                  # elements of ypred, so that after the loop over i completes,
    #                  # ypred will hold exactly the predicted labels of all test examples.
    # 	error_rate[t] = # call compute_error_rate to compute the error rate by 
    #                     comparing ypred against ytest
    #   print a line like '#tr = 100, error rate = 50.3%'
    # plot the figure:
    # f = plt.figure()
    # plt.plot(...)
    # plt.xlabel(...)
    # plt.ylabel(...)
    # plt.show()


    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(trials):
        trial_size = trial_sizes[i]
        trial_X = Xtrain[:trial_size]
        trial_Y = ytrain[:trial_size]

        for j in range(nbatches):
            ypred[batches[j]] = knn_predict(Xtest[batches[j]], trial_X, trial_Y, k)

        error_rates[i] = compute_error_rate(ypred, ytest)
        print('tr', i, 'error rate =', error_rates[i])

    #f = plt.figure()
    #plt.plot(error_rate)
    #plt.xlabel('Sample')
    #plt.ylabel('Error %')
    #plt.show()
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    #problem 2.2
    size = 1000
    k = 1
    # Here is the pseudo code:
    #
    # get the feature/label of the first 'size' (i.e., 1000) number of training examples
    # cvXtrain = Xtrain[...]  
    # cvytrain = ytrain[...]  

    # trial_folds   = [3, 10, 50, 100, 1000]
    # trials = # number of trials, i.e., get the length of trial_sizes
    # cverror_rates = [0]*trials

    # for t = 0 ... trials-1
    # 	error_rate = # call the 'cross_validation' function to get the error rate 
    #                #  for the current trial (of fold number)
    # 	cverror_rates[t] = error_rate
    #
    #   # print the error rate for the current trial.
    # 	print('{:d}-folds error rate: {:.2f}%\n'.format(trial_folds[t], error_rate)) 
    #
    # plot the figure:
    # f = plt.figure()
    # plt.plot(...)
    # plt.xlabel(...)
    # plt.ylabel(...)
    # plt.show()

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    cvXtrain = Xtrain[:size]
    cvytrain = ytrain[:size]

    trial_folds = [3, 10, 50, 100, 1000]
    trials = len(trial_folds)

    cverror_rates = [0] * trials

    for i in range(trials):
        error_rate = cross_validation(knn_predict, cvXtrain, cvytrain, trial_folds[i], k)
        cverror_rates[i] = error_rate
        print('{:d}-folds error rate: {:.2f}%\n'.format(trial_folds[i], error_rate))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return error_rates, cverror_rates

# Problem 3
def problem3():
    """
    copy your solutions of problem 3 to this function.
    DONOT copy the code for plotting 
    
    Outputs: 
    - err_ks: A numpy array of size (len(list_ks),) which store error rate for each k
    - best_k: An integer which gives lowest error rate on validation set
    - err_test: Error rate on test set
    - cm: confusion matrix
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # loading iris dataset
    iris = load_iris()
    # split dataset into training set and test set
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)



    # problem 3.1
    # Here is the pseudo code:
    # list_ks = 1,2,...,100
    # err_ks = 1D array of length 100
    # for k in list_ks:
    #   err_ks[k-1] = cross_validation under k 
    # best_k = argmin(err_ks)+1
    # plot err_ks versus list_ks

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    n_fold = 10
    nk = 100
    list_ks = list(range(1, nk + 1))
    err_ks = [0] * nk

    for k in list_ks:
        err_ks[k - 1] = cross_validation(knn_predict, X_train, Y_train, n_fold, k)

    best_k = np.argmin(err_ks) + 1
    # print(best_k)
    # print(err_ks)
    # print(list_ks)

    # f = plt.figure()
    # plt.plot(list_ks, err_ks)
    # plt.xlabel('K')
    # plt.ylabel('Error %')
    # plt.show()
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****






    # problem 3.2
    # Here is the pseudo code:
    # y_pred = knn_predict on X_test using X_train, Y_train, and best_k
    # use compute_error_rate to compute the error of y_pred compared with Y_test
    # Print the error rate with a line like 'The test error is x.y%'


    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    best_k_int = best_k.item()

    y_pred = knn_predict(X_test, X_train, Y_train, best_k_int)
    err_test = compute_error_rate(y_pred, Y_test)
    print('The test error is {:.1f}%\n'.format(err_test))


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # problem 3.3
    nclass = len(np.unique(Y_test))  # should be 3. Just be more adaptive to data.
    cm = np.zeros((nclass, nclass), dtype=int)  # confusion matrix is integer valued

    # Here is the pseudo code for Task 1: 
    # for t = 0...nte-1  # nte is the number of test examples
    #    cm[c1, c2] += 1  # c1 and c2 corresponds to the class of the t-th test example
    #                     # according to Y_test and y_pred, respectively
    #
    # Here is the pseudo code for Task 3:
    # Well, please consult the textbook, as I really hope you can do it yourself,
    # especially when the right answer is provided by sklearn for comparison


    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # sk = confusion_matrix(Y_test, y_pred)
    # print(sk)

    # part 1
    nte = len(X_test)

    for i in range(nte):
        cm[Y_test[i]][y_pred[i]] += 1

    # part 2
    cr = classification_report(Y_test, y_pred, [0, 1, 2])
    print(cr)

    # part 3
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score

    # f1 = f1_score(Y_test,y_pred, average='micro')
    # print(f1)

    prec = precision_score(Y_test, y_pred, average='micro')
    rec = recall_score(Y_test, y_pred, average='micro')

    f1_score = (2 * (prec * rec)) / (prec + rec)
    print(f1_score)

    # print(cm)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return err_ks, best_k, err_test, cm