## Project: Perception
____

This implementation of the Perception project was ran on a Ubuntu 16.04 device with the following specifications:

    Memory: 7.7 GiB
    Processor: Intel Core i7-3537U CPU @ 2.00GHz x 4
    Graphics: Intel Ivybridge Mobile
    OS Type: 64-bit


The objective of this project was to implement a perception pipeline from reading a point cloud to performing segmentation.

#### Structure of the project:

All the files and their content is described below:

- [*pr2_robot/scripts/perception_pipeline.py*](./pr2_robot/scripts/perception_pipeline.py): Contains the ROS node used to perform the perception.

- [*pr2_robot/scripts/features.py*](./pr2_robot/scripts/features.py): Contains the code to extract features histogram from an input point cloud.

- [*pr2_robot/scripts/train_svm.py*](./pr2_robot/scripts/train_svm.py): Contains the code to train the model.

- [*pr2_robot/scripts/capture_features.py*](./pr2_robot/scripts/capture_features.py): Contains the code to capture and serialize the training data for the model. The size of the dataset can be controlled by passing an integer to the script. This integer represent the number of sample to collect for each label.
- [*pr2_robot/scripts/output/*](./pr2_robot/scripts/output/): This folder contains the output yaml files for test1, test2 and test3.
- [*pr2_robot/scripts/estimators/*](./pr2_robot/scripts/estimators/): This folder the serialized trained prediction model that are supported by the perception_pipeline.

- [*notebooks/*](./notebooks/): Contains the ipython notebook used to experiment with various models as well as the training data.


##### Usage :

To run the perception_pipeline:

```bash
rosrun pr2_robot perception_pipeline <world_index> <estimator>  [--pickup]
```

with `<world_index>` and integer in 1-4 representing which world is currently in use, `<estimator>` the predicition model, and `--pickup` an optional indicating if the node should write to yaml (default) or send the pickup requests to the robot.

available models are: __kernel_logit, kernel_lda, svm_chi2, svm_sigmoid__
____

#### Pipeline Implementation:

##### Filtering, RANSAC and clustering
Please go the [perception_pipeline](./pr2_robot/scripts/perception_pipeline.py) to see the code relevant to this step in the method pcl_callback. 

##### Feature extraction and object recognition

To facilitate experimentation, we moved the code in train_svm.py to a notebook. train_svm is still provided but we recommend the reader to look at the notebook instead. 
We tested various models from Logistic Regression to different Kernels for SVM and Linear Discriminant Analysis. The confusion matrices and our comments are provided below.

We collected 400 data point, 50 by label.
The data was split at 50% between a development set used to for parameter tuning and an evaluation set used to test the performance of the model. 
For all models, we tuned hyperparameters by using grid search cross validation, using the function below:

```python
def parameter_search(estimator, X, y):
    """Uses a grid search to obtain the best hyperparameters of a given on the provided dataset.
    """
    estimator_ = {
        "scoring": "accuracy",
        "n_jobs": 2,
        "refit": True,
        "cv": 5
    }

    estimator_.update(estimator)
    # Set up 5-fold cross-validation
    clf = model_selection.GridSearchCV(**estimator_)
    clf.fit(X, y)
    print("Best estimator: {}".format(clf.best_estimator_))
    print("Score on development set: {}".format(clf.best_score_))
    return clf.best_estimator_, clf.best_score_, clf.best_params_
```


##### Logistic Regression
```
Best estimator: LogisticRegression(C=0.10000000000000001, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
Score on development set: 0.98
Scores: [ 0.975  1.     1.     1.     1.   ]
Accuracy: 0.99 (+/- 0.02)
accuracy score: 0.995
```
![](./media/confusion_matrices/logit.jpg?raw=true)
As one of the simplest off the shelve classifier available it was a good idea to start our exploration with this model. Input are standardized before being passed to the classifier. 
Logit already performs very well given enough data. 

##### Logistic Regression with RBF kernel.
```
Best estimator: LogisticRegression(C=0.5, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
Score on development set: 0.985
Scores: [ 0.9    0.85   0.975  1.     1.   ]
Accuracy: 0.94 (+/- 0.12)
accuracy score: 0.94
```
![](./media/confusion_matrices/kernel_logit.jpg?raw=true)
By passing the data through a kernel we hope to allow the estimator to take advantage of nonlinearity in the dataset. The performance of the model is slightly degraded, with a rather larger increase in standard deviation. This can imply that the model is starting to overfit. Given that we are estimating the standard deviation on a sample of 5 point, this comment is to be taken with a grain of salt.

##### LDA with RBF kernel.
```
Best estimator: LinearDiscriminantAnalysis(n_components=None, priors=None,
              shrinkage=0.22222222222222221, solver='lsqr',
              store_covariance=False, tol=0.0001)
Score on development set: 0.98
Scores: [ 1.     0.975  0.975  1.     0.975]
Accuracy: 0.98 (+/- 0.02)
accuracy score: 0.99
```
![](./media/confusion_matrices/kernel_lda.jpg?raw=true)
As one of the simplest linear classifier, it would be good to assess the performance of LDA before moving to SVM. The model is simpler to train and more parcimonious than the SVM, in term of data stored. On a robot that may have constraints on its storage, a more lightweight estimator is always preferable.
We see that LDA can provide stellar performance if the data is preprocessed correctly. In this case we are very sensitive to the kernel used in our Kernel PCA. RBF provides a very clear advantage over all the other kernel tested.
As a sidenote, one could say that combining Kernal PCA and LDA leads to a pipeline very similar to Kernel-SVM

##### SVM with Chi2 kernel.
```
Best estimator: SVC(C=0.67000000000000004, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1e-05,
  kernel=<function chi2_kernel at 0x7f1c2589a410>, max_iter=-1,
  probability=False, random_state=None, shrinking=True, tol=0.001,
  verbose=False)
Score on development set: 0.98
Scores: [ 1.     0.975  1.     1.     1.   ]
Accuracy: 0.99 (+/- 0.02)
accuracy score: 0.995
```
![](./media/confusion_matrices/svm_chi2.jpg?raw=true)
The Chi2 kernel is supposed to give good performances on histogram data. Because the kernel needs to work with positive data, we started by removing Standardizer from the pipeline. While this step is rather fast this simplifies the workflow.
The performance of the model is remarquable.

##### SVM with Sigmoid kernel.
```
Best estimator: SVC(C=0.45000000000000001, cache_size=200, class_weight='balanced', coef0=0,
  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='sigmoid',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Score on development set: 0.98
Scores: [ 1.     1.     0.975  1.     1.   ]
Accuracy: 0.99 (+/- 0.02)
accuracy score: 0.99
```
![](./media/confusion_matrices/svm_sigmoid.jpg?raw=true)
Very good performance overall, but it seems to be the most complex model of the collection. Nonetheless as it was the required model for this project, it was used during the pick and place.


Overall we can see that while SVM is a very good choice, one should experiment with various models to fine the best one for the job. Notably, the engineer should always look for the simplest model as this reduces the sources of errors in production.
____

#### Pick And Place

##### World 1
![](./media/world1.png?raw=true)

##### World 2
![](./media/world2.png?raw=true)

##### World 3
![](./media/world3.png?raw=true)

#####  Further improvements:
The perception_pipeline contains code to perform the pick and place operations but the logic to implement the sideways lookup and populate the collision map with the boxes is not complete. In addition several issues were encountered regarding the robots ability to grip the objects properly. For those reason while the pickup option is available, it may not word all the times.
