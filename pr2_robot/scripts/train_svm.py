#!/usr/bin/env python
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import model_selection
from sklearn import metrics

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{0:.2f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def evaluate_model(clf, X, y):
    # Set up 5-fold cross-validation
    kf = model_selection.KFold(n_splits=5,
                                shuffle=True)

    # Perform cross-validation
    scores = model_selection.cross_val_score(cv=kf,
                                             estimator=clf,
                                             X=X,
                                             y=y,
                                             scoring='accuracy'
                                            )
    print('Scores: ' + str(scores))
    print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), 2*scores.std()))

    # Gather predictions
    predictions = model_selection.cross_val_predict(cv=kf,
                                              estimator=clf,
                                              X=X,
                                              y=y
                                             )

    accuracy_score = metrics.accuracy_score(y, predictions)
    print('accuracy score: '+str(accuracy_score))

    confusion_matrix = metrics.confusion_matrix(y, predictions)

    class_names = encoder.classes_.tolist()
    return class_names, confusion_matrix


def parameter_search(estimator, X, y):
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


# Load training data from disk
training_set = pickle.load(open('../notebooks/training_data/training_set_project.sav', 'rb'))

# Format the features and labels for use with scikit learn
feature_list = []
label_list = []

for item in training_set:
    if np.isnan(item[0]).sum() < 1:
        feature_list.append(item[0])
        label_list.append(item[1])

print('Features in Training Set: {}'.format(len(training_set)))
print('Invalid Features in Training set: {}'.format(len(training_set)-len(feature_list)))

X = np.array(feature_list)
# Fit a per-column scaler
# X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
# X_train = X_scaler.transform(X)
y = np.array(label_list)

# Convert label strings to numerical encoding
encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_devel, X_train, y_devel, y_train = model_selection.train_test_split(X, y, test_size=0.5, stratify=y)

# Create classifier
estimator = {
    "estimator": svm.SVC(class_weight='balanced', kernel=metrics.pairwise.chi2_kernel),

    "param_grid" :
        {
            "C": np.linspace(0.01, 1.0, 10),
            "gamma": [0.00001,0.0001, 0.001,0.01,  0.5, 1],
        }
}


# Train the classifier

# Find hyper-parameters
clf, score, best_params = parameter_search(estimator, X_devel, y_devel)
# Evaluate the model
class_names, confusion_matrix = evaluate_model(clf, X_train, y_train)

# Train the classifier for use in production
clf.fit(X=X, y=y)

model = {'classifier': clf, 'classes': encoder.classes_, 'scaler': None}

# Save classifier to disk
pickle.dump(model, open('model_project.sav', 'wb'))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(confusion_matrix, classes=encoder.classes_,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(confusion_matrix, classes=encoder.classes_, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
