import os
from sklearn.linear_model import LogisticRegression

from data_loader import LabelledTextDS
from plotting import *

dataset = LabelledTextDS(os.path.join('data', 'labelled_movie_reviews.csv'))
train, valid, test = dataset.get_vector_representation()

model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)  # You can change the hyper-parameters of the model by passing args here
#n_jobs=-1,class_weight='balanced', penalty='l2', random_state=42
model.fit(train[0], train[1])
train_accuracy = (model.predict(train[0]) == train[1]).astype(float).mean()
valid_accuracy = (model.predict(valid[0]) == valid[1]).astype(float).mean()
test_accuracy = (model.predict(test[0]) == test[1]).astype(float).mean()

print_accuracies((train_accuracy, valid_accuracy, test_accuracy))
