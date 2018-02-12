#!/bin/bash

# In the conventional K-fold validation, the training data is split into K equal parts and K iterations of learning are run with every part getting a turn as the validation set. The performance over the validation sets is averaged over the K iterations to find the validation accuracy for the choice of hyperparameters. This process is repeated for various choices of hyperparameters and the best combination is found. Then, the model is training using the entire training set using the combination of hyperparameters found earlier. This model is evaluated on a hitherto unseen test set.
# The problem here is that there is no "unseen" test set and every ounce of data has been used for training the model.
# I can't use K-fold validatation coz the data has been split into validation and test for each fold.
# The handout suggests that I use the performance on the test set to tune the hyperparameters which is just not ok!
# So, I am going to consider one fold and use the validation set like it is supposed to i.e to find the optimal hyperparameters and then evaluate the performance on the test set and average it out.
# Actually, this isn't a whole lot different from what the handout suggests, I'll just do what is asked : train on training.txt and evaluate on test.txt

echo Fold1
./svm_rank_learn -c 0.01 MQ2008/Fold1/train.txt model > /dev/null
./svm_rank_classify MQ2008/Fold1/test.txt model | grep Zero
echo Fold2
./svm_rank_learn -c 0.01 MQ2008/Fold2/train.txt model > /dev/null
./svm_rank_classify MQ2008/Fold2/test.txt model | grep Zero
echo Fold3
./svm_rank_learn -c 0.01 MQ2008/Fold3/train.txt model > /dev/null
./svm_rank_classify MQ2008/Fold3/test.txt model | grep Zero
echo Fold4
./svm_rank_learn -c 0.01 MQ2008/Fold4/train.txt model > /dev/null
./svm_rank_classify MQ2008/Fold4/test.txt model | grep Zero
echo Fold5
./svm_rank_learn -c 0.01 MQ2008/Fold5/train.txt model > /dev/null
./svm_rank_classify MQ2008/Fold5/test.txt model | grep Zero
echo
