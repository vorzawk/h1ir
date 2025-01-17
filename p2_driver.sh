#!/bin/bash

# In the conventional K-fold validation, the training data is split into K equal parts and K iterations of learning are run with every part getting a turn as the validation set. The performance over the validation sets is averaged over the K iterations to find the validation accuracy for the choice of hyperparameters. This process is repeated for various choices of hyperparameters and the best combination is found. Then, the model is training using the entire training set using the combination of hyperparameters found earlier. This model is evaluated on a hitherto unseen test set.
# The problem here is that there is no "unseen" test set and every ounce of data has been used for training the model.
# I can't use K-fold validatation coz the data has been split into validation and test for each fold.
# The handout suggests that I use the performance on the test set to tune the hyperparameters which is just not ok!
# So, I am going to consider one fold and use the validation set like it is supposed to i.e to find the optimal hyperparameters and then evaluate the performance on the test set and average it out.
# Actually, this isn't a whole lot different from what the handout suggests, I'll just do what is asked : train on training.txt and evaluate on test.txt

echo Note: The first accuracy value is the training accuracy
echo Default parameters
 echo Fold1
 ./svm_rank_learn -c 0.01 MQ2008/Fold1/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold1/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold1/test.txt model | grep Zero
 echo Fold2
 ./svm_rank_learn -c 0.01 MQ2008/Fold2/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold2/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold2/test.txt model | grep Zero
 echo Fold3
 ./svm_rank_learn -c 0.01 MQ2008/Fold3/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold3/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold3/test.txt model | grep Zero
 echo Fold4
 ./svm_rank_learn -c 0.01 MQ2008/Fold4/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold4/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold4/test.txt model | grep Zero
 echo Fold5
 ./svm_rank_learn -c 0.01 MQ2008/Fold5/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold5/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold5/test.txt model | grep Zero
 echo
 echo Doesn\'t look like the model is overfitting, so may be increasing the model complexity can help
 echo Using an rbf kernel would have been a good idea but the model takes forever to run!
 
 echo Trying with c = 100.0
 echo Fold1
 ./svm_rank_learn -c 100.0 MQ2008/Fold1/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold1/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold1/test.txt model | grep Zero
 echo Fold2
 ./svm_rank_learn -c 100.0 MQ2008/Fold2/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold2/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold2/test.txt model | grep Zero
 echo Fold3
 ./svm_rank_learn -c 100.0 MQ2008/Fold3/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold3/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold3/test.txt model | grep Zero
 echo Fold4
 ./svm_rank_learn -c 100.0 MQ2008/Fold4/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold4/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold4/test.txt model | grep Zero
 echo Fold5
 ./svm_rank_learn -c 100.0 MQ2008/Fold5/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold5/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold5/test.txt model | grep Zero
 echo

echo c doesn\'t seem to have much impact, try varying the norm
echo Fold1
./svm_rank_learn -c 0.01 -p 2 MQ2008/Fold1/train.txt model > /dev/null
./svm_rank_classify MQ2008/Fold1/train.txt model | grep Zero
./svm_rank_classify MQ2008/Fold1/test.txt model | grep Zero
echo Fold2
./svm_rank_learn -c 0.01 -p 2 MQ2008/Fold2/train.txt model > /dev/null
./svm_rank_classify MQ2008/Fold2/train.txt model | grep Zero
./svm_rank_classify MQ2008/Fold2/test.txt model | grep Zero
echo Fold3
./svm_rank_learn -c 0.01 -p 2 MQ2008/Fold3/train.txt model > /dev/null
./svm_rank_classify MQ2008/Fold3/train.txt model | grep Zero
./svm_rank_classify MQ2008/Fold3/test.txt model | grep Zero
echo Fold4
./svm_rank_learn -c 0.01 -p 2 MQ2008/Fold4/train.txt model > /dev/null
./svm_rank_classify MQ2008/Fold4/train.txt model | grep Zero
./svm_rank_classify MQ2008/Fold4/test.txt model | grep Zero
echo Fold5
./svm_rank_learn -c 0.01 -p 2 MQ2008/Fold5/train.txt model > /dev/null
./svm_rank_classify MQ2008/Fold5/train.txt model | grep Zero
./svm_rank_classify MQ2008/Fold5/test.txt model | grep Zero

echo L2 norm gives a slight improvement
 echo try the other loss function
 
 echo Fold1
 ./svm_rank_learn -c 0.01 -p 2 -l 2 MQ2008/Fold1/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold1/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold1/test.txt model | grep Zero
 echo Fold2
 ./svm_rank_learn -c 0.01 -p 2 -l 2 MQ2008/Fold2/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold2/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold2/test.txt model | grep Zero
 echo Fold3
 ./svm_rank_learn -c 0.01 -p 2 -l 2 MQ2008/Fold3/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold3/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold3/test.txt model | grep Zero
 echo Fold4
 ./svm_rank_learn -c 0.01 -p 2 -l 2 MQ2008/Fold4/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold4/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold4/test.txt model | grep Zero
 echo Fold5
 ./svm_rank_learn -c 0.01 -p 2 -l 2 MQ2008/Fold5/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold5/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold5/test.txt model | grep Zero

 echo -o 1
 echo Fold1
 ./svm_rank_learn -c 100.0 -p 2 -o 1 MQ2008/Fold1/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold1/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold1/test.txt model | grep Zero
 echo Fold2
 ./svm_rank_learn -c 100.0 -p 2 -o 1 MQ2008/Fold2/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold2/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold2/test.txt model | grep Zero
 echo Fold3
 ./svm_rank_learn -c 100.0 -p 2 -o 1 MQ2008/Fold3/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold3/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold3/test.txt model | grep Zero
 echo Fold4
 ./svm_rank_learn -c 100.0 -p 2 -o 1 MQ2008/Fold4/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold4/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold4/test.txt model | grep Zero
 echo Fold5
 ./svm_rank_learn -c 100.0 -p 2 -o 1 MQ2008/Fold5/train.txt model > /dev/null
 ./svm_rank_classify MQ2008/Fold5/train.txt model | grep Zero
 ./svm_rank_classify MQ2008/Fold5/test.txt model | grep Zero

