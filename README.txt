For visualization using bar chart, i divided the group by gender and by acceptance rate, we can see 1335 people were not accepted and 157 were accepted.

I split the data and trained it on the models of Logistic Regression, Naive-Bayes and KNN, i did the evaluation using classifation report(accuracy,f1-score,precision and recall) and confusion matrix and those were the results.

Model: LogisticRegression
Accuracy: 0.92
F1 Score: 0.51
              precision    recall  f1-score   support

       False       0.94      0.97      0.95       268
        True       0.65      0.42      0.51        31

    accuracy                           0.92       299
   macro avg       0.79      0.70      0.73       299
weighted avg       0.91      0.92      0.91       299


[[261   7]
 [ 18  13]]
Model: GaussianNB
Accuracy: 0.87
F1 Score: 0.61
              precision    recall  f1-score   support

       False       1.00      0.86      0.92       268
        True       0.44      0.97      0.61        31

    accuracy                           0.87       299
   macro avg       0.72      0.91      0.76       299
weighted avg       0.94      0.87      0.89       299


[[230  38]
 [  1  30]]
Model: KNeighborsClassifier
Accuracy: 0.96
F1 Score: 0.79
              precision    recall  f1-score   support

       False       0.97      0.99      0.98       268
        True       0.85      0.74      0.79        31

    accuracy                           0.96       299
   macro avg       0.91      0.86      0.89       299
weighted avg       0.96      0.96      0.96       299


[[264   4]
 [  8  23]]
 
As we can see KNN gave the best result out of 3 models.