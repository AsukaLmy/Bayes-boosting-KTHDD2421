# Bayes-boosting-KTHDD2421
lab3 Bayes+boosting

*!ATTENTION!*

before you start, some modification should be made in labfun.py

-9-    from sklearn.datasets.samples_generator import make_blobs -> from sklearn.datasets import make_blobs 

-109, 113, 118, 122-    np.int -> np.int64 (or int)

-287-    rtn.classifier = tree.DecisionTreeClassifier(max_depth=Xtr.shape[1]/2+1) -> rtn.classifier = tree.DecisionTreeClassifier(max_depth=int(Xtr.shape[1]/2+1))

## testClassifier for vowels and iris datasets

*Answer the following questions:*

*(1) When can a feature independence assumption be reasonable and when not?*

*(2) How does the decision boundary look for the Iris dataset? How could one improve the classification results for this scenario by changing classfier or, alternatively, manipulating the data?*

***

(1). The reasonableness of this assumption depends on the specific problem and the nature of the features being considered:

When the features of data are not related, or not that strongly related, featuer independence assumption is reasonable. For example:

- text classification(like spam detection in email): though words in a document are often correlated, it can work surprisingly well in practice and simplifies the modeling process.
- Sparse Data:  In situations where you have limited data and many features, assuming feature independence can help to mitigate the curse of dimensionality. It simplifies the estimation of probabilities because you don't need to estimate high-dimensional joint probability distributions.


When the features are strongly related, featuer independence assumption is not reasonable anymore. 

- structured data: when data has clear dependencies or structures among features, the feature independence assumption is often violated. For example, in computer vision, adjacent pixels in an image are highly correlated, so assuming independence between them is not reasonable.

- time series data: time series data, where observations are taken at consecutive time points, typically exhibits temporal dependencies. Assuming independence between time steps would ignore crucial information

- specific knowledge:  domain knowledge may suggest that features are not independent. For instance, in physics or engineering, physical laws may imply dependencies between variables


(2). the boundary look as follow (donnot forget to attach the image！！！！！！！！！！！！！！！！！)

to improve the classification results for this scenario, there are some ways:

- choose different classifier or try ensemble methods like Random Forest, Gradient Boosting, or AdaBoost, which can often improve classification accuracy by combining multiple weak learners

- data preprocess like normalization or handle missing data using imputation techniques or removing instances with missing values.

- apply regularization techniques to prevent overfitting. Techniques like L1 or L2 regularization can help improve the classifier's generalization.

- understand your data, they are not dead stuff but living storytellor.


## implement Boosting functions

*Compute the classfication accuracy of the boosted classier on some data sets using testClassifier from labfuns.py and compare it with those of the basic classifier on the vowels and iris data sets (see Assignment 3):*

*(1) Is there any improvement in classification accuracy? Why/why not?*

*(2) Plot the decision boundary of the boosted classfier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?*

*(3) Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using boosting?*



