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


---

## implement Boosting functions

*Compute the classfication accuracy of the boosted classier on some data sets using testClassifier from labfuns.py and compare it with those of the basic classifier on the vowels and iris data sets (see Assignment 3):*

*(1) Is there any improvement in classification accuracy? Why/why not?*

*(2) Plot the decision boundary of the boosted classfier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?*

*(3) Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using boosting?*

---

### 1. the classification accuracy:

   basic classifier for data 'iris':
> Trial: 0 Accuracy 84.4
> 
> Trial: 10 Accuracy 95.6
>
> Trial: 20 Accuracy 93.3
>
> Trial: 30 Accuracy 86.7
>
> Trial: 40 Accuracy 88.9
>
> Trial: 50 Accuracy 91.1
>
> Trial: 60 Accuracy 86.7
>
> Trial: 70 Accuracy 91.1
>
> Trial: 80 Accuracy 86.7
>
> Trial: 90 Accuracy 91.1
>
> Final mean classification accuracy  89 with standard deviation 4.16


  boosted classifier for data 'iris':
>Trial: 0 Accuracy 95.6
>
>Trial: 10 Accuracy 100
>
>Trial: 20 Accuracy 93.3
>
>Trial: 30 Accuracy 91.1
>
>Trial: 40 Accuracy 97.8
>
>Trial: 50 Accuracy 93.3
>
>Trial: 60 Accuracy 93.3
>
>Trial: 70 Accuracy 97.8
>
>Trial: 80 Accuracy 95.6
>
>Trial: 90 Accuracy 93.3
>
>Final mean classification accuracy  94.7 with standard deviation 2.82



  basic classifier for data 'vowel':
>Trial: 0 Accuracy 61
>
>Trial: 10 Accuracy 66.2
>
>Trial: 20 Accuracy 74
>
>Trial: 30 Accuracy 66.9
>
>Trial: 40 Accuracy 59.7
>
>Trial: 50 Accuracy 64.3
>
>Trial: 60 Accuracy 66.9
>
>Trial: 70 Accuracy 63.6
>
>Trial: 80 Accuracy 62.3
>
>Trial: 90 Accuracy 70.8
>
>Final mean classification accuracy  64.7 with standard deviation 4.03


  boosted classifier for data 'vowel' (I set T=3 instead of 10 to make it have outcome faster):
>Trial: 0 Accuracy 65.6
>
>Trial: 10 Accuracy 77.3
>
>Trial: 20 Accuracy 76
>
>Trial: 30 Accuracy 68.2
>
>Trial: 40 Accuracy 63.6
>
>Trial: 50 Accuracy 66.2
>
>Trial: 60 Accuracy 73.4
>
>Trial: 70 Accuracy 70.8
>
>Trial: 80 Accuracy 69.5
>
>Trial: 90 Accuracy 72.7
>
>Final mean classification accuracy  72 with standard deviation 3.82


### 2. decision boundary

   decision boundary for basic classification of 'iris':

![iris_basic](https://github.com/AsukaLmy/Bayes-boosting-KTHDD2421/blob/main/lab3py/iris_basic_cf.png)



   decision boundary for boosted classification of 'iris':

![iris_boosted](https://github.com/AsukaLmy/Bayes-boosting-KTHDD2421/blob/main/lab3py/iris_boosted_cf.png)


### 3. Can we......?

of course we can 



---
   
## implement decision tree classifier


*(1) Is there any improvement in classification accuracy? Why/why not?*

*(2) Plot the decision boundary of the boosted classfier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?*

*(3) Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using decision tree?*

---

### 1. accuracy

   basic decision tree for 'iris'

>Trial: 0 Accuracy 95.6
>
>Trial: 10 Accuracy 100
>
>Trial: 20 Accuracy 91.1
>
>Trial: 30 Accuracy 91.1
>
>Trial: 40 Accuracy 93.3
>
>Trial: 50 Accuracy 91.1
>
>Trial: 60 Accuracy 88.9
>
>Trial: 70 Accuracy 88.9
>
>Trial: 80 Accuracy 93.3
>
>Trial: 90 Accuracy 88.9
>
>Final mean classification accuracy  92.4 with standard deviation 3.71


   boosted + decision tree for 'iris'

>Trial: 0 Accuracy 95.6
>
>Trial: 10 Accuracy 100
>
>Trial: 20 Accuracy 95.6
>
>Trial: 30 Accuracy 93.3
>
>Trial: 40 Accuracy 93.3
>
>Trial: 50 Accuracy 95.6
>
>Trial: 60 Accuracy 88.9
>
>Trial: 70 Accuracy 93.3
>
>Trial: 80 Accuracy 93.3
>
>Trial: 90 Accuracy 93.3
>
>Final mean classification accuracy  94.6 with standard deviation 3.65


   basic decision tree for 'vowel'

>Trial: 0 Accuracy 63.6
>
>Trial: 10 Accuracy 68.8
>
>Trial: 20 Accuracy 63.6
>
>Trial: 30 Accuracy 66.9
>
>Trial: 40 Accuracy 59.7
>
>Trial: 50 Accuracy 63
>
>Trial: 60 Accuracy 59.7
>
>Trial: 70 Accuracy 68.8
>
>Trial: 80 Accuracy 59.7
>
>Trial: 90 Accuracy 68.2
>
>Final mean classification accuracy  64.1 with standard deviation 4


   boosted + decision tree for 'vowel'

>Trial: 0 Accuracy 85.1
>
>Trial: 10 Accuracy 89.6
>
>Trial: 20 Accuracy 87
>
>Trial: 30 Accuracy 92.2
>
>Trial: 40 Accuracy 84.4
>
>Trial: 50 Accuracy 80.5
>
>Trial: 60 Accuracy 86.4
>
>Trial: 70 Accuracy 85.7
>
>Trial: 80 Accuracy 89
>
>Trial: 90 Accuracy 83.1
>
>Final mean classification accuracy  86.6 with standard deviation 3.06


### 2. boundary plot

   basic decision tree for 'iris':

![iris_basic_dt](https://github.com/AsukaLmy/Bayes-boosting-KTHDD2421/blob/main/lab3py/iris_dt_basic.png)


   boosted decision tree for 'iris':

![iris_basic_dt](https://github.com/AsukaLmy/Bayes-boosting-KTHDD2421/blob/main/lab3py/iris_dt_boosted.png)


### 3. Can ....?

I think we can 


