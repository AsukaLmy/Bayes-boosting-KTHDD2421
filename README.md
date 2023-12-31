# Bayes-boosting-KTHDD2421
lab3 Bayes+boosting

*GUIDE*

'lab3_assignment1-3.ipynb' is for assignment 1-3 (actually u can ignore and step for lab3_assignment4-9.ipynb directly)

'lab3_assignment4-9.ipynb' is for assignment 4-8 (superise, no assignment9)

'lab3_assignment4-9.ipynb 2.0' is the upgraded version of 'lab3_assignment4-9.ipynb'. I use the method mentioned in 'lab3.pdf' to replace the massive loop in function. (However, it doesn't be more efficient at all. Actually I think O(n) is same as before)

*!ATTENTION!*

before you start, some modification should be made in labfun.py

-9-    from sklearn.datasets.samples_generator import make_blobs -> from sklearn.datasets import make_blobs 

-109, 113, 118, 122-    np.int -> np.int64 (or int)

-287-    rtn.classifier = tree.DecisionTreeClassifier(max_depth=Xtr.shape[1]/2+1) -> 

rtn.classifier = tree.DecisionTreeClassifier(max_depth=int(Xtr.shape[1]/2+1))

## Bayes for vowels and iris datasets

*Answer the following questions:*

*(1) When can a feature independence assumption be reasonable and when not?*

*(2) How does the decision boundary look for the Iris dataset? How could one improve the classification results for this scenario by changing classfier or, alternatively, manipulating the data?*

***

(1). The reasonableness of this assumption depends on the specific problem and the nature of the features being considered:

When the features of data are **not related**, or not that strongly related, featuer independence assumption is reasonable. For example:

- text classification(like spam detection in email): though words in a document are often correlated, it can work surprisingly well in practice and simplifies the modeling process.
- Sparse Data:  In situations where you have limited data and many features, assuming feature independence can help to mitigate the curse of dimensionality. It simplifies the estimation of probabilities because you don't need to estimate high-dimensional joint probability distributions.


When the features are **strongly related**, featuer independence assumption is not reasonable anymore. 

- structured data: when data has clear dependencies or structures among features, the feature independence assumption is often violated. For example, in computer vision, adjacent pixels in an image are highly correlated, so assuming independence between them is not reasonable.

- time series data: time series data typically exhibits temporal dependencies. Assuming independence between time steps would ignore crucial information

- specific knowledge:  domain knowledge may suggest that features are not independent. For instance, in physics or engineering, physical laws may imply dependencies between variables


(2). the boundary can be seen in next part

to improve the classification results for this scenario, there are some ways:

- choose different classifier or try ensemble methods like Random Forest, Gradient Boosting, or AdaBoost.

- data preprocess like normalization or handle missing data using imputation techniques or removing instances with missing values.

- apply regularization techniques to prevent overfitting. (L1 or L2 regularization)

- understand your data, they are not dead stuff but living storytellor.


---

## implement Boosting functions

*Compute the classfication accuracy of the boosted classier on some data sets using testClassifier from labfuns.py and compare it with those of the basic classifier on the vowels and iris data sets (see Assignment 3):*

*(1) Is there any improvement in classification accuracy? Why/why not?*

*(2) Plot the decision boundary of the boosted classfier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?*

*(3) Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using boosting?*

---

### 1. the classification accuracy: 

'iris': 89 -> 94.7

'vowel': 64.7 -> 72

improved.  

Boosting creates a strong classifier by combining multiple weak classifiers, reducing the model's bias, and enhancing its generalization ability. 

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

After boosting, the boundary become more complex fittable.

   decision boundary for basic classification of 'iris':

![iris_basic](https://github.com/AsukaLmy/Bayes-boosting-KTHDD2421/blob/main/lab3py/iris_basic_cf.png)



   decision boundary for boosted classification of 'iris':

![iris_boosted](https://github.com/AsukaLmy/Bayes-boosting-KTHDD2421/blob/main/lab3py/iris_boosted_cf.png)


### 3. Can we......?

of course we can 

Sometimes, a more advanced model may be better. For example, by using decision tree the accuarcy reach 92.4 easily while it reach 94.7 by using boosted bayes.

---
   
## implement decision tree classifier


*(1) Is there any improvement in classification accuracy? Why/why not?*

*(2) Plot the decision boundary of the boosted classfier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?*

*(3) Can we make up for not using a more advanced model in the basic classifier (e.g. independent features) by using boosting?*

---

### 1. accuracy

'iris': 92.4 -> 94.6

'vowel': 64.1 -> 84.6

improved. 

bigger increase in 'vowel' 

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

---

## criteria for picking a classier, naive Bayes or a decision tree or the boosted versions of these


Actually I am not very clear about this question. I would apply first and see if it perform better. But chatgpt have his answer:

Your evaluation of Naive Bayes, Decision Trees, and their boosted versions based on the provided criteria is well-reasoned. Here's a summary of your analysis:

1. **Outliers**:
   - Naive Bayes is sensitive to outliers due to its probabilistic assumptions.
   - Decision Trees are less sensitive to outliers and can adapt to them.
   - Boosted version may handle outliers better than Naive Bayes.
   v2:
   - Decision trees and their boosted versions (like Random Forests or Gradient Boosted Trees) are generally more robust to outliers, as they partition the feature space into regions and don't rely on assumptions of a particular distribution for the data.
   - Naive Bayes classifiers might be more sensitive to outliers due to their assumptions (independence among features and certain probability distributions).

2. **Irrelevant Inputs**:
   - Naive Bayes may struggle with irrelevant inputs due to its feature independence assumption.
   - Decision Trees can automatically learn feature relevance.
   - Boosted version can adapt to the relevance of features.
   -    v2:
   - Decision trees naturally ignore irrelevant features as they build upon greedy algorithms based on information gain or Gini impurity, thus they typically perform better with irrelevant inputs.
   - Naive Bayes might suffer from irrelevant inputs as it tends to consider all features equally important, due to its assumption of independence between features.

3. **Predictive Power**:
   - Naive Bayes is simple and may not capture complex relationships well.
   - Decision Trees can model complex relationships but require proper tuning.
   - Boosted version generally have higher predictive power.
   -    v2:
   - Boosted versions of decision trees, like Random Forests or Gradient Boosting Machines (GBMs), typically have higher predictive power as they combine the strengths of multiple learners to reduce overfitting and variance.
   - Naive Bayes, while effective in certain scenarios, might have limited predictive power due to its strong assumptions.

4. **Mixed Types of Data**:
   - Naive Bayes can handle mixed data types.
   - Decision Trees can handle mixed types but may need special handling for categorical features.
   - Boosted version perform well with mixed data.
   -    v2:
   - Decision trees can naturally handle different types of data.
   - Naive Bayes requires different models for different data types (e.g., Gaussian Naive Bayes for continuous data, Multinomial Naive Bayes for categorical data), which increases implementation complexity.

5. **Scalability**:
   - Naive Bayes is highly scalable.
   - Decision Trees can be scalable with proper pruning but may overfit.
   - Boosted decision tree can be computationally expensive, especially for large datasets.
   - Boosted bayes depends on the basic bayes classifier it used and the scale of data
   -    v2:
   - Naive Bayes classifiers, due to their simplicity, generally have higher scalability when dealing with large datasets.
   - Decision trees and, in particular, their boosted versions might require more computational resources and time on large datasets.

Based on these considerations, your recommendations are sound:

- If scalability is a concern and you have a large dataset, Naive Bayes is a good choice.
- For mixed data types, all options are viable, with boosted versions likely offering better predictive power.
- When dealing with outliers and irrelevant inputs, Decision Trees or Boosted Decision Trees are more robust.
- If predictive power is a top priority and computational resources are available, Boosted Decision Trees are a strong choice.

Ultimately, the choice should align with your specific dataset, problem, and resource constraints. It's wise to experiment with multiple classifiers and perform thorough evaluations to determine the best-performing model for your task.
