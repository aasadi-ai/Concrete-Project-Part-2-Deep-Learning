# Concrete Classification
### Table of Contents:
  1. Project Introduction
  2. Methodology
  3. File Structure (Todo)
  3. Results (Todo)
  
## Project Introduction:
We aim to predict whether the compressive strength of concrete surpasses 34.4 MPa given its components and age (e.g. 3kg slag,1kg water,22 days old). 
The data we used for this project was intended for regression, so we set the threshold to surpass to be the median of the dataset for convenience, to balance the positive and negative classes. Even though it's relatively arbitrary, if we assume that our dataset is representative, than the result of classification would be being able to tell whether a  concrete mixture was "stonger or weaker than most mixtures".

## Methodology:
1. Handle Outliers<br/>
In our previous investigation, we visualized the data using boxplots and noticed that the features contained outliers. We chose to simply drop those outliers, but for this project we decided to **Winsorize the outliers**. 

2. Data Preprocessing<br/>
We noticed that the many of the features we're left skewed or right skewed, non-normal distributions. So we tried applying to log and sqrt transforms to the data as a preprocessing step. But ultimately found that the success of our ML algorithms was unaffected by the transforms so we simply **standardized the data**.

3. Baseline<br/>
We computed a basline by using the mode of our data,closest prototype(Averaged positive and negative class and selected label based on min distance to average of class), standard sklearn estimators including Logistic regression, Gradient Boosting, KNN etc.

4. Feature Generation<br/>
We generated several new features by combining and transforming the original features. **Some new features include log(A), A*B, A^2+4B+C etc** where A,B,C are old features.

5. Feature Selection<br/>
We used **Principle Component Analysis** and **Recursive Feature Elimination** to select from our new features.

6. Retest Baseline<br/>
We tested the baseline learners with the transformed data.

7. Simple Neural Network<br/>
We build a simple NN in pyTorch containing Linear,ReLU,dropout and batchNorm layers. 

8. Transform Tabular Data to Image Data<br/>
We used the generated features from step 4 to populate a 28X28 array. Pixels of functions that are close together in the function space are close together in the image.

9. Simple Convolutional Neural Network<br/>
We built and trained a simple CNN

10. Hyperparameter Tuning<br/>
We tuned params like layer size, dropout rate, learning rate, but did not rigorously test architectures.


