# Concrete Project: Part 2 (Deep Learning)
### Table of Contents:
  1. Project Introduction
  2. Methodology
  3. File Structure (Todo)
  3. Results (Todo)
 
## Recent Update:  
__Instead of haphazardly arranging the generated features into an image format, it would be worth trying a more sophisticated embedding into the image space by using a ConvTranspose2d generator for upsampling and an Autoencoder architecture.__
  
## Project Introduction:
We attempt to predict the compressive strength of concrete from its ingredients and age. This project uses the dataset found [here](https://www.kaggle.com/maajdl/yeh-concret-data).
As [maajdl](https://www.kaggle.com/maajdl) indicates:
> Concrete is the most important material in civil engineering.
> Concrete compressive strength is a highly nonlinear function of age and ingredients.<br/>

We have good reason to believe that the problem is important and non-trivial.<br/> 
* In __Part 1__, we took a naive approach to the regression problem, and yet still managed to achieve useful results.<br/>
* In __Part 2__, we attempt to address the slightly different problem of predicting whether the compressive strength of concrete surpasses 34.4 MPa given its components and age (e.g. 3kg slag,1kg water,22 days old). It turns out that the later problem is slightly more relavent in industy as generating data is quite cost effective because it involves low accuracy methods and machinery. The data we used for this project was intended for regression, so we set the threshold to surpass to be the median of the dataset for convenience, to balance the positive and negative classes. To choose another target would be an exercise in negative sampling, and would fall outside the scope of this project. 

## Methodology:
1. Handle Outliers<br/>
In our previous investigation, we visualized the data using boxplots and noticed that the features contained outliers. We chose to simply drop those outliers, but for this project we decided to **Winsorize the outliers**. 

2. Data Preprocessing<br/>
We noticed that the many of the features were left skewed or right skewed, non-normal distributions. So we tried applying to log and sqrt transforms to the data as a preprocessing step. But ultimately found that the success of our ML algorithms was unaffected by the transforms so we simply **standardized the data**. Theory suggests that this should be a suprising result as the success of many ML algorithms relies on the data being normally distributed.

3. Baseline<br/>
We computed a basline by using the mode of our data,closest prototype(Averaged positive and negative class and selected label based on min distance to average of class), standard sklearn estimators including Logistic regression, Gradient Boosting, KNN, Ensemble etc.

4. Feature Generation<br/>
We generated several new features by combining and transforming the original features. **Some new features include log(A), A*B, A^2+4B+C etc** where A,B,C are old features.

5. Feature Selection<br/>
We used **Principle Component Analysis** and **Recursive Feature Elimination** to select from our new features.

6. Retest Baseline<br/>
We tested the baseline learners with the transformed data.

7. Simple Linear Neural Network<br/>
We built a simple Neural Netowrk in PyTorch containing Linear,ReLU,Dropout and BatchNorm layers. 

8. Transform Tabular Data to Image Data<br/>
We used the generated features from step 4 to populate a 28X28 array. __Pixels of functions that are close together in the function space are close together in the image__.

9. Standard Convolutional Neural Network<br/>
We built and trained a simple Convolutional Neural Network. __Using a pretrained model or more complex architectures seemed out of the question given the orginality of the data and its relatively small size.__

10. Hyperparameter Tuning<br/>
We tuned parameters like layer size, dropout rate, learning rate, but did not rigorously test architectures. RayTune tuning was finicky, so we __wrote and tested custom hyperparamter tuning loops modelled after Scikit-Learns GridSearch__.


