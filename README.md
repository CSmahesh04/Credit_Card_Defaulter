<h1 align='center'>CREDIT CARD DEFAULER CLASSIFIER</h1>

Credit Card defaulters are a huge liability to any bank which offers the service. It can lead to millions of dollars in revenue loss. To mitigate this loss, companies employ Data Scientists to build models to predict whether a customer will be likely to default on the payment or not. Data Scientists in these companies utilize various historic and demographic data points such as Age, Gender, Education level, Marital Status, Credit limit and Previous payments.  The objective of this project is to build and implement a ML model which predicts whether a customer will default on a payment or not.

The dataset being used is from a bank in Taiwan. The dataset contains 25 variables. The dataset can be found here at the UCI Machine Learning Repository:  http://archive.ics.uci.edu/ml. For the sake of simplicity I did all aspects of this project in a single IPython notebook.

## Technologies Used
<details>
<a name="Technologies_Used"></a>
<summary>Show/Hide</summary>
<br>
  
* <strong>Python</strong>
* <strong>Pandas</strong>
* <strong>Numpy</strong>
* <strong>Matplotlib</strong>
* <strong>Seaborn</strong>
* <strong>Scikit-Learn</strong>
* <strong>XGBoost</strong>
</details>


## The Data
<details>
<a name="The Data"></a>
<summary>Show/Hide</summary>
<br>

After mounting my google drive onto Google collab, the first thing I did was look at the data and get a feel of what it comprises of. The dataset has 30,000 rows and 25 unique columns. Of these 25 columns 23 are feature variables, 1 is the target label and 1 is just an ID column, which is useless for our use case. The data is mostly numeric with datatypes of both int64 and float64.

<h5 align="center">Datatypes in the Dataset</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/info.PNG" width=600 height=500>
</p>

Next I performed a preliminary statistical analysis of all the columns,this can be seen in the picture below. The main insights from this are that the average credit limit of the customers is **1500 New Taiwan Dollar(TWD)**, and the maximum is **30,000 TWD**. The average age of the customers for this bank is 25 years with the youngest customer being 21 years old and oldest being 79. This shows that the bank's primary customer base is made up of young adults in their 20's. Historically young adults are considered to be very high risk to lend money to.

<h5 align="center">Statistical Information on the Dataset</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/describe.PNG" width=650 height=350>
</p>

</details>

## Visualizing the Data
<details>
<a name="Visualizing the Data"></a>
<summary>Show/Hide</summary>
<br>
  
Upon a quick inspection, the data seems to have no null/missing values. Below is the histogram plot of all the columns in the dataset. Nothing seems out of place and seems normal. The ID column is not required for our use case so it will be removed.

<h5 align="center">Histogram Plot of all Columns</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/red_hist.PNG" width=850>
</p>

Looking at the data closely, it is evident that the dataset with respect to the total number of defaulters is not balanced. Out of a total of 30,000 customers, only 6,636 have defaulted on their payments. That is approximately 22.12% only. The remaining 77.88 percent of customers have paid back their credit card debt. This unbalanced dataset is not ideal, but when it comes to fraud detection, an unbalanced dataset is a common occurrence. A heatmap is a very convenient and powerful way to find any obscure correlations between features.

<h5 align="center">Heatmap of Correlations</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/heatmap.PNG" width=850>
</p>

Next I wanted to check whether Age, Sex, Education or Marital Status have any bearing on a customer defaulting on their payment. It seems like there are more female customers than male customers and that women in general seem to be more careful in paying back their credit cards. There also seems to be credit card debt amongst married customers rather than single ones.

<h5 align="center">Bar Graphs of Age, Sex, Education and Marital Status</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/age.PNG" width=800>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/graph_1.PNG" width=850>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/graph_2.PNG" width=850>
</p>

I now want to check if the defaulter's credit limit tends to deviate from the general trend of those who don't default. The below graph is a combination of both a histogram and KDE. It seems like there is no outstanding feature with regards to credit limit when it comes to defaulting on payment.

<h5 align="center">Credit Limit vs Defaulting</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/kde1.PNG" width=750>
</p>

The below graph shows us the total amount paid by customers previously on their credit cards. From the graph it is evident that customers who default on payment usually tend to not pay a lot of their previous bills. This is shown by the high frequency of defaulters paying very little in their historic bill payment record.

<h5 align="center">Previous Payments by Customers</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/kde2.PNG" width=750>
</p>

The distribution of gender with respect to the total credit limit does not seem to have any credence. The same plot with marital status vs credit limit seems to show us that the average married customer has a higher limit than everyone else. There also seems to be many outliers for each status, which I find odd.

<h5 align="center">Boxplots of Sex and Marriage vs Credit Limit</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/box1.PNG" width=700>
</p>

<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/box2.PNG" width=700>
</p>
</details>

## Data Cleaning and Preprocessing
<details>
<a name="Data Cleaning and Preprocessing"></a>
<summary>Show/Hide</summary>
<br>
  
The Sex, Education and Marriage are categorical columns with more than 2 categories. I used the **OneHotEncoder** function from **Sci-kit Learn** library. This converts the below shows table from this:
<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/sex_normal.PNG" width=400>
</p>

To this table:

<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/sex_encoded.PNG" width=600>
</p>

Since the remaining all columns are continuous numerical in nature, we don't need to encode them, only scale the values. Before this step, I removed the label columns and stored it separately. For scaling I used the **MinMaxScaler** from **Sci-kit Learn**.
</details>

## Model Training and Evaluation
<details>
<a name="Model Training and Evaluation"></a>
<summary>Show/Hide</summary>
<br>
  
The whole EDA, data cleaning, preprocessing and model evaluation are in the same IPyhton notebook uploaded to this repository. I split the dataset with 30,000 rows into 3 sets; Train, Validation and Test. For cross validation, I used the Stratified 5-fold. I trained 5 different models for creating a very good classifier. I trained SVM, Naive Bayes, KNN, AdaBoost and XGBoost models. To no one's surprise, XGBoost performed the best out of all the models. 

For all the above models I trained them each without GridSearch initially and then later on with GridSearch. The SVM and KNN model improved drastically with GridSearch while the rest showed a slightly better performance. Below is the table which shows the train and test accuracy of all the models I implemented.

<h5 align="center">Models Accuracy Table</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/models2.PNG" width=600>
</p>

Since I realized XGBoost was performing the best I spent some more time trying to optimize the model by tuning the hyperparameters. But even after getting the most optimal performance on the dataset, the f1-score for the defaulters was low. This is definitely because of the severely unbalanced dataset. Below is the heatmap of the confusion matrix and classification report of the optimized XGBoost model on the final test data.

<h5 align="center">Confusion Matrix and Classification Report</h5>
<p align="center">
  <img src="https://github.com/CSmahesh04/Credit_Card_Defaulter/blob/main/Images/matrix_report.PNG" width=600>
</p>

</details>

