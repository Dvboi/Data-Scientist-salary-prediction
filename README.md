# Data Scientist Average Salary prediction: Project Overview
* Created a tool that estimates data science salaries (MAE ~ $ 26K) to help data scientists (specifically in U.S.) negotiate their income when they get a job.
* Collected Dataset from [Kaggle](https://www.kaggle.com/andrewmvd/data-scientist-jobs) which had almost 4000 job postings scraped from Glassdoor.   
* Engineered features from the text of each job description to quantify the value companies put on python, R, sql ..etc.   
* Used Lasso Regression and then Gradient Boosters to reach the best model. 
* Built a Flask API endpoint for the Productionization of model.   

## Installation Guide   

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Seaborn](https://seaborn.pydata.org/)
- [NLTK](https://www.nltk.org/)
- [Flask](https://flask.palletsprojects.com/)  
- [WordCloud](https://pypi.org/project/wordcloud/)
- [CatBoost](https://catboost.ai/docs/concepts/python-installation.html)
- [Requests](https://pypi.org/project/requests/)


OR if you just want to use the app you can clone the requirements.txt file from here and type `pip install -r requirements.txt` in the Command Line / IDE environment,then run the app as shown below in Productionization part.     


You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html) or any other IDE of your choice.  

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the IDE and more included.     


## CODE   


### Data Collection    
The Data was collected from [Kaggle](https://www.kaggle.com/andrewmvd/data-scientist-jobs)   
With each Job Posting we had the following:   
*	Job title
*	Salary Estimate
*	Job Description
*	Rating
*	Company Name
*	Location
*	Company Headquarters 
*	Company Size
*	Company Founded Year
*	Type of Ownership 
*	Industry
*	Sector
*	Revenue
*	Competitors  
* Easy Apply   


### Data Cleaning
After collecting the data, We need to clean it up so that it is usable for our model. I made the following changes and created the following variables:

*	Parsed numeric data out of salary into min and max salary and created a column of average salary.  
*	Made column for hourly wages   
*	Cleaned Company name text 
*	Made a new column for company state 
*	Added a column for if the job was at the company’s headquarters 
*	Transformed founded year into age of company 
*	Made columns for different skills that were listed in the job description:
    * Python  
    * R  
    * Sql  
    * Cloud Skills (aws,azure etc)  
    * Big data skills (Hadoop,spark etc)  
    * Visual skills (Tableau,Power BI etc)  
*	Created columns for simplified job title and Seniority 
*	Created columns for job description length.    

The entire code is available in the [Data Cleaning Notebook](https://github.com/Dvboi/Data-Scientist-salary-prediction/blob/master/Data_cleaning.ipynb)


### Exploratory Data Analysis   
After cleaning analysis was our next step. Ianalyzed the various distribution of the data and it's features.   
The entire notebook with dozens of Visualizations and code is available [here](https://github.com/Dvboi/Data-Scientist-salary-prediction/blob/master/Exploratory%20Data%20Analysis.ipynb)   
And a few are here:   

* The distribution of Ratings across Companies   
![rating](https://github.com/Dvboi/Data-Scientist-salary-prediction/blob/master/Rating.png)    
`-1 Rating is not possible, it was there to indicate missing values which we later imputed with median of the column in model building.`         


* Are the Age of Company and Average Salary related ?  
![ageVsal](https://github.com/Dvboi/Data-Scientist-salary-prediction/blob/master/ageVavg.png)    
   
   
Hmm... maybe slightly negative.   
   
   
* Let's look at all the Correlations    
    
![corr](https://github.com/Dvboi/Data-Scientist-salary-prediction/blob/master/Correlations.png)    

* What are the top jobs ?     
    
![job](https://github.com/Dvboi/Data-Scientist-salary-prediction/blob/master/jobs.png)     
  
  
* What are the top tools ?    

![tool](https://github.com/Dvboi/Data-Scientist-salary-prediction/blob/master/Top_tools.png)     


* What are the states that have most number of openings (Job-role wise) ?     

![sr](https://github.com/Dvboi/Data-Scientist-salary-prediction/blob/master/statewise_role.png)      


* Let's look at the top 3 roles statistics now:     
  
![3role](https://github.com/Dvboi/Data-Scientist-salary-prediction/blob/master/3role_sal.png)     


* Most common words in job description     
   
![c_wrds](https://github.com/Dvboi/Data-Scientist-salary-prediction/blob/master/common_words.png)    
  
  

**Remember to check out the notebook for more visuals and Pivot tables for more information**     


### Model Building    

The Phases here were:
* Data Preprocessing which included feature encoding (both label and one-hot) and feature scaling (for regression model).   
* Splitting data into 80-20 for Train-Test respectively.   
* Tried several models,the Best ones were-
  * As we had sparse data (sparse matrix is the one which has lots of zeros) , Lasso regression was used.  
  * Next were Gradient Boosters, Catboost Regressor was used as we had lots of categorical features(catboost is good for category heavy problems) and it takes care of feature encoding for us, and feature scaling is not important in Tree models (data preprocessing can be skipped).    
* I chose `MAE` for two reasons-
  > Good for Regression Tasks  
  > Easy to interpret   
* Finally chose Catboost as it yielded a better MAE than Lasso (26k).    

**Feature Importances for catboost model:**  
   
 ![feat_imp](https://github.com/Dvboi/Data-Scientist-salary-prediction/blob/master/feat_imp.png)    
 
 ### Productionization   
 
 In this step, I built a flask API endpoint that was hosted on a local webserver.The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary.   
 To run this on your machine,clone the [FlaskAPI](https://github.com/Dvboi/Data-Scientist-salary-prediction/tree/master/FlaskAPI) Folder and then after navigating to this directory on the Command line type `python wsgi.py` . After that open [getting_data.py](https://github.com/Dvboi/Data-Scientist-salary-prediction/blob/master/FlaskAPI/getting_data.py) in your favourite IDE and run it.   
 It takes the input automatically from sample_data.py file but you can enter your own input by just changing this `data = {'input':your input}`.     
 **Make sure your input is within a list and comma separated.**     
 
 
So,this was my end-to-end model for Data-role salary estimate prediction, hope you liked it!.  

 Want to connect? Here's my:   
 
 [![badge](https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devansh-verma-609218148/).     
 
       
        
### Happy Learning!!!













