# Databricks notebook source
dbutils.fs.ls("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/comments")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10 questions
# MAGIC 
# MAGIC #### We will mainly study the data under the subreddit 'CryptoCurrency'

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Business goal 1:
# MAGIC EDA: How much has the cryptocurrency community on Reddit grown over the period of time in the given data and which cryptocurrency receives the most attention? 
# MAGIC #### Technical proposal:
# MAGIC Technical proposal: Determine the total count of submissions/comments by month. Plot the total count against time to explore the trend in growth. Use Regex to find submissions and comments that include keywords such as “bitcoin”, “dogecoin”, “tether”, “ethereum”. Conduct counts on those submissions and comments by month and by year, together with the total count of each currency. Tabulate the result to see which which cryptocurrency behaves the best.
# MAGIC $$$$
# MAGIC 
# MAGIC #### Business goal 2:
# MAGIC EDA: Whether the cryptocurrency-related discussions are highly rated on Reddit, can we trust the labels given by Reddit such as gilded? 
# MAGIC #### Technical proposal： 
# MAGIC Technical proposal:  Determine the distribution of scores (upvotes - downvotes) for cryptocurrency subreddit submissions by distinguished and gilded variables. Produce a violine plot to visualize the distribution. Find the basic distribution statistics and check whether cryptocurrency subreddits are highly rated. 
# MAGIC $$$$
# MAGIC 
# MAGIC #### Business goal 3:
# MAGIC EDA: When are people more willing to talk about cryptocurrency one day(during work hours or after-work hours)? Or What time period is the cryptocurrency discussion more heated?
# MAGIC #### Technical proposal： 
# MAGIC Technical proposal: Explore the daily activity of subreddits and determine when is Reddit the most active throughout a week. Create three dummy variables during three time periods 0:00 am- 7:59 am (Sleep), 8:00 am-15:59 pm (working time), and 16:00 pm-23:59 pm(Afterwork). 
# MAGIC $$$$
# MAGIC 
# MAGIC #### Business goal 4:
# MAGIC EDA: How many words should a poster write in the comment to gain a high score, get distinguished, or get gilded under cryptocurrency-related subreddits? 
# MAGIC #### Technical proposal:
# MAGIC Technical proposal: First generate the new variable, the length of every comment under the subreddits related to cryptocurrencies. Then find the correlation between the comment length and variables distinguished/score/gilded by plotting the distribution of comment length based on scores(barplot), and dummy variables(distinguished/gilded)(violin plots). Also, compute the correlation coefficients between the comment length and variables distinguished/score/gilded. 
# MAGIC $$$$
# MAGIC 
# MAGIC #### Business goal 5:
# MAGIC NLP: With the popularity of cryptocurrencies, what are the most important topics or factors people care about regarding cryptocurrencies?
# MAGIC #### Technical proposal： 
# MAGIC Technical proposal: First split the sentences into words, stem, and delete stopwords. Then use CountVectorizer to count the number of words with real meanings. Finally, use TF-IDF to find the words’ importance and extract the most important words. 
# MAGIC $$$$
# MAGIC 
# MAGIC #### Business goal 6:
# MAGIC NLP:  What’s people’s attitude on Reddit posts and comments that mention the buzzwords?
# MAGIC #### Technical proposal:
# MAGIC Technical proposal: Reuse the most frequent words acquired earlier, and filter for the sentences that contain the buzzwords. Then perform sentiment analysis on the corpus. Use TextBlob and Vader to find the sentiment polarity. Also, do the data visualization of the sentimental polarity distribution.
# MAGIC 
# MAGIC $$$$
# MAGIC 
# MAGIC #### Business goal 7:
# MAGIC NLP: For one of the most popular branches of cryptocurrency, how does sentiment related to dogecoin change with respect to its price?
# MAGIC #### Technical proposal：
# MAGIC Technical proposal: Use NLP to identify posts that mention dogecoins, or look at future forms of dogecoins. Perform sentiment analysis on posts and assign a positive or negative value to each post. To determine the market development potential of dogecoin as an emerging virtual currency.
# MAGIC $$$$
# MAGIC 
# MAGIC #### Business goal 8:
# MAGIC ML: We want to automatically generate the community that a post belongs to rather than let the poster find it themselves. 
# MAGIC #### Technical proposal:
# MAGIC Technical proposal: Based on the user's theme, we recommend what subreddit theme is suitable for him. Build a Naive Bayes model to determine the user's topic, and finally, use the model to suggest his own suitable topic for the new post.
# MAGIC $$$$
# MAGIC 
# MAGIC #### Business goal 9:
# MAGIC ML: We want to help Reddit to find out whether a cryptocurrency’s comment is controversial or not. 
# MAGIC #### Technical proposal:
# MAGIC First, properly choose important and distinct variables that can be used to predict controversial variables(dummy variable). And then utilize regression and clustering algorithms to predict the values of controversial variables. Finally, compare the real variables and predicted variables and get the algorithm prediction accuracy and find out whether the algorithm can be put into real-world implementation. 
# MAGIC $$$$
# MAGIC 
# MAGIC #### Business goal 10:
# MAGIC ML: We want to help Reddit to judge a cryptocurrency’s comment’s quality based on the overall ratings it received.
# MAGIC #### Technical proposal:
# MAGIC First, properly choose important and distinct variables that can be used to predict the score variable(integer variable). And then utilize clustering algorithms to predict the values of the score variable. Finally, compare the real variables and predicted variables and get the algorithm prediction accuracy and find out whether the algorithm can be put into real-world implementation.
# MAGIC $$$$

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Data

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np

# COMMAND ----------

comments = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/comments")
submissions = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/submissions")

# COMMAND ----------

submissions_filtered = submissions.filter(submissions.subreddit == "CryptoCurrency")
comments_filtered = comments.filter(comments.subreddit == "CryptoCurrency")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Cleaning
# MAGIC #### Comments Dataset
# MAGIC 
# MAGIC According to the paper [The Pushshift Reddit Dataset](https://arxiv.org/pdf/2001.08435.pdf), There are some interesting columns in the comments:
# MAGIC + "Body" is the original text of the comments and is the main part of the dataset. Text analysis will be performed here.(String)
# MAGIC + "Created_utc" is the unix timestamp of when the comment was created.(Integer)
# MAGIC + "Score" shows the popularity of the comment. (Integer)
# MAGIC + "Gilded" shows how many times the comment has received reddit gold.(Long)
# MAGIC + "Controversial" refers to whether the comment was deemed controversial. 2% has a value of 1 and 98% has a value of 0.(Long)
# MAGIC + "Distinguished" shows whether the comment was distinguished or not. This would be either an administrator or a moderator. But in this dataset, about 10% are "Moderated" and 90% are not.(String)
# MAGIC + "Parent_id" links to the post to which the comment belongs. But still need other dataset to get the original text.(String)
# MAGIC + "Author_cakeday" tells if the author is on the first day of registration.(Boolean)

# COMMAND ----------

comments_filtered.printSchema()

# COMMAND ----------

comments_filtered.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We have 24477714 rows in total.

# COMMAND ----------

col_list = ["link_id", "body", "score", "controversiality", "can_gild", "gilded", \
            "created_utc", "distinguished", "parent_id", "author_cakeday"]
comment_cleaning = comments_filtered.select(col_list)

# COMMAND ----------

# Check Null values from comment body
comment_cleaning.select(count(when(isnull('body'), 'body'))).show()

# COMMAND ----------

# Check NaN again from comment body
comment_cleaning.select(count(when(isnan('body'), 'body'))).show()

# COMMAND ----------

# Remove Null and NaN from comment body
comment_cleaned = comment_cleaning.filter(col('body') != "[deleted]").filter(col('body') != "[removed]").filter(~isnull(col("body"))).filter(~isnan(col("body")))

# COMMAND ----------

# Check all column of comment dataframe for null values
col_null = ["link_id", "body", "score", "controversiality", "gilded", \
            "created_utc", "parent_id"] # boolean can not use isnan
comment_cleaned.select([count(when(isnull(v),v)).alias(v) for v in col_null]).show()
#Null value at distinguished means not distinguished.

# COMMAND ----------

# check the [deleted] and [removed]
comment_cleaned.filter(col("body") == "[deleted]").show()
comment_cleaned.filter(col("body") == "[removed]").show()

# COMMAND ----------

comment_cleaned.count()

# COMMAND ----------

# MAGIC %md
# MAGIC we have 21056509 rows in total for comment data after cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC #### Submission Dataset
# MAGIC According to the paper [The Pushshift Reddit Dataset](https://arxiv.org/pdf/2001.08435.pdf), There are some interesting columns in the submissions:
# MAGIC Many rows are similar to those in the comment
# MAGIC + "Title" The title that is associated with the submission, e.g., “What did you think of the ending of Rogue One?” (String)
# MAGIC + "selftext" The text that is associated with the submission (String).
# MAGIC + "Created_utc" is the unix timestamp of when the comment was created.(Integer)
# MAGIC + "Score" shows the popularity of the comment. (Integer)
# MAGIC + "num_comments" The number of comments associated with this submission, e.g., 7 (Integer).
# MAGIC + "over_18" Flag that indicates whether the submission is Not-Safe-For-Work, e.g., false (Boolean).
# MAGIC + "Gilded" shows how many times the comment has received reddit gold.(Long)
# MAGIC + "Distinguished" shows whether the comment was distinguished or not. This would be either an administrator or a moderator. But in this dataset, about 10% are "Moderated" and 90% are not.Flag to determine whether the submission is distinguished2 by moderators. “null” means not distinguished (String).
# MAGIC + "Author_cakeday" tells if the author is on the first day of registration. (Boolean)

# COMMAND ----------

submissions_filtered.printSchema()

# COMMAND ----------

submissions_filtered.count()

# COMMAND ----------

# MAGIC %md
# MAGIC We have 885844 rows in total.

# COMMAND ----------

# Select interested columns from submissions
col_list = ["title","selftext","num_comments", "over_18", "score", "gilded", \
            "created_utc", "distinguished", "author_cakeday"]
submissions_cleaning = submissions_filtered.select(col_list)

# COMMAND ----------

# Check for Null values in 'title'
submissions_cleaning.select(count(when(isnull('title'), 'title'))).show()

# COMMAND ----------

# Check for Null values for all columns
col_null = ["title","selftext","num_comments", "score", "gilded", \
            "created_utc"] # boolean can not use isnan
submissions_cleaning.select([count(when(isnull(v),v)).alias(v) for v in col_null]).show() 

# COMMAND ----------

# Remove [deleted] values in 'selftext' from submissions
submissions_cleaning.filter(col("selftext") == "[deleted]").show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Why not delete [deleted] and [removed] rows in the 'selftext' column of Submission dataframe？
# MAGIC 
# MAGIC Since for most reddit submissions(posts), the title contains sufficient information for our analysis, and 'selftext' column mainly serves as supporting statements, so we would lose a lot of meaningful data if we drop the empty rows from the dataframe based on the 'selftext' column.

# COMMAND ----------

#submissions_cleaned = submissions_cleaning.filter(col('selftext') != "[deleted]").filter(col('selftext') != "[removed]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Transformation
# MAGIC Based on our need, we created different variables for submission and comment dataset.
# MAGIC 
# MAGIC #### For submission dataset, new variables as follows:
# MAGIC + "is_gilded": Boolean variable, shows whether the submission is gilded or not. When gilded = 0, is_gilded = false. Otherwise, is_gilded = true. 
# MAGIC + "is_distinguished": Boolean variable, shows whether the submission is distinguished or not. When distinguished is Null, is_distinguished = false. Otherwise, is_distinguished = true. 
# MAGIC + "year": Integer variable, shows which year people post 
# MAGIC + "month": Integer variable, shows which month people post 
# MAGIC + "day_of_week": String variable, displays the reddit submissions and comments out of the days of the week they were posted
# MAGIC + "hour": Integer variable, shows what time people post 
# MAGIC + "dayhour_dummy": Boolean variable, hour 0-7 as "sleep", hour 8-15 "work time", hour 16-23 as "after-time" (dummy variable)
# MAGIC #### For comment dataset, new variables as folows:
# MAGIC + "year": Integer variable, shows which year people post 
# MAGIC + "month": Integer variable, shows which month people post 
# MAGIC + "day_of_week": String variable, displays the reddit submissions and comments out of the days of the week they were posted
# MAGIC + "hour": Integer variable, shows what time people post 
# MAGIC + "dayhour_dummy": Boolean variable, hour 0-7 as "sleep", hour 8-15 "work time", hour 16-23 as "after-time"

# COMMAND ----------

# df_submission = spark.read.csv("/FileStore/submissions1")
# col_list = ["title","selftext","num_comments", "over_18", "score", "gilded", \
#             "created_utc", "distinguished", "author_cakeday"]
# df_submission = df_submission.toDF(*col_list)
# df_submission.createOrReplaceTempView("cast")
# df_submission = spark.sql("SELECT STRING(title),STRING(selftext),INT(num_comments),BOOLEAN(over_18),INT(score),INT(gilded),INT(created_utc), STRING(distinguished), BOOLEAN(author_cakeday) from cast")
# df_submission.printSchema()
# df_submission.show()

# COMMAND ----------

# Create new dataframes for further processing
df_submission = submissions_cleaning
df_comment = comment_cleaned

# COMMAND ----------

df_submission.select("score","gilded","distinguished").summary().show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### As we only want to know whether the submission is gilded or not and is distinguished or not, (also because the gilded variable distribution is not balanced)
# MAGIC #### we will create new variables is_gilded and is_distinguished

# COMMAND ----------

from pyspark.sql.functions import col

df_submission.filter(col("distinguished").isNull()).count()

# COMMAND ----------

from pyspark.sql.functions import when
df_sub = df_submission.withColumn("is_gilded", \
   when((df_submission.gilded == 0), "false") \
     .otherwise("true") \
  )
df_sub = df_sub.withColumn("is_distinguished", \
   when((df_sub.distinguished.isNull()), "false") \
     .otherwise("true") \
  )
df_sub.show()

# COMMAND ----------

df_com = df_comment.withColumn("is_gilded", \
   when((df_comment.gilded == 0), "false") \
     .otherwise("true") \
  )
df_com = df_comment.withColumn("is_distinguished", \
   when((df_comment.distinguished.isNull()), "false") \
     .otherwise("true") \
  )
df_com.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### After data transformation, the summary table changes as below

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Table: show the summary statistics of scores 

# COMMAND ----------

df_sub.select("score").summary().toPandas()

# COMMAND ----------

### delete outliers in scores
df_sub = df_sub.filter(col("score") <= 2500)

# COMMAND ----------

from pyspark.sql.functions import col, asc,desc
submissions_by_score = df_sub.groupBy("score").count().orderBy(col("count"), ascending=False).collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Table: Show the distribution of score

# COMMAND ----------

score = spark.createDataFrame(submissions_by_score[:25]).toPandas()
score

# COMMAND ----------

submissions_by_gilded = df_sub.groupBy("is_gilded").count().collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Table: Show the distribution of is_gilded and is_distinguished 

# COMMAND ----------

gilded = spark.createDataFrame(submissions_by_gilded).toPandas()
gilded

# COMMAND ----------

submissions_by_distinguished = df_sub.groupBy("is_distinguished").count().collect()

# COMMAND ----------

dist = spark.createDataFrame(submissions_by_distinguished).toPandas()
dist

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create new variables about time:
# MAGIC + "year" in the format of YYYY, shows which year people post
# MAGIC + "month" in the format of MM, shows which month people post
# MAGIC + "day_of_week" to display the reddit submissions and comments out of the days of the week they were posted
# MAGIC + "hour" shows which hour people post
# MAGIC + "dayhour_dummy" Create dummy variables during three time periods 0:00 am- 7:59 am (Sleep), 8:00 am-15:59 pm (working time), and 16:00 pm-23:59 pm(Afterwork).

# COMMAND ----------

df_sub = df_sub.withColumn("year", from_unixtime(col("created_utc"),"yyyy").cast("int"))
df_sub = df_sub.withColumn("month", from_unixtime(col("created_utc"),"MM").cast("int"))
df_sub = df_sub.withColumn("input_timestamp",
    from_unixtime(col("created_utc"),"yyyy-MM-dd"))\
    .withColumn("day_of_week", date_format(col("input_timestamp"), "E"))
df_sub = df_sub.withColumn("hour", from_unixtime(col("created_utc"),"HH").cast("int"))

df_com = df_com.withColumn("year", from_unixtime(col("created_utc"),"yyyy").cast("int"))
df_com = df_com.withColumn("month", from_unixtime(col("created_utc"),"MM").cast("int"))
df_com = df_com.withColumn("input_timestamp",
    from_unixtime(col("created_utc"),"yyyy-MM-dd"))\
    .withColumn("day_of_week", date_format(col("input_timestamp"), "E"))
df_com = df_com.withColumn("hour", from_unixtime(col("created_utc"),"HH").cast("int"))

# COMMAND ----------

df_sub = df_sub.withColumn("dayhour_dummy", when(col("hour").isin([0,1,2,3,4,5,6,7]), "sleep")
                                           .when(col("hour").isin([8,9,10,11,12,13,14,15]), "work time")
                                           .when(col("hour").isin([16,17,18,19,20,21,22,23]), "after-work"))
df_com = df_com.withColumn("dayhour_dummy", when(col("hour").isin([0,1,2,3,4,5,6,7]), "sleep")
                                           .when(col("hour").isin([8,9,10,11,12,13,14,15]), "work time")
                                           .when(col("hour").isin([16,17,18,19,20,21,22,23]), "after-work"))

# COMMAND ----------

df_sub.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Table: Find out the distribution of submissions based on day of work

# COMMAND ----------

submissions_day_of_week = df_sub.groupby("day_of_week").count().toPandas()
submissions_day_of_week

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Table: Find out the distribution of submissions based on human activity

# COMMAND ----------

submissions_day_of_hour = df_sub.groupby("dayhour_dummy").count().toPandas()
submissions_day_of_hour

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join External Data 
# MAGIC ####(Bitcoin price data from https://www.investing.com/)

# COMMAND ----------

df = pd.read_csv('data/csv/Bitcoin-Historical-Data-21-22.csv')
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data cleaning

# COMMAND ----------

# Drop undesired columns
df_bit = df.drop(['Vol.', 'Change %', 'Open', 'High', 'Low'], axis=1)

# Check datatypes
print(f'Before: \n{df_bit.dtypes} \n')

# Change column types
df_bit["Price"] = df_bit["Price"].str.replace(",", "").astype('float')
df_bit['Date'] = pd.to_datetime(df_bit['Date'])

# Check datatypes again
print(f'After: \n{df_bit.dtypes} \n')

# Check missing values
print(df_bit.isna().sum()) # no missing values

df_bit.head()

# COMMAND ----------

# Aggregate to compute count by year and month for the subreddit as a whole
df_q1 = df_sub.groupby("input_timestamp").count().toPandas()

# COMMAND ----------

df_parent = df_q1
df_parent.head()
print(len(df_parent))

# Sort parent dataframe by date descending to check the time range
df_parent = df_parent.sort_values(by=['input_timestamp'], ascending=False)
df_parent = df_parent.reset_index(drop=True)
print(df_parent.head(), '\n')
print(df_parent.tail(), '\n')
# Parent data is from 2021-01-01 to 2022-08-31
# Child data is from 2021-01-01 to 2022-10-31

# Subset the external data to fit in the range of the parent data
id = df_bit.index[df_bit['Date'] == '2022-08-31'][0]
df_child_clean = df_bit.iloc[id:]

print(len(df_child_clean))
print(df_child_clean.head(), '\n')
print(df_child_clean.tail(), '\n')


# COMMAND ----------

# MAGIC %md
# MAGIC #### Join dataframes

# COMMAND ----------

# Rename column for joining
df_parent = df_parent.rename(columns={'input_timestamp': 'Date'})
df_parent = df_parent.rename(columns={'count': 'Count'})
df_child_clean = df_child_clean.rename(columns={'Price': 'Bitcoin_Price'})

# Convert parent dataframe "Date" column to datetime format
df_parent['Date'] = pd.to_datetime(df_parent['Date'])
df_big = pd.merge(df_parent, df_child_clean, on='Date')
df_big.head()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Visualization

# COMMAND ----------

import matplotlib.pyplot as plt
import plotly.express as px
# ! pip install altair
import altair as alt
import plotly.graph_objects as go

# COMMAND ----------

# MAGIC %md
# MAGIC #### EDA question 1: How much has the cryptocurrency community on Reddit grown over the period of time in the given data and which cryptocurrency receives the most attention?

# COMMAND ----------

# Get time from row names
rowName=[]
for row in df_q1.index:
    rowName.append(row)
name = pd.Series(rowName, name='time')

# COMMAND ----------

# plot1
X1 = list(df_q1['input_timestamp'])
Y1 = list(df_q1['count'])

d = {'Date': X1, 'Number of Submission': Y1}


data = pd.DataFrame(data = d)
#create feature

# Sort the Date because there some inconsistent dates
data = data.sort_values(by=['Date'])

fig = px.line(data,x = "Date", y= 'Number of Submission', title = 'Count of Cryptocurrency Subreddit Submissions Over Time' )
# fig = go.Figure()

# fig.add_trace(go.Scatter(x=list(data.Date), y=list(data['Number of Submission'])))

# #Set Title
# fig.update_layout(
#     title = {
#         'text':'Count of Cryptocurrency Subreddit Submissions Over Time',
#         'y':0.93,
#         'x':0.5,
#         'xanchor':'center',
#         'yanchor':'top'
#     })


# Add range slider
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()
fig.write_html("figure_line.html")

# COMMAND ----------

# MAGIC %md
# MAGIC The above line plot shows that there are several significant spikes in the number of posts during Feb 21, May/Jun 21, Aug/Sep 21, and Dec 21. However, the number of posts has been declining since then. To investigate the probable reason behind this decrease in activity, we plan to incoporate external data on the price of cryptocurrencies and determine if there is any correlation between market price of cryptocurrencies and their reddit popularity.

# COMMAND ----------

fig.write_html("figure_line.html")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Implement regex searches for specific keywords of items from cryptocurrencies

# COMMAND ----------

# Use Regex to find submissions that mention different cryptocurrencies
from operator import add
from functools import reduce

df = df_sub

df = df.withColumn('BTC', when(df.title.rlike('(?i)bitcoin|(?i)btc'), 1).otherwise(0))
df = df.withColumn('ETH', when(df.title.rlike('(?i)etherium|ETH'), 1).otherwise(0))
df = df.withColumn('USDT', when(df.title.rlike('(?i)USDT|(?i)tether'), 1).otherwise(0))
df = df.withColumn('USDC', when(df.title.rlike('(?i)USDC'), 1).otherwise(0))
df = df.withColumn('BNB', when(df.title.rlike('(?i)BNB'), 1).otherwise(0))
df = df.withColumn('XRP', when(df.title.rlike('(?i)XRP'), 1).otherwise(0))
df = df.withColumn('BUSD', when(df.title.rlike('(?i)BUSD|(?i)Binance USD'), 1).otherwise(0))
df = df.withColumn('ADA', when(df.title.rlike('(?i)cardano|(?i)ADA'), 1).otherwise(0))
df = df.withColumn('SOL', when(df.title.rlike('(?i)solana|(?i)SOL'), 1).otherwise(0))
df = df.withColumn('DOG', when(df.title.rlike('(?i)dogecoin|(?i)DOGE'), 1).otherwise(0))

df_sml = df.select("BTC","ETH","USDT",'USDC','BNB','XRP','BUSD','ADA','SOL','DOG')

res = df_sml.groupBy().sum().collect()

# COMMAND ----------

# Table 1
dict1 = {'Name' : ['BTC','ETH','USDT','USDC','BNB','XRP','BUSD','ADA','SOL','DOG'],
        'Submission Count' : res[0]}
df_coinCnt = pd.DataFrame(dict1).sort_values(by=['Submission Count'], ascending=False).reset_index(drop=True) 
df_coinCnt.head(10)

# COMMAND ----------

#plot 2
fig = px.bar(df_coinCnt, y="Submission Count", x="Name",
          template='seaborn', title="Number of Submissions for Different Cryptocurrencies",
            width=700, height=500,
            labels={ # replaces default labels by column name
                "Submission Count": "Count of Submissions",  "Name":"Type of Cryptocurrency"
            })
fig.update_layout( # customize font and legend orientation & position
    font_family="Rockwell"
)
fig.show()
fig.write_html('fig_2.html')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC From the above bar chart, we found the top 5 mentioned crytocurrencies are Bitcoin(BTC), ETH(Ethereum), Dogecoin(DOGE), Cardano(ADA), and Solana(SOL). 

# COMMAND ----------

# MAGIC %md
# MAGIC #### EDA question 2: Whether the crpytocurrency discussion on Reddit is highly rated,, can we trust the labels given by Reddit such as distinguished, and gilded?? 

# COMMAND ----------

df_1 = df_sub.sample(0.005)
df_1 = df_1.toPandas()

# COMMAND ----------

df_1.shape

# COMMAND ----------

df_commment = comment_cleaned.sample()
df_commment = df_commment.toPandas()

# COMMAND ----------

fig = px.violin(df_1, y="score", x="is_gilded", box=True, points="all",
          hover_data=["score"],template='seaborn', title="The Distribution of CryptoCurrency Submission Scores Based on Gilded",
            width=700, height=500,
            labels={ # replaces default labels by column name
                "is_gilded": "Whether Gilded",  "score": "Cryptocurrency Subreddit Scores"
            },category_orders={ # replaces default order by column name
                "is_gilded": ["not gilded", "gilded"]
            },)
fig.update_layout( # customize font and legend orientation & position
    font_family="Rockwell"
)
fig.show()
fig.write_html('fig_violin.html')

# COMMAND ----------

# MAGIC %md
# MAGIC From the distribution plot, we can see that the score of cryptocurrency discussions can go up to 5000, which means that most of the cryptocurrency submission is highly rated. In the meanwhile, if we compare the gilded submissions and ungilded submissions, we can find that the gilded submissions have higher scores than ungilded submissions. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### EDA question 3 : When are people more willing to talk about cryptocurrency one day(during work hours or after-work hours)? Or What time period is the cryptocurrency discussion more heated?

# COMMAND ----------

# alt.renderers.enable('html')
alt.Chart(submissions_day_of_week).mark_point().encode(
    x='day_of_week',
    y='hour',
    size='score')

# COMMAND ----------

#create a new data frame about the hour and day of week
df_time = df_sub.sample(0.005)
df_time = df_time.toPandas()

# COMMAND ----------

df_time1 = df_time.groupby(["day_of_week","dayhour_dummy"]).count().reset_index()

# COMMAND ----------

df_time1

# COMMAND ----------

fig = px.line(df_time1, x="day_of_week", y="title", color='dayhour_dummy',
              template='seaborn', title="Number of Submissions by Day of A Week towards Three Periods",
              width=700, height=500,
            labels={ # replaces default labels by column name
                "day_of_week": "day of week",  "title": "Number of CryptoCurrency Submission"
            },)
fig.update_layout( # customize font and legend orientation & position
    font_family="Rockwell",
    legend_title="three periods"
)
fig.show()
fig.write_html('fig_4.html')

# COMMAND ----------

# MAGIC %md
# MAGIC The graph shows the amount of submissions by each day in a week, from three periods of a day. In this figure, it indicates people are most active during the after-work time whatever in any day of week. Also, people like to talk about crptocurrency in monday and thursday. There is a big drop during weekends time, thus, we guess cyptocurrency is not a popular topic in spare time. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### EDA question 4: How many words should a poster write in the comment to gain a high score under CryptoCurrency subreddits?

# COMMAND ----------

df_sub_highscore = df_sub.filter(col("score") > 10)

# COMMAND ----------

df_sub_highscore.count()

# COMMAND ----------

df_4 = df_sub_highscore.sample(0.05)

# COMMAND ----------

df_4 = df_4.withColumn('wordCount', f.size(f.split(f.col('title'), ' ')))
df_4.show()

# COMMAND ----------

df_4 = df_4.toPandas()

# COMMAND ----------

import plotly.express as px
fig = px.bar(df_4, x='wordCount', y='score', title="Words Count for High Score",height=500)
fig.update_layout( # customize font and legend orientation & position
    font_family="Rockwell"
)
fig.show()
fig.write_html('fig_5.html')
# plt.bar('wordCount','score', data = df_4)
# plt.title("Distribution of scores of wordcount")
# plt.xlabel("Count of Word")
# plt.ylabel("Score")
# plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC This graph clearly expresses how many words there are in the high scoring post of reddit. It is interesting to note that the higher the score the more post tends to use few words to describe their problems.

# COMMAND ----------

df_sub.write.parquet("/FileStore/submissions2")
df_com.write.parquet("/FileStore/comments2")
df_big.to_csv('data/csv/Merged_bitcoin.csv')
