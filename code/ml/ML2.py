# Databricks notebook source
# MAGIC %md
# MAGIC #### Business goal 9:
# MAGIC ML: We want to help Reddit to find out whether a cryptocurrency’s comment is controversial or not. 
# MAGIC #### Technical proposal:
# MAGIC First, properly choose important and distinct variables that can be used to predict controversial variables(dummy variable). And then utilize regression and clustering algorithms to predict the values of controversial variables. Finally, compare the real variables and predicted variables and get the algorithm prediction accuracy and find out whether the algorithm can be put into real-world implementation. 
# MAGIC $$$$

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler, RobustScaler,PCA
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml import Pipeline, Model
from pyspark.ml.pipeline import PipelineModel

# COMMAND ----------

#import library
import pyspark.sql.functions as f
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pyspark.ml import Pipeline
from operator import add
from functools import reduce
from pyspark.ml.feature import CountVectorizer,IDF


# COMMAND ----------

import pandas as pd
import numpy as np
import json

import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import roc_curve, roc_auc_score

# COMMAND ----------

spark = SparkSession.builder \
        .appName("RedditNLP") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.4.2") \
    .master('yarn') \
    .getOrCreate()

# COMMAND ----------

spark

# COMMAND ----------

# Read in data
df_sub = spark.read.parquet("/FileStore/submissions2")
df_com = spark.read.parquet("/FileStore/sentimentcomment")
df_bit = pd.read_csv("../../data/csv/Merged_bitcoin.csv", index_col=False)

# COMMAND ----------

df_com.printSchema()

# COMMAND ----------

gilded = df_com.groupby("gilded").count().collect()

# COMMAND ----------

gilded

# COMMAND ----------

scores = df_com.groupby("score").count().collect()

# COMMAND ----------

scores

# COMMAND ----------

contro = df_com.groupby("controversiality").count().collect()

# COMMAND ----------

contro

# COMMAND ----------

df_com = df_com.withColumnRenamed('labels', 'sentiment')

# COMMAND ----------

df = df_com

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Selection

# COMMAND ----------

df = df.drop('link_id','gilded','distinguished','author_cakeday','year','input_timestamp','author', 'author_created_utc','finished_final' ,'created_utc', 'body', 'id', 'parent_id', 'author_created_time_full', 'author_created_time', 'created_time_full', 'created_time', 'text')

# COMMAND ----------

df.printSchema()

# COMMAND ----------

### give weights
# Compute the weights
n = df.count()
n_con_0 = df.filter(col('controversiality')==0).count()
n_con_1 = n - n_con_0

c = 2
weight_con_0 = n/(2*n_con_0)
weight_con_1 = n/(2*n_con_1)

# COMMAND ----------

# Assign the weights to a new column
df = df.withColumn('weight_con', f.when(col('controversiality')==0, weight_con_0).otherwise(weight_con_1))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split the dataset into train and test

# COMMAND ----------

## 
train_data, test_data, predict_data = df.randomSplit([0.7, 0.2, 0.1], 138)

# COMMAND ----------

print("Number of training records: " + str(train_data.count()))
print("Number of testing records : " + str(test_data.count()))
print("Number of prediction records : " + str(predict_data.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create pipeline

# COMMAND ----------

train_data.printSchema()

# COMMAND ----------

df = df.withColumn("controversiality",df.controversiality.cast('int'))
df = df.withColumn("score",df.score.cast('int'))

# COMMAND ----------

otherF = ["can_gild","BTC","ETH","polygon","algorand","robinhood","score"]

# COMMAND ----------

# category to index
stringIndexer_DOW = StringIndexer(inputCol="day_of_week", outputCol="day_index")
stringIndexer_hour = StringIndexer(inputCol="dayhour_dummy", outputCol="hour_index")

stringIndexer_sent = StringIndexer(inputCol="sentiment", outputCol="sent_index")
stringIndexer_dis =StringIndexer(inputCol="is_distinguished",outputCol="dis_index")
stringIndexer_crypt = StringIndexer(inputCol="crypto_term",outputCol="crypto_index")
#label
# stringIndexer_con = StringIndexer(inputCol="controversiality",outputCol="con_index")


# encode month, dayofweek, hour into one vector
onehot_time = OneHotEncoder(inputCols=["month","day_index","hour_index"],
                           outputCols=["monthVec","dayVec","hourVec"])



vectorAssembler_time = VectorAssembler(
    inputCols=[ "monthVec","dayVec","hourVec" ], 
    outputCol= "time")


vector_feature = VectorAssembler(
    inputCols=['time', 'sent_index','dis_index','crypto_index'] + otherF, 
    outputCol= "features")



# COMMAND ----------

# MAGIC %md
# MAGIC ### Model selection 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model 1 - Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC ##### First set of hyperparameters

# COMMAND ----------

# first set of hyperparemeters
rf_sen_1 = RandomForestClassifier(labelCol="controversiality", featuresCol="features", weightCol='weight_con',
                                  numTrees=50, maxDepth=10, seed=22)

rf_sen_1_pipeline = Pipeline(stages= [stringIndexer_DOW, 
                                      stringIndexer_hour,
                                      stringIndexer_sent,
                                      stringIndexer_dis,
                                      stringIndexer_crypt,
                                      stringIndexer_con,
                                      onehot_time, 
                                      vectorAssembler_time,
                                      vector_feature,
                                      rf_sen_1])

rf_sen_1_model = rf_sen_1_pipeline.fit(train_data)
rf_sen_1_train = rf_sen_1_model.transform(train_data)

# COMMAND ----------

## save and load model
# Save random forest pipeline into DBFS
rf_sen_1_model.write().overwrite().save('/FileStore/rf_sen_1_model')

# COMMAND ----------

# Read the pipeline
rf_sen_1_model = PipelineModel.load('/FileStore/rf_sen_1_model')

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Test and Evaluation

# COMMAND ----------

# Transform the test data by the training model
rf_sen_1_test = rf_sen_1_model.transform(test_data)

# COMMAND ----------

# Confusion Matrix
rf_sen_1_test.crosstab('controversiality', 'prediction').show()

# COMMAND ----------

# ACC
rf_sen_1_acc = MulticlassClassificationEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                                 metricName="accuracy").evaluate(rf_sen_1_test)

# MSE
rf_sen_1_mse = RegressionEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                    metricName="mse").evaluate(rf_sen_1_test)

# VAR
rf_sen_1_var = RegressionEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                   metricName="var").evaluate(rf_sen_1_test)

# COMMAND ----------

print("Accuracy = %g" % rf_sen_1_acc)
print("Test Error = %g" % (1.0 - rf_sen_1_acc))
print("MSE = %g" % rf_sen_1_mse)
print("Var = %g" % rf_sen_1_var)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Second set of hyperparameters

# COMMAND ----------

rf_sen_2 = RandomForestClassifier(labelCol="controversiality", featuresCol="features", weightCol='weight_con',
                                  numTrees=50, maxDepth=15, seed=22)

rf_sen_2_pipeline = Pipeline(stages= [stringIndexer_DOW, 
                                      stringIndexer_hour,
                                      stringIndexer_sent,
                                      stringIndexer_dis,
                                      stringIndexer_crypt,
                                      stringIndexer_con,
                                      onehot_time, 
                                      vectorAssembler_time,
                                      vector_feature,
                                      rf_sen_2])

rf_sen_2_model = rf_sen_2_pipeline.fit(train_data)
rf_sen_2_train = rf_sen_2_model.transform(train_data)

# COMMAND ----------

## save and load model
# Save random forest pipeline into DBFS
rf_sen_2_model.write().overwrite().save('/FileStore/rf_sen_2_model')

# COMMAND ----------

# Read the pipeline
rf_sen_2_model = PipelineModel.load('/FileStore/rf_sen_2_model')

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Test and Evaluation

# COMMAND ----------

# Transform the test data by the training model
rf_sen_2_test = rf_sen_2_model.transform(test_data)

# COMMAND ----------

# Confusion Matrix
rf_sen_2_test.crosstab('controversiality', 'prediction').show()

# COMMAND ----------

# ACC
rf_sen_2_acc = MulticlassClassificationEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                                 metricName="accuracy").evaluate(rf_sen_2_test)

# MSE
rf_sen_2_mse = RegressionEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                    metricName="mse").evaluate(rf_sen_2_test)

# VAR
rf_sen_2_var = RegressionEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                   metricName="var").evaluate(rf_sen_2_test)

# COMMAND ----------

print("Accuracy = %g" % rf_sen_2_acc)
print("Test Error = %g" % (1.0 - rf_sen_2_acc))
print("MSE = %g" % rf_sen_2_mse)
print("Var = %g" % rf_sen_2_var)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model 2 - Gradient Boosted Tree Classifier

# COMMAND ----------

# MAGIC %md
# MAGIC ##### First set of hyperparameters

# COMMAND ----------

gb_sen_1 = GBTClassifier(labelCol="controversiality", featuresCol="features", weightCol='weight_con',
                                  maxIter=10, maxDepth=10, seed=22)

gb_sen_1_pipeline = Pipeline(stages= [stringIndexer_DOW, 
                                      stringIndexer_hour,
                                      stringIndexer_sent,
                                      stringIndexer_dis,
                                      stringIndexer_crypt,
                                      stringIndexer_con,
                                      onehot_time, 
                                      vectorAssembler_time,
                                      vector_feature,
                                      gb_sen_1])

gb_sen_1_model = gb_sen_1_pipeline.fit(train_data)
gb_sen_1_train = gb_sen_1_model.transform(train_data)

# COMMAND ----------

gb_sen_1_model.write().overwrite().save('/FileStore/gb_sen_1_model')

# COMMAND ----------

gb_sen_1_train.printSchema()

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Test and Evaluation

# COMMAND ----------

# Transform the test data by the training model
gb_sen_1_test = gb_sen_1_model.transform(test_data)

# COMMAND ----------

# Confusion Matrix
gb_sen_1_test.crosstab('controversiality', 'prediction').show()

# COMMAND ----------

# ACC
gb_sen_1_acc = MulticlassClassificationEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                                 metricName="accuracy").evaluate(gb_sen_1_test)

# MSE (less than 1 so cannot use RMSE)
gb_sen_1_mse = RegressionEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                    metricName="mse").evaluate(gb_sen_1_test)

# VAR
gb_sen_1_var = RegressionEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                   metricName="var").evaluate(gb_sen_1_test)

# COMMAND ----------

print("Accuracy = %g" % gb_sen_1_acc)
print("Test Error = %g" % (1.0 - gb_sen_1_acc))
print("MSE = %g" % gb_sen_1_mse)
print("Var = %g" % gb_sen_1_var)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Second set of hyperparameters

# COMMAND ----------

gb_sen_2 = GBTClassifier(labelCol="controversiality", featuresCol="features", weightCol='weight_con',
                                  maxIter=15, maxDepth=15, seed=22)

gb_sen_2_pipeline = Pipeline(stages= [stringIndexer_DOW, 
                                      stringIndexer_hour,
                                      stringIndexer_sent,
                                      stringIndexer_dis,
                                      stringIndexer_crypt,
                                      stringIndexer_con,
                                      onehot_time, 
                                      vectorAssembler_time,
                                      vector_feature,
                                      gb_sen_2])

gb_sen_2_model = gb_sen_2_pipeline.fit(train_data)
gb_sen_2_train = gb_sen_2_model.transform(train_data)

# COMMAND ----------

gb_sen_2_model.write().overwrite().save('/FileStore/gb_sen_2_model')

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Test and Evaluation

# COMMAND ----------

# Transform the test data by the training model
gb_sen_2_test = gb_sen_2_model.transform(test_data)

# COMMAND ----------

# Confusion Matrix
gb_sen_2_test.crosstab('controversiality', 'prediction').show()

# COMMAND ----------

# ACC
gb_sen_2_acc = MulticlassClassificationEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                                 metricName="accuracy").evaluate(gb_sen_2_test)

# MSE (less than 1 so cannot use RMSE)
gb_sen_2_mse = RegressionEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                    metricName="mse").evaluate(gb_sen_2_test)

# VAR
gb_sen_2_var = RegressionEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                   metricName="var").evaluate(gb_sen_2_test)



# COMMAND ----------

print("Accuracy = %g" % gb_sen_2_acc)
print("Test Error = %g" % (1.0 - gb_sen_2_acc))
print("MSE = %g" % gb_sen_2_mse)
print("Var = %g" % gb_sen_2_var)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model 3 PCA + Logistic Regression Model

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Modify feature transformation

# COMMAND ----------

# ## transfer bool variable to int for logistic regression
# df1 = df.withColumn('gild_index', f.when(col('can_gild')==True, 1).otherwise(0))
# # train_test split
# train_data, test_data, predict_data = df1.randomSplit([0.7, 0.2, 0.1], 138)

# COMMAND ----------

otherF = ["BTC","ETH","polygon","algorand","robinhood","score"]

vector_feature = VectorAssembler(
    inputCols=['time', 'sent_index','dis_index','crypto_index'] + otherF, 
    outputCol= "features")

#standardize but also robust to outliers
robustScaler_f = RobustScaler(inputCol="features", outputCol="scaled_features",
                           withScaling=True, withCentering=False,
                           lower=0.25,upper=0.75)
# Add PCA in piplines
pca = PCA(k=3, inputCol="scaled_features", outputCol="pcaFeatures")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### First set of hyperparameter

# COMMAND ----------

lr_sen_1 = LogisticRegression(labelCol='controversiality', featuresCol="pcaFeatures", 
                              weightCol="weight_con", maxIter=100, regParam=0.3)

lr_sen_1_pipeline = Pipeline(stages= [stringIndexer_DOW, 
                                      stringIndexer_hour,
                                      stringIndexer_sent,
                                      stringIndexer_dis,
                                      stringIndexer_crypt,
                                      stringIndexer_con,
                                      onehot_time, 
                                      vectorAssembler_time,
                                      vector_feature,
                                      robustScaler_f,
                                      pca,
                                      lr_sen_1])

lr_sen_1_model = lr_sen_1_pipeline.fit(train_data)
lr_sen_1_train = lr_sen_1_model.transform(train_data)

# COMMAND ----------

lr_sen_1_model.write().overwrite().save('/FileStore/lr_sen_1_model')

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Test and Evaluation

# COMMAND ----------

# Transform the test data by the training model
lr_sen_1_test = lr_sen_1_model.transform(test_data)

# COMMAND ----------

# Confusion Matrix
lr_sen_1_test.crosstab('controversiality', 'prediction').show()

# COMMAND ----------

# ACC
lr_sen_1_acc = MulticlassClassificationEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                                 metricName="accuracy").evaluate(lr_sen_1_test)

# MSE
lr_sen_1_mse = RegressionEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                    metricName="mse").evaluate(lr_sen_1_test)

# VAR
lr_sen_1_var = RegressionEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                   metricName="var").evaluate(lr_sen_1_test)

# COMMAND ----------

print("Accuracy = %g" % lr_sen_1_acc)
print("Test Error = %g" % (1.0 - lr_sen_1_acc))
print("MSE = %g" % lr_sen_1_mse)
print("Var = %g" % lr_sen_1_var)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Second set of hyperparameter (noPCA)

# COMMAND ----------

lr_sen_2 = LogisticRegression(labelCol='controversiality', featuresCol="scaled_features", 
                              weightCol="weight_con", maxIter=100, regParam=0.05)

lr_sen_2_pipeline = Pipeline(stages= [stringIndexer_DOW, 
                                      stringIndexer_hour,
                                      stringIndexer_sent,
                                      stringIndexer_dis,
                                      stringIndexer_crypt,
                                      stringIndexer_con,
                                      onehot_time, 
                                      vectorAssembler_time,
                                      vector_feature,
                                      robustScaler_f,
                                      lr_sen_2])

lr_sen_2_model = lr_sen_2_pipeline.fit(train_data)
lr_sen_2_train = lr_sen_2_model.transform(train_data)

# COMMAND ----------

lr_sen_2_model.write().overwrite().save('/FileStore/lr_sen_2_model')

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Test and Evaluation

# COMMAND ----------

# Transform the test data by the training model
lr_sen_2_test = lr_sen_2_model.transform(test_data)

# COMMAND ----------

# Confusion Matrix
lr_sen_2_test.crosstab('controversiality', 'prediction').show()

# COMMAND ----------

# ACC
lr_sen_2_acc = MulticlassClassificationEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                                 metricName="accuracy").evaluate(lr_sen_2_test)

# MSE
lr_sen_2_mse = RegressionEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                    metricName="mse").evaluate(lr_sen_2_test)

# VAR
lr_sen_2_var = RegressionEvaluator(labelCol="controversiality", predictionCol="prediction", 
                                   metricName="var").evaluate(lr_sen_2_test)

# COMMAND ----------

print("Accuracy = %g" % lr_sen_2_acc)
print("Test Error = %g" % (1.0 - lr_sen_2_acc))
print("MSE = %g" % lr_sen_2_mse)
print("Var = %g" % lr_sen_2_var)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation (Vis)

# COMMAND ----------

# True Labels
d_true = {'rf_sen_1': rf_sen_1_test.select('controversiality').rdd.flatMap(lambda x: x).collect(),
          'rf_sen_2': rf_sen_2_test.select('controversiality').rdd.flatMap(lambda x: x).collect(),
          'gb_sen_1': gb_sen_1_test.select('controversiality').rdd.flatMap(lambda x: x).collect(),
          'gb_sen_2': gb_sen_2_test.select('controversiality').rdd.flatMap(lambda x: x).collect(),
          'lr_sen_1': lr_sen_1_test.select('controversiality').rdd.flatMap(lambda x: x).collect(),
          'lr_sen_2': lr_sen_2_test.select('controversiality').rdd.flatMap(lambda x: x).collect()}

y_true = pd.DataFrame(data=d_true)

# COMMAND ----------

# Predicted Labels
d_pred = {'rf_sen_1': rf_sen_1_test.select('prediction').rdd.flatMap(lambda x: x).collect(),
          'rf_sen_2': rf_sen_2_test.select('prediction').rdd.flatMap(lambda x: x).collect(),
          'gb_sen_1': gb_sen_1_test.select('prediction').rdd.flatMap(lambda x: x).collect(),
          'gb_sen_2': gb_sen_2_test.select('prediction').rdd.flatMap(lambda x: x).collect(),
          'lr_sen_1': lr_sen_1_test.select('prediction').rdd.flatMap(lambda x: x).collect(),
          'lr_sen_2': lr_sen_2_test.select('prediction').rdd.flatMap(lambda x: x).collect()}

y_pred = pd.DataFrame(data=d_pred)

# COMMAND ----------

fig = go.Figure()
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

for i in range(y_pred.shape[1]):
    true = y_true.iloc[:, i]
    pred = y_pred.iloc[:, i]

    fpr, tpr, _ = roc_curve(true, pred)
    auc_score = roc_auc_score(true, pred)

    name = f"{y_true.columns[i]} (AUC={auc_score:.2f})"
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

fig.update_layout(
    title = {
        'text': 'ROC for Controversiality Classification',
        'y':0.9,
        'x':0.42,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=800, height=500
)


# COMMAND ----------

fig.write_html('../../images/figure_con_roc.html')
