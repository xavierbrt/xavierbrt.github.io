---
layout: post
title:  Kickstarter - Successful Projects
date:   2019-12-09 11:00:20 +0300
description: In this tutorial, we will predict the success of a fundraising campaign on Kickstarter, based on its characteristics.
img: post-2.jpg # Add image post (optional)
tags: [ML, Spark, Data Cleaning]
author: Xavier Bracquart # Add name author (optional)
---

In this tutorial, we will predict the success of a fundraising campaign on Kickstarter, based on its characteristics.<br /> 
An important part of the tutorial will be dedicated to the data cleaning, and to the preparation of the different stages of the pipeline to train the model.

The objective of the project is to develop a pipeline based on the **Spark environment** and coded in **Scala**.

The subject comes from a Kaggle competition. The data can be downloaded from the [associated page](https://www.kaggle.com/codename007/funding-successful-projects).

## Creation of a SparkSession

First, we create a Spark session.


```scala
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession, Row}
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, CountVectorizer, CountVectorizerModel, IDF, StringIndexer, OneHotEncoderEstimator, VectorAssembler}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.attribute.AttributeGroup

val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

 val spark = SparkSession
      .builder
      .config(conf)
      .appName("Lab - Kickstarter campaign")
      .getOrCreate()

import spark.implicits._  // to use the symbol $
```

    Intitializing Scala interpreter ...

    Spark Web UI available at http://xavier-linux.home:4040
    SparkContext available as 'sc' (version = 2.4.4, master = local[*], app id = local-1580508273855)
    SparkSession available as 'spark'



# Part 1: Data preparation

We preprocess the data, in order to later obtain better results.

## Load and clean the data


```scala
 val df: DataFrame = spark
        .read
        .option("header", true)
        .option("inferSchema", "true")
        .option("quote", "\"")
        .option("escape", "\"")
        .csv("./train_clean.csv")

println(s"Number of lines: ${df.count}")
println(s"Number of columns: ${df.columns.length}")
```

    Number of lines: 108129
    Number of columns: 14





We begin by selecting the interesting columns and casting them to the correct type.


```scala
 val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))

dfCasted
      .select("goal", "deadline", "state_changed_at", "created_at", "launched_at", "backers_count", "final_status")
      .describe()
      .show

dfCasted.printSchema()
```

    +-------+-----------------+--------------------+--------------------+--------------------+--------------------+------------------+------------------+
    |summary|             goal|            deadline|    state_changed_at|          created_at|         launched_at|     backers_count|      final_status|
    +-------+-----------------+--------------------+--------------------+--------------------+--------------------+------------------+------------------+
    |  count|           108129|              108129|              108129|              108129|              108129|            108129|            108129|
    |   mean|36726.22826438791|1.3802484980048554E9|1.3801529957698119E9|1.3740368577694051E9|1.3772990047093103E9|123.51666065532835|0.3196274819891056|
    | stddev|971902.7051687709|4.2702221220911644E7| 4.266401844467795E7|4.2723097677902974E7| 4.294421262600033E7| 1176.745162158387|0.4663343928283478|
    |    min|                0|          1241333999|          1241334017|          1240335335|          1240602723|                 0|                 0|
    |    max|        100000000|          1433096938|          1433096940|          1432325200|          1432658473|            219382|                 1|
    +-------+-----------------+--------------------+--------------------+--------------------+--------------------+------------------+------------------+

    root
     |-- project_id: string (nullable = true)
     |-- name: string (nullable = true)
     |-- desc: string (nullable = true)
     |-- goal: integer (nullable = true)
     |-- keywords: string (nullable = true)
     |-- disable_communication: boolean (nullable = true)
     |-- country: string (nullable = true)
     |-- currency: string (nullable = true)
     |-- deadline: integer (nullable = true)
     |-- state_changed_at: integer (nullable = true)
     |-- created_at: integer (nullable = true)
     |-- launched_at: integer (nullable = true)
     |-- backers_count: integer (nullable = true)
     |-- final_status: integer (nullable = true)
    


## Drop columns we will not use

We remove the column *disable_communication*, which contains few data, and the columns *backers_count* and *state_changed* that we don't have at the launch of a Kickstarter campaign. If we use it to train the model, it represents a data leakage.


```scala
val df2: DataFrame = df.drop("disable_communication", "backers_count", "state_changed_at")
df2.show()
```

    +--------------+--------------------+--------------------+-------+--------------------+-------+--------+----------+----------+-----------+------------+
    |    project_id|                name|                desc|   goal|            keywords|country|currency|  deadline|created_at|launched_at|final_status|
    +--------------+--------------------+--------------------+-------+--------------------+-------+--------+----------+----------+-----------+------------+
    |kkst1451568084| drawing for dollars|I like drawing pi...|   20.0| drawing-for-dollars|     US|     USD|1241333999|1240600507| 1240602723|           1|
    |kkst1474482071|Sponsor Dereck Bl...|I  Dereck Blackbu...|  300.0|sponsor-dereck-bl...|     US|     USD|1242429000|1240960224| 1240975592|           0|
    | kkst183622197|       Mr. Squiggles|So I saw darkpony...|   30.0|        mr-squiggles|     US|     USD|1243027560|1242163613| 1242164398|           0|
    | kkst597742710|Help me write my ...|Do your part to h...|  500.0|help-me-write-my-...|     US|     USD|1243555740|1240963795| 1240966730|           1|
    |kkst1913131122|Support casting m...|I m nearing compl...| 2000.0|support-casting-m...|     US|     USD|1243769880|1241177914| 1241180541|           0|
    |kkst1085176748|        daily digest|I m a fledgling v...|  700.0|        daily-digest|     US|     USD|1243815600|1241050799| 1241464468|           0|
    |kkst1468954715|iGoozex - Free iP...|I am an independe...|  250.0|igoozex-free-ipho...|     US|     USD|1243872000|1241725172| 1241736308|           0|
    | kkst194050612|Drive A Faster Ca...|Drive A Faster Ca...| 1000.0|drive-a-faster-ca...|     US|     USD|1244088000|1241460541| 1241470291|           1|
    | kkst708883590|"""""""""""""""""...|Opening Friday  J...| 5000.0|lostles-at-tinys-...|     US|     USD|1244264400|1241415164| 1241480901|           0|
    | kkst890976740|Choose Your Own A...|This project is f...| 3500.0|choose-your-own-a...|     US|     USD|1244946540|1242268157| 1242273460|           0|
    |kkst2053381363|Anatomy of a Cred...|I am an independe...|30000.0|anatomy-of-a-cred...|     US|     USD|1245026160|1241829376| 1242056094|           0|
    | kkst918550886|No-bit: An artist...|I want to create ...|  300.0|no-bit-an-artist-...|     US|     USD|1245038400|1242523061| 1242528805|           0|
    | kkst934689279|Indie Nerd Board ...|pictured here is ...| 1500.0|indie-nerd-board-...|     US|     USD|1245042600|1242364202| 1242369560|           1|
    | kkst191414809|Icons for your iP...|I make cool icons...|  500.0|awesome-icons-for...|     US|     USD|1245092400|1241034764| 1241039475|           1|
    | kkst569584443|HAPPY VALLEY: Dex...|I am a profession...|  500.0|help-me-make-my-w...|     US|     USD|1245528660|1242072711| 1242333869|           0|
    | kkst485555421|       Project Pedal|Project Pedal is ...| 1000.0|       project-pedal|     US|     USD|1245556740|1242682134| 1242690018|           1|
    |kkst1537563608|Frank Magazine Er...|We are throwing a...|  600.0|frank-magazine-er...|     US|     USD|1245882360|1244579167| 1244742156|           0|
    |kkst1261713500|  Crossword Puzzles!|I create crosswor...| 1500.0|   crossword-puzzles|     US|     USD|1246354320|1240997554| 1241005923|           1|
    | kkst910550425|Run, Blago Run! Show|A 3-day pop-up ar...| 3500.0|  run-blago-run-show|     US|     USD|1246420800|1244299453| 1244388012|           0|
    | kkst139451001|It Might Become a...|We are broke film...| 1000.0|it-might-become-a...|     US|     USD|1246420800|1243272026| 1243616180|           1|
    +--------------+--------------------+--------------------+-------+--------------------+-------+--------+----------+----------+-----------+------------+
    only showing top 20 rows
    




## Preprocessing

### Date processing

We create the column *days_campaign* with the (truncated) number of days between launch time and deadline. We also create the column *hours_prepa* with the number of hours between creation time and launch time. We then drop *launched_at*, *created_at* and *deadline* that we will not be used anymore.


```scala
val df3: DataFrame = df2.withColumn("days_campaign", datediff(from_unixtime($"deadline"), from_unixtime($"launched_at")))
val df4: DataFrame = df3.withColumn("hours_prepa", round(($"launched_at" - $"created_at")/3600,3))
val df5: DataFrame = df4.drop("launched_at", "created_at", "deadline")
df5.show()
```

    +--------------+--------------------+--------------------+-------+--------------------+-------+--------+------------+-------------+-----------+
    |    project_id|                name|                desc|   goal|            keywords|country|currency|final_status|days_campaign|hours_prepa|
    +--------------+--------------------+--------------------+-------+--------------------+-------+--------+------------+-------------+-----------+
    |kkst1451568084| drawing for dollars|I like drawing pi...|   20.0| drawing-for-dollars|     US|     USD|           1|            9|      0.616|
    |kkst1474482071|Sponsor Dereck Bl...|I  Dereck Blackbu...|  300.0|sponsor-dereck-bl...|     US|     USD|           0|           17|      4.269|
    | kkst183622197|       Mr. Squiggles|So I saw darkpony...|   30.0|        mr-squiggles|     US|     USD|           0|           10|      0.218|
    | kkst597742710|Help me write my ...|Do your part to h...|  500.0|help-me-write-my-...|     US|     USD|           1|           30|      0.815|
    |kkst1913131122|Support casting m...|I m nearing compl...| 2000.0|support-casting-m...|     US|     USD|           0|           30|       0.73|
    |kkst1085176748|        daily digest|I m a fledgling v...|  700.0|        daily-digest|     US|     USD|           0|           28|    114.908|
    |kkst1468954715|iGoozex - Free iP...|I am an independe...|  250.0|igoozex-free-ipho...|     US|     USD|           0|           24|      3.093|
    | kkst194050612|Drive A Faster Ca...|Drive A Faster Ca...| 1000.0|drive-a-faster-ca...|     US|     USD|           1|           31|      2.708|
    | kkst708883590|"""""""""""""""""...|Opening Friday  J...| 5000.0|lostles-at-tinys-...|     US|     USD|           0|           32|      18.26|
    | kkst890976740|Choose Your Own A...|This project is f...| 3500.0|choose-your-own-a...|     US|     USD|           0|           31|      1.473|
    |kkst2053381363|Anatomy of a Cred...|I am an independe...|30000.0|anatomy-of-a-cred...|     US|     USD|           0|           35|     62.977|
    | kkst918550886|No-bit: An artist...|I want to create ...|  300.0|no-bit-an-artist-...|     US|     USD|           0|           29|      1.596|
    | kkst934689279|Indie Nerd Board ...|pictured here is ...| 1500.0|indie-nerd-board-...|     US|     USD|           1|           31|      1.488|
    | kkst191414809|Icons for your iP...|I make cool icons...|  500.0|awesome-icons-for...|     US|     USD|           1|           47|      1.309|
    | kkst569584443|HAPPY VALLEY: Dex...|I am a profession...|  500.0|help-me-make-my-w...|     US|     USD|           0|           37|     72.544|
    | kkst485555421|       Project Pedal|Project Pedal is ...| 1000.0|       project-pedal|     US|     USD|           1|           33|       2.19|
    |kkst1537563608|Frank Magazine Er...|We are throwing a...|  600.0|frank-magazine-er...|     US|     USD|           0|           14|     45.275|
    |kkst1261713500|  Crossword Puzzles!|I create crosswor...| 1500.0|   crossword-puzzles|     US|     USD|           1|           62|      2.325|
    | kkst910550425|Run, Blago Run! Show|A 3-day pop-up ar...| 3500.0|  run-blago-run-show|     US|     USD|           0|           24|       24.6|
    | kkst139451001|It Might Become a...|We are broke film...| 1000.0|it-might-become-a...|     US|     USD|           1|           33|     95.598|
    +--------------+--------------------+--------------------+-------+--------------------+-------+--------+------------+-------------+-----------+
    only showing top 20 rows
    




### Textual information processing

We prepare the textual informations, mainly by transforming the text in lowercase and by concatening the columns containing textual information.


```scala
// Process textual information
val df6: DataFrame = df5.withColumn("name", lower($"name"))
                        .withColumn("desc", lower($"desc"))
                        .withColumn("keywords", regexp_replace(lower($"keywords"), "-", " "))

// Concatenate text columns
val df7: DataFrame = df6.withColumn("text", concat_ws(" ",$"name", $"desc",$"keywords"))
val df8: DataFrame = df7.drop("name", "desc", "keywords")
df8.show()
```

    +--------------+-------+-------+--------+------------+-------------+-----------+--------------------+
    |    project_id|   goal|country|currency|final_status|days_campaign|hours_prepa|                text|
    +--------------+-------+-------+--------+------------+-------------+-----------+--------------------+
    |kkst1451568084|   20.0|     US|     USD|           1|            9|      0.616|drawing for dolla...|
    |kkst1474482071|  300.0|     US|     USD|           0|           17|      4.269|sponsor dereck bl...|
    | kkst183622197|   30.0|     US|     USD|           0|           10|      0.218|mr. squiggles so ...|
    | kkst597742710|  500.0|     US|     USD|           1|           30|      0.815|help me write my ...|
    |kkst1913131122| 2000.0|     US|     USD|           0|           30|       0.73|support casting m...|
    |kkst1085176748|  700.0|     US|     USD|           0|           28|    114.908|daily digest i m ...|
    |kkst1468954715|  250.0|     US|     USD|           0|           24|      3.093|igoozex - free ip...|
    | kkst194050612| 1000.0|     US|     USD|           1|           31|      2.708|drive a faster ca...|
    | kkst708883590| 5000.0|     US|     USD|           0|           32|      18.26|"""""""""""""""""...|
    | kkst890976740| 3500.0|     US|     USD|           0|           31|      1.473|choose your own a...|
    |kkst2053381363|30000.0|     US|     USD|           0|           35|     62.977|anatomy of a cred...|
    | kkst918550886|  300.0|     US|     USD|           0|           29|      1.596|no-bit: an artist...|
    | kkst934689279| 1500.0|     US|     USD|           1|           31|      1.488|indie nerd board ...|
    | kkst191414809|  500.0|     US|     USD|           1|           47|      1.309|icons for your ip...|
    | kkst569584443|  500.0|     US|     USD|           0|           37|     72.544|happy valley: dex...|
    | kkst485555421| 1000.0|     US|     USD|           1|           33|       2.19|project pedal pro...|
    |kkst1537563608|  600.0|     US|     USD|           0|           14|     45.275|frank magazine er...|
    |kkst1261713500| 1500.0|     US|     USD|           1|           62|      2.325|crossword puzzles...|
    | kkst910550425| 3500.0|     US|     USD|           0|           24|       24.6|run, blago run! s...|
    | kkst139451001| 1000.0|     US|     USD|           1|           33|     95.598|it might become a...|
    +--------------+-------+-------+--------+------------+-------------+-----------+--------------------+
    only showing top 20 rows
    






## Save the dataframe to access it later


```scala
println(s"Number of lines: ${df8.count}")
println(s"Number of columns: ${df8.columns.length}")
// Save the dataframe in parquet format
df8.write.mode("overwrite").parquet("./prepared_dataset")
```

    Number of lines: 108129
    Number of columns: 8


# Part 2: Pipeline creation

## Load the dataframe


```scala
val df: DataFrame = spark
            .read
            .option("header", true)
            .option("inferSchema", "true")
            .parquet("./prepared_dataset")

println(s"Number of lines: ${df.count}")
println(s"Number of columns: ${df.columns.length}")
df.show(5)
```

    Number of lines: 108129
    Number of columns: 8
    +--------------+-------+-------+--------+------------+-------------+-----------+--------------------+
    |    project_id|   goal|country|currency|final_status|days_campaign|hours_prepa|                text|
    +--------------+-------+-------+--------+------------+-------------+-----------+--------------------+
    |kkst1815328106| 1000.0|     US|     USD|           1|           50|   3261.394|pnmini - positive...|
    | kkst960290677| 7500.0|     US|     USD|           1|           30|    162.408|make the """"""""...|
    |kkst1831828189|12500.0|     US|     USD|           1|           30|    285.209|treasures untold ...|
    | kkst805323690|  750.0|     US|     USD|           1|           30|   1462.793|give it up for th...|
    | kkst753800887|20000.0|     US|     USD|           1|           22|    193.536|escape big box es...|
    +--------------+-------+-------+--------+------------+-------------+-----------+--------------------+
    only showing top 5 rows
    




## Prepare the stages of the pipeline

We transform the textual data:


```scala
// Stage 1 : transform words from text to tokens
val tokenizer = new RegexTokenizer()
  .setPattern("\\W+")
  .setGaps(true)
  .setInputCol("text")
  .setOutputCol("tokens")
```


```scala
// Stage 2 : drop the stop words (from this list : StopWordsRemover.loadDefaultStopWords("english"))
val stopWordsRemover = new StopWordsRemover()
    .setInputCol("tokens")
    .setOutputCol("text_filtered")
```


```scala
// Stage 3 : compute TF
val cvModel: CountVectorizer = new CountVectorizer()//Model(Array("a", "b", "c"))
    .setInputCol("text_filtered")
    .setOutputCol("cv_features")
```

```scala
// Stage 4 : compute IDF
val idf = new IDF()
    .setInputCol("cv_features")
    .setOutputCol("tfidf")
```





We convert categorical variables to numeric variables:


```scala
// Stage 5 : convert country to numeric variables
val indexer_country = new StringIndexer()
    .setInputCol("country")
    .setOutputCol("country_indexed")
    .setHandleInvalid("keep")
```



```scala
// Stage 6 : convert currency to numeric variables
val indexer_currency = new StringIndexer()
    .setInputCol("currency")
    .setOutputCol("currency_indexed")
```




```scala
// Stages 7 and 8: one-hot encoding for these variables
val encoder = new OneHotEncoderEstimator()
    .setInputCols(Array("country_indexed", "currency_indexed"))
    .setOutputCols(Array("country_onehot", "currency_onehot"))
```




We format the data in a format usable by Spark.ML:


```scala
// Stage 9: format the data
val assembler = new VectorAssembler()
  .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
  .setOutputCol("features")
```





We implement the last stage, the classifiction model, which is a logistic regression:


```scala
// Stage 10 : create classification model
val lr = new LogisticRegression()
  .setElasticNetParam(0.0)
  .setFitIntercept(true)
  .setFeaturesCol("features")
  .setLabelCol("final_status")
  .setStandardization(true)
  .setPredictionCol("predictions")
  .setRawPredictionCol("raw_predictions")
  .setThresholds(Array(0.7, 0.3))
  .setTol(1.0e-6)
  .setMaxIter(20)
```





We create the pipeline by referencing all the stages:


```scala
val pipeline = new Pipeline()
    .setStages(Array(tokenizer, stopWordsRemover, 
                     cvModel, idf, indexer_country, indexer_currency,
                     encoder, assembler, lr))
```




## Train the model and get predictions

We divide the dataset into training and test sets:


```scala
val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed=8)
```





We train the model and save it:


```scala
val model = pipeline.fit(training)
model.write.overwrite().save("./spark-logistic-regression-model")
```





We compute predictions for the test set:


```scala
val dfWithSimplePredictions = model.transform(test)
dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()
```

    +------------+-----------+-----+
    |final_status|predictions|count|
    +------------+-----------+-----+
    |           1|        0.0| 1850|
    |           0|        1.0| 2249|
    |           1|        1.0| 1605|
    |           0|        0.0| 5065|
    +------------+-----------+-----+
    





We compute the f1_score to evaluate the performance of the model:


```scala
// Compute the f1_score
val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("final_status")
    .setPredictionCol("predictions")
    .setMetricName("f1")
val f1_score = evaluator.evaluate(dfWithSimplePredictions)
```

    f1_score: Double = 0.6244230648375184




## Adjustment of model hyper-parameters

We will improve the model by tuning the hyperparameters (see [documentation](https://spark.apache.org/docs/2.2.0/ml-tuning.html)).<br />
We begin by setting a grid containing multiple values for two parameters:


```scala
val paramGrid = new ParamGridBuilder()
    .addGrid(cvModel.minDF, Array(55.0, 75.0, 95.0))
    .addGrid(lr.elasticNetParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
    .build()
```




    paramGrid: Array[org.apache.spark.ml.param.ParamMap] =
    Array({
    	logreg_19b714373adf-elasticNetParam: 1.0E-7,
    	cntVec_cb28b7d870ca-minDF: 55.0
    }, {
    	logreg_19b714373adf-elasticNetParam: 1.0E-5,
    	cntVec_cb28b7d870ca-minDF: 55.0
    }, {
    	logreg_19b714373adf-elasticNetParam: 0.001,
    	cntVec_cb28b7d870ca-minDF: 55.0
    }, {
    	logreg_19b714373adf-elasticNetParam: 0.1,
    	cntVec_cb28b7d870ca-minDF: 55.0
    }, {
    	logreg_19b714373adf-elasticNetParam: 1.0E-7,
    	cntVec_cb28b7d870ca-minDF: 75.0
    }, {
    	logreg_19b714373adf-elasticNetParam: 1.0E-5,
    	cntVec_cb28b7d870ca-minDF: 75.0
    }, {
    	logreg_19b714373adf-elasticNetParam: 0.001,
    	cntVec_cb28b7d870ca-minDF: 75.0
    }, {
    	logreg_19b714373adf-elasticNetParam: 0.1,
    	cntVec_cb28b7d870ca-minDF: 75.0
    }, {
    	logreg_19b714373adf-elasticNetParam: 1.0E-7,
    	cntVec_cb28b7d870ca-min...




```scala
val validation = new TrainValidationSplit()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setTrainRatio(0.7)
```






We compute the model, with a method that will use the grid to assess the parameters and choose the better ones:


```scala
val model_improved = validation.fit(training)
val dfWithPredictions = model_improved.transform(test)
dfWithPredictions.groupBy("final_status", "predictions").count.show()
```

    +------------+-----------+-----+
    |final_status|predictions|count|
    +------------+-----------+-----+
    |           1|        0.0| 1048|
    |           0|        1.0| 2764|
    |           1|        1.0| 2407|
    |           0|        0.0| 4550|
    +------------+-----------+-----+
    





We obtain a better score:


```scala
val f1_score = evaluator.evaluate(dfWithPredictions)
```




    f1_score: Double = 0.6577082809521838




# Part 3: Explanation of the model

We list the names of the features that have the most impact on the model:


```scala
val model_lr = model_improved.bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[LogisticRegressionModel]
// Extract the attributes of the input (features)
val schema = model_improved.transform(test).schema
val featureAttrs = AttributeGroup.fromStructField(schema(model_lr.getFeaturesCol)).attributes.get
val features = featureAttrs.map(_.name.get)

// Add "(Intercept)" to list of feature names if the model was fit with an intercept
val featureNames: Array[String] = if (model_lr.getFitIntercept) {
  Array("(Intercept)") ++ features
} else {
  features
}

// Get array of coefficients
val lrModelCoeffs = model_lr.coefficients.toArray
val coeffs = if (model_lr.getFitIntercept) {
  lrModelCoeffs ++ Array(model_lr.intercept)
} else {
  lrModelCoeffs
}
val coeffs_abs = coeffs.map(num => Math.abs(num))

// Print feature names & coefficients together
println("Feature\tCoefficient")
featureNames.zip(coeffs_abs).sortBy(_._2)(Ordering[Double].reverse).foreach { case (feature, coeff) =>
  println(s"$feature\t$coeff")
}
```

    Feature	Coefficient
    country_onehot_IE	5.024040617101225
    tfidf_30	3.4832059890928075
    currency_onehot_DKK	0.7183471470889083
    tfidf_1303	0.6961178597790534
    country_onehot_NO	0.5485997398987462
    tfidf_2216	0.39300772402867395
    country_onehot_AU	0.32637625467165815
    tfidf_893	0.30953915220258327
    tfidf_1977	0.2787557693237988
    goal	0.2521459471160525
    country_onehot_DE	0.2521459471160525
    tfidf_835	0.23280331768470497
    currency_onehot_AUD	0.21729016364193784
    tfidf_3233	0.21654273945517713
    country_onehot_DK	0.19829213034296503
    tfidf_43	0.17094565628856292
    tfidf_1827	0.15373178890587805
    tfidf_2469	0.14934003219817943
    tfidf_300	0.14911742526095986
    tfidf_2234	0.14802420525366458
    tfidf_3166	0.1451474554636832
    tfidf_3846	0.14327533176878213
    tfidf_3216	0.1410202390940732
    tfidf_45	0.1408017133166311
    tfidf_3796	0.1384818201835908
    tfidf_3376	0.13790889286926125
    country_onehot_SE	0.13762570986696357
    currency_onehot_SEK	0.13762570986696357
    tfidf_3845	0.13306450189797106
    tfidf_2615	0.13268855701197949
    tfidf_3724	0.12790143828453865
    ...

 It turns out that the parameters most impacting the regression are the **country**, many terms coming from the **transformation into tf-idf** and the **amount** of the project.



# Conclusion

We set a pipeline on Spark ML to clean a dataset, train a classification model, evaluate the model and improve it by tuning and explaining the model.

We note that tuning the model by looking for the best hyperparameters helped improve the F1 score from **0.624** to **0.658**.<br />
We also tried (outside of this tutorial), to train the model on the raw data, before the preprocessing. We obtained a score of **0.616**.<br />
To finish, we implemented a random forest model, but the result was a F1 score of **0.543**.

The score obtained is not very high. It is not a surprise, because the success of a crowdfunding campaigns depends a lot on communication and on the reason of the campaign. However, our model can be improved with :
* Better processing of the textual information (separation of the title and the keywords, extraction and clustering based on the reason of the campaign, etc.);
* Better improvement of the model, with a larger grid search;
* Test of other models.


You can see the complete code and project explanations on the [GitHub repository of the project](https://github.com/xavierbrt/spark_project_kickstarter_2019_2020).


--------------------
Illustration photo by Nattanan Kanchanaprat from Pixabay.