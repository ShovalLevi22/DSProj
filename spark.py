### Part 4 imports
from pyspark.shell import spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.clustering import BisectingKMeans, SparkSession
from scipy.cluster import hierarchy

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as shc
import numpy as np
from main import fancy_dendrogram

def spark_decision_tree(sc, df):
    train_data, test_data = df.randomSplit([0.7, 0.3])
    dtc = DecisionTreeClassifier(labelCol='Discount Code', featuresCol='features', maxBins=10000)
    dtc_model = dtc.fit(train_data)
    dtc_predictions = dtc_model.transform(test_data)

    acc_evaluator = MulticlassClassificationEvaluator(labelCol="Discount Code", predictionCol="prediction", metricName="accuracy")
    dtc_acc = acc_evaluator.evaluate(dtc_predictions)

    print(f"Spark DecisionTreeClassifier Accuracy: {dtc_acc*100}")
    print(f"Spark DecisionTreeClassifier Test Error: {(1.0 - dtc_acc)}")
def spark_naive_bayes(sc, df):
    # Split the data into train and test
    splits = df.randomSplit([0.7, 0.3])
    train = splits[0]
    test = splits[1]

    # create the trainer and set its parameters
    nb = NaiveBayes(labelCol='Discount Code', featuresCol='features')

    # train the model
    model = nb.fit(train)

    # select example rows to display.
    predictions = model.transform(test)
    predictions.show()

    # compute accuracy on the test set
    evaluator = MulticlassClassificationEvaluator(labelCol="Discount Code", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Spark NaiveBayes accuracy:{accuracy}")
def create_df():
    df = spark.read.csv('Hotels_data_Changed.csv', header=True, inferSchema=True)
    df = df.select(['WeekDay','Snapshot Date','Checkin Date','DayDiff','Hotel Name','Discount Code'])
    print(df.printSchema)
    print(df.columns)
    print(df.show(3))

    # Get a list of columns that are string type
    categoricalColumns = [item[0] for item in df.dtypes if item[1].startswith('string')]

    # List of stages in pipeline
    stages = []

    # Iterate through all categorical values
    for categoricalCol in categoricalColumns:
        # create a string indexer for those categorical values and assign a new name including the word 'Index'
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=f"{categoricalCol}Index")
        print(f"Creating a string indexer for {categoricalCol}")
        # append the string Indexer to our list of stages
        stages += [stringIndexer]

    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    df = df.select(['DayDiff', 'Discount Code', 'WeekDayIndex', 'Snapshot DateIndex', 'Checkin DateIndex', 'Hotel NameIndex'])
    print(df.show(3))

    assembler = VectorAssembler(
        inputCols=['DayDiff', 'WeekDayIndex', 'Snapshot DateIndex', 'Checkin DateIndex', 'Hotel NameIndex'],
        outputCol="features")

    df = assembler.transform(df)
    print(df.show(3))
    return df


def spark_kmeans():
    df = spark.read.csv('Clustering_data.csv',header=True, inferSchema=True)
    print(df.printSchema)
    print(df.columns)
    print(df.show(3))

    indexer = StringIndexer(inputCol='Hotel Name', outputCol=f"Hotel NameIndex")
    df = indexer.fit(df).transform(df)
    df = df.drop('Hotel Name')
    df = df.drop('ID')
    print(df.show(3))

    assembler = VectorAssembler(
        inputCols=df.columns,
        outputCol="features")
    df = assembler.transform(df)

    from pyspark.ml.feature import StandardScaler
    scaler = StandardScaler(inputCol='features',
                            outputCol='scaledFeatures',
                            withStd=True, withMean=True)
    df = scaler.fit(df).transform(df)

    df.show()
    # Trains a bisecting k-means model.
    bkm = BisectingKMeans(predictionCol='Test').setK(4)
    model = bkm.fit(df)

    # Evaluate clustering.
    cost = model.computeCost(df)
    print("Within Set Sum of Squared Errors = " + str(cost))

    # Shows the result.
    print("Cluster Centers: ")
    centers = model.clusterCenters()
    for center in centers:
        print(center)
    df.filter(df.contains)
    array = np.array((centers[0],centers[1],centers[2],centers[3])).transpose()
    plt.figure(figsize=(25, 10))
    plt.title("Dendrograms")
    plt.xlabel('sample index')
    dendrogram(
        array,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8,  # font size for the x axis labels
    )
    plt.show()

def filter_spark_dataframe_by_list(df, column_name, filter_list):
    """ Returns subset of df where df[column_name] is in filter_list """
    spark = SparkSession.builder.getOrCreate()
    filter_df = spark.createDataFrame(filter_list, df.schema[column_name].dataType)
    return df.join(filter_df, df[column_name] == filter_df["value"])

def spark_clustering_data():
    df = spark.read.csv(f"hotels_data.csv",header=True, inferSchema=True)

    # 150 Hotels with most rows
    temp_hotels = df.groupBy("Hotel Name").count().orderBy("Count", ascending=False).head(150)
    df = df.join(df.groupBy('Hotel Name').count(), on='Hotel Name')
    hotels = df.filter(df["Count"] >= temp_hotels[149][1])
    hotels = hotels.drop('count')

    # 40 Checkin dates with most rows
    temp_checkins = hotels.groupBy("Checkin Date", "Hotel Name").count().orderBy("Count", ascending=False).head(40)
    hotels = hotels.join(hotels.groupBy('Checkin Date').count(), on='Checkin Date')
    checkins = hotels.filter(hotels["count"] >= temp_checkins[39][2])
    print(checkins.show())





if __name__ == "__main__":
    # Get the sparkContext that is created by spark
    sc = spark.sparkContext
    # data = create_df()
    # data.show()

    # Task4 - Classification
    # spark_decision_tree(sc,data)
    # spark_naive_bayes(sc,data)

    # Task4 - Clustering
    df = spark_clustering_data()
    spark_kmeans()

