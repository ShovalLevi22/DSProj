### Part 4 imports
from pyspark.shell import spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.clustering import BisectingKMeans, SparkSession

from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import numpy as np

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
    bkm = BisectingKMeans().setK(4)
    model = bkm.fit(df)

    # Evaluate clustering.
    cost = model.computeCost(df)
    print("Within Set Sum of Squared Errors = " + str(cost))

    # Shows the result.
    print("Cluster Centers: ")
    centers = model.clusterCenters()
    for center in centers:
        print(center)

    array = np.array((centers[0], centers[1], centers[2], centers[3])).transpose()
    plt.figure(figsize=(25, 10))
    plt.title("Dendrograms")
    plt.xlabel('sample index')
    dendrogram(
        array,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8,  # font size for the x axis labels
    )
    plt.show()

if __name__ == "__main__":
    # Get the sparkContext that is created by spark
    sc = spark.sparkContext
    # data = create_df()
    # data.show()

    # Task4 - Classification
    # spark_decision_tree(sc,data)
    # spark_naive_bayes(sc,data)

    # Task4 - Clustering
    spark_kmeans()

