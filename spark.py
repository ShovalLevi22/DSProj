### Part 4 imports
from pyspark.shell import spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Imputer
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
    df = spark.read.csv('Clustering_data12.csv',header=True, inferSchema=True)
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
    imputer = Imputer(
        inputCols=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160'],
        outputCols=['1_x','2_x','3_x','4_x','5_x','6_x','7_x','8_x','9_x','10_x','11_x','12_x','13_x','14_x','15_x','16_x','17_x','18_x','19_x','20_x','21_x','22_x','23_x','24_x','25_x','26_x','27_x','28_x','29_x','30_x','31_x','32_x','33_x','34_x','35_x','36_x','37_x','38_x','39_x','40_x','41_x','42_x','43_x','44_x','45_x','46_x','47_x','48_x','49_x','50_x','51_x','52_x','53_x','54_x','55_x','56_x','57_x','58_x','59_x','60_x','61_x','62_x','63_x','64_x','65_x','66_x','67_x','68_x','69_x','70_x','71_x','72_x','73_x','74_x','75_x','76_x','77_x','78_x','79_x','80_x','81_x','82_x','83_x','84_x','85_x','86_x','87_x','88_x','89_x','90_x','91_x','92_x','93_x','94_x','95_x','96_x','97_x','98_x','99_x','100_x','101_x','102_x','103_x','104_x','105_x','106_x','107_x','108_x','109_x','110_x','111_x','112_x','113_x','114_x','115_x','116_x','117_x','118_x','119_x','120_x','121_x','122_x','123_x','124_x','125_x','126_x','127_x','128_x','129_x','130_x','131_x','132_x','133_x','134_x','135_x','136_x','137_x','138_x','139_x','140_x','141_x','142_x','143_x','144_x','145_x','146_x','147_x','148_x','149_x','150_x','151_x','152_x','153_x','154_x','155_x','156_x','157_x','158_x','159_x','160_x']
).setStrategy("median").setMissingValue(0)
    test = imputer.fit(df)
    test.transform(df).show()

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

def filter_spark_dataframe_by_list(df, column_name, filter_list):
    """ Returns subset of df where df[column_name] is in filter_list """
    spark = SparkSession.builder.getOrCreate()
    filter_df = spark.createDataFrame(filter_list, df.schema[column_name].dataType)
    return df.join(filter_df, df[column_name] == filter_df["value"])


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

