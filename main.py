from typing import Tuple

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
import argparse
import csv
import datetime


def parseargs() -> Tuple[str, str, str]:
    parser = argparse.ArgumentParser()

    parser.add_argument("name", choices=["linear", "doubling"], help="Algorithm version: linear or doubling")
    parser.add_argument("input_path", type=str, help="Input file path")
    parser.add_argument("output_path", type=str, help="Output file path")

    args = parser.parse_args()

    algorithm_version = args.name
    input_path = args.input_path
    output_path = args.output_path

    return algorithm_version, input_path, output_path


def update_paths(paths_df, edges_dataframe) -> pyspark.sql.DataFrame:
    paths_df = paths_df.alias('df1')
    edges_dataframe = edges_dataframe.alias('df2')
    return paths_df \
        .join(edges_dataframe, [col('df1.edge_2') == col('df2.edge_1'),
                                col('df1.edge_1') != col('df2.edge_2')]) \
        .select(
            col("df1.edge_1").alias('edge_1'),
            col("df2.edge_2").alias('edge_2'),
            (col("df1.length") + col("df2.length")).alias("length")
        ).groupBy(
            "edge_1", "edge_2"
        ).agg(pyspark.sql.functions.min("length").alias("length"))


def join_edges_simple(edges_df: pyspark.sql.DataFrame, max_iter=999999):
    paths_df: pyspark.sql.DataFrame = edges_df.selectExpr("edge_1 as paths_edge_1",
                                                          "edge_2 as paths_edge_2",
                                                          "length as paths_length").repartition(12)

    best_paths_df = edges_df.selectExpr("edge_1 as best_edge_1",
                                        "edge_2 as best_edge_2",
                                        "length as best_length").repartition(12)

    edges_df = edges_df.cache()
    for i in range(max_iter):
        print("Iteration {}".format(i))

        paths_df = paths_df \
            .join(edges_df, on=[col('paths_edge_2') == col('edge_1'),
                                col('paths_edge_1') != col('edge_2')], how="inner") \
            .join(best_paths_df, on=[col('paths_edge_1') == col('best_edge_1'),
                                     col('edge_2') == col('best_edge_2')],
                  how="left") \
            .where("best_length IS NULL OR (paths_length + length < best_length)") \
            .selectExpr("paths_edge_1", "edge_2 as paths_edge_2", "paths_length + length as paths_length") \
            .groupBy(["paths_edge_1", "paths_edge_2"]) \
            .agg(pyspark.sql.functions.min("paths_length").alias("paths_length")) \
            .repartition(12) \
            .checkpoint()

        if paths_df.isEmpty():
            break

        best_paths_df = best_paths_df.join(paths_df, on=[
            col("paths_edge_1") == col("best_edge_1"),
            col("paths_edge_2") == col("best_edge_2")
        ], how="outer").selectExpr(
            "CASE WHEN best_edge_1 IS NOT NULL then best_edge_1 ELSE paths_edge_1 END AS best_edge_1",
            "CASE WHEN best_edge_2 IS NOT NULL then best_edge_2 ELSE paths_edge_2 END AS best_edge_2",
            "CASE WHEN paths_edge_1 IS NOT NULL then paths_length ELSE best_length END AS best_length"
        ).repartition(12).checkpoint()

    return best_paths_df.selectExpr("best_edge_1 as edge_1", "best_edge_2 as edge_2", "best_length as length")


def join_paths(edges_df, max_iter=999999):
    best_paths_df: pyspark.sql.DataFrame = edges_df.repartition(12)
    paths_df = edges_df.repartition(12)
    for i in range(max_iter):
        print("Iteration {}".format(i))
        # Check if there are any changes in paths
        # What we could do here instead is use a temp variable
        # dummy = paths_df
        # to do dummy.unpersist() right after paths_df.checkpoint() is called
        # To tell spark it can free the memory/disk reserved in the previous iteration step
        # But since unpersist() is just a hint to spark
        # Unless we operate on big dataframes, or do a lot of steps and encounter issues
        # We can clean the checkpoint dir manually after the script ends
        paths_df = update_paths(paths_df, best_paths_df).alias("paths").join(
            best_paths_df.alias("best"),
            [
                col("best.edge_1") == col("paths.edge_1"),
                col("best.edge_2") == col("paths.edge_2")
            ],
            how="left"
        ).where(
            col("best.length").isNull() | (col("paths.length") < col("best.length"))
        ).selectExpr("paths.edge_1 as edge_1", "paths.edge_2 as edge_2", "paths.length as length") \
            .repartition(12).checkpoint()

        is_not_changed = paths_df.isEmpty()
        if is_not_changed:
            break

        # Update the best paths for the next iteration
        best_paths_df = best_paths_df.alias("best").join(paths_df.alias("paths"), on=[
            col("best.edge_1") == col("paths.edge_1"),
            col("best.edge_2") == col("paths.edge_2")
        ], how="outer").selectExpr(
            "CASE WHEN best.edge_1 IS NOT NULL then best.edge_1 ELSE paths.edge_1 END AS edge_1",
            "CASE WHEN best.edge_2 IS NOT NULL then best.edge_2 ELSE paths.edge_2 END AS edge_2",
            "CASE WHEN paths.edge_1 IS NOT NULL then paths.length ELSE best.length END AS length"
        ).repartition(12).checkpoint()
    return best_paths_df


def get_paths(algorithm_version: str, edges_df, output_path: str):
    if algorithm_version == 'linear':
        outcome = join_edges_simple(edges_df)
    elif algorithm_version == 'doubling':
        outcome = join_paths(edges_df)
    else:
        raise Exception("provided algorithm_version {} is invalid! Choose from linear or doubling"
                        .format(algorithm_version)
                        )

    print("End time: {}".format(datetime.datetime.now()))
    print("Outcome size: {}".format(outcome.count()))
    spark_to_csv(outcome, output_path)


def spark_to_csv(df: pyspark.sql.DataFrame, file_path):
    """ Converts spark dataframe to CSV file """
    with open(file_path, "w+") as f:
        writer = csv.DictWriter(f, fieldnames=df.columns)
        writer.writerow(dict(zip(df.columns, df.columns)))
        for row in df.toLocalIterator():
            writer.writerow(row.asDict())


if __name__ == '__main__':
    algorithm_version, input_path, output_path = parseargs()

    # .config("spark.executor.memory", "2g") \
    # .config("spark.driver.memory", "3g") \

    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("mlibs") \
        .getOrCreate()

    spark.sparkContext.setCheckpointDir("checkpoints")

    edges_df = spark.read.csv(input_path, schema=StructType([
        StructField("edge_1", IntegerType(), True),
        StructField("edge_2", IntegerType(), True),
        StructField("length", DoubleType(), True)
    ]), header=True).groupBy("edge_1", "edge_2") \
        .agg(pyspark.sql.functions.min("length").alias("length")
             )


    print("Start time: {}".format(datetime.datetime.now()))

    get_paths(algorithm_version, edges_df, output_path)

    print("End time: {}".format(datetime.datetime.now()))
