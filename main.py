from typing import Tuple

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import argparse

from pyspark.sql.types import StructType, StructField, LongType


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


# 1st way - add a single edge at a time
# After each join, we keep only the shortest paths (agg min)
# We stop when there's no changes in the dataset?
# How to efficiently detect changes?

# 2nd way - join paths with themselves
# Same as above?

# Function to update paths
def update_paths(paths_df, edges_dataframe) -> pyspark.sql.DataFrame:
    paths_df = paths_df.alias('df1')
    edges_dataframe = edges_dataframe.alias('df2')
    # TODO: join returns only new paths, we'd also like to keep the existing ones...
    new_paths = paths_df \
        .join(edges_dataframe, col('df1.edge_2') == col('df2.edge_1')) \
        .select(
           col("df1.edge_1").alias('edge_1'),
            col("df2.edge_2").alias('edge_2'),
            (col("df1.length") + col("df2.length")).alias("length")
        )

    return paths_df.union(new_paths).groupBy(
        "edge_1", "edge_2"
    ).agg(pyspark.sql.functions.min("length").alias("length"))


# TODO: can the length be negative? What if we have a negative-length cycle?
# TODO: cache initial paths df!
# TODO: checkpoint updated_paths_df before checking the changes?
def join_edges(edges_df, max_iter=999999):
    edges_df = edges_df.cache()
    initial_paths_df = edges_df.selectExpr("edge_1 as edge_1", "edge_2 as edge_2", "length")
    for i in range(max_iter):
        # Update paths
        updated_paths_df = update_paths(initial_paths_df, edges_df)
        # Check if there are any changes in paths
        # We need to find paths that are either in right but not in left or are in right but with a different value, meaning
        # We need to find any row from df2 that does not exist in df1

        # TODO: there must be a better version to do this - after all, we don't have to execute the whole exceptAll,
        # we can just stop at the first difference encountered
        is_not_changed = updated_paths_df.selectExpr("edge_1 as edge_1", "edge_2 as edge_2", "length") \
            .exceptAll(initial_paths_df).isEmpty()
        # If no changes, break the loop
        if is_not_changed:
            break

        # Update the paths for the next iteration
        initial_paths_df = updated_paths_df
    return initial_paths_df


# TODO: cache? checkpoint? Unpersist?
def join_paths(edges_df, max_iter=999999):
    initial_paths_df = edges_df.selectExpr("edge_1 as edge_1", "edge_2 as edge_2", "length")
    for i in range(max_iter):
        # Update paths
        updated_paths_df = update_paths(initial_paths_df, initial_paths_df)
        # Check if there are any changes in paths
        is_not_changed = updated_paths_df.selectExpr("edge_1 as edge_1", "edge_2 as edge_2", "length") \
            .exceptAll(initial_paths_df).isEmpty()
        if is_not_changed:
            break

        # Update the paths for the next iteration
        initial_paths_df = updated_paths_df
    return initial_paths_df


def get_paths(algorithm_version: str, edges_df, output_path: str):
    if algorithm_version == 'linear':
        outcome = join_edges(edges_df)
    elif algorithm_version == 'doubling':
        outcome = join_paths(edges_df)
    else:
        raise Exception("provided algorithm_version {} is invalid! Choose from linear or doubling"
                        .format(algorithm_version)
                        )

    outcome.show()
    outcome.write.csv(output_path)


if __name__ == '__main__':
    algorithm_version, input_path, output_path = parseargs()
    # Initialize Spark session
    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "1g") \
        .appName("mlibs") \
        .getOrCreate()

    edges_df = spark.read.csv(input_path, schema=StructType([
        StructField("edge_1", LongType(), True),
        StructField("edge_2", LongType(), True),
        StructField("length", LongType(), True)
    ]), header=True)

    get_paths(algorithm_version, edges_df, output_path)
