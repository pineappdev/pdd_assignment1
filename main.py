from typing import Tuple

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, collect_list
import argparse


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
def update_paths(paths_df, edges_dataframe):
    paths_df \
        .join(edges_dataframe, paths_df.edge_2 == edges_dataframe.edge_1) \
        .select(
        col("edge_1"),
        col("edge_2"),
        (col("length") + col("weight")).alias("new_length")
    ) \
        .groupBy("edge_1", "edge_2").min("new_length").alias("weight")

    # We're not doing the anti-join here
    return paths_df


# TODO: can the length be negative? What if we have a negative-length cycle?
# TODO: cache initial paths df!
# TODO: checkpoint updated_paths_df before checking the changes?
def join_edges(edges_df, max_iter=999999):
    edges_df.cache()
    initial_paths_df = edges_df.selectExpr("edge_1 as edge_1", "edge_2 as edge_2", "length as weight")
    for i in range(max_iter):
        # Update paths
        updated_paths_df = update_paths(initial_paths_df, edges_df)
        # Check if there are any changes in paths
        changes_df = initial_paths_df.join(updated_paths_df, on="node", how="left_anti")
        # If no changes, break the loop
        if changes_df.count() == 0:
            break

        # Update the paths for the next iteration
        initial_paths_df = updated_paths_df
    return initial_paths_df


# TODO: cache? checkpoint? Unpersist?
def join_paths(edges_df, max_iter=999999):
    initial_paths_df = edges_df.selectExpr("edge_1 as edge_1", "edge_2 as edge_2", "length as weight")
    for i in range(max_iter):
        # Update paths
        updated_paths_df = update_paths(initial_paths_df, initial_paths_df)
        # Check if there are any changes in paths
        changes_df = initial_paths_df.join(updated_paths_df, on="node", how="left_anti")
        # If no changes, break the loop
        if changes_df.count() == 0:
            break

        # Update the paths for the next iteration
        initial_paths_df = updated_paths_df
    return initial_paths_df


if __name__ == '__main__':
    # Initialize Spark session
    spark = SparkSession.builder.appName("ShortestPaths").getOrCreate()
