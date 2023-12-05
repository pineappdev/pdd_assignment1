from typing import Tuple

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import argparse

from pyspark.sql.types import StructType, StructField, LongType, DoubleType


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


def update_paths(paths_df, edges_dataframe) -> pyspark.sql.DataFrame:
    paths_df = paths_df.alias('df1')
    edges_dataframe = edges_dataframe.alias('df2')
    new_paths = paths_df \
        .join(edges_dataframe, col('df1.edge_2') == col('df2.edge_1'), col('df1.edge_1') != col('df2.edge_2')) \
        .select(
            col("df1.edge_1").alias('edge_1'),
            col("df2.edge_2").alias('edge_2'),
            (col("df1.length") + col("df2.length")).alias("length")
    )

    return paths_df.union(new_paths).groupBy(
        "edge_1", "edge_2"
    ).agg(pyspark.sql.functions.min("length").alias("length"))


def update_paths_2(paths_df, edges_dataframe) -> pyspark.sql.DataFrame:
    paths_df = paths_df.alias('df1')
    edges_dataframe = edges_dataframe.alias('df2')
    return paths_df \
        .join(edges_dataframe, [col('df1.edge_2') == col('df2.edge_1'), col('df1.edge_1') != col('df2.edge_2')]) \
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
        # TODO: which checkpoint to choose?
        # best_paths_df.checkpoint()
        # paths_df.checkpoint()

        # TODO: use aliases, not different column names...
        # TODO: print num partitions

        # TODO: what about paths 1,1,1? Should we filter them? Ignore them?
        paths_df = paths_df \
            .join(edges_df, on=[col('paths_edge_2') == col('edge_1'),
                                col('paths_edge_1') != col('edge_2')], how="inner") \
            .join(best_paths_df, on=[col('paths_edge_1') == col('best_edge_1'),
                                     col('edge_2') == col('best_edge_2')],
                  how="left") \
            .where("best_length IS NULL OR (paths_length < best_length)") \
            .selectExpr("paths_edge_1", "edge_2 as paths_edge_2", "paths_length + length as paths_length") \
            .groupBy(["paths_edge_1", "paths_edge_2"]) \
            .agg(pyspark.sql.functions.min("paths_length").alias("paths_length")) \

        if paths_df.isEmpty():
            break

        print("Iteration {}, paths_df size: {}".format(i, paths_df.count()))
        paths_df.localCheckpoint()

        # TODO: for some reason, it stops here... when paths_df's size is 58
        # we get java out of memory error
        # but why?
        best_paths_df = best_paths_df.join(paths_df, on=[
            col("paths_edge_1") == col("best_edge_1"),
            col("paths_edge_2") == col("best_edge_2")
        ], how="outer").selectExpr(
            "CASE WHEN best_edge_1 IS NOT NULL then best_edge_1 ELSE paths_edge_1 END AS best_edge_1",
            "CASE WHEN best_edge_2 IS NOT NULL then best_edge_2 ELSE paths_edge_2 END AS best_edge_2",
            "CASE WHEN paths_edge_1 IS NOT NULL then paths_length ELSE best_length END AS best_length"
        )
        # best_paths_df = best_paths_df.union(paths_df.selectExpr("paths_edge_1 as best_edge_1",
        #                                                         "paths_edge_2 as best_edge_2",
        #                                                         "paths_length as best_length")) \
        #     .groupBy(["best_edge_1", "best_edge_2"]) \
        #     .agg(pyspark.sql.functions.min("best_length").alias("best_length"))

        best_paths_df.localCheckpoint()
        print("Iteration {}, best_paths size: {}".format(i, best_paths_df.count()))

    return best_paths_df


# TODO: checkpoint updated_paths_df before checking the changes?
# If we added edge X,X,0 to our edges then we wouldn't have to do union at all
def join_edges(edges_df, max_iter=999999):
    solution_df = edges_df.selectExpr("edge_1 as edge_1", "edge_2 as edge_2", "length")
    paths_df = edges_df.selectExpr("edge_1 as edge_1", "edge_2 as edge_2", "length")
    edges_df = edges_df.cache()
    for i in range(max_iter):
        # TODO: when to call unpersist?
        # solution_df.unpe # TODO: do we want to do that?

        # This gives us paths generated by appending 1 edge to new paths from the previous step
        updated_paths_df = update_paths_2(paths_df, edges_df)
        # Now we want to get rid of paths that are already in old df and have <= length
        # To get only new updated paths
        paths_df = updated_paths_df.alias('df1').join(
            solution_df.alias('df2'),
            on=[col('df1.edge_1') == col('df2.edge_1'), col('df1.edge_2') == col('df2.edge_2')],
            how='left'
        ) \
            .where(col('df2.length').isNull() | (col('df1.length') < col('df2.length'))) \
            .select(col('df1.edge_1').alias('edge_1'), col('df1.edge_2').alias('edge_2'),
                    col('df1.length').alias('length'))

        if paths_df.isEmpty():
            break

        # TODO: Block rdd_3_0 already exists on this machine, not re-adding it...? What?
        solution_df = solution_df \
            .union(paths_df) \
            .groupBy(
            "edge_1", "edge_2"
        ) \
            .agg(
            pyspark.sql.functions.min("length").alias("length")
        )

        # solution_df.checkpoint()

    return solution_df


def join_edges_2(edges_df, max_iter=999999):
    # edges_df.repartition(12)
    edges_df = edges_df.cache()

    # TODO: change num of partitions...
    # 2 to 4 times more than cores
    # we have 4 cores, that gives 8 - 16 partitions
    solution_df = edges_df.selectExpr("edge_1 as edge_1", "edge_2 as edge_2", "length")
    paths_df = edges_df.selectExpr("edge_1 as edge_1", "edge_2 as edge_2", "length")
    edges_df = edges_df.repartition(12)
    for i in range(max_iter):
        updated_paths_df = update_paths_2(paths_df, edges_df)
        paths_df = updated_paths_df.alias('df1').join(
            solution_df.alias('df2'),
            on=[col('df1.edge_1') == col('df2.edge_1'), col('df1.edge_2') == col('df2.edge_2')],
            how='outer'
        )

        new_paths = paths_df \
            .where('df1.length IS NOT NULL AND (df2.length IS NULL OR df1.length < df2.length)') \
            .select(col('df1.edge_1').alias('edge_1'),
                    col('df1.edge_2').alias('edge_2'),
                    col('df1.length').alias('length'))

        # TODO: do we have to do checkpoints if we already have the 'isEmpty' condition here?
        if new_paths.isEmpty():
            break

        solution_df = paths_df \
            .selectExpr("COALESCE(df1.edge_1, df2.edge_1) as edge_1",
                        "COALESCE(df1.edge_2, df2.edge_2) as edge_2",
                        "least(df1.length, df2.length) as length")  # TODO: least? Will it really work?

        solution_df.checkpoint()

        paths_df = new_paths
        paths_df.checkpoint()

    return solution_df


# TODO: cache? checkpoint? Unpersist?
# We can't do the similar approach for avoiding creating the same paths as in the edges, because
# Updating a path from X to Y means we have to join that path now with all the paths from previous version of our path dataset
# We can't just add it to only paths we updated in the step before
# So how do we avoid adding the same paths over and over again?
# Well
#
def join_paths(edges_df, max_iter=999999):
    initial_paths_df = edges_df.selectExpr("edge_1 as edge_1", "edge_2 as edge_2", "length")
    for i in range(max_iter):
        # Update paths
        updated_paths_df = update_paths(initial_paths_df, initial_paths_df)
        # Check if there are any changes in paths
        is_not_changed = updated_paths_df.selectExpr("edge_1 as edge_1", "edge_2 as edge_2", "length") \
            .exceptAll(initial_paths_df).isEmpty()
        # TODO: that exceptAll does not remove new paths with length <= old length !!!
        if is_not_changed:
            break

        # Update the paths for the next iteration
        initial_paths_df = updated_paths_df
    return initial_paths_df


def get_paths(algorithm_version: str, edges_df, output_path: str):
    if algorithm_version == 'linear':
        outcome = join_edges_simple(edges_df)
    elif algorithm_version == 'doubling':
        outcome = join_paths(edges_df)
    else:
        raise Exception("provided algorithm_version {} is invalid! Choose from linear or doubling"
                        .format(algorithm_version)
                        )

    outcome.show()
    # TODO: change this to create a single cvs, not some weird folders
    outcome.repartition(1).write.csv(output_path, header=True, mode='overwrite')

if __name__ == '__main__':
    algorithm_version, input_path, output_path = parseargs()
    # Initialize Spark session
    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "3g") \
        .appName("mlibs") \
        .getOrCreate()

    # spark = SparkSession.builder \
    #     .master("spark://master:7077") \
    #     .config("spark.executor.memory", "2g") \
    #     .config("spark.driver.memory", "3g") \
    #     .appName("mlibs") \
    #     .getOrCreate()

    # spark.sparkContext.setCheckpointDir("checkpointDir")

    # TODO: setting this to -1 shouldn't help since it's 10MB by default
    # spark.sql.autoBroadcastJoinThreshold

    # TODO: maybe we shouldn't read it with spark directly...
    # As spark expects the file to be on all worker nodes

    # TODO: use explain !!!

    edges_df = spark.read.csv(input_path, schema=StructType([
        StructField("edge_1", LongType(), True),
        StructField("edge_2", LongType(), True),
        StructField("length", DoubleType(), True)
    ]), header=True)

    get_paths(algorithm_version, edges_df, output_path)
