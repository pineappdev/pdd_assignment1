# Shortest paths in Spark

This is the Spark implementation of a shortest path finding algorithm in big graphs.
The algorithm comes in two flavours:
1. starting with a set of paths initiated to edges, keep adding single edges to existing paths to the paths until no new (shortest) paths are created.
2. starting with a set of paths initiated to edges, keep extending the paths by joining a path with all possible paths until no new (shortest) paths are created.

How to run:
- pip install pyspark
- python main.py (the script will give you description on required command line arguments if missing)
