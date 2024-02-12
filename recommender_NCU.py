import inspect
import numpy as np
import pandas as pd
import os
import time
import warnings

from colorama import Fore, init
from rich import console
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

# clear the output
os.system('cls') # for windows
os.system('clear') # for linux/mac

startTime = time.time()

# Initialize colorama
init(autoreset=True)

print(Fore.LIGHTGREEN_EX + "The Force has awaken >>>")

warnings.filterwarnings("ignore")
# For more control over rich terminal content, import and construct a Console object
console = console.Console()

# ### GLOBAL VARIABLES ###
CONFIDENCE_LEVEL = 0.95
ALPHA = 1 - CONFIDENCE_LEVEL
ITEM_TITLE = "Angels &amp; Demons"
ITEM_ISBN = "0671027360"

# ### METHODS ###
def lineno():
    """
    Returns the current line number in our program.

    This function uses the inspect module to get the current frame and returns the line number of the caller's frame.

    Returns:
        int: The line number of the caller's frame.

    Raises:
        AttributeError: If inspect.currentframe() returns None.
    """
    frame = inspect.currentframe()
    if frame is None:
        raise AttributeError("inspect.currentframe() returned None")
    return frame.f_back.f_lineno


def item_item_collaborative_filtering(books_ratings, isbn, n: int = 20, cluster_type: str = "kmeans"):
    startTime_cpu = time.process_time() # CPU time

    _result = pd.DataFrame()

    # Get the cluster type
    _cluster_type = "kmeans_cluster" if cluster_type.lower() == "kmeans" else "dbscan_cluster"
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: cluster type: {_cluster_type} >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # keep a copy of the original dataframe
    books_ratings_ori = books_ratings.copy()
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: copy of the original dataframe created >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # Filter the books_ratings dataframe to include only the specified ISBN
    books_ratings = books_ratings[books_ratings["isbn"] == isbn]
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: books_ratings filtered to include only the specified ISBN >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
 
    # Get the MAX cluster label of the specified book
    max_rating_cluster = books_ratings.loc[books_ratings["rating"].idxmax(), _cluster_type]
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: max cluster label of the specified book >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
 
    # Filter the books_ratings_ori dataframe to include only the books in the same Max cluster
    cluster_books = books_ratings_ori[books_ratings_ori[_cluster_type] == max_rating_cluster]
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: books_ratings_ori filtered to include only the books in the same Max cluster >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
 
    # Create a Pivot Matrix: Users as rows and Books as columns
    cluster_books_pivot = cluster_books.pivot(index="user_id", columns="isbn", values="rating").fillna(0)
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: pivot matrix created >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # Create a Sparse Matrix
    #   if most of the elements of the matrix have 0 value, then it is called a sparse matrix
    #   Representing a sparse matrix by a 2D array leads to wastage of lots of memory as zeroes in the matrix are of no use in most 
    #   of the cases. So, instead of storing zeroes with non-zero elements, we only store non-zero elements. This means storing 
    # non-zero elements with triples- (Row, Column, value)
    cluster_book_matrix = csr_matrix(cluster_books_pivot.values)
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: sparse matrix created >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    item_similarity = pairwise_distances(cluster_book_matrix.T, metric="cosine")
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: item similarity computed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    item_similarity_df = pd.DataFrame(item_similarity, index=cluster_books_pivot.columns, columns=cluster_books_pivot.columns)
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: item similarity dataframe created >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    _result = item_similarity_df.loc[isbn].sort_values(ascending=True)[1:n+1].to_frame()
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: top similar books computed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    _result.columns = ['similarity_score']
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: column name changed to 'similarity_score' >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    _result['rank'] = _result['similarity_score'].rank(ascending=True)
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: rank computed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    return _result


def recommend_similar_books(books_ratings, isbn, n: int = 20, cluster_type: str = "kmeans"):
    startTime_cpu = time.process_time() # CPU time

    book = books_ratings[books_ratings['isbn'] == isbn]
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: book found >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    _cluster_type = "kmeans_cluster" if cluster_type.lower() == "kmeans" else "dbscan_cluster"
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: cluster type: {_cluster_type} >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    if book.empty:
        print(Fore.RED + f">>> >>>>> line {lineno()}: book with ISBN {isbn} not found >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
        return None
    else:
        book_cluster = book[_cluster_type].values[0]
        print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: book cluster found >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

        similar_books = books_ratings[books_ratings[_cluster_type] == book_cluster]
        print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: similar books found >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

        similar_books = similar_books[similar_books["isbn"] != isbn]
        print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: similar books filtered >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

        similar_books = similar_books.sort_values(by="rating", ascending=False).head(n)
        print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: top similar books found >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

        similar_books.drop_duplicates(inplace=True)
        print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: duplicates removed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    return similar_books[["isbn", "title", _cluster_type, "rating", "rating_normalized"]]


# ### LOAD DATA ###
startTime_cpu = time.process_time() # CPU time
ratings = pd.read_csv(r"data/bx_ratings.csv", sep=";", on_bad_lines="skip", encoding="latin-1", usecols=["User-ID", "ISBN", "Book-Rating"], dtype={"User-ID": "int", "ISBN": "str", "Book-Rating": "int"}, nrows=50000, )
books = pd.read_csv(r"data/bx_books.csv", sep    =";", on_bad_lines="skip", encoding="latin-1", usecols=["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"], dtype={"ISBN": "str", "Book-Title": "str", "Book-Author": "str", "Year-Of-Publication": "str", "Publisher": "str"}, nrows=50000, )
users = pd.read_csv(r"data/bx_users.csv", sep=";", on_bad_lines="skip", encoding="latin-1", usecols=["User-ID", "Location", "Age"], dtype={"User-ID": "int", "Location": "str", "Age": "str"}, nrows=50000, )
print(Fore.CYAN + f">>> line {lineno()}: data loaded >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# ### [START] DATA CLEANING ###
# rename columns
startTime_cpu = time.process_time() # CPU time
ratings.columns = ["user_id", "isbn", "rating", ]
books.columns = ["isbn", "title", "author", "year", "publisher", ]
users.columns = ["user_id", "location", "age", ]
print(Fore.CYAN + f">>> line {lineno()}: primary columns renamed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Get the count of times a book has been rated
startTime_cpu = time.process_time() # CPU time
ratings_count = ratings["isbn"].value_counts()
print(Fore.CYAN + f">>> line {lineno()}: count of times a book has been rated >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Add the count of times a book has been rated to the books dataframe
startTime_cpu = time.process_time() # CPU time
books = books.merge(ratings_count, left_on="isbn", right_index=True, how="left")
books.rename(columns={"isbn": "isbn", "title": "title", "author": "author", "year": "year", "publisher": "publisher", "count": "rating_count", }, inplace=True)
print(Fore.CYAN + f">>> line {lineno()}: count of times a book has been rated added to the books dataframe >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# console.print(books["034545104X" == books["isbn"]])
# console.print(books.head(5))

# Remove duplicates and books with missing values
startTime_cpu = time.process_time() # CPU time
books.drop_duplicates(["isbn"], inplace=True, )
ratings.drop_duplicates(["user_id", "isbn"], inplace=True, )
users.drop_duplicates(["user_id"], inplace=True, )
print(Fore.CYAN + f">>> line {lineno()}: duplicates and missing values removed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Remove columns with NaN values
startTime_cpu = time.process_time() # CPU time
books.dropna(inplace=True)
ratings.dropna(inplace=True)
users.dropna(inplace=True)
print(Fore.CYAN + f">>> line {lineno()}: columns with NaN values removed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Deleting rows with wrong values in column "year" ("year_of_publication")
startTime_cpu = time.process_time() # CPU time
bad_year_books = books[books["year"].isin(list(filter(lambda x : (x.isnumeric() is False), books["year"])))].index
books.drop(bad_year_books, inplace=True)
print(Fore.CYAN + f">>> line {lineno()}: wrong values in column 'year' removed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Convert Users' Age to int
# We have decide to let users.age as a numerical (quantitative) data ==> "numerical" we add a continuity to the data. 
#   It might, for example, show that people read more books or rate them lower as they age...
#   I am going to leave it as it is for now (numerical) and see how it goes...
# Side Note: On the other hand, if we had chosen to change it to categorical (string == categorical (qualitative)) then it will be better to 
#   talk about "age group" and we will treat the users of each group as a class
startTime_cpu = time.process_time() # CPU time
users["age"] = users["age"].astype("int")
print(Fore.CYAN + f">>> line {lineno()}: users' age converted to int >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Convert Publishers' Year of Publication to int
startTime_cpu = time.process_time() # CPU time
books["year"] = books["year"].astype("int")
print(Fore.CYAN + f">>> line {lineno()}: publishers' year of publication converted to int >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Deleting rows with "ages" less than 5 and greater than 100
startTime_cpu = time.process_time() # CPU time
bad_age_users = users[users["age"].isin(list(filter(lambda x : (x < 5 or x > 100), users["age"])))].index
users.drop(bad_age_users, inplace=True)
print(Fore.CYAN + f">>> line {lineno()}: ages less than 5 and greater than 100 removed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Merge Books and Ratings
startTime_cpu = time.process_time() # CPU time
books_ratings = pd.merge(left=books, right=ratings, on="isbn")
print(Fore.CYAN + f">>> line {lineno()}: books and ratings merged >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Keeping the following columns: isbn, title, user_id, rating
startTime_cpu = time.process_time() # CPU time
books_ratings = books_ratings[["isbn", "title", "user_id", "rating", "rating_count"]]
print(Fore.CYAN + f">>> line {lineno()}: keeping the following columns: isbn, title, user_id, rating >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
# ### [END] DATA CLEANING ###

# Normalizing data for clustering - normalizing over the standard deviation: 
#   scale the data so that all features have a mean of 0 and a standard deviation of 1
# In Python, it is important to normalize data for clustering before applying a clustering algorithm because it helps to ensure that all 
#   features are on an equal scale. This is important because clustering algorithms work by calculating the distance between data points, 
#   and if one feature has a much larger range of values than another, it will have a greater impact on the distance calculation. This can 
#   lead to the algorithm clustering the data based on that one feature, even if it is not the most important feature.
startTime_cpu = time.process_time() # CPU time
book_ratings_norm = StandardScaler(with_mean=False).fit_transform(books_ratings[['rating']])
books_ratings['rating_normalized'] = book_ratings_norm

book_ratings_count_norm = StandardScaler(with_mean=False).fit_transform(books_ratings[['rating_count']])
books_ratings['rating_count_normalized'] = book_ratings_count_norm

# console.print(books_ratings.head(5))

print(Fore.CYAN + f">>> line {lineno()}: normalizing data for clustering: completed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# ### [START] CLUSTERS ###
# ### [START] K-Means Clustering

# ### [END] K-Means Clustering
# K-Means wants numerical columns, with no null/infinite values and avoid categorical data
startTime_cpu = time.process_time() # CPU time
print(Fore.LIGHTCYAN_EX + f">>> line {lineno()}: START: K-MEANS CLUSTERING >>>")

# console.print(books_ratings.head(5))
# console.print(books_ratings["034545104X" == books_ratings["isbn"]])

kmeans = KMeans(init="k-means++", n_clusters=15, ).fit(books_ratings[["rating"]])
print(Fore.CYAN + f">>> line {lineno()}: kmeans similarity computed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

books_ratings["kmeans_cluster"] = kmeans.labels_
print(Fore.CYAN + f">>> line {lineno()}: kmeans cluster added to books_ratings >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# console.print(books_ratings.head(5))
# console.print(books_ratings["034545104X" == books_ratings["isbn"]])

# console.print(books_ratings[books_ratings["kmeans_cluster"] == 0])
# console.print(books_ratings[books_ratings["kmeans_cluster"] == 1])
# console.print(books_ratings[books_ratings["kmeans_cluster"] == 2])
# console.print(books_ratings[books_ratings["kmeans_cluster"] == 3])
# console.print(books_ratings[books_ratings["kmeans_cluster"] == 4])
# console.print(books_ratings[books_ratings["kmeans_cluster"] == 5])
# console.print(books_ratings[books_ratings["kmeans_cluster"] == 6])
# console.print(books_ratings[books_ratings["kmeans_cluster"] == 7])
# console.print(books_ratings[books_ratings["kmeans_cluster"] == 8])
# console.print(books_ratings[books_ratings["kmeans_cluster"] == 9])
# console.print(books_ratings[books_ratings["kmeans_cluster"] == 10])
# console.print(books_ratings[books_ratings["kmeans_cluster"] == 11])

# console.print(books_ratings["kmeans_cluster"].value_counts())
# console.print(books_ratings["kmeans_cluster"].unique())

# compute an average silhouette score for each point
# The silhouette score is a measure of how SIMILAR an object is to its own cluster (cohesion) compared to other clusters (separation)
# The silhouette score ranges from -1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. 
silhouette_scores_rating = silhouette_score(books_ratings[['rating_normalized']], books_ratings['kmeans_cluster'])
silhouette_scores_rating_count = silhouette_score(books_ratings[['rating_count_normalized']], books_ratings['kmeans_cluster'])

# console.print(silhouette_scores_rating)
# console.print(silhouette_scores_rating_count)
print(Fore.CYAN + f">>> line {lineno()}: k-means silhouette score computed for each point >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# compute the confidence interval for the k-means model
# confidence score is a metric for the confidence level of the assignment of an entity to a cluster
# The confidence score ranges from 0 to 1, where a high value indicates that the object is well matched to its own cluster and 
# poorly matched to neighboring clusters
kmeans_confidence_interval = kmeans.score(books_ratings[['rating']]) 

# console.print(kmeans_confidence_interval)
print(Fore.CYAN + f">>> line {lineno()}: confidence scores computed for k-means model >>> {(time.process_time() - startTime_cpu) * 10**3}ms")


print(Fore.LIGHTCYAN_EX + f">>> line {lineno()}: END: K-MEANS CLUSTERING >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# ### [START] DBSCAN Clustering
startTime_cpu = time.process_time() # CPU time
print(Fore.LIGHTCYAN_EX + f">>> line {lineno()}: START: DBSCAN CLUSTERING >>>")

dbscan = DBSCAN(eps=0.5, min_samples=5, ).fit(books_ratings[["rating"]])
print(Fore.CYAN + f">>> line {lineno()}: dbscan similarity computed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

books_ratings["dbscan_cluster"] = dbscan.labels_
print(Fore.CYAN + f">>> line {lineno()}: dbscan cluster added to books_ratings >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# console.print(books_ratings.head(5))
# console.print(books_ratings["034545104X" == books_ratings["isbn"]])

# console.print(books_ratings[books_ratings["dbscan_cluster"] == 0])
# console.print(books_ratings[books_ratings["dbscan_cluster"] == 1])
# console.print(books_ratings[books_ratings["dbscan_cluster"] == 2])
# console.print(books_ratings[books_ratings["dbscan_cluster"] == 3])
# console.print(books_ratings[books_ratings["dbscan_cluster"] == 4])
# console.print(books_ratings[books_ratings["dbscan_cluster"] == 5])
# console.print(books_ratings[books_ratings["dbscan_cluster"] == 6])
# console.print(books_ratings[books_ratings["dbscan_cluster"] == 7])
# console.print(books_ratings[books_ratings["dbscan_cluster"] == 8])
# console.print(books_ratings[books_ratings["dbscan_cluster"] == 9])
# console.print(books_ratings[books_ratings["dbscan_cluster"] == 10])
# console.print(books_ratings[books_ratings["dbscan_cluster"] == 11])

# console.print(books_ratings["dbscan_cluster"].value_counts())
# console.print(books_ratings["dbscan_cluster"].unique())

# compute an average silhouette score for each point
# The silhouette score is a measure of how SIMILAR an object is to its own cluster (cohesion) compared to other clusters (separation)
# The silhouette score ranges from -1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. 
silhouette_scores_db_rating = silhouette_score(books_ratings[['rating_normalized']], books_ratings['dbscan_cluster'])
silhouette_scores_db_rating_count = silhouette_score(books_ratings[['rating_count_normalized']], books_ratings['dbscan_cluster'])

# console.print(silhouette_scores_db_rating)
# console.print(silhouette_scores_db_rating_count)
print(Fore.CYAN + f">>> line {lineno()}: dbscan silhouette score computed for each point >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# compute the confidence interval for the dbscan model
dbscan_confidence_interval = np.percentile(silhouette_scores_db_rating, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])

# console.print(f"DBSCAN Confidence Interval: {dbscan_confidence_interval}")
print(Fore.CYAN + f">>> line {lineno()}: confidence scores computed for dbscan model >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

print(Fore.LIGHTCYAN_EX + f">>> line {lineno()}: END: DBSCAN CLUSTERING >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
# ### [END] DBSCAN Clustering
# ### [END] CLUSTERS ###

# ### [START] RECOMMENDATION SYSTEM ###    
startTime_cpu = time.process_time() # CPU time
# console.print(item_item_collaborative_filtering(books_ratings, ITEM_ISBN, ))
console.print(recommend_similar_books(books_ratings, ITEM_ISBN, ))
print(Fore.CYAN + f">>> line {lineno()}: top similar books recommended >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
# ### [END] RECOMMENDATION SYSTEM ###

executionTime = (time.time() - startTime) * 10**3
print(Fore.LIGHTGREEN_EX + f"Execution time in milliseconds: {executionTime}ms")