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
ITEM_ISBN = "0671027360"
PRINT_CLUSTER_LABELS = False


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


def rmse(predictions, targets):
  return np.sqrt(np.mean((predictions - targets)**2))


def recommend_similar_books(books_ratings: pd.DataFrame, isbn: str, n: int = 11, cluster_type: str = "kmeans"):
    startTime_cpu = time.process_time() # CPU time
    print(Fore.LIGHTGREEN_EX + f">>> line {lineno()}: START METHOD: RECOMMEND_SIMILAR_BOOKS >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # Get the cluster label for the book
    _cluster_type = "kmeans_cluster" if cluster_type.lower() == "kmeans" else "dbscan_cluster"
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: cluster type: {_cluster_type} >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # find the book in the dataframe given the isbn
    book = books_ratings[books_ratings['isbn'] == isbn]
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: book found >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # if the book is not found, return None
    if book.empty:
        print(Fore.RED + f">>> >>>>> line {lineno()}: book with ISBN {isbn} not found >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
        return None

    # get the MAX cluster label for the book
    max_book_cluster = book[_cluster_type].values[0]
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: book cluster found >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # get all books in the same cluster
    books_in_cluster = books_ratings[books_ratings[_cluster_type] == max_book_cluster]
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: similar books found >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # create a pivot table of the books in the same cluster and their ratings
    pivot_table = books_in_cluster.pivot(index='isbn', columns='user_id', values='rating_normalized').fillna(0)
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: pivot table created >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # remove the book itself from the list
    books_in_cluster = books_in_cluster[books_in_cluster['isbn'] != isbn]

    # Calculate the similarity between the selected book and each similar book
    similarities = pairwise_distances(pivot_table.loc[isbn].values.reshape(1, -1), pivot_table.loc[books_in_cluster['isbn']].values)
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: similarity calculated >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # Attach the similarity scores to the similar_books dataframe
    books_in_cluster['similarity'] = similarities[0]
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: similarity scores attached to the similar_books dataframe >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # remove the user_id column
    books_in_cluster.drop(columns=["user_id"], inplace=True)
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: clean up complete >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # sort the books by rating_count descending, and get the top n books
    similar_books = books_in_cluster.sort_values(by="rating_count", ascending=False).head(n)
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: top similar books found >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # remove duplicate books
    similar_books.drop_duplicates(inplace=True)
    print(Fore.LIGHTGREEN_EX + f">>> >>>>> line {lineno()}: duplicates removed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # create an array of predictions
    predictions = books_in_cluster[['isbn', 'similarity', 'rating_count_normalized']].groupby('isbn').mean()

    # sort the predictions by rating_count_normalized descending, and get the top n books
    predictions.sort_values(by="rating_count_normalized", ascending=False).head(n)

    # calculate the accuracy using np.mean, predictions, and the actual ratings
    accuracy = np.mean(predictions['similarity'])
    similar_books['accuracy'] = accuracy

    # Calculate the RMSE
    rmse_score = rmse(predictions['similarity'], predictions['rating_count_normalized'])
    similar_books['rmse'] = rmse_score

    print(Fore.LIGHTGREEN_EX + f">>> line {lineno()}: END METHOD: RECOMMEND_SIMILAR_BOOKS >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

    # return similar books
    return similar_books


# ### LOAD DATA ###
startTime_cpu = time.process_time() # CPU time
ratings = pd.read_csv(r"data/bx_ratings.csv", sep=";", on_bad_lines="skip", encoding="latin-1", usecols=["User-ID", "ISBN", "Book-Rating"], dtype={"User-ID": "int", "ISBN": "str", "Book-Rating": "int"}, nrows=10000, )
books = pd.read_csv(r"data/bx_books.csv", sep=";", on_bad_lines="skip", encoding="latin-1", usecols=["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"], dtype={"ISBN": "str", "Book-Title": "str", "Book-Author": "str", "Year-Of-Publication": "str", "Publisher": "str"}, nrows=10000, )
users = pd.read_csv(r"data/bx_users.csv", sep=";", on_bad_lines="skip", encoding="latin-1", usecols=["User-ID", "Location", "Age"], dtype={"User-ID": "int", "Location": "str", "Age": "str"}, nrows=10000, )
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
print(Fore.CYAN + f">>> line {lineno()}: normalizing data for clustering: completed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# ### [START] CLUSTERS ###
# ### [START] K-Means Clustering
# K-Means wants numerical columns, with no null/infinite values and avoid categorical data
startTime_cpu = time.process_time() # CPU time
print(Fore.LIGHTCYAN_EX + f">>> line {lineno()}: START: K-MEANS CLUSTERING >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# maximum number of clusters => max rating
# plus 1 because the range of the rating is from 0 to 10
max_rating = books_ratings['rating'].max()+1
print(Fore.CYAN + f">>> line {lineno()}: maximum number of clusters => max rating: {max_rating} >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

kmeans = KMeans(init="k-means++", n_clusters=max_rating, ).fit(books_ratings[["rating"]])
print(Fore.CYAN + f">>> line {lineno()}: kmeans set >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

books_ratings["kmeans_cluster"] = kmeans.labels_
print(Fore.CYAN + f">>> line {lineno()}: kmeans cluster added to books_ratings >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# compute an average silhouette score for each point
# The silhouette score is a measure of how SIMILAR an object is to its own cluster (cohesion) compared to other clusters (separation)
# The silhouette score ranges from -1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. 
kmeans_silhouette_scores_rating = silhouette_score(books_ratings[['rating_normalized']], books_ratings['kmeans_cluster'])
kmeans_silhouette_scores_rating_count = silhouette_score(books_ratings[['rating_count_normalized']], books_ratings['kmeans_cluster'])
print(Fore.CYAN + f">>> line {lineno()}: silhouette scores computed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# compute the confidence interval for the k-means model
# confidence score is a metric for the confidence level of the assignment of an entity to a cluster
# The confidence score ranges from 0 to 1, where a high value indicates that the object is well matched to its own cluster and 
# poorly matched to neighboring clusters
kmeans_confidence_interval = np.percentile(kmeans_silhouette_scores_rating, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
print(Fore.CYAN + f">>> line {lineno()}: confidence interval for the k-means model computed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# PRINT the list of the unique cluster labels
if PRINT_CLUSTER_LABELS:
    cluster_labels = books_ratings["kmeans_cluster"].unique()
    for label in cluster_labels:
        print(books_ratings[books_ratings["kmeans_cluster"] == label])

print(Fore.LIGHTCYAN_EX + f">>> line {lineno()}: END: K-MEANS CLUSTERING >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
# ### [END] K-Means Clustering

# ### [START] DBSCAN Clustering
startTime_cpu = time.process_time() # CPU time
print(Fore.LIGHTCYAN_EX + f">>> line {lineno()}: START: DBSCAN CLUSTERING >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

dbscan = DBSCAN(eps=0.5, min_samples=5, ).fit(books_ratings[["rating"]])
print(Fore.CYAN + f">>> line {lineno()}: dbscan similarity computed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

books_ratings["dbscan_cluster"] = dbscan.labels_
print(Fore.CYAN + f">>> line {lineno()}: dbscan cluster added to books_ratings >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# compute an average silhouette score for each point
# The silhouette score is a measure of how SIMILAR an object is to its own cluster (cohesion) compared to other clusters (separation)
# The silhouette score ranges from -1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. 
dbscan_silhouette_scores_rating = silhouette_score(books_ratings[['rating_normalized']], books_ratings['dbscan_cluster'])
dbscan_silhouette_scores_rating_count = silhouette_score(books_ratings[['rating_count_normalized']], books_ratings['dbscan_cluster'])
print(Fore.CYAN + f">>> line {lineno()}: dbscan silhouette score computed for each point >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# compute the confidence interval for the dbscan model
dbscan_confidence_interval = np.percentile(dbscan_silhouette_scores_rating, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
print(Fore.CYAN + f">>> line {lineno()}: confidence scores computed for dbscan model >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# PRINT the list of the unique cluster labels
if PRINT_CLUSTER_LABELS:
    cluster_labels = books_ratings["dbscan_cluster"].unique()
    for label in cluster_labels:
        print(books_ratings[books_ratings["dbscan_cluster"] == label])

print(Fore.LIGHTCYAN_EX + f">>> line {lineno()}: END: DBSCAN CLUSTERING >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
# ### [END] DBSCAN Clustering
# ### [END] CLUSTERS ###

# ### [START] RECOMMENDATION SYSTEM ###  
startTime_cpu = time.process_time() # CPU time
console.print(recommend_similar_books(books_ratings, ITEM_ISBN, ))
print()
console.print(recommend_similar_books(books_ratings, ITEM_ISBN, cluster_type='dbscan' ))
print(Fore.CYAN + f">>> line {lineno()}: top similar books recommended >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
# ### [END] RECOMMENDATION SYSTEM ###

executionTime = (time.time() - startTime) * 10**3
print(Fore.LIGHTGREEN_EX + f"Execution time in milliseconds: {executionTime}ms")
