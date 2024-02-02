import inspect
import numpy as np
import pandas as pd
import time
import warnings

from colorama import Fore, init
from rich import console
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

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


def kmeans_elbow(data, low:int = 1, high:int = 10):
    if high <= 1:
        return 1

    _elbow  = 1
    _value  = 0
    _last   = 0
    _k      = range(low, high)
    _sum_of_squared_distances = []

    for num_clusters in _k:
        _kmeans = KMeans(init="k-means++", n_clusters=num_clusters, n_init=10, max_iter=300, random_state=42).fit(data)
        _sum_of_squared_distances.append(_kmeans.inertia_)

    for i in range(1, len(_sum_of_squared_distances)):
        _value = _sum_of_squared_distances[i - 1] - _sum_of_squared_distances[i]
        if _value > _last:
            _elbow += 1 
        _last = _value

    return _elbow


def get_recommendations(item:str, matrix:pd.DataFrame=None, n:int=10, result_type:str='similar'):
    _result = pd.DataFrame()

    if matrix is None:
        raise ValueError("matrix cannot be None")

    if item not in matrix.index:
        raise ValueError(f"{item} does not exist in matrix")

    if result_type.lower() == 'similar':
        _result = matrix.loc[item].sort_values(ascending=True)[1:n+1].to_frame()
        _result.columns = ['similarity_score']
        _result['rank'] = _result['similarity_score'].rank(ascending=True)

    elif result_type.lower() == 'dissimilar':
        # _result = matrix.loc[item].sort_values(ascending=False)[-(n+1):].to_frame()
        # _result.columns = ['dissimilarity_score']
        # _result['rank'] = _result['dissimilarity_score'].rank(ascending=False)
        # _result.drop(_result.index[-1], inplace=True)

        _result = 1 - matrix.loc[item].sort_values(ascending=False)[-(n+1):].to_frame()
        _result.columns = ['dissimilarity_score']
        _result['rank'] = _result['dissimilarity_score'].rank(ascending=True)
        _result.drop(_result.index[-1], inplace=True)

    return _result


def kmeans_confidence(data:pd.DataFrame, k:int, confidence=CONFIDENCE_LEVEL):
    """Calculates the confidence interval for a k-means model

    Args:
        data (pd.DataFrame): The data to be clustered
        k (int): The number of clusters
        confidence (float, optional): The desired confidence level. Defaults to 0.95

    Returns:
        A tuple of two floats, the lower and upper bounds of the confidence interval
    """

    # Fit the k-means model
    model = KMeans(n_clusters=k).fit(data)

    # Calculate the inertia
    inertia = model.inertia_

    # Calculate the degrees of freedom
    degree_of_freedom = k - 1

    # Calculate the critical value
    t_critical = stats.t.ppf((1 - confidence) / 2, degree_of_freedom)

    # Calculate the confidence interval
    lower_bound = inertia - t_critical * np.sqrt(2 * inertia / degree_of_freedom)
    upper_bound = inertia + t_critical * np.sqrt(2 * inertia / degree_of_freedom)

    return lower_bound, upper_bound


def calculate_rmse(set1, set2):
    # Convert the sets to NumPy arrays for ease of calculation
    set1 = np.array(set1)
    set2 = np.array(set2)

    # Calculate the squared differences
    squared_diff = (set1 - set2) ** 2

    # Calculate the mean of squared differences
    mean_squared_diff = np.mean(squared_diff)

    # Calculate the square root to get the RMSE
    rmse = np.sqrt(mean_squared_diff)

    return rmse
# ### [END] METHODS ###

# Load data
startTime_cpu = time.process_time() # CPU time
ratings = pd.read_csv(r"data/bx_ratings.csv", sep=";", on_bad_lines="skip", encoding="latin-1", usecols=["User-ID", "ISBN", "Book-Rating"], dtype={"User-ID": "int", "ISBN": "str", "Book-Rating": "int"}, nrows=50000, )
books = pd.read_csv(r"data/bx_books.csv", sep=";", on_bad_lines="skip", encoding="latin-1", usecols=["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"], dtype={"ISBN": "str", "Book-Title": "str", "Book-Author": "str", "Year-Of-Publication": "str", "Publisher": "str"}, nrows=50000, )
users = pd.read_csv(r"data/bx_users.csv", sep=";", on_bad_lines="skip", encoding="latin-1", usecols=["User-ID", "Location", "Age"], dtype={"User-ID": "int", "Location": "str", "Age": "str"}, nrows=50000, )
print(Fore.CYAN + f">>> line {lineno()}: data loaded >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# ### [START] DATA CLEANING ###
# rename columns
startTime_cpu = time.process_time() # CPU time
ratings.columns = ["user_id", "isbn", "rating", ]
books.columns = ["isbn", "title", "author", "year", "publisher", ]
users.columns = ["user_id", "location", "age", ]
print(Fore.CYAN + f">>> line {lineno()}: primary columns renamed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

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

# Filter out users and books with low ratings
startTime_cpu = time.process_time() # CPU time
ratings_count = ratings.groupby(["isbn"]).count()
popular_books = ratings_count[ratings_count["rating"] >= 10].index
ratings = ratings[ratings["isbn"].isin(popular_books)]
ratings = ratings[ratings["user_id"].isin(users["user_id"])]
print(Fore.CYAN + f">>> line {lineno()}: users and books with low ratings filtered out >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
# ### [END] DATA CLEANING ###

# Merge Books and Ratings
startTime_cpu = time.process_time() # CPU time
books_ratings = pd.merge(left=books, right=ratings, on="isbn")
print(Fore.CYAN + f">>> line {lineno()}: books and ratings merged >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Keeping the following columns: isbn, title, user_id, rating
startTime_cpu = time.process_time() # CPU time
books_ratings = books_ratings[["isbn", "title", "user_id", "rating"]]
print(Fore.CYAN + f">>> line {lineno()}: keeping the following columns: isbn, title, user_id, rating >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Create a Pivot Matrix: Users as rows and Books as columns
startTime_cpu = time.process_time() # CPU time
books_pivot = books_ratings.pivot(index="user_id", columns="isbn", values="rating").fillna(0)
print(Fore.CYAN + f">>> line {lineno()}: pivot matrix created >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Create a Sparse Matrix
# A CSR is a compressed sparse row or compressed row storage matrix. It’s just a fancy way of storing only the non-zero entries in a matrix
startTime_cpu = time.process_time() # CPU time
book_matrix = csr_matrix(books_pivot.values)
print(Fore.CYAN + f">>> line {lineno()}: sparse matrix created >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Normalizing data for clustering - normalizing over the standard deviation
startTime_cpu = time.process_time() # CPU time
book_ratings_norm = StandardScaler(with_mean=False).fit_transform(books_ratings[['rating']])
print(Fore.CYAN + f">>> line {lineno()}: normalizing data for clustering: completed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Compute item-item similarity (using cosine similarity)
startTime_cpu = time.process_time() # CPU time
item_similarity = pairwise_distances(book_matrix.T, metric="cosine") # Transpose to get item-item similarity
item_similarity_df = pd.DataFrame(item_similarity, index=books_pivot.columns, columns=books_pivot.columns)
print(Fore.CYAN + f">>> line {lineno()}: item-item similarity computed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Normalize similarity values
startTime_cpu = time.process_time() # CPU time
item_similarity_norm = StandardScaler(with_mean=False).fit_transform(item_similarity)
item_similarity_norm_df = pd.DataFrame(item_similarity_norm, index=books_pivot.columns, columns=books_pivot.columns)
print(Fore.CYAN + f">>> line {lineno()}: similarity values normalized >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Compute item-item dissimilarity (using cosine dissimilarity)
startTime_cpu = time.process_time() # CPU time
item_dissimilarity = 1 - pairwise_distances(book_matrix.T, metric="cosine")
item_dissimilarity_df = pd.DataFrame(item_dissimilarity, index=books_pivot.columns, columns=books_pivot.columns)
print(Fore.CYAN + f">>> line {lineno()}: item-item dissimilarity computed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# Normalize dissimilarity values
startTime_cpu = time.process_time() # CPU time
item_dissimilarity_norm = StandardScaler(with_mean=False).fit_transform(item_dissimilarity)
item_dissimilarity_norm_df = pd.DataFrame(item_dissimilarity_norm, index=books_pivot.columns, columns=books_pivot.columns)
print(Fore.CYAN + f">>> line {lineno()}: dissimilarity values normalized >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

# ### [START] K-Means Clustering
startTime_cpu = time.process_time() # CPU time
print(Fore.LIGHTCYAN_EX + f">>> line {lineno()}: START: K-MEANS CLUSTERING >>>")

num_clusters = kmeans_elbow(book_ratings_norm) # this is the number of clusters per the elbow method
print(Fore.CYAN + f">>> line {lineno()}: number of clusters: {num_clusters} >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

kmeans = KMeans(init="k-means++", n_clusters=num_clusters, n_init=10, max_iter=300, random_state=42).fit(book_ratings_norm)
print(Fore.CYAN + f">>> line {lineno()}: kmeans similarity computed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

kmeans_dissimilarity = KMeans(init="k-means++", n_clusters=num_clusters, n_init=10, max_iter=300, random_state=42).fit(item_dissimilarity_norm)
print(Fore.CYAN + f">>> line {lineno()}: kmeans dissimilarity computed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

item_similarity_df["kmeans_label"] = pd.Series(kmeans.labels_)
item_similarity_norm_df["kmeans_label"] = pd.Series(kmeans.labels_)
print(Fore.CYAN + f">>> line {lineno()}: kmeans labels added to similarity dataframes >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

item_dissimilarity_df["kmeans_label"] = pd.Series(kmeans_dissimilarity.labels_)
item_dissimilarity_norm_df["kmeans_label"] = pd.Series(kmeans_dissimilarity.labels_)
print(Fore.CYAN + f">>> line {lineno()}: kmeans labels added to dissimilarity dataframes >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

_kmeans_sim = get_recommendations(item=ITEM_ISBN, matrix=item_similarity_df, result_type='similar')
_kmeans_dissim = get_recommendations(item=ITEM_ISBN, matrix=item_dissimilarity_df, result_type='dissimilar')

console.print()
console.print(_kmeans_sim)
console.print()
console.print(_kmeans_dissim)
console.print()

# compute an average silhouette score for each point
kmeans_silhouette = silhouette_score(book_ratings_norm, kmeans.labels_)

console.print(f"K-means Silhouette Score: {kmeans_silhouette}")
console.print()

# compute the confidence interval for the k-means model
kmeans_confidence_interval = kmeans_confidence(book_ratings_norm, num_clusters)

console.print(f"K-means Confidence Interval: {kmeans_confidence_interval}")
console.print()

print(Fore.LIGHTCYAN_EX + f">>> line {lineno()}: END: K-MEANS CLUSTERING >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
# ### [END] K-Means Clustering

# ### [START] DBSCAN Clustering
startTime_cpu = time.process_time() # CPU time
print(Fore.LIGHTCYAN_EX + f">>> line {lineno()}: START: DBSCAN CLUSTERING >>>")

dbscan = DBSCAN(eps=0.5, min_samples=5, ).fit(book_ratings_norm)
print(Fore.CYAN + f">>> line {lineno()}: dbscan similarity computed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

dbscan_dissimilarity = DBSCAN(eps=0.5, min_samples=5, ).fit(item_dissimilarity_norm)
print(Fore.CYAN + f">>> line {lineno()}: dbscan dissimilarity computed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

item_similarity_df.drop(columns=["kmeans_label"], inplace=True)
item_similarity_norm_df.drop(columns=["kmeans_label"], inplace=True)
print(Fore.CYAN + f">>> line {lineno()}: similarity dataframes reset: kmeans labels removed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

item_dissimilarity_df.drop(columns=["kmeans_label"], inplace=True)
item_dissimilarity_norm_df.drop(columns=["kmeans_label"], inplace=True)
print(Fore.CYAN + f">>> line {lineno()}: dissimilarity dataframes reset: kmeans labels removed >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

item_similarity_df["dbscan_label"] = pd.Series(dbscan)
item_similarity_norm_df["dbscan_label"] = pd.Series(dbscan)
print(Fore.CYAN + f">>> line {lineno()}: dbscan labels added to similarity dataframes >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

item_dissimilarity_df["dbscan_label"] = pd.Series(dbscan_dissimilarity)
item_dissimilarity_norm_df["dbscan_label"] = pd.Series(dbscan_dissimilarity)
print(Fore.CYAN + f">>> line {lineno()}: dbscan labels added to dissimilarity dataframes >>> {(time.process_time() - startTime_cpu) * 10**3}ms")

_dbscan_sim = get_recommendations(item=ITEM_ISBN, matrix=item_similarity_df, result_type='similar')
_dbscan_dissim = get_recommendations(item=ITEM_ISBN, matrix=item_dissimilarity_df, result_type='dissimilar')

console.print()
console.print(_dbscan_sim)
console.print()
console.print(_dbscan_sim.describe())
console.print()
console.print(_dbscan_dissim)
console.print()
console.print(_dbscan_dissim.describe())
console.print()

# compute an average silhouette score for each point
dbscan_silhouette = silhouette_score(books_pivot, dbscan.fit_predict(books_pivot))
dbscan_silhouette_individual = silhouette_samples(books_pivot, dbscan.fit_predict(books_pivot))

console.print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")
console.print()

# compute the confidence interval for the dbscan model
dbscan_confidence_interval = np.percentile(dbscan_silhouette_individual, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])

console.print(f"DBSCAN Confidence Interval: {dbscan_confidence_interval}")
console.print()

print(Fore.LIGHTCYAN_EX + f">>> line {lineno()}: END: DBSCAN CLUSTERING >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
# ### [END] DBSCAN Clustering

### [START] Hypothesis Testing #1
startTime_cpu = time.process_time() # CPU time
print(Fore.LIGHTCYAN_EX + f">>> line {lineno()}: START: HYPOTHESIS TESTING #1 >>>")

# Perform the two sample t-test with equal variances
h1_t_stat, h1_p_value = stats.ttest_ind(list(_kmeans_sim['similarity_score'][0:9]), list(_dbscan_sim['similarity_score'][0:9]), equal_var=True)
console.print("t-statistic: ", np.nan_to_num(h1_t_stat))
console.print("p-value: ", np.nan_to_num(h1_p_value))

print(Fore.CYAN + f">>> line {lineno()}: Hypothesis Testing #1 >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
console.print()

### [START] Hypothesis Testing #2
startTime_cpu = time.process_time() # CPU time
print(Fore.LIGHTCYAN_EX + f">>> line {lineno()}: START: HYPOTHESIS TESTING #2 >>>")

# Calculate the accuracy
accuracy = calculate_rmse(list(_kmeans_sim['similarity_score'][0:9]), list(_dbscan_sim['similarity_score'][0:9]))
console.print("rmse: ", np.nan_to_num(accuracy))

print(Fore.CYAN + f">>> line {lineno()}: Hypothesis Testing #2 >>> {(time.process_time() - startTime_cpu) * 10**3}ms")
console.print()

executionTime = (time.time() - startTime) * 10**3
print(Fore.LIGHTGREEN_EX + f"Execution time in milliseconds: {executionTime}ms")
