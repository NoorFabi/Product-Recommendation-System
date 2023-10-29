from django.shortcuts import render
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
from sklearn.decomposition import TruncatedSVD
import numpy as np
# Create your views here.

# Load the saved model from the file
with open('kmeans_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the previously fitted vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def homepage(request):
    return render(request,'homepage.html')

def trending(request):
    if request.method == "POST":
        selected_value = request.POST.get('one')
        s = int(selected_value)
        df = pd.read_csv('popularity.csv')
        a = df.values.tolist()
        a = a[:s]
    else:
        df = pd.read_csv('popularity.csv')
        a = df.values.tolist()
        a = a[:5]  # Provide an empty list as a fallback when no form submission occurs
    return render(request, 'trending.html', {'rows': a})

def groupsitem(request):
    df = pd.read_csv("testing.csv")
    clusters = {}
    if request.method == "POST":
        value = request.POST.get('abc')
        Y = vectorizer.transform([value])
        prediction = loaded_model.predict(Y)
        a =  prediction[0]
        clusters[a] = df[df['cluster_id'] == a].values.tolist()
    else:
        a = ""
    context = {'clusters': clusters, 'my_list': a}
    return render(request,'groupsitem.html',context)


def categories(request):
    df = pd.read_csv("testing.csv")
    clusters = {}  # Dictionary to store rows for each cluster_id
    if request.method == "POST":
        selected_value = request.POST.get('one')
        s = int(selected_value)
        clusters[s] = df[df['cluster_id'] == s].values.tolist()
    else:
        for cluster_id in range(0,11):
            clusters[cluster_id] = df[df['cluster_id'] == cluster_id].values.tolist()

    context = {'clusters': clusters}
    return render(request, 'categories.html', context)


def product_names(request):
    df = pd.read_csv("content_based_cosine.csv")
    a = df.values.tolist()
    return render(request,'productnames.html',{'rows': a})


def recommendation_page(request):
    if request.method == "POST":
        value = request.POST.get('abc')
        value = value.strip()  #To remove extra space 
        product_names = []
        images = []
        df = pd.read_csv("content_based_cosine.csv")
        cv = CountVectorizer(max_features=5000,stop_words='english')
        vectors = cv.fit_transform(df['tags']).toarray()
        similarity = cosine_similarity(vectors)
        index = df[df['title'] == value].index[0]
        distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
        for i in distances[0:8]:
            product_names.append(df.iloc[i[0]].title)
            images.append(df.iloc[i[0]].images)
    
    else:
        product_names = []
        images = []
    combined_data = zip(product_names, images)
    return render(request,'recommendationpage.html',{'combined_data': combined_data,})


def recommendation(request):
    if request.method == "POST":
        value = request.POST.get('abc')
        value = value.strip()  #To remove extra space 
        product_names = []
        images = []
        df = pd.read_csv("second_cosine.csv")
        cv = CountVectorizer(max_features=5000,stop_words='english')
        vectors = cv.fit_transform(df['tags']).toarray()
        similarity = cosine_similarity(vectors) 
        index = df[df['product_name'] == value].index[0]
        distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
        for i in distances[0:8]:
            product_names.append(df.iloc[i[0]].product_name)
            images.append(df.iloc[i[0]].img_link)
    
    else:
        product_names = []
        images = []
    combined_data = zip(product_names, images)
    return render(request,'recommendation.html',{'combined_data': combined_data,})

def collaborativebased(request):
    df = pd.read_csv('collaborative.csv')
    amazon_ratings = df
    ratings_utility_matrix = amazon_ratings.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)
    X = ratings_utility_matrix.T
    SVD = TruncatedSVD(n_components=10) #The n_components parameter specifies the number of components or dimensions to which the original data will be reduced.
    decomposed_matrix = SVD.fit_transform(X) #The fit_transform() method fits the TruncatedSVD model to the input data and then performs the dimensionality reduction by transforming the data into the lower-dimensional space.
    correlation_matrix = np.corrcoef(decomposed_matrix)
    product_IDs = list(X.index)
    if request.method == "POST":
        value = request.POST.get('abc')
        value = value.strip()
        i = str(value)
        index_of_product_ID = product_IDs.index(i)
        correlation_product_ID = correlation_matrix[index_of_product_ID]
        Recommend = list(X.index[correlation_product_ID > 0.90])
        a = Recommend[0:8]
    else:
        i = ""
        a = ""
    context = {'product_list':product_IDs, 'recommendation':a}
    return render(request,'collaborativebased.html',context)
