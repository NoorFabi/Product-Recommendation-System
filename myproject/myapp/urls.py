from django.urls import path 
from . import views

urlpatterns = [
    path("",views.homepage,name="homepage"),
    path("trending",views.trending,name="trending"),
    path('group',views.groupsitem, name="groupsitem"),
    path('categories',views.categories, name="categories"),
    path('productnames', views.product_names, name="product_names"),
    path('recommendationpage', views.recommendation_page, name="recommendation_page"),
    path('recommendation', views.recommendation, name="recommendation"),
    path('collaborative', views.collaborativebased, name="collaborativebased")
]