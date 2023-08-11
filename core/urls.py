from django.urls import path
from .views import home, productsList, aboutPage, productDetail, search_herb

urlpatterns = [
    path('', home, name='home'),
    path('products/', productsList, name="products"),
    path('product/<str:pk>/', productDetail, name='detail'),
    path('about/', aboutPage, name="about"),
    path('search-herb/', search_herb , name='search-herb'),
]