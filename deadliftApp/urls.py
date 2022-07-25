
from django.urls import path
from deadliftApp import views

urlpatterns = [
    path('',views.index, name='index'),
    path('start/',views.start, name='start'),


]
