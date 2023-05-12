from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("select_program", views.home, name = "home2"),
    path("home", views.home, name="home"),
    path("usdcad", views.usdcad, name="usdcad"),
    path("usdjpy", views.usdjpy, name="usdjpy"),
    path("usdeuro", views.usdeuro, name="usdeuro"),
    path("audjpy", views.audjpy, name="audjpy"),
    path("dashboard", views.dashboard, name="dashboard")
]