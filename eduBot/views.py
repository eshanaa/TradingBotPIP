from django.shortcuts import render
from django.http import HttpResponse
from . import arima_usdjpy, arima_audjpy
import subprocess
#from eduBot.ArimaModel import arima_usdjpy, arima_usdcad, arima_audjpy, arima_usdeuro

def home(request):
    return render(request, 'home.html', context = {"home": "page"})

def usdcad(request):
    #subprocess.run(["python", "eduBot/arima_usdcad.py"])
    return render(request, 'usdcad.html', context = {"usd": "cad"})

def usdjpy(request):
    subprocess.run(["python", "eduBot/arima_usdjpy.py"])
    return render(request, 'usdjpy.html', context = {"usd": "jpy"})

def usdeuro(request):
    #subprocess.run(["python", "eduBot/arima_usdeuro.py"])
    return render(request, 'usdeuro.html', context = {"usd": "euro"})

def audjpy(request):
    subprocess.run(["python", "eduBot/arima_audjpy.py"])
    return render(request, 'audjpy.html', context = {"aud": "jpy"})

def dashboard(request):
    return render(request, 'dashboard.html', context = {"dash": "board"})