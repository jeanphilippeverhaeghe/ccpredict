#from django.http import HttpResponse
from django.shortcuts import render

def home(request):
    #return HttpResponse('<h1>Hello Customer Cluster Predictor</h1>')
    return render(request, 'home.html')

