from django.shortcuts import redirect, render

# Create your views here.
from tkinter import *
import cv2
import mediapipe as mp
import numpy as np
from numpy import savetxt
import time
import matplotlib.pyplot as plt
from django.shortcuts import redirect
from mpl_toolkits.mplot3d import Axes3D



def index(request):
    return render(request,'index.html')



def start(request):
    
    return redirect('/')


