
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib   #to save model as binary file, also can use pickle
from matplotlib.colors import ListedColormap
import os
plt.style.use("fivethirtyeight")
from sklearn import preprocessing

def prepare_data(df):
  X = df.drop("y", axis=1)
  y = df["y"]
  return X, y


def save_model(model, filename):
  model_dir = "My_Models"
  os.makedirs(model_dir, exist_ok=True)     #Only create if model_dir not exist
  filepath = os.path.join(model_dir, filename)  #model path/filename
  joblib.dump(model, filepath)


def save_plot(df, file_name, model):

  def _create_base_plot(df):       #internal function
    df.plot(kind = "scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=2)  #Plotting Horizontal Axis - X Axis
    plt.axvline(x=0, color="black", linestyle="--", linewidth=2)  #Plotting Horizontal Axis - y Axis
    figure = plt.gcf()    #get current figure
    figure.set_size_inches(10,8)    #Set size in inches

  def _plot_decision_regions(X, y, classfier, resolution=0.02):
    colors = ("red", "blue", "cyan", "green", "gray")
    cmap = ListedColormap(colors[ : len(np.unique(y))])

    X = X.values   #we are passing X as dataframe but here we want value only as array
    x1 =  X[:,0]      #First column
    x2 =  X[:,1]      #Second column
    x1_min, x1_max = x1.min()-1  ,  x1.max()+1    #First column Min & Max value
    x2_min, x2_max = x2.min()-1  ,  x2.max()+1    #First column Min & Max value 

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))  
    
    Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)       #If Z is 1 - Blue,   0 - Red
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.plot()

  X, y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  plotPath = os.path.join(plot_dir, file_name) # model/filename
  plt.savefig(plotPath)
