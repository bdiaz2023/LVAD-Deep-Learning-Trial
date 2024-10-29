import numpy as np
import tqdm
import pandas as pd
import glob
import os
from scipy.interpolate import griddata as gd
from sklearn.neighbors import KNeighborsRegressor

#define fucntion to get rdf
def centerline_rdf(coordinates, centerline_points):

  x =coordinates[:, 0]
  y= coordinates[:, 1]
  z= coordinates[:, 2]

  non_z = np.sum(abs(coordinates), axis = 1) > 0
  rdf = np.zeros(len(coordinates))

  i=0
  for point in coordinates[non_z]:
 

    #calculate minimum distance to centerline point
    
    dist = np.min(np.sum((point - centerline_points)**2, axis = 1)**0.5)
    rdf[i] = dist
    i+=1

  return rdf

#define function for putting into regular grid
def makeGrid(xdim, ydim, zdim, coordinates, vx, vy, vz, centerline_coords):

  x = coordinates[:,0]
  y = coordinates[:,1]
  z = coordinates[:,2]

  #find space between evenly separated points in our regular grid
  x_step = (np.max(x) - np.min(x))/xdim
  z_step = (np.max(z) - np.min(z))/zdim
  y_step = (np.max(y) - np.min(y))/ydim
  rdfgrid = np.zeros([xdim, ydim, zdim])

  #find centerpoint of the regular grid
  centerpoint = np.array([(np.max(x) + np.min(x))/2 , (np.max(y) + np.min(y))/2, (np.max(z) + np.min(z))/2])


  xi,yi,zi=np.ogrid[np.min(x):np.max(x):32j, np.min(y):np.max(y):32j, 
                    np.min(z):np.max(z):32j]

  X1=xi.reshape(xi.shape[0],)
  Y1=yi.reshape(yi.shape[1],)
  Z1=zi.reshape(zi.shape[2],)
  ar_len=len(X1)*len(Y1)*len(Z1)
  X=np.arange(ar_len,dtype=float)
  Y=np.arange(ar_len,dtype=float)
  Z=np.arange(ar_len,dtype=float)
  l=0

  #find coordinates of regular grid points
  for i in range(0,len(X1)):
      for j in range(0,len(Y1)):
          for k in range(0,len(Z1)):
              X[l]=X1[i]
              Y[l]=Y1[j]
              Z[l]=Z1[k]
              l=l+1



  #interpolate data from irregular coordinates to grid coordinates
  Vx = gd((x,y,z), vx, (X,Y,Z), method='linear', fill_value=np.nan,)
  Vx[np.isnan(Vx)]=0
  Vx=Vx.reshape([1, xdim, ydim, zdim])

  Vy = gd((x,y,z), vy, (X,Y,Z), method='linear', fill_value=np.nan)
  Vy[np.isnan(Vy)]=0
  Vy=Vy.reshape([1, xdim, ydim, zdim])

  Vz = gd((x,y,z), vz, (X,Y,Z), method='linear', fill_value=np.nan)
  Vz[np.isnan(Vz)]=0
  Vz=Vz.reshape([1, xdim, ydim, zdim])
    
  #CALL RDF FUNCTION
  rdf = centerline_rdf(coordinates, centerline_coords)
    
#   #fit interpotalo
#   x_step = (np.max(x) - np.min(x))/xdim
#   y_step = (np.max(y) - np.min(y))/ydim
#   z_step = (np.max(z) - np.min(z))/zdim
#   knnr = KNeighborsRegressor(n_neighbors = 3, weights='distance',leaf_size=10)
#   knnr.fit(coordinates, rdf)
    
#   #reg_grid_coords 
#   rdf_grid = np.zeros((xdim, ydim, zdim))
    
#   xrange = np.arange(0, xdim, 1)
#   yrange = np.arange(0, ydim, 1)
#   zrange = np.arange(0, zdim, 1)
#   for i in xrange:
#         for j in yrange:
#             for k in zrange:
#                 regular_grid_coordinate = np.array([i*x_step, j*y_step, k*z_step])
#                 rdf_grid[i,j,k] = knnr.predict(regular_grid_coordinate.reshape(1, -1))
                
                
  rdf_grid = gd((x,y,z), rdf, (X,Y,Z), method='nearest', fill_value=np.nan)
  rdf_grid=rdf_grid.reshape([1, xdim, ydim, zdim])
  rdf_grid=rdf_grid.reshape([1, xdim, ydim, zdim])
  mask = np.where(Vz == 0)
  rdf_grid[mask] = 0
  rdf_grid[np.isnan(rdf_grid)]=0
  


#   #use velocity mask to do rdf calculations
#   for i in range(0,len(X1)):
#     for j in range(0,len(Y1)):
#         for k in range(0,len(Z1)):

#             # do rdf calculations and store
#             #only do if we are within binary mask
#             if np.sum(np.abs(Vx [0, i, j, k])) > 0:
#             #find point by multiplying grid integer coordinate by step
#               point = np.array([i * x_step, j * y_step,  k * z_step])
#               dist = np.abs(np.sum((point - centerpoint)**2)**0.5)
#               rdfgrid[i, j, k] = dist

#   rdfgrid=rdfgrid.reshape([1, xdim, ydim, zdim])

  return np.stack((Vx, Vy, Vz, rdf_grid), axis = -1)

#define fucntion to get rdf
def centerline_rdf(coordinates, centerline_points):

  x =coordinates[:, 0]
  y= coordinates[:, 1]
  z= coordinates[:, 2]

  non_z = np.sum(abs(coordinates), axis = 1) > 0
  rdf = np.zeros(len(coordinates))

  i=0
  for point in coordinates[non_z]:
 

    #calculate minimum distance to centerline point
    
    dist = np.min(np.sum((point - centerline_points)**2, axis = 1)**0.5)
    rdf[i] = dist
    i+=1

  return rdf

def centerpoint_rdf(coordinates):
      x =coordinates[:, 0]
      y= coordinates[:, 1]
      z= coordinates[:, 2]
        
      centerpoint = [np.median(x), np.median(y), np.median(z)] 

      non_z = np.sum(abs(coordinates), axis = 1) > 0
      rdf = np.zeros(len(coordinates))

      i=0
      for point in coordinates[non_z]:


        
        #calculate minimum distance to centerline point

        dist = np.sum((point - centerpoint)**2)**0.5
        rdf[i] = dist
        i+=1

      return rdf
    
    

#define function for putting into regular grid
def makeGrid(coordinates, vx, vy, vz, xdim = 32, ydim = 32, zdim = 32, centerline_coords = None, centerline = False):

  x = coordinates[:,0]
  y = coordinates[:,1]
  z = coordinates[:,2]

  #find space between evenly separated points in our regular grid
  x_step = (np.max(x) - np.min(x))/xdim
  z_step = (np.max(z) - np.min(z))/zdim
  y_step = (np.max(y) - np.min(y))/ydim
  rdfgrid = np.zeros([xdim, ydim, zdim])

  #find centerpoint of the regular grid
  centerpoint = np.array([(np.max(x) + np.min(x))/2 , (np.max(y) + np.min(y))/2, (np.max(z) + np.min(z))/2])


  xi,yi,zi=np.ogrid[np.min(x):np.max(x):32j, np.min(y):np.max(y):32j, 
                    np.min(z):np.max(z):32j]

  X1=xi.reshape(xi.shape[0],)
  Y1=yi.reshape(yi.shape[1],)
  Z1=zi.reshape(zi.shape[2],)
  ar_len=len(X1)*len(Y1)*len(Z1)
  X=np.arange(ar_len,dtype=float)
  Y=np.arange(ar_len,dtype=float)
  Z=np.arange(ar_len,dtype=float)
  l=0

  #find coordinates of regular grid points
  for i in range(0,len(X1)):
      for j in range(0,len(Y1)):
          for k in range(0,len(Z1)):
              X[l]=X1[i]
              Y[l]=Y1[j]
              Z[l]=Z1[k]
              l=l+1



  #interpolate data from irregular coordinates to grid coordinates
  Vx = gd((x,y,z), vx, (X,Y,Z), method='linear', fill_value=np.nan,)
  Vx[np.isnan(Vx)]=0
  Vx=Vx.reshape([1, xdim, ydim, zdim])

  Vy = gd((x,y,z), vy, (X,Y,Z), method='linear', fill_value=np.nan)
  Vy[np.isnan(Vy)]=0
  Vy=Vy.reshape([1, xdim, ydim, zdim])

  Vz = gd((x,y,z), vz, (X,Y,Z), method='linear', fill_value=np.nan)
  Vz[np.isnan(Vz)]=0
  Vz=Vz.reshape([1, xdim, ydim, zdim])
    
  #CALL RDF FUNCTION
  if centerline == True:
      rdf = centerline_rdf(coordinates, centerline_coords)
  else:
      rdf = centerpoint_rdf(coordinates)
    
    
#  
                
                
  rdf_grid = gd((x,y,z), rdf, (X,Y,Z), method='nearest', fill_value=np.nan)
  rdf_grid=rdf_grid.reshape([1, xdim, ydim, zdim])
  rdf_grid=rdf_grid.reshape([1, xdim, ydim, zdim])
  mask = np.where(Vz == 0)
  rdf_grid[mask] = 0
  rdf_grid[np.isnan(rdf_grid)]=0
  


#   #use velocity mask to do rdf calculations
#   for i in range(0,len(X1)):
#     for j in range(0,len(Y1)):
#         for k in range(0,len(Z1)):

#             # do rdf calculations and store
#             #only do if we are within binary mask
#             if np.sum(np.abs(Vx [0, i, j, k])) > 0:
#             #find point by multiplying grid integer coordinate by step
#               point = np.array([i * x_step, j * y_step,  k * z_step])
#               dist = np.abs(np.sum((point - centerpoint)**2)**0.5)
#               rdfgrid[i, j, k] = dist

#   rdfgrid=rdfgrid.reshape([1, xdim, ydim, zdim])

  return np.stack((Vx, Vy, Vz, rdf_grid), axis = -1)