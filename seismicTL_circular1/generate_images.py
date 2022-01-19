import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import plotly.graph_objects as go
import pathlib
import wiggle
from io import BytesIO
import base64
import h5py
import numpy as np
# import dash_colorscales as dcs
import json
import os
import time

from datetime import datetime

pdir="."
expt="seismicTL_circular1"



if not os.path.exists(pdir):
    os.mkdir(pdir)

now = datetime.now()
dir = os.path.join(pdir,"images_"+expt)
# dir = os.path.join(pdir,now.strftime("%d-%m-%Y_%H"))

if not os.path.exists(dir): os.mkdir(dir)


matplotlib.use('agg')

PATH = pathlib.Path(__file__).parent
# DATA_PATH = PATH.joinpath("data").resolve()

pdir="seismicTL_circular1"
DATA_PATH= PATH.joinpath(expt).resolve()
# mh5=h5py.File(DATA_PATH.joinpath("medium.h5"), 'r')
# m={}
# epsilon=mh5.keys()
# for key in mh5.keys():
    # m[key]=np.array(mh5[key])
X=h5py.File(PATH.joinpath("Xotestdash.h5"), 'r')
Xot=np.array(X["data"][:,:,:,:,:])
nsamp, ntau, nr, nt, nfield=Xot.shape
# with open(DATA_PATH.joinpath('Xotestdash.json')) as f:
    # labels = json.load(f)
# labelindices={eps: [] for eps in epsilon}
# for isamp in range(nsamp):
    # if(expt=="mnist"):
        # labelindices[labels[str(isamp+1)]["1"]].append(isamp)
    # else:
        # labelindices[labels[str(isamp+1)]["1"]["Medium"]].append(isamp)

encoderr=tf.keras.models.load_model(PATH.joinpath("_TFencoderr"), compile=False)
encodertau=tf.keras.models.load_model(PATH.joinpath("_TFencodertau"), compile=False)
decoder=tf.keras.models.load_model(PATH.joinpath("_TFdecoder"), compile=False)



def save_wiggle(x,s,c='k', txt='none'):
    fig=plt.figure(dpi=150, figsize=(6,3.5))
    sf=np.max(np.std(x,axis=1))
    wiggle.wiggle(x, color=c, sf= sf/0.5/10)#
    # plt.imshow(x)
    if(txt != 'none'):
        tc='lightsteelblue'
        if(txt>0.1):
            tc='lightcoral'
        txt="mse="+str(txt)
        plt.text(0.5, 8, txt, size=20, color='black', bbox=dict(facecolor=tc, alpha=1,))
    plt.axis('off')
    plt.savefig(os.path.join(dir,s), bbox_inches='tight')
    plt.close()



    return None

    
def reconstruct(i,j,itau):
   from sklearn.metrics import mean_squared_error
   x=Xot[[i,j]]
   zr=encoderr.predict(x)
   ztau=encodertau.predict(x)
   z=tf.concat([zr, ztau],1)
   xhat=decoder.predict(z)
   # xhat=x

   jtau=itau

#    mse=tf.keras.losses.MeanSquaredError(tf.keras.losses.MeanSquaredError())
   xi=x[0,itau,:,:,0]
   xj=x[-1,jtau,:,:,0]
   save_wiggle(np.transpose(xi),str(itau)+"_itrue")
   save_wiggle(np.transpose(xj),str(itau)+"_jtrue")
   xihat=xhat[0,itau,:,:,0]
   xjhat=xhat[-1,jtau,:,:,0]
   from math import dist
#    save_wiggle(np.transpose(xihat),str(itau)+"_ii", 'k', "mse="+str((np.square(xi - xihat)).mean(axis=None)))
   save_wiggle(np.transpose(xihat),str(itau)+"_ii", 'k', round(mean_squared_error(xi,xihat),3))
   save_wiggle(np.transpose(xjhat),str(itau)+"_jj", 'k', round(mean_squared_error(xj,xjhat),3))
   dxihat=xihat-xi
   dxjhat=xjhat-xj
   save_wiggle(np.transpose(dxihat),str(itau)+"_dii")
   save_wiggle(np.transpose(dxjhat),str(itau)+"_djj")
   

   zr=np.flip(zr, axis=[0]);
   z=tf.concat([zr, ztau],1)
   xhat=decoder.predict(z)

   xjihat=xhat[0,itau,:,:,0]
   xijhat=xhat[-1,jtau,:,:,0]
   save_wiggle(np.transpose(xjihat),str(itau)+"_ji",'k',round(mean_squared_error(xjihat,xj),3))
   save_wiggle(np.transpose(xijhat),str(itau)+"_ij", 'k',round(mean_squared_error(xijhat,xi),3)) 


   return None
 


for itau in range(7):
   reconstruct(0,1,itau)
