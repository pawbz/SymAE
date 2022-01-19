
#%% Load packages
import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers
tfkltd= tf.keras.layers.TimeDistributed

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

print(tf.__version__)

class Downsampler(tf.keras.Model):
  def __init__(self, kernel_sizes, filters, fact=4):
    super(Downsampler, self).__init__(name='')
    k1,k2=kernel_sizes

    self.conv1 = tfkltd(tfkl.Conv2D(filters, (k1,k2),padding='same',activation='elu',name='conv1'))
    self.conv2 = tfkltd(tfkl.Conv2D(filters, (k1,k2),padding='same',activation='elu',name='conv2'))
    self.mp = tfkltd(tfkl.MaxPool2D(pool_size=(1,fact)))
    self.conv3 = tfkltd(tfkl.Conv2D(filters//2, (k1,k2),padding='same',activation='elu',name='conv3'))
    self.conv4 = tfkltd(tfkl.Conv2D(1, (k1,k2),padding='same'))
    self.bn = tfkltd(tf.keras.layers.BatchNormalization())

  def call(self, input_tensor, training=False):
    # n, ntau, nr, nt, nc = input_tensor.get_shape()
    # x=tfkl.Reshape(target_shape=((ntau*nr,nt,nc)))(input_tensor)
    x=self.conv1(input_tensor)
    x=self.conv2(x)
    x=self.mp(x)
    x=self.conv3(x)
    x=self.conv4(x)
    out=self.bn(x, training=training)
    # nt_in=x.get_shape()[2]
    # out=tfkl.Reshape(target_shape=((ntau,nr,nt_in,nc)))(x)
    return out

class DownsamplerDense(tf.keras.Model):
  def __init__(self, nt_out, nt_in):
    super(DownsamplerDense, self).__init__(name='')
    self.nt_in=nt_in
    self.nt_out=nt_out
    self.d1=tfkltd(tfkl.Dense(nt_out,activation='elu'))
    self.d2=tfkltd(tfkl.Dense(nt_out))
    self.bn=tfkltd(tf.keras.layers.BatchNormalization())

  def call(self, input_tensor, training=False):
    n, ntau, nr, nt, nc = input_tensor.get_shape()
    x=tfkl.Reshape(target_shape=((ntau*nr,nt)))(input_tensor)
    x=self.d1(x)
    x=self.d2(x)
    x=tfkl.Reshape(target_shape=((ntau,nr,self.nt_out,1)))(x)
    out=self.bn(x, training=training)
    return out




class NuisanceEncoder(tf.keras.Model):
  def __init__(self, kernel_sizes, filters, rdown=[2,2,1,1], tdown=[1,1,2,2], fstep=[2,4,8], latent_dim=512):
    super(NuisanceEncoder, self).__init__(name='')
    k1,k2=kernel_sizes

    self.c1=tfkltd(tfkl.Conv2D(filters,(k1,k2),padding='same',activation='elu'))
    self.c2=tfkltd(tfkl.Conv2D(filters,(k1,k2),padding='same',activation='elu'))
    self.mp1=tfkltd(tfkl.MaxPool2D(pool_size=(rdown[0],tdown[0])))
    self.c3=tfkltd(tfkl.Conv2D(filters//fstep[0],(k1,k2),padding='same',activation='elu'))
    self.c4=tfkltd(tfkl.Conv2D(filters//fstep[0],(k1,k2),padding='same',activation='elu'))
    self.mp2=tfkltd(tfkl.MaxPool2D(pool_size=(rdown[1],tdown[1])))
    self.c5=tfkltd(tfkl.Conv2D(filters//fstep[1],(k1,k2),padding='same',activation='elu'))
    self.c6=tfkltd(tfkl.Conv2D(filters//fstep[1],(k1,k2),padding='same',activation='elu'))
    self.mp3=tfkltd(tfkl.MaxPool2D(pool_size=(rdown[2],tdown[2])))
    self.c7=tfkltd(tfkl.Conv2D(filters//fstep[2],(k1,k2),padding='same',activation='elu'))
    self.c8=tfkltd(tfkl.Conv2D(filters//fstep[2],(k1,k2),padding='same'))
    self.bn=tfkltd(tfkl.BatchNormalization())
    self.a=tfkltd(tfkl.Activation('elu'))
    self.mp4=tfkltd(tfkl.MaxPool2D(pool_size=(rdown[3],tdown[3])))
    self.f=tfkltd(tfkl.Flatten())
    self.d=tfkltd(tfkl.Dense(latent_dim))

  def call(self, input_tensor, training=False):
    # n, ntau, nr, nt, nc = input_tensor.get_shape()
    x=self.c1(input_tensor)
    x=self.c2(x)
    x=self.mp1(x)
    x=self.c3(x)
    x=self.c4(x)
    x=self.mp2(x)
    x=self.c5(x)
    x=self.c6(x)
    x=self.mp3(x)
    x=self.c7(x)
    x=self.c8(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    x=self.mp4(x)
    x=self.f(x)
    out=self.d(x)
    return out
  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))


class NuisanceEncoder1D(tf.keras.Model):
  def __init__(self, kernel_size, filter,  fstep=[2,4,8], tdown=[1,1,2,2], latent_dim=512):
    super(NuisanceEncoder1D, self).__init__(name='')
    k1=kernel_size

    self.c1=tfkltd(tfkl.Conv1D(filter,((k1)),padding='same',activation='elu'))
    self.c2=tfkltd(tfkl.Conv1D(filter,((k1)),padding='same',activation='elu'))
    self.mp1=tfkltd(tfkl.MaxPool1D(pool_size=(tdown[0])))
    self.c3=tfkltd(tfkl.Conv1D(filter//fstep[0],((k1)),padding='same',activation='elu'))
    self.c4=tfkltd(tfkl.Conv1D(filter//fstep[0],((k1)),padding='same',activation='elu'))
    self.mp2=tfkltd(tfkl.MaxPool1D(pool_size=(tdown[1])))
    self.c5=tfkltd(tfkl.Conv1D(filter//fstep[1],((k1)),padding='same',activation='elu'))
    self.c6=tfkltd(tfkl.Conv1D(filter//fstep[1],((k1)),padding='same',activation='elu'))
    self.mp3=tfkltd(tfkl.MaxPool1D(pool_size=(tdown[2])))
    self.c7=tfkltd(tfkl.Conv1D(filter//fstep[2],((k1)),padding='same',activation='elu'))
    self.c8=tfkltd(tfkl.Conv1D(filter//fstep[2],((k1)),padding='same'))
    self.bn=tfkltd(tfkl.BatchNormalization())
    self.a=tfkltd(tfkl.Activation('elu'))
    self.mp4=tfkltd(tfkl.MaxPool1D(pool_size=(tdown[3])))
    self.f=tfkltd(tfkl.Flatten())
    self.d=tfkltd(tfkl.Dense(latent_dim))

  def call(self, input_tensor, training=False):
    # n, ntau, nr, nt, nc = input_tensor.get_shape()
    x=self.c1(input_tensor)
    x=self.c2(x)
    x=self.mp1(x)
    x=self.c3(x)
    x=self.c4(x)
    x=self.mp2(x)
    x=self.c5(x)
    x=self.c6(x)
    x=self.mp3(x)
    x=self.c7(x)
    x=self.c8(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    x=self.mp4(x)
    x=self.f(x)
    out=self.d(x)
    return out
  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))

class NuisanceEncoderDense1D(tf.keras.Model):
  def __init__(self, nt_out, nt_in,steps=[2,4,8]):
    super(NuisanceEncoderDense1D, self).__init__(name='')
    self.steps=steps
    self.nt_out=nt_out
    self.nt_in=nt_in
    self.d1=tfkltd(tfkl.Dense(nt_in//steps[0],activation='elu'))
    self.d2=tfkltd(tfkl.Dense(nt_in//steps[1],activation='elu'))
    # self.d3=tfkltd(tfkl.Dense(nt_in,activation='elu'))
    # self.d4=tfkltd(tfkl.Dense(nt_in,activation='elu'))
    self.d5=tfkltd(tfkl.Dense(nt_in//steps[2],))
    self.a=tfkltd(tfkl.Activation('elu'))
    self.d6=tfkltd(tfkl.Dense(nt_out))
    self.bn=tfkltd(tf.keras.layers.BatchNormalization())

  def call(self, input_tensor, training=False):
    # n, ntau, nr, nt, nc = input_tensor.get_shape()
    x=self.d1(input_tensor)
    x=self.d2(x)
    # x=self.d3(x)
    # x=self.d4(x)
    x=self.d5(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    out=self.d6(x)
    return out
  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))

class SymmetricEncoder(tf.keras.Model):
  def __init__(self, kernel_sizes, filters, rdown=[2,2,2,1], tdown=[2,4,4,4],latent_dim=8, fstep=[2,4,8]):
    super(SymmetricEncoder, self).__init__(name='')
    k1,k2=kernel_sizes

    self.c11=tfkltd(tfkl.Conv2D(filters,(k1,k2),padding='same',activation='elu'))
    self.c12=tfkltd(tfkl.Conv2D(filters//fstep[0],(k1,k2),padding='same',activation='elu'))
    self.mp11=tfkltd(tfkl.MaxPool2D(pool_size=(rdown[0],tdown[0])))
    self.c13=tfkltd(tfkl.Conv2D(filters//fstep[1],(k1,k2),padding='same',activation='elu'))
    self.c14=tfkltd(tfkl.Conv2D(filters//fstep[2],(k1,k2),padding='same',activation='elu'))
    self.mp12=tfkltd(tfkl.MaxPool2D(pool_size=(rdown[1],tdown[1])))


    self.c21=tfkl.Conv2D(filters,(k1,k2),padding='same',activation='elu')
    self.c22=tfkl.Conv2D(filters//fstep[0],(k1,k2),padding='same',activation='elu')
    self.mp21=tfkl.MaxPool2D(pool_size=(rdown[2],tdown[2]))
    self.c23=tfkl.Conv2D(filters//fstep[1],(k1,k2),padding='same',activation='elu')
    self.c24=tfkl.Conv2D(filters//fstep[2],(k1,k2),padding='same')
    self.bn=tfkl.BatchNormalization()
    self.a=tfkl.Activation('elu')
    self.mp22=tfkl.MaxPool2D(pool_size=(rdown[3],tdown[3]))
    self.f=tfkl.Flatten()
    self.d=tfkl.Dense(latent_dim)

  def call(self, input_tensor, training=False):
    # n, ntau, nr, nt, nc = input_tensor.get_shape()
    x=self.c11(input_tensor)
    x=self.c12(x)
    x=self.mp11(x)
    x=self.c13(x)
    x=self.c14(x)
    x=self.mp12(x)
    x=tf.math.reduce_mean(x,axis=1)
    x=self.c21(x)
    x=self.c22(x)
    x=self.mp21(x)
    x=self.c23(x)
    x=self.c24(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    x=self.mp22(x)
    x=self.f(x)
    out=self.d(x)
    return out
  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))

class SymmetricEncoder1D(tf.keras.Model):
  def __init__(self, kernel_size, filters, fstep=[2,4,8], tdown=[2,4,4,4],latent_dim=8):
    super(SymmetricEncoder1D, self).__init__(name='')
    k1=kernel_size

    self.c11=tfkltd(tfkl.Conv1D(filters,(k1),padding='same',activation='elu'))
    self.c12=tfkltd(tfkl.Conv1D(filters//fstep[0],(k1),padding='same',activation='elu'))
    self.mp11=tfkltd(tfkl.MaxPool1D(pool_size=(tdown[0])))
    self.c13=tfkltd(tfkl.Conv1D(filters//fstep[1],(k1),padding='same',activation='elu'))
    self.c14=tfkltd(tfkl.Conv1D(filters//fstep[2],(k1),padding='same',activation='elu'))
    self.mp12=tfkltd(tfkl.MaxPool1D(pool_size=(tdown[1])))


    self.c21=tfkl.Conv1D(filters,(k1),padding='same',activation='elu')
    self.c22=tfkl.Conv1D(filters//fstep[0],(k1),padding='same',activation='elu')
    self.mp21=tfkl.MaxPool1D(pool_size=(tdown[2]))
    self.c23=tfkl.Conv1D(filters//fstep[1],(k1),padding='same',activation='elu')
    self.c24=tfkl.Conv1D(filters//fstep[2],(k1),padding='same')
    self.bn=tfkl.BatchNormalization()
    self.a=tfkl.Activation('elu')
    self.mp22=tfkl.MaxPool1D(pool_size=(tdown[3]))
    self.f=tfkl.Flatten()
    self.d=tfkl.Dense(latent_dim)

  def call(self, input_tensor, training=False):
    # n, ntau, nr, nt, nc = input_tensor.get_shape()
    x=self.c11(input_tensor)
    x=self.c12(x)
    x=self.mp11(x)
    x=self.c13(x)
    x=self.c14(x)
    x=self.mp12(x)
    x=tf.math.reduce_mean(x,axis=1)
    x=self.c21(x)
    x=self.c22(x)
    x=self.mp21(x)
    x=self.c23(x)
    x=self.c24(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    x=self.mp22(x)
    x=self.f(x)
    out=self.d(x)
    return out

  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))


class SymmetricEncoderDense1D(tf.keras.Model):
  def __init__(self, nt_out, nt_in,steps=[2,4,8]):
    super(SymmetricEncoderDense1D, self).__init__(name='')

    self.nt_out=nt_out
    self.steps=steps
    self.nt_in=nt_in
    self.d1=tfkltd(tfkl.Dense(nt_in//steps[0],activation='elu'))
    self.d2=tfkltd(tfkl.Dense(nt_in//steps[1],activation='elu'))
    # self.d3=tfkltd(tfkl.Dense(nt_in,activation='elu'))
    self.d4=tfkltd(tfkl.Dense(nt_in//steps[1],activation='elu'))
    # self.d5=tfkl.Dense(nt_in,activation='elu')
    # self.d6=tfkl.Dense(nt_in,activation='elu')
    self.d7=tfkl.Dense(nt_in//steps[2],)
    self.d8=tfkl.Dense(nt_out,)
    self.bn=tfkl.BatchNormalization()
    self.a=tfkl.Activation('elu')

  def call(self, input_tensor, training=False):
    # n, ntau, nr, nt, nc = input_tensor.get_shape()
    x=self.d1(input_tensor)
    x=self.d2(x)
    # x=self.d3(x)
    x=self.d4(x)
    x=tf.math.reduce_mean(x,axis=1)
    # x=self.d5(x)
    # x=self.d6(x)
    x=self.d7(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    out=self.d8(x)
    return out

  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))

class DistributeZsym(tf.keras.Model):
  def __init__(self, ntau, nz0, nzi):
    super(DistributeZsym, self).__init__(name='')

    self.nz0=nz0
    self.nzi=nzi
    self.ntau=ntau
    self.ri=tfkl.Reshape(target_shape=(ntau,nzi))
    self.repeat=tfkl.RepeatVector(ntau)
#     Xhatr=tfkl.Reshape(target_shape=(ntau,latent_dimr))(Xhatr)

  def call(self, z, training=False):

    z0,zi=tf.split(z,[self.nz0, self.ntau*self.nzi],axis=1)
    zi=self.ri(zi)
    z0=self.repeat(z0)
    out=tfkl.concatenate([z0, zi],axis=2)
    return out
  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))


class LatentCat(tf.keras.Model):
  def __init__(self, alpha=1.0):
    super(LatentCat, self).__init__(name='')

    self.drop=tfkl.GaussianDropout(alpha)
    # self.drop=tfkl.Dropout(alpha)

  def call(self, zsym, znuisance,training=False):
    znuisance=self.drop(znuisance,training=training)
    z=tfkl.concatenate([zsym, znuisance])
    return z

class DroppedLatentCat(tf.keras.Model):
  def __init__(self,len):
    super(DroppedLatentCat, self).__init__(name='')
    self.len=len
    # self.pad=tfkl.ZeroPadding1D(padding=(0,len))

  def call(self, zsym): 
    z=tf.pad(zsym,paddings=[[0,0],[0,self.len]])
    return z







class Mixer(tf.keras.Model):
  def __init__(self, kernel_sizes, filters, upfacts, nt, nr, fstep=[8,4,2]):
    super(Mixer, self).__init__(name='')
    k1,k2=kernel_sizes
    rup,tup=upfacts

    self.d1=tfkltd(tfkl.Dense(units=((nr//rup)*(nt//tup)*1),activation='elu'))
    self.r2=tfkltd(tfkl.Reshape(target_shape=((nr//rup),(nt//tup),1)))
    self.c1=tfkltd(tfkl.Conv2D(filters//fstep[0],(k1,k2),padding='same',activation='elu'))
    self.us1=tfkltd(tfkl.UpSampling2D(size=(rup,tup)))
    self.c2=tfkltd(tfkl.Conv2D(filters//fstep[1],(k1,k2),padding='same',activation='elu'))
    self.c3=tfkltd(tfkl.Conv2D(filters//fstep[1],(k1,k2),padding='same'))
    self.bn=tfkltd(tfkl.BatchNormalization())
    self.a=tfkltd(tfkl.Activation('elu'))
    self.c4=tfkltd(tfkl.Conv2D(filters//fstep[2],(k1,k2),padding='same',activation='elu'))
    self.c5=tfkltd(tfkl.Conv2D(filters//fstep[2],(k1,k2),padding='same',activation='elu'))
    self.c6=tfkltd(tfkl.Conv2D(1,(k1,k2),padding='same'))

  def call(self, z, training=False):
    x=self.d1(z)
    x=self.r2(x)
    x=self.c1(x)
    x=self.us1(x)
    x=self.c2(x)
    x=self.c3(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    x=self.c4(x)
    x=self.c5(x)
    out=self.c6(x)
    return out
  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))

class Mixer1D(tf.keras.Model):
  def __init__(self, kernel_size, filters, upfact, nt):
    super(Mixer1D, self).__init__(name='')
    (k1)=kernel_size
    tup=upfact

    self.d1=tfkltd(tfkl.Dense(units=((nt//tup)*1),activation='elu'))
    self.r2=tfkltd(tfkl.Reshape(target_shape=((nt//tup),1)))
    self.c1=tfkltd(tfkl.Conv1D(filters,((k1)),padding='same',activation='elu'))
    self.us1=tfkltd(tfkl.UpSampling1D(size=(tup)))
    self.c2=tfkltd(tfkl.Conv1D(filters,((k1)),padding='same',activation='elu'))
    self.c3=tfkltd(tfkl.Conv1D(filters,((k1)),padding='same'))
    self.bn=tfkltd(tfkl.BatchNormalization())
    self.a=tfkltd(tfkl.Activation('elu'))
    self.c4=tfkltd(tfkl.Conv1D(filters,((k1)),padding='same',activation='elu'))
    self.c5=tfkltd(tfkl.Conv1D(filters,((k1)),padding='same',activation='elu'))
    self.c6=tfkltd(tfkl.Conv1D(1,((k1)),padding='same'))

  def call(self, z, training=False):
    x=self.d1(z)
    x=self.r2(x)
    x=self.c1(x)
    x=self.us1(x)
    x=self.c2(x)
    x=self.c3(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    x=self.c4(x)
    x=self.c5(x)
    out=self.c6(x)
    return out
  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))


class MixerDense1D(tf.keras.Model):
  def __init__(self, nt_out, nt_in,steps=[8,4,2]):
    super(MixerDense1D, self).__init__(name='')
    self.nt_out=nt_out
    self.steps=steps
    self.nt_in=nt_in
    self.d1=tfkltd(tfkl.Dense(nt_out//steps[0],activation='elu'))
    self.d2=tfkltd(tfkl.Dense(nt_out//steps[1],activation='elu'))
    # self.d3=tfkltd(tfkl.Dense(nt_in,activation='elu'))
    # self.d4=tfkltd(tfkl.Dense(nt_in,activation='elu'))
    # self.d5=tfkltd(tfkl.Dense(nt_in,activation='elu'))
    # self.d6=tfkltd(tfkl.Dense(nt_in,activation='elu'))
    self.d7=tfkltd(tfkl.Dense(nt_out//steps[2],))
    self.d8=tfkltd(tfkl.Dense(nt_out,))
    self.bn=tfkl.BatchNormalization()
    self.a=tfkl.Activation('elu')

    self.bn=tfkltd(tfkl.BatchNormalization())
    self.a=tfkltd(tfkl.Activation('elu'))

  def call(self, z, training=False):
    x=self.d1(z)
    x=self.d2(x)
    # x=self.d3(x)
    # x=self.d4(x)
    # x=self.d5(x)
    # x=self.d6(x)
    x=self.d7(x)
    x=self.bn(x, training=training)
    x=self.a(x)
    out=self.d8(x)
    return out
  def model(self, x):
    return tfk.Model(inputs=x, outputs=self.call(x))


class Upsampler(tf.keras.Model):
  def __init__(self, kernel_sizes, filters, fact=4):
    super(Upsampler, self).__init__(name='')
    k1,k2=kernel_sizes

    self.conv1 = tfkltd(tfkl.Conv2D(filters, (k1,k2),padding='same',activation='elu'))
    self.us1 = tfkltd(tfkl.UpSampling2D(size=(1,fact)))
    self.conv2 = tfkltd(tfkl.Conv2D(filters, (k1,k2),padding='same',activation='elu'))
    self.conv3 = tfkltd(tfkl.Conv2D(filters//2, (k1,k2),padding='same',activation='elu'))
    self.conv4 = tfkltd(tfkl.Conv2D(1, (k1,k2),padding='same'))

  def call(self, input_tensor, training=False):
    # n, ntau, nr, ntd, nc = input_tensor.get_shape()
    # x=tfkl.Reshape(target_shape=((ntau*nr,ntd,nc)))(input_tensor)
    x=self.conv1(input_tensor)
    x=self.us1(x)
    x=self.conv2(x)
    x=self.conv3(x)
    out=self.conv4(x)
    # nt=x.get_shape()[2]
    # out=tfkl.Reshape(target_shape=((ntau,nr,nt,nc)))(x)
    return out

class UpsamplerDense(tf.keras.Model):
  def __init__(self, nt_out, nt_in):
    super(UpsamplerDense, self).__init__(name='')
    self.nt_in=nt_in
    self.nt_out=nt_out
    self.d1=tfkltd(tfkl.Dense(nt_in,activation='elu'))
    self.d2=tfkltd(tfkl.Dense(nt_out))

  def call(self, input_tensor, training=False):
    n, ntau, nr, ntd, nc = input_tensor.get_shape()
    x=tfkl.Reshape(target_shape=((ntau*nr,ntd)))(input_tensor)
    x=self.d1(x)
    x=self.d2(x)
    out=tfkl.Reshape(target_shape=((ntau,nr,self.nt_out,1)))(x)
    return out

#


# %%
