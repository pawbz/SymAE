

from numpy.core.fromnumeric import sort


def plot_mnist(X, labels, symencoder, nencoder, decoder):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import tensorflow as tf
    from datetime import datetime
    now = datetime.now()
    dir = now.strftime("%d-%m-%Y_%H")

    if not os.path.exists(dir): os.mkdir(dir)

    nsamp=np.shape(X)[0]

    i = np.random.randint(0, nsamp-1)
    j = np.random.randint(0, nsamp-1)
    print("test_indices"+str(i)+","+str(j))

    Gi=symencoder.predict(X[i:i+1])
    Wi=nencoder.predict(X[i:i+1])
    Gj=symencoder.predict(X[j:j+1])
    Wj=nencoder.predict(X[j:j+1])

    Zii=np.concatenate((Gi,Wi),axis=1)
    Zjj=np.concatenate((Gj,Wj),axis=1)
    Zij=np.concatenate((Gi,Wj),axis=1)
    Zji=np.concatenate((Gj,Wi),axis=1)

    Xhatii=decoder.predict(Zii)
    Xhatij=decoder.predict(Zij)
    Xhatjj=decoder.predict(Zjj)
    Xhatji=decoder.predict(Zji)


    def plot(x, name):
        # num=1
        print(name)
        plt.imsave(name,x[:,:,0],cmap='gray')#,vmin=-1, vmax=1)
        # plot images

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # i=0
        # # fig, ax = plt.plot(figsize=(2,2))
        # # for i in range(num):
        #     # ax = axes[i]
        # ax.imshow(x[0][i], cmap='gray')
        #     # ax.set_title('Label: {}'.format(labels[i]))
        #     # Hide grid lines
        # ax.grid(False)

        #     # Hide axes ticks
        # ax.set_xticks([])
        # ax.set_yticks([])
        # plt.savefig(os.path.join(dir,'books_read.png'),  bbox_inches = 'tight', pad_inches = 0)
        # plt.tight_layout()
        # plt.show()

    losses={}
    for itau in range(20):
        lossi=np.mean(np.square(X[i][itau]-Xhatii[0][itau]))
        lossj=np.mean(np.square(X[j][itau]-Xhatjj[0][itau]))
        losses[str(itau)]={"i":lossi, "j":lossj}


        plot(X[i][itau], os.path.join(dir,str(labels[i]["1"]))+'_'+str(i)+'-true-'+str(itau)+'_itrue.png')
        plot(X[j][itau], os.path.join(dir,str(labels[j]["1"]))+'_'+str(j)+'-true-'+str(itau)+'_jtrue.png')
        plot(Xhatii[0][itau], os.path.join(dir,str(labels[i]["1"]))+'_'+str(i)+'-'+str(i)+'-'+str(itau)+'_ii.png')
        plot(Xhatij[0][itau], os.path.join(dir,str(labels[i]["1"]))+'_'+str(i)+'-'+str(j)+'-'+str(itau)+'_ij.png')
        plot(Xhatjj[0][itau], os.path.join(dir,str(labels[j]["1"]))+'_'+str(j)+'-'+str(j)+'-'+str(itau)+'_jj.png')
        plot(Xhatji[0][itau], os.path.join(dir,str(labels[j]["1"]))+'_'+str(j)+'-'+str(i)+'-'+str(itau)+'_ji.png')
    

    import json
    with open(os.path.join(dir,str(labels[i]["1"])+'-'+str(labels[j]["1"])+'_'+str(i)+'-'+str(j)+'_losses.txt'), 'w') as outfile:
        json.dump(str(losses), outfile)
    # plot(Xhatij)

    # plot(Xhatjj)
    # plot(Xhatjj)
    # plot(Xhatji)




def plot_poly_sweep(X, labels, symencoder, nencoder, decoder):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import tensorflow as tf
    from datetime import datetime
    from scipy import signal
    from sklearn.metrics import mean_squared_error
    now = datetime.now()
    dir = now.strftime("%d-%m-%Y_%H")

    if not os.path.exists(dir): os.mkdir(dir)

    nsamp=np.shape(X)[0]

    i = np.random.randint(0, nsamp-1)
    j = np.random.randint(0, nsamp-1)
    print("test_indices"+str(i)+","+str(j))

    Gi=symencoder.predict(X[i:i+1])
    Wi=nencoder.predict(X[i:i+1])
    Gj=symencoder.predict(X[j:j+1])
    Wj=nencoder.predict(X[j:j+1])

    Zii=np.concatenate((Gi,Wi),axis=1)
    Zjj=np.concatenate((Gj,Wj),axis=1)
    Zij=np.concatenate((Gi,Wj),axis=1)
    Zji=np.concatenate((Gj,Wi),axis=1)

    Xhatii=decoder.predict(Zii)
    Xhatij=decoder.predict(Zij)
    Xhatjj=decoder.predict(Zjj)
    Xhatji=decoder.predict(Zji)




    # if strategy.num_replicas_in_sync > 1:
    #     # grab only batch assigned to first GPU
    #     trn_batch = [x.values[0].numpy() for x in trn_batch]
    # else:
    #     trn_batch = [x.numpy() for x in trn_batch]
    # trn_batch = [x.numpy() for x in trn_batch]
    # trn_batch = trn_batch.numpy()

    plt.figure(figsize=(50,10), facecolor='w')
    num_instance_plot = 5
    for kk in np.arange(num_instance_plot):
        plt.subplot(2, num_instance_plot, kk+1)
        a=X[i,kk,:]
        plt.plot(a)
        # plt.xticks([]); plt.yticks([]);
        if kk == 0:
            plt.ylabel('Exact')
                
        plt.subplot(2, num_instance_plot, num_instance_plot+kk+1)
        b=Xhatii[0,kk,:]
        plt.plot(b)
        # plt.xticks([]); plt.yticks([]);
        if kk == 0:
            plt.ylabel('Pred')
        mse=mean_squared_error(a,b)

        plt.text(0.1, 0.9, str(mse), size=15, color='purple')
        
    plt.show()
    
    # """ plot styling of replicates """
    nsplot = 2 # number of style plots

    # content_code = symae.content_encode(trn_batch).numpy()
    # style_code = symae.style_encode(trn_batch).numpy()
    
    plt.figure(figsize=(57,20), facecolor='w')
    for jj in np.arange(nsplot):
        # styled_mean = symae.style_decode(content_code[[jj],:,:], style_code[[0],:,:])
        # styled_mean = styled_mean.numpy()
        for kk in np.arange(nsplot):
            if jj == 0:
                # plot the style of each instance inside the 0th bag
                plt.subplot(nsplot+1, nsplot+1, kk+2)
                plt.plot(X[i,kk,:])
                plt.xticks([]); plt.yticks([]);
                if kk==0:
                    plt.subplot(nsplot+1, nsplot+1, kk+1)
                    plt.plot(labels[i][str(1)]["s"])
                    plt.plot(labels[i][str(0)]["s"])
            else:
                if kk==0:
                    # plot an instance to show the "content"
                    plt.subplot(nsplot+1, nsplot+1, jj*(nsplot+1)+kk+1)
                    plt.plot(labels[j][str(1)]["s"])
                    plt.xticks([]); plt.yticks([]);
                    if jj == 1:
                        plt.title('Content')

                # style the jth bag with styles from the 0th bag
                plt.subplot(nsplot+1, nsplot+1, jj*(nsplot+1)+kk+2)
                plt.plot(Xhatji[0,kk,:])
                # plt.plot(X[i,kk,:])

                dtrue=signal.convolve(labels[j][str(kk)]["s"], labels[i][str(kk)]["g"], mode='full')
                dtrue = dtrue/np.std(dtrue)
                plt.plot(dtrue)

                plt.xticks([]); plt.yticks([]);
     


def plot_eq2(X, labels, symencoder, nencoder, decoder):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import random
    import tensorflow as tf
    from datetime import datetime
    from scipy import signal
    from sklearn.metrics import mean_squared_error
    now = datetime.now()
    dir = now.strftime("%d-%m-%Y_%H")

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px

    if not os.path.exists(dir): os.mkdir(dir)

    nsamp=np.shape(X)[0]
    bin_names=np.unique([labels[ii][str(0)]['binname'] for ii in range(nsamp)])
    bin_names=[float(item) for item in bin_names]
    bin_names=np.sort(bin_names)
    bin_names=[str(item) for item in bin_names]

    # predict sym codes
    G=[]
    for (k,bin_name) in enumerate(bin_names):
        j=random.choice([ii for ii in range(nsamp) if labels[ii][str(0)]['binname']==bin_name])
        Gj=symencoder.predict(X[j:j+1])
        G.append(Gj)
    # print(G.keys())


    # randomly choose a station and datapoint

    i = np.random.randint(0, nsamp-1)
    j = np.random.randint(0, 19)
    stname=labels[i][str(j)]['trc']

    Gtest=symencoder.predict(X[[ii for ii in range(nsamp) if labels[ii][str(0)]['binname']==labels[i][str(0)]['binname']]])
    fig = px.imshow(Gtest)
    fig.update_layout(width=1000, height=500, title="testing coherent code")
    fig.show()



    print("plotting trc "+ stname +" from datapoint index "+str(i)+" at "+labels[i][str(0)]['binname'])

    # get stn code
    Wi=nencoder.predict(X[i:i+1])

    t=np.linspace(-200, 200, np.shape(X)[2])

    fig = make_subplots(rows=1, cols=1, x_title="Time [s] relative to PREM P", shared_xaxes=True, vertical_spacing=0.02)


    Gi=symencoder.predict(X[i:i+1])
    Wi=nencoder.predict(X[i:i+1])
    Zii=np.concatenate((Gi,Wi),axis=1)
    Xhatii=decoder.predict(Zii)

    a=X[i,j,:].flatten()
    fig.add_trace(
        go.Scatter(x=t,y=a,name="true trace",line=dict(color='rgb(10,10,10)',dash='dot')),
        row=1, col=1
    )

    b=Xhatii[0,j,:].flatten()
    fig.add_trace(
        go.Scatter(x=t,y=b,name="predicted trace"),
        row=1, col=1
    )


    fig.update_layout(height=500, width=1000, title_text="True Vs. Predict", legend_title=" ",
    font=dict(
        family="Courier New, monospace",        size=18,
        color="RebeccaPurple"
    ))
    fig.show()




    fig = make_subplots(rows=len(bin_names), cols=2, x_title="Time [s] relative to PREM P", shared_xaxes=True, vertical_spacing=0.02)

    for (ibin,bin) in enumerate(bin_names):
        a=X[i,j,:].flatten()
        stname=labels[i][str(j)]['trc']
        fig.add_trace(
            go.Scatter(x=t,y=a,name="true trace",line=dict(color='rgb(10,10,10)',dash='dot')),
            row=ibin+1, col=1
        )

        Zji=np.concatenate((G[ibin],Wi),axis=1)
        Xhatji=decoder.predict(Zji)

        b=Xhatji[0,j,:].flatten()
        fig.add_trace(
            go.Scatter(x=t,y=b,name="virtual trace in "+bin),
            row=ibin+1, col=1
        )



        B = np.fft.rfft(b)
        absB = np.abs(B)
        pB = np.square(absB)
        dbpB= 20 * np.log10(pB/1000)
        f = np.linspace(0, 1/(t[2]-t[1])/2, len(absB))

        fig.add_trace(
            go.Scatter(x=f,y=dbpB,name="spectrum virtual trace in "+bin),
            row=ibin+1, col=2
        )



    fig.update_layout(height=1000, width=2000, title_text="Virtual "+stname+" at all azimuths ", legend_title=" ",
    font=dict(
        family="Courier New, monospace",        size=18,
        color="RebeccaPurple"
    ))
    fig.update_yaxes(col=2, range=[-50,50])
    fig.update_yaxes(col=1, range=[-5,5])
    fig.show()




 


def plot_eq1(X, labels, symencoder, nencoder, decoder):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import random
    import tensorflow as tf
    from datetime import datetime
    from scipy import signal
    from sklearn.metrics import mean_squared_error
    now = datetime.now()
    dir = now.strftime("%d-%m-%Y_%H")

    if not os.path.exists(dir): os.mkdir(dir)

    nsamp=np.shape(X)[0]

    i = np.random.randint(0, nsamp-1)

    j=random.choice([ii for ii in range(nsamp) if labels[ii][str(0)]['eqname']!=labels[i][str(0)]['eqname']])
    # j = np.random.randint(0, nsamp-1)
    print("test_indices"+str(i)+","+str(j))

    Gi=symencoder.predict(X[i:i+1])
    Wi=nencoder.predict(X[i:i+1])
    Gj=symencoder.predict(X[j:j+1])
    Wj=nencoder.predict(X[j:j+1])

    Zii=np.concatenate((Gi,Wi),axis=1)
    Zjj=np.concatenate((Gj,Wj),axis=1)
    Zij=np.concatenate((Gi,Wj),axis=1)
    Zji=np.concatenate((Gj,Wi),axis=1)

    Xhatii=decoder.predict(Zii)
    Xhatij=decoder.predict(Zij)
    Xhatjj=decoder.predict(Zjj)
    Xhatji=decoder.predict(Zji)


    eqnamei=labels[i][str(0)]['eqname']
    eqnamej=labels[j][str(0)]['eqname']

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    num_instance_plot = 5
    fig = make_subplots(rows=num_instance_plot, cols=1, 
    x_title="Time [s] relative to PREM P",
    shared_xaxes=True,
                    vertical_spacing=0.02)
    t=np.linspace(-200, 200, np.shape(X)[2])
    for kk in np.arange(num_instance_plot):
        a=X[i,kk,:].flatten()
        stname=labels[i][str(kk)]['trc']
        fig.add_trace(
            go.Scatter(x=t,y=a, name=eqnamei+"_"+stname),
            row=kk+1, col=1
        )

        b=Xhatii[0,kk,:].flatten()
        fig.add_trace(
            go.Scatter(x=t,y=b,name="out "+eqnamei+'_'+stname),
            row=kk+1, col=1
        )


    fig.update_layout(height=1000, width=1500, title_text="True Vs. Pred", legend_title=" ",
    font=dict(
        family="Courier New, monospace",        size=18,
        color="RebeccaPurple"
    ))
    fig.show()

   
    fig = make_subplots(rows=num_instance_plot, cols=1,
    x_title="Time [s] relative to PREM P",
    shared_xaxes=True,
                    vertical_spacing=0.02)
    for kk in np.arange(num_instance_plot):
        a=Xhatji[0,kk,:].flatten()
        stname=labels[i][str(kk)]['trc']
        fig.add_trace(
            go.Scatter(x=t,y=a/np.std(a), name='virtual '+eqnamej+"_"+stname),
            row=kk+1, col=1
        )

        b=X[i,kk,:].flatten()
        fig.add_trace(
            go.Scatter(x=t,y=b, name=eqnamei+"_"+stname),
            row=kk+1, col=1
        )




    fig.update_layout(height=1000, width=1500, title_text="Virtual Earthquakes", legend_title=" ",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    ))
    fig.show()



    # # """ plot styling of replicates """
    # nsplot = 2 # number of style plots

    # # content_code = symae.content_encode(trn_batch).numpy()
    # # style_code = symae.style_encode(trn_batch).numpy()
    
    # plt.figure(figsize=(57,20), facecolor='w')
    # for kk in np.arange(nsplot):
    #     # styled_mean = symae.style_decode(content_code[[jj],:,:], style_code[[0],:,:])
    #     # styled_mean = styled_mean.numpy()
    #     # for kk in np.arange(nsplot):
    #         # if jj == 0:
    #             # plot the style of each instance inside the 0th bag
    #         plt.subplot(nsplot, 2, 1+kk*2)
    #         plt.plot(X[i,kk,:])
    #         plt.xticks([]); plt.yticks([]);
    #         plt.title(labels[i][str(0)]['eqname'])
    #             # if kk==0:
    #                 # plt.subplot(nsplot+1, nsplot+1, kk+1)
    #                 # plt.plot(labels[i][str(1)]["s"])
    #                 # plt.plot(labels[i][str(0)]["s"])
    #         # else:
    #             # if kk==0:
    #                 # plot an instance to show the "content"
    #                 # plt.subplot(nsplot+1, nsplot+1, jj*(nsplot+1)+kk+1)
    #                 # plt.plot(labels[j][str(1)]["s"])
    #                 # plt.xticks([]); plt.yticks([]);
    #                 # plt.title(labels[j][str(0)]['eqname'])

    #             # style the jth bag with styles from the 0th bag
    #         plt.subplot(nsplot, 2, kk*2+2)
    #         plt.plot(Xhatji[0,kk,:], label='fake; hatji')
    #         plt.plot(Xhatii[0,kk,:], label='hatii')
    #         plt.title('faked '+labels[j][str(0)]['eqname'])
    #         plt.legend()
    #             # plt.plot(X[i,kk,:])

    #             # dtrue=signal.convolve(labels[j][str(kk)]["s"], labels[i][str(kk)]["g"], mode='full')
    #             # dtrue = dtrue/np.std(dtrue)
    #             # plt.plot(dtrue)

    #         plt.xticks([]); plt.yticks([]);
 



def plot_small_norb(X, labels, symencoder, nencoder, decoder):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import tensorflow as tf
    from datetime import datetime
    from scipy import signal
    from sklearn.metrics import mean_squared_error
    now = datetime.now()
    dir = now.strftime("%d-%m-%Y_%H")

    if not os.path.exists(dir): os.mkdir(dir)

    nsamp=np.shape(X)[0]

    plt.figure(figsize=(18,6), facecolor='w')


    i = np.random.randint(0, nsamp-1)
    # print("test_indices"+str(i)+","+str(j))

    Gi=symencoder.predict(X[i:i+1])
    Wi=nencoder.predict(X[i:i+1])

    Zii=np.concatenate((Gi,Wi),axis=1)

    Xhatii=decoder.predict(Zii)






    num_instance_plot = 5
    for kk in np.arange(num_instance_plot):
        plt.subplot(2, num_instance_plot, kk+1)
        plt.imshow(X[i,kk,:,:,0])
        plt.xticks([]); plt.yticks([]);
        if kk == 0:
            plt.ylabel('Exact')
                
        plt.subplot(2, num_instance_plot, num_instance_plot+kk+1)
        plt.imshow(Xhatii[0,kk,:,:,0])
        plt.xticks([]); plt.yticks([]);
        if kk == 0:
            plt.ylabel('Pred')
        
    plt.show()
        
        # """ plot styling of replicates """v
    nsplot = 5 # number of style plots

    plt.figure(figsize=(18,18), facecolor='w')
        # content_code = symae.content_encode(trn_batch).numpy()
        # style_code = symae.style_encode(trn_batch).numpy()
        
        # plt.figure(figsize=(10,10), facecolor='w')
    for jj in np.arange(nsplot):
        j = np.random.randint(0, nsamp-1)
        Gj=symencoder.predict(X[j:j+1])
        Wj=nencoder.predict(X[j:j+1])
        Zjj=np.concatenate((Gj,Wj),axis=1)
        Zij=np.concatenate((Gi,Wj),axis=1)
        Zji=np.concatenate((Gj,Wi),axis=1)
        Xhatij=decoder.predict(Zij)
        Xhatjj=decoder.predict(Zjj)
        Xhatji=decoder.predict(Zji)
        #     styled_mean = symae.style_decode(content_code[[jj],:,:], style_code[[0],:,:])
        #     styled_mean = styled_mean.numpy()
        for kk in np.arange(nsplot):
            if jj == 0:
                # plot the style of each instance inside the 0th bag
                plt.subplot(nsplot+1, nsplot+1, kk+2)
                plt.imshow(X[i,kk,:,:,0])
                plt.xticks([]); plt.yticks([]);
                if kk==0:
                    plt.ylabel('Style')
            else:
                if kk==0:
                    # plot an instance to show the "content"
                    plt.subplot(nsplot+1, nsplot+1, jj*(nsplot+1)+kk+1)
                    plt.imshow(X[j,0,:,:,0])
                    plt.xticks([]); plt.yticks([]);
                    if jj == 1:
                        plt.title('Content')

                # style the jth bag with styles from the 0th bag
                plt.subplot(nsplot+1, nsplot+1, jj*(nsplot+1)+kk+2)
                plt.imshow(Xhatji[0,kk,:,:,0])
                plt.xticks([]); plt.yticks([]);
                    
    plt.show()


def plot_seismic(X, labels, symencoder, nencoder, decoder):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import random 
    import wiggle
    import tensorflow as tf
    from datetime import datetime
    from scipy import signal
    from sklearn.metrics import mean_squared_error
    now = datetime.now()
    # dir = now.strftime("%d-%m-%Y_%H")

    # if not os.path.exists(dir): os.mkdir(dir)

    nsamp=np.shape(X)[0]

    plt.figure(figsize=(40,40), facecolor='w')


    i = np.random.randint(0, nsamp-1)
    # j=random.choice([ii for ii in range(nsamp) if labels[ii][str(0)]['medium']!=labels[i][str(0)]['medium']])
    # print("test_indices"+str(i)+","+str(j))

    i=0
    j=1

    eqnamei=labels[i][str(0)]['medium']
    eqnamej=labels[j][str(0)]['medium']

    Gi=symencoder.predict(X[i:i+1])
    Wi=nencoder.predict(X[i:i+1])

    Zii=np.concatenate((Gi,Wi),axis=1)

    Xhatii=decoder.predict(Zii)

    num_instance_plot = 5
    kki=random.choices(range(0,20),k=num_instance_plot)
    for ikk,kk in enumerate(kki):
        plt.subplot(5, num_instance_plot, ikk+1)
        plot_wiggle(X[i,kk,:,:,0])
        plt.xticks([]); plt.yticks([]);
        if kk == 0:
            plt.ylabel('Exact')
                
        plt.subplot(5, num_instance_plot, num_instance_plot+ikk+1)
        plot_wiggle(Xhatii[0,kk,:,:,0])
        plt.xticks([]); plt.yticks([]);
        if kk == 0:
            plt.ylabel('Pred')
        
    Gj=symencoder.predict(X[j:j+1])
    Wj=nencoder.predict(X[j:j+1])
    Zjj=np.concatenate((Gj,Wj),axis=1)
    Zij=np.concatenate((Gi,Wj),axis=1)
    Zji=np.concatenate((Gj,Wi),axis=1)
    Xhatij=decoder.predict(Zij)
    Xhatjj=decoder.predict(Zjj)
    Xhatji=decoder.predict(Zji)
    for ikk, kk in enumerate(kki):
        # style the jth bag with styles from the 0th bag
        plt.subplot(5, num_instance_plot, 2*num_instance_plot+ikk+1)
        a=Xhatji[0,kk,:,:,0]
        plot_wiggle(a,'r')
        plt.xticks([]); plt.yticks([]);

        plt.subplot(5, num_instance_plot, 3*num_instance_plot+ikk+1)
        b=X[j,kk,:,:,0]
        plot_wiggle(b)
        plt.xticks([]); plt.yticks([]);

        plt.subplot(5, num_instance_plot, 4*num_instance_plot+ikk+1)
        plot_wiggle(a-b)
        plt.xticks([]); plt.yticks([]);
        plt.title(str(mean_squared_error(a,b)))

                    
    plt.show()


def plot_wiggle(x,c='k', txt='none'):
    import numpy as np
    import wiggle as wiggle
    import matplotlib.pyplot as plt
    # fig=plt.figure(dpi=150, figsize=(6,12))
    sf=np.max(np.std(x,axis=1))
    wiggle.wiggle(x.T, color=c, sf= sf/0.5/10)#
    # if(txt != 'none'):
        # plt.text(0.5, 8, txt, size=20, color='black', bbox=dict(facecolor='red', alpha=1,))
    plt.axis('off')
    # plt.savefig(os.path.join(dir,s), bbox_inches='tight')
    # plt.close()



    return None
