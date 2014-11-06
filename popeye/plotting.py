from __future__ import division

def polar_angle_plot(x, y, voxel_dim, num_radians, rlim, plot_color, label_name, fig=None, ax=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from pylab import find
    
    # arguments can include figure and axes handles for plotting multiple ROIs
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
    else:
        ax = fig.add_subplot(111, projection='polar')
    
    # compute the polar angle
    polar_angle = np.mod(np.arctan2(y,x),2*np.pi)
    
    # calculate the volume of ROI at a given polar angle sector ...
    n,bins = np.histogram(polar_angle,bins=np.arange(0,(2*np.pi)+2*np.pi/num_radians,2*np.pi/num_radians))
    mu = (n*voxel_dim**3)/(len(polar_angle)*voxel_dim**3)
    sme = mu / np.sqrt(n)
    bincenters = list(0.5*(bins[1:]+bins[:-1]))
    
    # wrap it to make it circular & then recast back to ndarray
    mu = np.append(mu, mu[0])
    sme = np.append(sme, sme[0])
    bincenters = np.append(bincenters, bincenters[0])
    
    label_name += ',n=%d' %(len(polar_angle))
    ax.plot(bincenters,mu,c=plot_color,lw=2, label=label_name)
    ax.fill_between(bincenters,mu-sme,mu+sme,color=plot_color,alpha=0.5)
    
    
    ax.set_rlim(0,rlim)
    
    # make labels
    labels = []
    for label_val in np.arange(5,rlim*100,5):
        labels.append('%s%%' %(int(label_val)))
    
    
    ax.set_rgrids(np.arange(0.05,rlim,0.05),labels,angle=90,fontsize=24)
    ax.set_thetagrids(np.arange(0,360,360/8),[],fontsize=24)
    ax.legend(fancybox=True,loc=4)
    
    # show and return the fig and ax handles
    plt.show()
    
    # return handles
    return fig,ax


def eccentricity_sigma_fill(ecc,sigma,plot_color,label_name,fig=None,ax=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from pylab import find
    
    ecc = np.array(ecc)
    sigma = np.array(sigma)
    
    # arguments can include figure and axes handles for plotting multiple ROIs
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111)
    
    # fit a line
    p = np.polyfit(ecc,sigma,1)
    [y1,y2] = np.polyval(p,[0,14])
    
    mus = []
    errs = []
    
    # bin and plot the errors
    for e in np.arange(0.5,14.5,1):
        b0 = e-0.5
        b1 = e+0.5
        idx0 = find(ecc>=b0)
        idx1 = find(ecc<=b1)
        idx = np.intersect1d(idx0,idx1)
        mu = np.mean(sigma[idx])
        err = np.std(sigma[idx])/np.sqrt(len(idx))
        mus.append(mu)
        errs.append(err)
    
    xs = np.arange(0,len(mus))
    ys = np.linspace(y1,y2,len(mus))
    ax.plot(xs, ys, c = '%s' %(plot_color), lw=2)
    ax.fill_between(xs, ys+errs, ys-errs, color='%s' %(plot_color), alpha=0.5)
    
    # beautify
    ax.set_xlim((0,13))
    ax.set_ylim([-1,5])
    ax.set_xticks(np.arange(0,14))
    ax.set_yticks(np.arange(0,6))
    ax.set_xticklabels(np.arange(0,14),fontsize='18')
    ax.set_yticklabels(np.arange(0,6),fontsize='18')
    ax.set_ylabel('pRF Size (deg)',size='18')
    ax.set_xlabel('Eccentricity (deg)',size='18')
    ax.legend(fancybox=True,loc=2)
    
    # show and return the fig and ax handles
    plt.show()
    return fig,ax


def eccentricity_sigma_scatter(x,y,sigma,xlim,ylim,plot_color,label_name,fig=None,ax=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from pylab import find
    
    ecc = np.sqrt(x**2+y**2)
    
    # arguments can include figure and axes handles for plotting multiple ROIs
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111)
    
    # fit a line
    p = np.polyfit(ecc,sigma,1)
    [y1,y2] = np.polyval(p,xlim)
    label_name += ',n=%d' %(len(ecc))
    ax.plot([0,0],[0,0],c='%s' %(plot_color),lw=5,label='%s' %(label_name))
    
    # bin and plot the errors
    for e in np.arange(xlim[0]+0.5,xlim[1]+0.5,1):
        b0 = e-0.5
        b1 = e+0.5
        idx0 = find(ecc>=b0)
        idx1 = find(ecc<=b1)
        idx = np.intersect1d(idx0,idx1)
        mu = np.mean(sigma[idx])
        err = np.std(sigma[idx])/np.sqrt(len(idx))
        ax.errorbar(e,mu,yerr=err,color='%s' %(plot_color), mec='%s' %(plot_color),capsize=0,lw=4)
        ax.scatter(e,mu,c='%s' %(plot_color),s=100,edgecolor='%s' %(plot_color))
    
    # beautify
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(xlim[0],xlim[1]+1))
    ax.set_yticks(np.arange(ylim[0],ylim[1]+1))
    ax.set_xticklabels(np.arange(xlim[0],xlim[1]+1),fontsize='18')
    ax.set_yticklabels(np.arange(ylim[0],ylim[1]+1),fontsize='18')
    ax.set_ylabel('pRF Size (deg)',size='18')
    ax.set_xlabel('Eccentricity (deg)',size='18')
    ax.legend(fancybox=True,loc=2)
    
    # show and return the fig and ax handles
    plt.show()
    return fig,ax
    
def hrf_delay_kde(delays,kernel_width,plot_color,label_name,fig=None,ax=None):
    from scipy.stats import gaussian_kde
    import numpy as np
    import matplotlib.pyplot as plt
    
    # arguments can include figure and axes handles for plotting multiple ROIs
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111)
    
    # compute the kernel density estimation
    density = gaussian_kde(delays)
    xs = np.linspace(np.min(delays) + np.min(delays)*0.5,np.max(delays)*1.5,200)
    xs = np.linspace(-4,4,200)
    density.covariance_factor = lambda : kernel_width
    density._compute_covariance()
    
    # plot it
    ax.plot(xs,density(xs),lw=3,color=plot_color,label=r'%s: $ \mu $ = %.02f $ \sigma $ = %.02f' %(label_name.split("_")[-1].split(".nii.gz")[0],np.mean(delays),np.std(delays)))
    
    # beautify it
    ax.set_xlabel(r'HRF Delay $\tau$ $\pm$ 5s',fontsize=18)
    ax.set_xlim([-3,3])
    ax.set_xticks(np.arange(-3,4))
    ax.set_xticklabels(np.arange(-3,4),fontsize=18)
    ax.set_ylabel('Probabilty',fontsize=18)
    ax.set_yticks(np.arange(0.1,1.1,.1))
    ax.set_yticklabels(np.arange(0.1,1.1,.1),fontsize=18)
    ax.legend(fancybox=True,loc=0)
    
    # show and return the fig and ax handles
    plt.show()
    return fig,ax
    

def location_estimate_jointdist(x,y,plot_color):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    
    nullfmt   = NullFormatter()         # no labels
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a rectangular Figure
    plt.figure(1, figsize=(12,9))
    
    axScatter = plt.axes(rect_scatter)
    plt.xticks(fontsize='16')
    plt.xlabel('Degrees of visual angle in X',fontsize='20')
    plt.yticks(fontsize='16')
    plt.yticks(np.arange(10,-15,-5),np.arange(-10,15,5),fontsize='16')
    plt.ylabel('Degrees of visual angle in Y',fontsize='20')
    
    axHistx = plt.axes(rect_histx)
    plt.yticks(fontsize='16')
    axHisty = plt.axes(rect_histy)
    plt.xticks(fontsize='16',rotation=45)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:
    axScatter.scatter(x, y, facecolor=plot_color, edgecolor=plot_color)
    
    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim = ( int(xymax/binwidth) + 1) * binwidth
    
    axScatter.set_xlim( (-lim, lim) )
    axScatter.set_ylim( (-lim, lim) )
    
    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins, histtype='stepfilled',edgecolor=plot_color,facecolor=plot_color)
    axHisty.hist(y, bins=bins, orientation='horizontal', edgecolor=plot_color,facecolor=plot_color,histtype='stepfilled')
    
    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )
    
    plt.show()
    
    return None

def location_and_size_map(x,y,s, plot_color):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle
    
    ind = np.argsort(s)[::-1]
    
    fig = plt.figure(figsize=(8,8),dpi=100)
    ax = fig.add_subplot(111, aspect='equal')
    for i in ind:
        ax.add_artist(Circle(xy=(x[i], y[i]),radius=s[i],fc=plot_color,ec='k',alpha=0.25))
    
    # beautify
    plt.xticks(np.arange(-12,13,3),fontsize='16')
    plt.yticks(np.arange(-12,13,3),fontsize='16')
    plt.xlabel('Degrees of visual angle in X',fontsize='20')
    plt.ylabel('Degrees of visual angle in Y',fontsize='20')
    plt.xlim((-15,15))
    plt.ylim((-15,15))
    plt.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=.06)
    plt.show()
    
    return None
