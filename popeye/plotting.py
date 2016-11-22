from __future__ import division
import numpy as np
from scipy.stats import gaussian_kde, linregress
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.patches import Circle
from matplotlib import cm

from popeye.spinach import generate_og_receptive_field

# Python 3 compatibility:
try:
    xrange
except NameError:  # python3
    xrange = range


def make_movie_static(stim_arr, dims, vmin=0, vmax=255, dpi=100, fps=60, write=False, fname=None):

    # this function requires ffmpeg -- https://www.ffmpeg.org/
    # http://matplotlib.org/examples/animation/dynamic_image2.html

    # set up the figure
    fig = plt.figure(figsize=(dims[1]/dpi,dims[0]/dpi),dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = plt.axes(xlim=(0,dims[1]),ylim=(0,dims[0]))

    ims = []

    # loop over each frame
    for frame in xrange(stim_arr.shape[-1]):

        # create figure
        im = ax.imshow(stim_arr[:,:,frame],cmap=cm.gray,
                       vmin=vmin, vmax=vmax, interpolation=None, origin='lower')
        plt.axis('off')

        # stash it
        ims.append([im])

    # animate it
    anim = animation.ArtistAnimation(fig, ims, interval=fps)

    plt.show()

    # this is a bug see here -- http://stackoverflow.com/questions/20137792/using-ffmpeg-and-ipython
    mywriter = animation.FFMpegWriter(fps=fps)

    if write and fname:
        anim.save('%s.mp4' %(fname), writer=mywriter, extra_args=['-vcodec', 'libx264'])

    return anim

def make_movie_calleable(stim_arr, dims, vmin, vmax, fps, bitrate, dpi, fname=None):

    # this function requires ffmpeg -- https://www.ffmpeg.org/
    # https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

    # set up the figure
    fig = plt.figure(figsize=(dims[1]/dpi,dims[0]/dpi),dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = plt.axes(xlim=(0,dims[1]),ylim=(0,dims[0]))
    im = ax.imshow(np.zeros(dims),cmap=cm.gray,
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    # define the animator
    def animate(frame, stim_arr):
        print("doing frame %d" %(frame))
        im.set_data(stim_arr[:,:,frame])
        plt.axis('off')
        return im,

    # initialization function: plot the background of each frame
    def init():
        im.set_data(())
        return im,

    anim = animation.FuncAnimation(fig=fig, func=animate, blit=False, repeat=False,
                                   fargs=(stim_arr,), interval=fps*20, frames=stim_arr.shape[-1])

    # this is a bug see here -- http://stackoverflow.com/questions/20137792/using-ffmpeg-and-ipython
    mywriter = animation.FFMpegWriter(fps=fps)

    if fname:
        anim.save('%s.mp4' %(fname), writer=mywriter, bitrate=bitrate, extra_args=['-vcodec', 'libx264'])

    return anim

def eccentricity_hist(x, y, xlim, voxel_dim, dof,
                     plot_alpha=1.0, plot_color='k',
                     label_name=None, show_legend=False,
                     fig=None, ax=None):


    # arguments can include figure and axes handles for plotting multiple ROIs
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111)

    ecc = np.sqrt(x**2+y**2)
    n,bins = np.histogram(ecc,bins=np.arange(xlim[0],xlim[1]))
    bincenters = list(0.5*(bins[1:]+bins[:-1]))
    mu = (n*voxel_dim**3)/(len(ecc)*voxel_dim**3)
    sme = mu / np.sqrt(dof)
    cumsum = np.cumsum(mu)

    ax.plot(bincenters,cumsum,plot_color,lw=2,label=label_name, alpha=plot_alpha)
    ax.fill_between(bincenters,cumsum-sme,cumsum+sme,color=plot_color,alpha=0.5)
    plt.xticks(np.arange(xlim[0],xlim[1]+1,5),fontsize=18)
    plt.yticks(np.arange(.2,1.2,.2),['20%','40%','60%','80%','100%'],fontsize=18)
    ax.bar(bincenters,mu,color=plot_color,alpha=plot_alpha)
    plt.xlim(0.5,xlim[1]+0.5)
    plt.ylim(0,1)

    if show_legend:
        ax.legend(loc=0)

    return fig, ax

def hrf_delay_hist(measure, bins, voxel_dim, dof,
                   plot_alpha=1.0,plot_color='k',label_name=None,
                   show_legend=False, fig=None, ax=None):

    # arguments can include figure and axes handles for plotting multiple ROIs
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111)

    n,bins = np.histogram(measure,bins=bins)
    bincenters = list(0.5*(bins[1:]+bins[:-1]))
    mu = (n*voxel_dim**3)/(len(measure)*voxel_dim**3)
    sme = mu / np.sqrt(dof)
    cumsum = np.cumsum(mu)

    ax.plot(bincenters,cumsum,plot_color,lw=2,label=label_name, alpha=plot_alpha)
    ax.fill_between(bincenters,cumsum-sme,cumsum+sme,color=plot_color,alpha=0.5)
    plt.xticks(bins,fontsize=18)
    plt.yticks(np.arange(.2,1.2,.2),['20%','40%','60%','80%','100%'],fontsize=18)
    ax.bar(bincenters,mu,color=plot_color,alpha=plot_alpha,width=0.33)
    plt.xlim(0.5,xlim[1]+0.5)
    plt.ylim(0,1)

    if show_legend:
        ax.legend(loc=0)

    return fig, ax

def polar_angle_plot(x, y, voxel_dim, num_radians, rlim, dof,
                    plot_alpha=1.0, plot_color='k', label_name=None,
                    show_legend=False, fig=None, ax=None):

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
    sme = mu / np.sqrt(dof)
    bincenters = list(0.5*(bins[1:]+bins[:-1]))

    # wrap it to make it circular & then recast back to ndarray
    mu = np.append(mu, mu[0])
    sme = np.append(sme, sme[0])
    bincenters = np.append(bincenters, bincenters[0])

    ax.plot(bincenters, mu, c=plot_color,lw=2, alpha=plot_alpha, label=label_name)
    ax.fill_between(bincenters, mu-sme, mu+sme,color=plot_color, alpha=plot_alpha)


    ax.set_rlim(0,rlim)

    # make labels
    labels = []
    for label_val in np.arange(5,rlim*100,5):
        labels.append('%s%%' %(int(label_val)))


    ax.set_rgrids(np.arange(0.05,rlim,0.05),labels,angle=90,fontsize=24)
    ax.set_thetagrids(np.arange(0,360,360/8),[],fontsize=24)

    if show_legend:
        ax.legend(loc=0)

    # show and return the fig and ax handles
    plt.show()

    # return handles
    return fig,ax

def XY_scatter(x, y, xlim, ylim, min_n, dof,
               plot_alpha=1.0, plot_color='k',label_name=None,
               show_legend=False, show_fit = False, fig=None, ax=None):

    diameter = y

    # arguments can include figure and axes handles for plotting multiple ROIs
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111)

    # fit a line
    linfit = linregress(x,y)

    # show fit
    if show_fit:
        ax.text(xlim[0]+1,ylim[1]-1,r'$\sigma = %.02f * \rho + %.02f$' %(linfit[0],linfit[1]) ,fontsize=24)

    mus = []
    es = []

    # bin and plot the errors
    for e in np.arange(xlim[0]+0.5,xlim[1]+0.5,1):
        b0 = e-0.5
        b1 = e+0.5
        idx0 = np.nonzero(x>=b0)
        idx1 = np.nonzero(x<=b1)
        if len(idx0[0]) > min_n and len(idx1[0]) > min_n:
            idx = np.intersect1d(idx0[0],idx1[0])
            mu = np.mean(diameter[idx])
            err = np.std(diameter[idx])/np.sqrt(dof)
            ax.errorbar(e,mu,yerr=err,color='%s' %(plot_color), mec='%s' %(plot_color),capsize=0,lw=4,alpha=plot_alpha)
            ax.scatter(e,mu,c='%s' %(plot_color),s=100,edgecolor='%s' %(plot_color),alpha=plot_alpha)
            mus.append(mu)
            es.append(e)

    # fit a line
    p = np.polyfit(es,mus,1)
    [y1,y2] = np.polyval(p,xlim)
    ax.plot(xlim,[y1,y2],c='%s' %(plot_color),lw=5,label=label_name)

    # beautify
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(xlim[0],xlim[1]+1))
    ax.set_yticks(np.arange(ylim[0],ylim[1]+1))
    ax.set_xticklabels(np.arange(xlim[0],xlim[1]+1),fontsize='18')
    ax.set_yticklabels(np.arange(ylim[0],ylim[1]+1),fontsize='18')

    if show_legend:
        ax.legend(loc=0)

    # show and return the fig and ax handles
    plt.show()

    return fig,ax

def eccentricity_sigma_scatter(x, y, sigma, xlim, ylim, min_n, dof,
                              plot_alpha=1.0, plot_color='k',label_name=None,
                              show_legend=False, show_fit = False, fig=None, ax=None):

    ecc = np.sqrt(x**2+y**2)
    diameter = sigma

    # arguments can include figure and axes handles for plotting multiple ROIs
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111)

    # fit a line
    linfit = linregress(ecc,sigma)

    # show fit
    if show_fit:
        ax.text(xlim[0]+1,ylim[1]-1,r'$\sigma = %.02f * \rho + %.02f$' %(linfit[0],linfit[1]) ,fontsize=24)

    mus = []
    es = []

    # bin and plot the errors
    for e in np.arange(xlim[0]+0.5,xlim[1]+0.5,1):
        b0 = e-0.5
        b1 = e+0.5
        idx0 = np.nonzero(ecc>=b0)
        idx1 = np.nonzero(ecc<=b1)
        if len(idx0[0]) > min_n and len(idx1[0]) > min_n:
            idx = np.intersect1d(idx0[0],idx1[0])
            mu = np.mean(diameter[idx])
            err = np.std(diameter[idx])/np.sqrt(dof)
            ax.errorbar(e,mu,yerr=err,color='%s' %(plot_color), mec='%s' %(plot_color),capsize=0,lw=4,alpha=plot_alpha)
            ax.scatter(e,mu,c='%s' %(plot_color),s=100,edgecolor='%s' %(plot_color),alpha=plot_alpha)
            mus.append(mu)
            es.append(e)

    # fit a line
    p = np.polyfit(es,mus,1)
    [y1,y2] = np.polyval(p,xlim)
    ax.plot(xlim,[y1,y2],c='%s' %(plot_color),lw=5,label=label_name)

    # beautify
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(xlim[0],xlim[1]+1))
    ax.set_yticks(np.arange(ylim[0],ylim[1]+1))
    ax.set_xticklabels(np.arange(xlim[0],xlim[1]+1),fontsize='18')
    ax.set_yticklabels(np.arange(ylim[0],ylim[1]+1),fontsize='18')
    ax.set_ylabel('pRF Size (deg)',size='18')
    ax.set_xlabel('Eccentricity (deg)',size='18')

    if show_legend:
        ax.legend(loc=0)

    # show and return the fig and ax handles
    plt.show()

    return fig,ax


def sigma_hrf_delay_scatter(sigma, hrf_delay, xlim, ylim,
                            min_vox, n, plot_color, label_name,
                            show_legend=False, fig=None, ax=None):

    # arguments can include figure and axes handles for plotting multiple ROIs
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111)

    # fit a line
    p = np.polyfit(sigma,hrf_delay,1)
    [y1,y2] = np.polyval(p,xlim)
    ax.plot(xlim,[y1,y2],c='%s' %(plot_color),lw=5,label='%s' %(label_name))

    # bin and plot the errors
    for e in np.arange(xlim[0]+0.25,xlim[1]+0.25,1):
        b0 = e-0.25
        b1 = e+0.25
        idx0 = np.nonzero(sigma>=b0)
        idx1 = np.nonzero(sigma<=b1)
        idx = np.intersect1d(idx0,idx1)
        if len(idx) > min_vox:

            # if its only 1 subject, c
            if num_subjects == 1:
                n = len(idx)
            else:
                n = num_subjects

            mu = np.mean(hrf_delay[idx])
            err = np.std(hrf_delay[idx])/np.sqrt(len(n))
            ax.errorbar(e,mu,yerr=err,color='%s' %(plot_color), mec='%s' %(plot_color),capsize=0,lw=4)
            ax.scatter(e,mu,c='%s' %(plot_color),s=100,edgecolor='%s' %(plot_color))

    # beautify
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(xlim[0],xlim[1]+1))
    ax.set_yticks(np.arange(ylim[0],ylim[1]+1))
    ax.set_xticklabels(np.arange(xlim[0],xlim[1]+1),fontsize='18')
    ax.set_yticklabels(np.arange(ylim[0]+5,ylim[1]+1+5),fontsize='18')
    ax.set_ylabel('HRF Delay (secs)',size='18')
    ax.set_xlabel('pRF Size (deg)',size='18')

    if show_legend:
        ax.legend(loc=2)

    # show and return the fig and ax handles
    plt.show()
    return fig,ax



def eccentricity_sigma_fill(ecc,sigma,plot_color,label_name,fig=None,ax=None):

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
        idx0 = np.nonzero(ecc>=b0)
        idx1 = np.nonzero(ecc<=b1)
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

def beta_hist(beta, xlim, voxel_dim, plot_color, label_name, fig=None, ax=None):

    # arguments can include figure and axes handles for plotting multiple ROIs
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111)

    spacing = np.linspace(xlim[0],xlim[1],xlim[2],endpoint=False)

    n,bins = np.histogram(beta,bins=spacing)
    bincenters = list(0.5*(bins[1:]+bins[:-1]))
    mu = (n*voxel_dim**3)/(len(beta)*voxel_dim**3)
    sme = mu / np.sqrt(n)
    cumsum = np.cumsum(mu)

    ax.plot(bincenters,cumsum,plot_color,lw=2,label=label_name)
    ax.fill_between(bincenters,cumsum-sme,cumsum+sme,color=plot_color,alpha=0.5)
    plt.xticks(spacing,fontsize=18)
    plt.yticks(np.arange(.2,1.2,.2),['20%','40%','60%','80%','100%'],fontsize=18)
    ax.bar(bincenters,mu,color=plot_color,width=0.33)
    plt.xlim(spacing[0]+0.15,spacing[-1]+0.25)
    plt.ylim(0,1)

    return fig, ax




def hrf_delay_kde(delays,kernel_width,plot_color,label_name,fig=None,ax=None):
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

def hexbin_location_map(x,y):
    from matplotlib import pyplot as plt

    # set up figure
    fig = plt.figure(figsize=(8,8),dpi=100)
    ax = fig.add_subplot(111, aspect='equal')

    # get the upper bound
    ub = max(np.max(x),np.max(y))*0.9

    im = ax.hexbin(x,y,gridsize=25,extent=(-12,12,-12,12),vmax=ub,cmap='hot')

    # beautify
    plt.xticks(np.arange(-12,13,3),fontsize='16')
    plt.yticks(np.arange(-12,13,3),fontsize='16')
    plt.xlabel('degrees in X',fontsize='20')
    plt.ylabel('degrees in Y',fontsize='20')
    plt.xlim((-12,12))
    plt.ylim((-12,12))
    plt.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=.06)
    plt.show()

    return None

def field_coverage(x, y, s, deg_x, deg_y, log=False, polar=False):

    # set up figure
    fig = plt.figure(figsize=(8,8),dpi=100)
    if polar:
        ax = fig.add_subplot(111, projection='polar')
    else:
        ax = fig.add_subplot(111, aspect='equal')

    # set up a blank field
    field = np.zeros_like(deg_x)

    # create the RFs
    for r in np.arange(len(x)):
        rf = generate_og_receptive_field(x[r], y[r], s[r], deg_x, deg_y)
        # d = np.sqrt((x[r]-deg_x)**2 + (y[r]-deg_y)**2)<s[r]
        field += rf

    # normalize
    field /= np.max(field)
    field *= 100

    # create image
    if log:
        im = ax.imshow(field,cmap='viridis',extent=(-12,12,-12,12),norm=LogNorm(),vmin=1e0,vmax=1e2)
        cb = plt.colorbar(im, format=LogFormatterMathtext(),)
    else:
        im = ax.imshow(field,cmap='viridis',extent=(-12,12,-12,12),vmin=1e0,vmax=1e2)
        cb = plt.colorbar(im)

    # beautify
    plt.xticks(np.arange(-12,13,3),fontsize='16')
    plt.yticks(np.arange(-12,13,3),fontsize='16')
    plt.xlabel('degrees in X',fontsize='20')
    plt.ylabel('degrees in Y',fontsize='20')
    plt.xlim((-12,12))
    plt.ylim((-12,12))
    plt.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=.06)
    plt.grid('on',c='w')
    plt.show()

    return field

def lazy_field_coverage(field, log=False, polar=False):

    # set up figure
    fig = plt.figure(figsize=(8,8),dpi=100)
    if polar:
        ax = fig.add_subplot(111, projection='polar')
    else:
        ax = fig.add_subplot(111, aspect='equal')

    limit = np.ceil(np.max([np.abs(field.min()),np.abs(field.max())]))

    # create image
    if log:
        im = ax.imshow(field+1,cmap='jet',extent=(-12,12,-12,12),norm=LogNorm(),vmin=-limit,vmax=limit)
        cb = plt.colorbar(im, format=LogFormatterMathtext(),)
    else:
        im = ax.imshow(field,cmap='jet',extent=(-12,12,-12,12),vmin=-limit,vmax=limit)
        cb = plt.colorbar(im)

    # beautify
    plt.xticks(np.arange(-12,13,3),fontsize='16')
    plt.yticks(np.arange(-12,13,3),fontsize='16')
    plt.xlabel('degrees in X',fontsize='20')
    plt.ylabel('degrees in Y',fontsize='20')
    plt.xlim((-12,12))
    plt.ylim((-12,12))
    plt.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=.06)
    plt.grid('on',c='w')
    plt.show()

    return None

def location_and_size_map(x,y,s, plot_color):

    ind = np.argsort(s)[::-1]

    fig = plt.figure(figsize=(8,8),dpi=100)
    ax = fig.add_subplot(111, aspect='equal')
    for i in ind:
        ax.add_artist(Circle(xy=(x[i], y[i]),radius=s[i],fc=plot_color,ec=plot_color,alpha=0.25))

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
