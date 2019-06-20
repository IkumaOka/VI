import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import *
from matplotlib import animation as ani

sigma = 1
mu     = 3

def norm_dist_neg(x):
    return -1./(np.sqrt(2 * np.pi) * sigma)* np.exp(-0.5*((x-mu)**2)/((sigma**2)))

def norm_dist_1st_div_neg(x):
    return (x-float(mu))/(np.sqrt(2 * np.pi) * sigma**3) * np.exp(-0.5*(x-mu)**2/((sigma**2)))

x_low  = -3.5
x_high = 3.5
x_low  += mu
x_high += mu

x    = np.linspace(x_low,x_high,100)
y    = norm_dist_neg(x) 
y1  = norm_dist_1st_div_neg(x)

def plot_line(slope, x, y):
    
    xbase = np.linspace(x_low, x_high, 100)
    b = - x * slope + y
    y1 = slope * xbase + b
    plt.plot(xbase,y1, c="r")
    
def calculate(init_x = 0,nmax=1000, learning_ratio = 1, precision=8):
    
    list_xs    = []
    list_ys    = []
    list_slope = []
    list_xdiff = []
    xs = init_x
    i = 0
    
    for i in range(nmax):
        ys = norm_dist_neg(xs)
        slope = norm_dist_1st_div_neg(xs)
        list_xs.append(xs)
        list_ys.append(ys)
        list_slope.append(slope)
        
        x_diff = learning_ratio * slope
        xs -= x_diff
        list_xdiff.append(x_diff)
        
        if abs(x_diff) < (0.1**precision) and (i != 0) :
            break
    
    ret_dict = {}
    ret_dict['num'] = i
    ret_dict['list_xs'] = list_xs
    ret_dict['list_ys'] = list_ys
    ret_dict['list_slope'] = list_slope
    ret_dict['list_xdiff'] = list_xdiff
    
    return ret_dict



def animate(nframe):
    xs = ret_dict['list_xs'][nframe] 
    ys = ret_dict['list_ys'][nframe] 
    slope = ret_dict['list_slope'][nframe] #norm_dist_1st_div_neg(xs)
    xdiff = ret_dict['list_xdiff'][nframe]
    
    plt.clf()
    
    # display norm dist 
    plt.subplot(2, 1, 1)
    plt.title("n=%d, x=%.5f, y=%.5f, xdiff=%.5f" % (nframe,xs, ys, xdiff))
    plot_line(slope, xs, ys)
    plt.scatter(xs, ys, c="b", s=20, alpha=0.8)
    plt.plot([xs, xs-xdiff],[ys,ys], c="k")
    plt.plot([xs-xdiff, xs-xdiff],[ys,ys-(xdiff*slope)], c="k")
    plt.plot(x, y, c="b")
    plt.plot([x_low,x_high],[0,0],  "--", c="k")
    plt.plot([0,0],[-1, 1], "--", c="k")
    plt.xlim(x_low,x_high)
    plt.ylim(-0.45, 0.05)
    
    # display deviation of norm dist
    plt.subplot(2, 1, 2)
    plt.plot(x, y1, c="g")
    plt.xlim(x_low,x_high)
    plt.ylim(-0.3, 0.3)
    plt.title("n=%d, slope=%.5f" % (nframe,xdiff))
    plt.scatter(xs, slope, c="g", s=20, alpha=0.8)
    plt.plot([x_low,x_high],[0,0],  "--", c="k")
    plt.plot([0,0],[-1, 1], "--", c="k")

for i in [0.0, 6.0]:
    init = 6.0
    ret_dict = calculate(init_x=init)
    print ("calc finish.")
    fig = plt.figure(figsize=(6.5,6.5))
    print (ret_dict['num'])
    anim = ani.FuncAnimation(fig, animate, frames=ret_dict['num'])
    anim.save('normdist_decent_%.1f_anim.mp4' % init, fps=5)
    
    clip = VideoFileClip("normdist_decent_%.1f_anim.mp4" % init)
    clip.write_gif("normdist_decent_%.1f_anim.gif" % init)
