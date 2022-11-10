#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 16:05:09 2022

@author: richard
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab



plt.close('all')
figsize = [12, 8]
fontsize = 18
dpi = 200

pylab.rcParams.update({
    'figure.figsize': figsize,
    'legend.fontsize': fontsize,
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'savefig.dpi': dpi
})



zip_datapath = r"../../data/social_capital_zip.csv"

investigated_series_ids = ["exposure_grp_mem_zip", "nbhd_ec_zip", "bias_grp_mem_zip"]


raw_data = pd.read_csv(zip_datapath)

ds1 = raw_data[investigated_series_ids[0]]
ds2 = raw_data[investigated_series_ids[1]]
ds3 = raw_data[investigated_series_ids[2]]

series = [ds1, ds2, ds3]






# plotting population size
fig = plt.figure(figsize=figsize)
plt.hist(raw_data["pop2018"], bins=20)
plt.title('pop2018')
plt.savefig(r'../report/images/pop2018_hist.png')



# plotting histogramms of three chosen datasets
for id, series in zip(investigated_series_ids, series):
    fig = plt.figure()
    plt.hist(series, bins=20)
    plt.title(id)
    plt.savefig(r'../report/images/{}_hist.png'.format(id))



# plotting a 3D scatter plot to see if 3D data still appears gaussian
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel(investigated_series_ids[0])
ax.set_ylabel(investigated_series_ids[1])
ax.set_zlabel(investigated_series_ids[2])

ax.scatter(ds1, ds2, ds3)
plt.title("3D scatter plot of the chosen data-series")
plt.savefig(r'../report/images/3Dscatter.png')




# calculating the mean of the chosen 3D data-set
df = ds1.to_frame().join(ds2).join(ds3)

mean = df.mean()


