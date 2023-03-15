import numpy as np
import pylab as plt
import matplotlib as mpl

if __name__ == '__main__':
	fig = plt.figure(figsize=(1, 15))
	cmap = mpl.cm.Spectral_r
	ax3 = fig.add_axes([0, 0, 0.2, 1]) # 四个参数分别是左、下、宽、长
	norm = mpl.colors.Normalize(vmin=0, vmax=9)
	#bounds = [ round(elem, 2) for elem in np.linspace(0, 9, 14)] #
	cb3 = mpl.colorbar.ColorbarBase(
								ax3,
								cmap=cmap,
								norm=norm,
								# to use 'extend', you must
								# specify two extra boundaries:
								#boundaries= [1.2] + bounds + [2.6],
								#extend='both',
								#ticks=bounds,  # optional
								spacing='proportional',
								orientation='vertical')
	plt.show()
