# Copyright (C) 2017 Vladimir Nadvornik
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import numpy as np
import cv2

import tifffile
import tempfile
import subprocess
from astropy.io import fits

def normalize(img):
        dst = np.empty_like(img)
        return cv2.normalize(img, dst, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

cam_params = {
	'Canon EOS 40D' : { 'cols': slice(30, 30 + 3908), 'rows': slice(18, 18 + 2600), "masked": slice(0, 30), 'min': 1024, 'max': 16383},
	'Canon EOS 7D' : { 'cols': slice(158, 158 + 5202), 'rows': slice(51, 51 + 3465), "masked": slice(8, 156), 'min': 2048, 'max': 16383}
}

def imgread(name):
	if name.endswith('.tif') or name.endswith('.tiff'):
		img = tifffile.TiffFile(name).asarray(memmap=True)
		return np.atleast_3d(img), 0, 65535

	if name.endswith('.fits'):
		hdul = fits.open(name)
		img = hdul[0].data
		return np.atleast_3d(img), 0, 65535


def rawread(name):
	if name.endswith('.tif') or name.endswith('.tiff'):
		img = tifffile.TiffFile(name).asarray(memmap=True)
		return np.atleast_3d(img)[:,:, 0], 0, 65535

	if name.endswith('.fits'):
		hdul = fits.open(name)
		img = hdul[0].data
		return np.atleast_3d(img), 0, 65535

	camera = None
	for line in subprocess.check_output(['dcraw', '-i', '-v', name], env={'LANG':'C'}, universal_newlines=True).split('\n'):
		print(line)
		if line.startswith('Camera: '):
			camera = line[len('Camera: '):]
	
	with tempfile.NamedTemporaryFile() as f:
		subprocess.check_call(['dcraw', '-E', '-T', '-4', '-t', '0', '-c', name], stdout = f)
		img = tifffile.imread(f.name)
		img = np.atleast_3d(img)[:,:, 0]
		
		maxval = None
		minval = None
		
		if camera in cam_params:
			p = cam_params[camera]
			mask = img[p['rows'], p['masked']]
			
			mask_m = np.median(mask, axis = 1)
			w = np.ones_like(mask)
			for i in range(0, 4):
				d2 = (mask - mask_m[:, None])**2
				s2 = np.average(d2, axis = 1, weights = w)
				w = np.ones_like(mask)
				w[np.where(d2 > (s2 * 4)[:, None])] = 0
				mask_m = np.average(mask, axis = 1, weights = w)
			
			mask_m -= np.amin(mask_m)
			print(mask_m)
			
			img = img[p['rows'], p['cols']]
			
			img -= np.array(mask_m, dtype = img.dtype)[:, None]
		
			maxval = p['max']
			minval = p['min']
		if maxval is None:
			maxval = int(np.amax(img))
		if minval is None:
			minval = int(np.amin(img))

		
		return img, minval, maxval

def debayer(img, filt = False):
	return cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR)


#def debayer(img, filt = False):
#	b = img[0::2, 0::2]
#	r = img[1::2, 1::2]
#	
#	#if filt:
#	#	b = cv2.medianBlur(b, 3)
#	#	r = cv2.medianBlur(r, 3)
#	
#	b = cv2.resize(b, (img.shape[1], img.shape[0]))
#	r = cv2.resize(r, (img.shape[1], img.shape[0]))
#	
#	
#	g = np.array(img, copy = True)
##	print g[2:-1:2, 2:-1:2].shape, g[1:-2:2, 2:-1:2].shape, g[3::2, 2:-1:2].shape, g[2:-1:2, 1:-1:2].shape, g[2:-1:2, 3::2].shape
#	
#	g[2:-1:2, 2:-1:2] = np.mean([g[1:-2:2, 2:-1:2], g[3::2, 2:-1:2], g[2:-1:2, 1:-1:2], g[2:-1:2, 3::2]], axis = 0)
#        #g[2:-1:2, 0] = np.median([g[1:-2:2, 0], g[3::2, 0], g[2:-1:2, 1]], axis = 0)
#	#g[0, 2:-1:2] = np.median([g[1, 2:-1:2], g[0, 1:-1:2], g[0, 3::2]], axis = 0)
#	#g[0, 0] = np.median([g[1, 0], g[0, 1]], axis = 0)
#
#
#	g[1:-1:2, 1:-1:2] = np.mean([g[0:-2:2, 1:-1:2], g[2::2, 1:-1:2], g[1:-1:2, 0:-2:2], g[1:-1:2, 2:-1:2]], axis = 0)
#	#g[-1, 1:-1:2] = np.median([g[-2, 1:-1:2], g[-1, 0:-2:2], g[-1, 2:-1:2]], axis = 0)
#	#g[1:-1:2, -1] = np.median([g[0:-2:2, -1], g[2::2, -1], g[1:-1:2, -2]], axis = 0)
#	#g[-1, -1] = np.median([g[-2, -1], g[-1, -2]], axis = 0)
#
##	if filt:
##		g = cv2.medianBlur(g, 3)
#
#
#	res = np.empty((img.shape[0], img.shape[1], 3), dtype = np.uint16)
#	
##	print img.shape, res.shape
#	
#	res[:, :, 0] = b
#	res[:, :, 1] = g
#	res[:, :, 2] = r
#	
#	return res


def hp_filt_bayer(img, size = 9):
	#col = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
	col = debayer(img)
	col = np.array(col, dtype=np.float32)
	
	col = cv2.GaussianBlur(col, (size, size), 0)
	
	sub = np.empty_like(img)
	sub[0::2, 0::2] = col[0::2, 0::2, 0]
	sub[1::2, 1::2] = col[1::2, 1::2, 2]
	sub[0::2, 1::2] = col[0::2, 1::2, 1]
	sub[1::2, 0::2] = col[1::2, 0::2, 1]
	res = cv2.subtract(img, sub, dtype=cv2.CV_32FC1)
	return res

def hp_filt_rgb(img, size = 9):
	#col = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)

	r_bl = cv2.GaussianBlur(img[0::2, 0::2], (size, size), 0)
	b_bl = cv2.GaussianBlur(img[1::2, 1::2], (size, size), 0)
	
	g_bl = (cv2.GaussianBlur(img[0::2, 1::2], (size, size), 0) + cv2.GaussianBlur(img[1::2, 0::2], (size, size), 0)) / 2.0
	
	res = [
		cv2.GaussianBlur(img[0::2, 0::2] - r_bl, (3,3), 0),
		(img[0::2, 1::2] + img[1::2, 0::2]) / 2.0 - g_bl,
		cv2.GaussianBlur(img[1::2, 1::2] - b_bl, (3,3), 0)
		]
	
	return res

def hp_filt_mono(img, size = 21):
	bl = cv2.GaussianBlur(img, (size, size), 0)
	res = cv2.subtract(img, bl, dtype=cv2.CV_32FC1)
	return res

def poly_array(X, Y, order = 3):
#	return np.polynomial.polynomial.polyvander2d(X, Y, (order-1,order-1))[:, np.where(np.flipud(np.tri(order)).ravel())[0]]
	return np.polynomial.polynomial.polyvander2d(X, Y, (order-1,order-1))

def poly_res(shape, coef, order = 3, scale = 1, darkframes = []):
	YX = np.indices(shape, dtype = np.float64) * scale / 1000.0
	X = YX[1]
	Y = YX[0]
	
	A = poly_array(X, Y, order)
	
	if len(darkframes) > 0:
		nA = np.empty((A.shape[0], A.shape[1], A.shape[2] + len(darkframes)), dtype = np.float64)
		nA[:, :, 0:A.shape[2]] = A
		for i, df in enumerate(darkframes):
			nA[:, :, A.shape[2] + i] = df
		A = nA
	
	return np.dot(A, coef)

def poly_fit(img, mask = None, order = 3, scale = 1, darkframes = [], sqrt_w = None):
	YX = np.indices(img.shape, dtype = np.float64) * scale / 1000.0
	X = YX[1]
	Y = YX[0]
	
	
	Xf = X.ravel()
	Yf = Y.ravel()
	af = img.ravel()
	
	df_ravel = []

	if sqrt_w is not None:
		sqrt_w = sqrt_w.ravel()
	
	if mask is not None:
		nz = np.nonzero(mask.ravel())
		Xf = Xf[nz]
		Yf = Yf[nz]
		af = af[nz]
		
		if sqrt_w is not None:
			sqrt_w = sqrt_w[nz]
		
		for df in darkframes:
			df_ravel.append(df.ravel()[nz])
	else:
		for df in darkframes:
			df_ravel.append(df.ravel())

	A = poly_array(Xf, Yf, order = order)
	
	if len(df_ravel) > 0:
		A = np.append(A, np.asarray(df_ravel).T, axis = 1)
	
	if sqrt_w is not None:
		A = A * sqrt_w[:,np.newaxis]
		af = af * sqrt_w
	
	coef = np.linalg.lstsq(A, af, rcond=-1)[0]
	
	ret = poly_res(img.shape, coef, order = order, scale = scale, darkframes = darkframes)
	return ret, coef

def poly_bg(img, order = 3, scale = 8, it = 4, erode = 0, kappa = 2, transpmask = None, get_mask = False, darkframes = [], save_mask = None, get_stddev = False):
	iimg = np.atleast_3d(img)
	move_a = False
	if iimg.shape[0] > iimg.shape[2]:
		iimg = np.moveaxis(iimg, -1, 0)
		move_a = True
	src_shape = iimg.shape[1:]
	resize_w = int((iimg.shape[2] + scale - 1) / scale)
	resize_h = int((iimg.shape[1] + scale - 1) / scale)
	n_ch = iimg.shape[0]


	print(iimg.shape)
	img = []
	for c in range(n_ch):
		img1 = cv2.resize(iimg[c], (resize_w, resize_h), interpolation=cv2.INTER_AREA)
		if erode > 0:
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
			img1 = cv2.erode(img1, kernel)
		img.append(np.array(img1, dtype = np.float64))
	del iimg
	
	df_res = []
	for src_df in darkframes:
		df = cv2.resize(src_df, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
		df = np.array(df, dtype = np.float64)
		df = cv2.GaussianBlur(df, (9, 9), 0)
		df_res.append(df)
		
	
	lowmask = np.ones((resize_h, resize_w), dtype = np.uint8)
	if transpmask is not None:
		lowmask = cv2.bitwise_and(lowmask, transpmask)
	
	eps = np.finfo(np.float32).eps * 2
	
	c_stddev = [0] * n_ch
	sqrt_w = [None] * n_ch

	for i in range(0, it):
		coef_l = []
		new_lowmask = np.ones((resize_h, resize_w), dtype = np.uint8) * 255
		for c in range(0,n_ch):
			grad, coef = poly_fit(img[c], mask = lowmask, order = min(i + 1, order), scale = scale, darkframes = df_res, sqrt_w = sqrt_w[c])
			coef_l.append(coef)
			diff = img[c] - grad
			if i > it // 2:
				sqrt_w[c] = (np.abs(diff) + 1e-12)** -0.25
			mean, stddev = cv2.meanStdDev(diff, mask = lowmask)
			
			c_stddev[c] = float(stddev)
			if stddev < eps:
				continue
			
			new_lowmask = cv2.bitwise_and(new_lowmask, cv2.compare(img[c], grad + float(stddev) * kappa, cv2.CMP_LE))
		if transpmask is not None:
			new_lowmask = cv2.bitwise_and(new_lowmask, transpmask)
		lowmask = new_lowmask

		if save_mask is not None:
			tifffile.imsave("%s%d.tif" % (save_mask, i), lowmask)
			print("%s%d.tif" % (save_mask, i))
	
	if get_mask == True:
		return lowmask, stddev
	
	res = []
	for c in range(0,n_ch):
		res.append(poly_res(src_shape, coef_l[c], order = order, scale = 1, darkframes = darkframes))
	if move_a:
		res = np.moveaxis(res, 0, -1)
	res = np.array(res)
	
	if get_stddev:
		return res, c_stddev
	return res

def combine_images(lst, coefs, sub = None):
	if sub is None:
		sub = np.zeros_like(coefs)
	buf = np.zeros_like(lst[0], dtype=np.float32)
	for a, c, s in zip(lst, coefs, sub):
		af = np.array(a, dtype=np.float32)
		if s != 0:
			af = cv2.subtract(af, s)
		buf = cv2.scaleAdd(af, c, buf)
	return buf

def fit_images(src_list, target, it = 10, mask = None, kappa = None, kappa_plus = None, init_weight = None):
	solv_a = np.array([i.ravel() for i in src_list]).T
	solv_b = target.ravel()
	
	if mask is not None:
		keep = np.where(mask.ravel() > 0)
		solv_a = solv_a[keep]
		solv_b = solv_b[keep]
	
	if init_weight is not None:
		weights = init_weight.ravel()
	else:
		weights = np.ones_like(solv_b)
	
	for i in range(0, it):
#		print "a", solv_a
#		print "b", solv_b

		sqrtw = np.sqrt(weights)
		solv_aw = solv_a * sqrtw[:, None]
		solv_bw = solv_b * sqrtw
		coefs = np.linalg.lstsq(solv_aw, solv_bw)[0]
	
		d = np.dot(solv_a, coefs)

		diff2 = (d - solv_b) ** 2
		var = np.average(diff2, weights = weights)
		print(i, coefs, "var:", var)
		if var == 0:
			return coefs, len(solv_b)
			
		weights = 1 / (1 + diff2 / var)
		
		if kappa is not None:
			weights[np.where(diff2 > var * kappa ** 2)] = 0

		if kappa_plus is not None:
			weights[np.where((diff2 > var * kappa_plus ** 2) & (solv_b > d))] = 0
		
		
	return coefs, len(solv_b)



def sigma_clip(images, zero = 0, over = 50000, scales = None, adds = None, weights = None, kappa = 2):
	h, w, channels = np.atleast_3d(images[0]).shape
	col = [1,1,1,3,3][channels]
	n = len(images)

	outc = np.zeros([h, w, col], np.float32)
	outsigma = np.zeros([h, w, col], np.float32)
	num = np.zeros([h, w], np.uint16)


	for y in range(0, h):
		weights0 = np.ones([n, w], np.float32)
		if (channels > col):
			for i in range(0, n):
				weights0[i, :] = np.atleast_3d(images[i])[y, :, col] / 65535.0 + 0.0000001

		#print(weights)
		if (weights is not None):
			for i in range(0, n):
				weights0[i, :] *= weights[i][y, :]
        
		for c in range(0, col):
			img_row = np.empty([n, w], np.float32)
        
			for i in range(0, n):
				img_row[i, :] = np.array(np.atleast_3d(images[i])[y, :, c], dtype = np.float32)
				over_w = (img_row[i, :] - over) / (65535.0 - over)
				if scales is not None:
					over_w /= scales[i]**2
				over_w = np.clip(1.0 - over_w, 0.01, 1)
				weights0[i, :] *= over_w
				img_row[i, :] -= zero
				if scales is not None:
					img_row[i, :] *= scales[i]
				if adds is not None:
					img_row[i, :] += adds[i]


				#over_w = (np.clip(img_row[i, :], zero, over) - zero) / (over - zero)
				#over_w **= 2
				#over_w = over_w * (1 - over_w) * 10
				#over_w = np.clip(over_w, 0.000000001, 3)
				#weights0[i, :] *= over_w
				
			cur_weights = weights0
			
			for it in range(0,10):
				avg = np.average(img_row, weights = cur_weights, axis = 0)
				delta_sq = (img_row - avg)**2
				variance = np.average(delta_sq, weights=cur_weights, axis = 0)
				
				clip = np.ones([n, w], np.float32)
				#clip = 1.0 / (1 + delta_sq / np.tile(variance + 1, (n,1)))
				clip[np.where(delta_sq > np.tile(variance * kappa**2, (n,1)))] = 0
				
				cur_weights = weights0 * clip
			
			#print(y, c)
			#print "avg", avg
			#print "sigma", variance ** 0.5
        
			outc[y, :, c] = avg

			outsigma[y, :, c] = variance ** 0.5
		num[y, :] = np.sum(weights0, axis = 0)

	return outc, num, outsigma

class ExpDiff:
	def __init__(self, img, zero, low, high, bg_dist = 0, name = 0):

	
		img = np.array(img, dtype = np.float32)
		img = cv2.GaussianBlur(img, (5, 5), 0)
	
		img[np.where(img <= 0)] = 0

		dilkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
	
		dil = cv2.dilate(img, dilkernel)

		mask = cv2.compare(img, dil, cv2.CMP_GE)

		mask[np.where(dil <= low)] = 0
		mask[np.where(dil > high)] = 0

		if bg_dist > 0:
			ekernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
			bg = cv2.erode(img, ekernel)
			mask[np.where(bg <= zero)] = 0
			where = np.where(mask > 0)
			mask[where] = np.where((dil[where] - zero) / (bg[where] - zero) < bg_dist, 0, 255)
	
		tifffile.imsave("mask_%d.tif" % name, mask)

		self.where = np.argwhere(mask > 0)
		
		mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
		dil[np.where(mask == 0)] = 0
		self.dil = dil
		self.zero = zero
		
		
	def min_level(self):
		return np.min(self.dil[(self.where[:,0], self.where[:,1])])


	def diff(self, other):
		pts = np.array([[self.dil[y,x] - self.zero, other.dil[y,x] - other.zero] for y,x in self.where if other.dil[y,x] > other.zero])
		
		lma = np.log(pts[:,1]/pts[:,0])
		w = pts[:,0] * pts[:,1]
		lm = np.average(lma, weights = w)
		m = np.exp(lm)
		b = 0
		for i in range(5):
			w2 = 1 / (pts[:,0] * pts[:,1] + 1)
			w2[np.where(w == 0)] = 0
			if np.sum(w2) == 0:
				break
			#b = np.average(pts[:,1] - m * pts[:,0], weights = w2)
			print(m, b)
			ma = (pts[:,1] - b)/pts[:,0]
			ma[np.where(ma <= 0)] = 0.00000001
			lma = np.log(ma)

			diff2 = (lma - lm)**2
			s2 = np.average(diff2, weights = w)
			w = pts[:,0] * pts[:,1]
			w[np.where(diff2 > s2 * 9)] = 0
			if np.sum(w) == 0:
				break
			lm = np.average(lma, weights = w)
			m = np.exp(lm)
		
		print(m, b)

		#pts = pts[np.where(w > 0)]
		#import matplotlib.pyplot as plt
		#plt.plot(pts[:,0], pts[:,1], 'ro')
		#plt.plot([0, 100],  [b, 100*m + b])
		#plt.show()
		return m, b, len(np.argwhere(w> 0))

	
	
	


def noise_level(src):
	blur = cv2.GaussianBlur(src,(11, 11),0)

	noise = cv2.subtract(src, blur, dtype = cv2.CV_64FC1)
	noise2 = cv2.pow(noise, 2)

	var = cv2.mean(noise2)
	try:
		var = var.get()
	except:
		pass
	var = np.amax(var)
	for i in range(0, 10):
		print(var ** 0.5)
		mask = cv2.inRange(noise2, 0, var * 4)
		#mask = np.amax(np.atleast_3d(mask), axis = 2)
		var = cv2.mean(noise2, mask)
		try:
			var = var.get()
		except:
			pass
		var = np.amax(var)

	return var ** 0.5


def extrapolate_transp(img, transpmask, add = False):
	img = np.array(img)
	transpmask = cv2.compare(transpmask, 0, cv2.CMP_GT)
	
	transpmask = cv2.erode(transpmask, np.ones((3,3), dtype=np.uint8))
	
	bgtranspmask = np.array(transpmask, copy = True)

	
	i = 0
	while(np.amin(bgtranspmask) == 0):
		bgblur = np.array(img, copy=True, dtype=np.float64)

		bgblurmask = np.ones((bgblur.shape[0], bgblur.shape[1]))
		transp = np.where(bgtranspmask == 0)
		bgblur[transp] = 0
		bgblurmask[transp] = 0

		bgblur = cv2.blur(bgblur, (5 + i * i, 5 + i * i))
		bgblurmask = cv2.blur(bgblurmask, (5 + i * i, 5 + i * i))
		
		upd = cv2.compare(bgblurmask, 0.5, cv2.CMP_GT)
		upd = cv2.bitwise_and(upd, cv2.bitwise_not(bgtranspmask))
		upd = np.where(upd > 0)
		
		for c in range(0, img.shape[2]):
			if add:
				img[:,:,c][upd] = np.amax([np.atleast_3d(bgblur)[:,:,c][upd] / bgblurmask[upd], img[:,:,c][upd]], axis = 0)
			else:
				img[:,:,c][upd] = np.atleast_3d(bgblur)[:,:,c][upd] / bgblurmask[upd]
			bgtranspmask[upd] = 255
		
		i += 1
	return img
	


if __name__ == "__main__":
	a = np.array([[1, 2, 3], [4, 5,6], [7, 8, 9]])
	print(poly_fit(a, order = 2))