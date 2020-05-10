import numpy as np
import cv2
import tifffile


def line_filters(r):
	res = []
	for i in range(0, 3 * r):
		a = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.uint8)
	
		s = i / 3.0 / r * np.pi
		rs = int(round(r * 50 * np.sin(s)))
		rc = int(round(r * 50 * np.cos(s)))
		cv2.line(a,(r - rs, r - rc), (r + rs, r + rc), (1), 1)
		#a /= np.sum(a)
		#a -= np.mean(a)
		res.append(a)
	return res

line_filters_cache = None

def detect_lines(dif):
	global line_filters_cache
	if line_filters_cache is None:
		line_filters_cache = line_filters(10)

	a = 30
	
	mean = cv2.blur(dif, (a, a))
	mean = cv2.blur(mean, (a, a))
	dif = cv2.subtract(mean, dif)
	
	dif2 = cv2.pow(dif, 2)
	var = cv2.blur(dif2, (a, a))
	
	stddev = cv2.pow(var, 0.5)
	
	dif = cv2.medianBlur(dif, 3)
	dif = cv2.medianBlur(dif, 3)
	
	dmax = None
	for l in line_filters_cache:
		l1 = np.array(l, dtype=np.float32) / np.sum(l) * 3
		l1 -= cv2.flip(cv2.transpose(l1), flipCode=0)
		
		dl = cv2.filter2D(dif, -1, l1)
		#dl = cv2.dilate(dl, l, 1)
		dl = cv2.erode(dl, l, 1)
		
		#cv2.imshow("l", dl.get() * 100)
		#cv2.waitKey(1000)
		
		if dmax is None:
			dmax = dl
		else:
			dmax = cv2.max(dmax, dl)
	
	#dif = cv2.subtract(dmax, dmin)
	dif = dmax

	dif = cv2.medianBlur(dif, 3)




	
	mask = cv2.compare(dif, stddev, cv2.CMP_GT)
	
	mask_np = mask.get()
	#cv2.imshow("e", mask)
	
	lines = cv2.HoughLines(mask,1,np.pi/360, 400) 

	try:
		lines = lines.get()
	except:
		pass


	res = []
	
	if lines is not None:
		points = np.where(mask_np)
		points = np.array([points[1], points[0]]).T
		
		for ((r,theta),) in lines:
			a = np.cos(theta) 
			b = np.sin(theta) 
			x0 = a*r 
			y0 = b*r 
			x1 = int(x0 + 100000*(-b)) 
			y1 = int(y0 + 100000*(a)) 
		
			l = fit_line(points, (x0, y0, x1, y1), 30)
			l = fit_line(points, l, 10)
			
			res.append(l)
	print("lines1", res)

	keep = [True] * len(res)
	
	for i in range(len(res)):
		for j in range(i + 1, len(res)):
			#print(res[i], res[j])
			if np.amax(np.abs(np.array(res[i]) - np.array(res[j]))) < 1e-5:
				keep[j] = False
				
			dx1 = res[i][2] - res[i][0]
			dy1 = res[i][3] - res[i][1]

			dx2 = res[j][2] - res[j][0]
			dy2 = res[j][3] - res[j][1]
			#print(dx1,dy1,dx2,dy2)
			
			c = (dx1 * dx2 + dy2 * dy1) / (dx1 * dx1 + dy1 * dy1)**0.5 / (dx2 * dx2 + dy2 * dy2)**0.5
			print("cos ", c)
			if np.abs(c) < 0.01:
				keep[i] = False
				keep[j] = False
	print("keep", keep)
	res = [l for i,l in enumerate(res) if keep[i]]
	print("lines2", res)

	return res, mask_np
	


def fit_line(points, line, max_dist):

	x1, y1, x2, y2 = line

	x0 = points[:, 0]
	y0 = points[:, 1]
	
	dx = x2 - x1
	dy = y2 - y1
	
	dist = np.abs((dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / (dx * dx  + dy * dy)**0.5)
	
	
	points = points[dist < max_dist, :]


	#x0 = points[:, 0]
	#y0 = points[:, 1]
	#dist = (dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / (dx * dx  + dy * dy)**0.5
	
	
	vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L1, 0, 0.01, 0.01)
	
	return float(x0), float(y0), float(x0+vx*1000), float(y0+vy*1000)
	
	
def draw_lines(img, lines, col=(0,)):

	for (x0,y0,x1,y1) in lines: 
		dx = x1 - x0
		dy = y1 - y0
		
		
		xa = int(x0 + 100*(dx)) 
		ya = int(y0 + 100*(dy)) 
		xb = int(x0 - 100*(dx)) 
		yb = int(y0 - 100*(dy)) 
		cv2.line(img,(xa,ya), (xb,yb), col, 10) 
      


if __name__ == "__main__":
	dif = cv2.UMat(0.5 - tifffile.imread('dif03_0001.tif'))


	lines, mask = detect_lines(dif)

	mask = 255 - mask
	draw_lines(mask, lines, 127)
	cv2.imshow("e", mask)

	cv2.waitKey(0)



#fit_line(np.array([[0, 1], [1, 0], [1, 0.1]]), (0, 1, 1, 0))
