#!/usr/bin/python3

# Copyright (C) 2017 Vladimir Nadvornik
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import cv2
import numpy as np
import io
import queue
import threading
import atexit
import time

#from stacktraces import stacktraces

class MyGUI_CV2(threading.Thread):

	def __init__(self):
		threading.Thread.__init__(self)
		self.queue = queue.Queue()
		self.key_queue = queue.Queue()
		self.stop = False
		self.daemon = True
		atexit.register(self.terminate)
		self.start()

	def namedWindow(self, name):
		self.queue.put((name, None))
	
	
	def imshow(self, name, img):
		self.queue.put((name, img))

	def run(self):
		while not self.stop:
#			stacktraces()
			key = cv2.waitKey(1)
			if key != -1:
				self.key_queue.put(key)

			try:
				(name, img) = self.queue.get(block=False)
			except queue.Empty:
				time.sleep(0.1)
				continue
			if img is None:
				cv2.namedWindow(name, cv2.WINDOW_NORMAL)
			else:
				cv2.imshow(name, img)

	def set_status(self, status):
		self.status = status

	def terminate(self):
		self.stop = True
	
	def waitKey(self, delay = 0):
		try:
			timeout = None
			if delay > 0:
				timeout = delay / 1000.0
			key = key_queue.get(block = True, timeout = timeout)
			return key
		except:
			return -1
		

cv2t = MyGUI_CV2()
