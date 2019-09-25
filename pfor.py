# Copyright (C) 2017 Vladimir Nadvornik
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import queue
import threading
from line_profiler import LineProfiler


def execute(q, func, lock):
	profiler = LineProfiler()
	profiler.add_function(func)
	profiler.enable_by_count()
	res = False

	while True:
		try:
			i = q.get(block = False)
		except:
			break
		func(i, lock)
		res = True
#	if res:
#		with lock:
#			profiler.print_stats()

def pfor(func, r, threads = 8):
	lock = threading.Lock()
	q = queue.Queue()
	for i in r:
		q.put(i)
	
	tt = []
	for i in range(threads):
		if threads == 1:
			execute(q,func, lock)
		else:
			t = threading.Thread(target=execute, args = (q,func, lock))
			t.start()
			tt.append(t)
	
	for t in tt:
		t.join()
		



if __name__ == "__main__":
	import time
	def proc(i, lock):
		print("start %d" % i)
		time.sleep(2)
		print("stop %d" % i)
	
	pfor(proc, list(range(0, 10)))