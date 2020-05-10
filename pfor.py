# Copyright (C) 2017 Vladimir Nadvornik
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import queue
import threading
from line_profiler import LineProfiler
import sys
import traceback
import time
import gc

def stacktraces():
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n#\n# ThreadID: %s" % threadId)
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))

    print("\n".join(code))


def execute(q, func, lock, i):
	profiler = LineProfiler()
	profiler.add_function(func)
	profiler.enable_by_count()
	res = False

	while True:
		try:
			task = q.get(block = False)
		except:
			break
		func(task, lock)
		res = True
		if i == 0:
			gc.collect()
	if res:
		with lock:
			profiler.print_stats()
	stacktraces()

def pfor(func, r, threads = 6):
	lock = threading.Lock()
	q = queue.Queue()
	for i in r:
		q.put(i)
	
	tt = []
	for i in range(threads):
		if threads == 1:
			execute(q,func, lock, i)
		else:
			t = threading.Thread(target=execute, args = (q,func, lock, i))
			t.start()
			tt.append(t)
			time.sleep(1)
	
	for t in tt:
		t.join()
		



if __name__ == "__main__":
	import time
	def proc(i, lock):
		print("start %d" % i)
		time.sleep(2)
		print("stop %d" % i)
	
	pfor(proc, list(range(0, 10)))