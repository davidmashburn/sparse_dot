'''Benchmarks using /usr/bin/time'''

import os
import sparse_dot

cmd = '/usr/bin/time --format="time: %E peak_memory: %MkB" python -c "import sparse_dot.testing_utils as t; print t.run_timing_test_vs_sparse({}, {}, {})"'

os.system(cmd.format(1000, 1000, 100000))
os.system(cmd.format(2000, 2000, 100000))
os.system(cmd.format(5000, 5000, 100000))
os.system(cmd.format(10000, 10000, 100000))
os.system(cmd.format(10000, 10000, 1000000))
os.system(cmd.format(10000, 10000, 10000000))
os.system(cmd.format(1000, 20000, 10000000))
os.system(cmd.format(5000, 20000, 10000))
