'''Benchmarks using /usr/bin/time'''

import os
import sparse_dot

cmd = '/usr/bin/time --format="time: %E peak_memory: %MkB" python3 -c "import sparse_dot.testing_utils as t; print(t.run_timing_test_cython_csr_or_coo({}, {}, {}, {}))"'

timing_test_params = [(1000, 1000, 100000),
                      (2000, 2000, 100000),
                      (5000, 5000, 100000),
                      (10000, 10000, 100000),
                      (10000, 10000, 1000000),
                      (10000, 10000, 10000000),
                      (1000, 20000, 10000000),
                      (5000, 20000, 10000),
                     ]

for num_rows, num_cols, num_entries in timing_test_params:
    for algo in ['cython', 'csr', 'coo']:
        os.system(cmd.format("'{}'".format(algo), num_rows, num_cols, num_entries))
