import metrics

# tests pairwise sum
try:
    metrics._pairwise_sum(None)
except TypeError:
    print("Pairwise sum won't work with None value.")

assert metrics._pairwise_sum([]) == 0.0  # empty array
assert metrics._pairwise_sum([10]) == 10.0  # one element
assert metrics._pairwise_sum([10, 20]) == 30.0  # even number of elements
assert metrics._pairwise_sum([10, 20, -30, -0.5]) == -0.5  # even number of elements
assert metrics._pairwise_sum([10, 20, -30]) == 0.0  # odd number of elements
assert metrics._pairwise_sum([10, 20, -30, -0.5, 0.5]) == 0.0  # odd number of elements

# tests calculation of variance
try:
    metrics._compute_var(None)
except TypeError:
    print("Var calculation won't work with None value.")

assert not float('-inf') < metrics._compute_var([]) < float('inf')  # empty array, not a number
assert metrics._compute_var([10]) == 0.0  # one element
assert float('-inf') < metrics._compute_var([10, 20]) < float('inf')  # even number of elements
assert float('-inf') < metrics._compute_var([10, 20, -30, -0.5]) < float('inf')  # even number of elements
assert float('-inf') < metrics._compute_var([10, 20, -30]) < float('inf')  # odd number of elements
assert float('-inf') < metrics._compute_var([10, 20, -30, -0.5, 0.5]) < float('inf')  # odd number of elements

# tests loading of CSV file
try:
    metrics.load_data('nofile.csv')
except FileNotFoundError:
    print("Can't read a file that doesn't exist.")

output = metrics.load_data('test.csv')
assert len(output) > 0
print('observations:', output)

# tests d-prime computation
try:
    metrics.compute_d_prime(None)
except TypeError:
    print("D-prime calculation won't work on None value.")

try:
    metrics.compute_d_prime([0])
except TypeError:
    print("D-prime calculation won't work on arrays not containing (<label>,<score>) elements.")

assert not float('-inf') < metrics.compute_d_prime([]) < float('inf')  # empty array, not a number
assert not float('-inf') < metrics.compute_d_prime([(0, 0.1)]) < float('inf')  # missing genuine, not a number
assert not float('-inf') < metrics.compute_d_prime([(1, 0.1)]) < float('inf')  # missing impostors, not a number

dprime = metrics.compute_d_prime(output)
assert float('-inf') < dprime < float('inf')
print('d-prime:', dprime)

# tests FMR computation
try:
    metrics.compute_sim_fmr(None, 0.0)
except TypeError:
    print("FMR calculation won't work on None value.")

try:
    metrics.compute_sim_fmr([0], 0.0)
except TypeError:
    print("FMR calculation won't work on arrays not containing (<label>,<score>) elements.")

assert not float('-inf') < metrics.compute_sim_fmr([], 0.0) < float('inf')  # empty array, not a number
assert float('-inf') < metrics.compute_sim_fmr([(0, 0.1)], 0.0) < float('inf')  # missing genuine, ok
assert not float('-inf') < metrics.compute_sim_fmr([(1, 0.1)], 0.0) < float('inf')  # missing impostors, not a number

fmr = metrics.compute_sim_fmr(output, 0.25)
assert float('-inf') < fmr < float('inf')
print('FMR @ 0.25:', fmr)

fmr = metrics.compute_sim_fmr(output, 0.5)
assert float('-inf') < fmr < float('inf')
print('FMR @ 0.50:', fmr)

fmr = metrics.compute_sim_fmr(output, 0.75)
assert float('-inf') < fmr < float('inf')
print('FMR @ 0.75:', fmr)

# tests FNMR computation
try:
    metrics.compute_sim_fnmr(None, 0.0)
except TypeError:
    print("FNMR calculation won't work on None value.")

try:
    metrics.compute_sim_fnmr([0], 0.0)
except TypeError:
    print("FNMR calculation won't work on arrays not containing (<label>,<score>) elements.")

assert not float('-inf') < metrics.compute_sim_fnmr([], 0.0) < float('inf')  # empty array, not a number
assert not float('-inf') < metrics.compute_sim_fnmr([(0, 0.1)], 0.0) < float('inf')  # missing genuine, not a number
assert float('-inf') < metrics.compute_sim_fnmr([(1, 0.1)], 0.0) < float('inf')  # missing impostors, ok

fnmr = metrics.compute_sim_fnmr(output, 0.25)
assert float('-inf') < fnmr < float('inf')
print('FNMR @ 0.25:', fnmr)

fnmr = metrics.compute_sim_fnmr(output, 0.5)
assert float('-inf') < fnmr < float('inf')
print('FNMR @ 0.50:', fnmr)

fnmr = metrics.compute_sim_fnmr(output, 0.75)
assert float('-inf') < fnmr < float('inf')
print('FNMR @ 0.75:', fnmr)

# tests FNMR and FMR at EER
try:
    metrics.compute_sim_fmr_fnmr_eer(None)
except TypeError:
    print("FNMR, FMR @ EER calculation won't work on None value.")

try:
    metrics.compute_sim_fmr_fnmr_eer([0])
except TypeError:
    print("FNMR, FMR @ EER calculation won't work on arrays not containing (<label>,<score>) elements.")

assert not float('-inf') < metrics.compute_sim_fmr_fnmr_eer([])[0] < float('inf')  # empty array, not a number
assert not float('-inf') < metrics.compute_sim_fmr_fnmr_eer([])[1] < float('inf')  # empty array, not a number
assert not float('-inf') < metrics.compute_sim_fmr_fnmr_eer([])[2] < float('inf')  # empty array, not a number

assert not float('-inf') < metrics.compute_sim_fmr_fnmr_eer([(0, 0.1)])[0] < float('inf')  # missing genuine, nan
assert not float('-inf') < metrics.compute_sim_fmr_fnmr_eer([(0, 0.1)])[1] < float('inf')  # missing genuine, nan
assert not float('-inf') < metrics.compute_sim_fmr_fnmr_eer([(0, 0.1)])[2] < float('inf')  # missing genuine, nan

assert not float('-inf') < metrics.compute_sim_fmr_fnmr_eer([(1, 0.1)])[0] < float('inf')  # missing impostors, nan
assert not float('-inf') < metrics.compute_sim_fmr_fnmr_eer([(1, 0.1)])[1] < float('inf')  # missing impostors, nan
assert not float('-inf') < metrics.compute_sim_fmr_fnmr_eer([(1, 0.1)])[2] < float('inf')  # missing impostors, nan

fnmr_fmr_eer = metrics.compute_sim_fmr_fnmr_eer(output)
assert float('-inf') < fnmr_fmr_eer[0] < float('inf')
assert float('-inf') < fnmr_fmr_eer[1] < float('inf')
assert float('-inf') < fnmr_fmr_eer[2] < float('inf')
print('FNMR, FMR, THR@ EER:', fnmr_fmr_eer)

# tests FMR x FMR AUC
try:
    metrics.compute_sim_fmr_tmr_auc(None)
except TypeError:
    print("FMR x TMR AUC calculation won't work on None value.")

try:
    metrics.compute_sim_fmr_tmr_auc([0])
except TypeError:
    print("FMR x TMR AUC calculation won't work on arrays not containing (<label>,<score>) elements.")

assert not float('-inf') < metrics.compute_sim_fmr_tmr_auc([])[0] < float('inf')  # empty array, not a number
assert metrics.compute_sim_fmr_tmr_auc([])[1] is None  # empty array, nothing to do
assert metrics.compute_sim_fmr_tmr_auc([])[2] is None  # empty array, nothing to do

assert not float('-inf') < metrics.compute_sim_fmr_tmr_auc([(0, 0.1)])[0] < float('inf')  # missing genuine, nan
assert metrics.compute_sim_fmr_tmr_auc([(0, 0.1)])[1] is None  # missing genuine, nothing to do
assert metrics.compute_sim_fmr_tmr_auc([(0, 0.1)])[2] is None  # missing genuine, nothing to do

assert not float('-inf') < metrics.compute_sim_fmr_tmr_auc([(1, 0.1)])[0] < float('inf')  # missing impostors, nan
assert metrics.compute_sim_fmr_tmr_auc([(1, 0.1)])[1] is None  # missing impostors, nothing to do
assert metrics.compute_sim_fmr_tmr_auc([(1, 0.1)])[2] is None  # missing impostors, nothing to do

auc, fmrs, tmrs = metrics.compute_sim_fmr_tmr_auc(output)
assert float('-inf') < auc < float('inf')
assert len(fmrs) > 0
assert len(tmrs) > 0
assert len(fmrs) == len(tmrs)
print('AUC:', auc)
print('FMR:', fmrs)
print('TMR:', tmrs)

# plots
metrics.plot_hist(output)  # close window to see next...
metrics.plot_sim_fmr_tmr_auc(output)
