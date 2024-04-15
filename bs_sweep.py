from run import Parameters, Sweeps
import cProfile
import pstats
from pstats import SortKey

def main():
    params = Parameters()
    params.default_params['device'] = 'cpu'
    sweep = Sweeps(params.default_params)
    sweep.bs_sweep()

if __name__ == '__main__':
    # cProfile.run('main()', 'restats')
    # p = pstats.Stats('restats')
    # p.sort_stats(SortKey.CUMULATIVE).print_stats(20)
    main()