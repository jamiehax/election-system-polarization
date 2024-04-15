from run import Parameters, OpinionDynamics
import cProfile
import pstats
from pstats import SortKey

def main():
    params = Parameters()
    params.default_params['device'] = 'cpu'
    sim = OpinionDynamics(params.default_params)
    sim.run_simulation('bc', 'bc1', 'plurality')

if __name__ == '__main__':
    # cProfile.run('main()', 'restats')
    # p = pstats.Stats('restats')
    # p.sort_stats(SortKey.CUMULATIVE).print_stats(50)
    main()