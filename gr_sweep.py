from run import Parameters, Sweeps

def main():
    params = Parameters()
    params.default_params['device'] = 'cpu'
    sweep = Sweeps(params.default_params)
    sweep.gr_sweep()

if __name__ == '__main__':
    main()