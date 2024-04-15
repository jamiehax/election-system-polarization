from run import Parameters, Sweeps

def main():
    params = Parameters()
    sweep = Sweeps(params.default_params)
    sweep.make_figure_grids()

if __name__ == '__main__':
    main()