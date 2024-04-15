from run import Parameters, OpinionDynamics

def main():
    params = Parameters()
    params.default_params['device'] = 'cpu'
    dyn = OpinionDynamics(params.default_params)
    dyn.compare_election_system_variance()

if __name__ == '__main__':
    main()