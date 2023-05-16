import os
import warnings
import config
import sys
import getopt

import compute as comp
import plot
import metrics as met


def generate_configs(experiment):
    all_confs = []
    setups = []
    nb_repeats = 50
    taumax = 2
    pc_alpha = 0.05

    if experiment == "1":
        functional_form = "linear"
        ci_test = 'parcorr_mult'
        frac_observed = 1.
        N = 2
        L = 1
        K_time = 1
        K_space = 1
        for sample_size in [10, 20, 30, 50, 60, 70, 100, 150, 200]:
            for nb_domains in [5, 10, 20, 30, 50, 60, 100, 150, 200]:
                para_setup = (experiment, sample_size, nb_domains, N, L, K_time, K_space,
                              frac_observed, functional_form,
                              ci_test, pc_alpha, taumax, nb_repeats)
                setups += [para_setup]

                name = '%s-' * len(para_setup) % para_setup
                name = name[:-1]
                all_confs += [(nb_repeats, name)]

    elif experiment == "2":
        functional_form = "nonlinear"
        ci_test = 'gpdc'
        frac_observed = 1.
        N = 2
        L = 1
        K_time = 1
        K_space = 1
        for sample_size in [20]:  # [10, 20, 30, 50, 60, 70, 100, 150, 200]:
            for nb_domains in [5, 10, 20, 30, 50, 60]:  # , 100, 150, 200]:
                para_setup = (experiment, sample_size, nb_domains, N, L, K_time, K_space,
                              frac_observed, functional_form,
                              ci_test, pc_alpha, taumax, nb_repeats)
                setups += [para_setup]

                name = '%s-' * len(para_setup) % para_setup
                name = name[:-1]
                all_confs += [(nb_repeats, name)]
    else:
        functional_form = "linear"
        ci_test = 'parcorr_mult'
        for sample_size in [10, 20, 30, 50, 60, 70, 100, 150, 200]:
            for nb_domains in [5, 10, 20, 30, 50, 60, 100, 150, 200]:
                for N in [5]:
                    if N == 2:
                        L = 1
                    else:
                        L = N
                    for K_time in [2]:
                        for K_space in [2]:
                            for frac_observed in [0.5, 1.]:
                                para_setup = (experiment, sample_size, nb_domains, N, L, K_time, K_space,
                                              frac_observed, functional_form,
                                              ci_test, pc_alpha, taumax, nb_repeats)
                                setups += [para_setup]

                                name = '%s-' * len(para_setup) % para_setup
                                name = name[:-1]
                                all_confs += [(nb_repeats, name)]

    return all_confs, setups


def check_existing_configs(all_configs, mypath, overwrite=False):
    print(mypath)
    current_results_files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    print(current_results_files)
    already_there = []
    configs = []
    for conf in all_configs:
        if conf not in configs:
            name = conf[1].replace("'", "")
            if (overwrite is False) and (name + '.dat' in current_results_files):
                already_there.append(conf)
                pass
            else:
                configs.append(conf)

    print("number of existing configs ", len(already_there))
    return configs


def parse_arguments(argv):
    arg_experiment = 0
    arg_help = "{0} -e <experiment>".format(
        argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "he:",
                                   ["help", "experiment="])
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-e", "--experiment"):
            arg_experiment = arg
    return arg_experiment


if __name__ == "__main__":
    exp_type = parse_arguments(sys.argv)

    if exp_type == 0:
        warnings.warn("WARNING: Computing all experimental results. This might take a while!")

    all_configurations, mypath = generate_configs(exp_type)
    foldername = config.home_path + config.interim_result_path
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    configurations = check_existing_configs(all_configurations, foldername)

    num_configs = len(configurations)
    print("number of todo configs ", num_configs)
    if num_configs == 0:
        warnings.warn("No configs to do...")

    # run experiments with the different parameter configurations
    for index, config in enumerate(configurations):
        config_string = str(config[1]).replace(',', '').replace('"', '')
        comp.assemble_experiments(config[0], [config_string])

    # calculate metrics on the obtained results
    met.calculate_all_metrics(all_configurations)

    # plot
    plot.generate_all_plots(all_configurations)