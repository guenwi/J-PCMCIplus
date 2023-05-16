import os

import numpy as np
import time
import sys
import pickle
from numpy.random import SeedSequence
import getopt

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.toymodels import structural_causal_processes as toys

from parcorr_mult_new import ParCorrMultNew
from gpdc_new import GPDCNew
import generate_data as gd
import config
from j_pcmci import J_PCMCI


def unpack_params(params_str):
    para_setup_string, sam = params_str
    para_setup_string = para_setup_string.replace("'", "")
    paras = para_setup_string.split('-')
    experiment = int(paras[0])
    sample_size = int(paras[1])
    nb_domains = int(paras[2])
    N = int(paras[3])
    L = int(paras[4])
    K_time = int(paras[5])
    K_space = int(paras[6])
    frac_observed = float(paras[7])
    functional_form = str(paras[8])
    ci_test = str(paras[9])
    pc_alpha = float(paras[10])
    tau_max = int(paras[11])
    nb_repeats = int(paras[12])

    return experiment, sample_size, nb_domains, N, L, K_time, K_space, frac_observed, \
        functional_form, ci_test, pc_alpha, tau_max, nb_repeats


def define_nodes(T, ens, N, nb_context_nodes):
    process_vars = list(range(N))  # [0, 1]
    time_dummy = list(range(N + nb_context_nodes, N + nb_context_nodes + T))
    space_dummy = list(range(N + nb_context_nodes + T, N + nb_context_nodes + T + ens))
    return process_vars, time_dummy, space_dummy


def embed_dummy(T, ens, i):
    time = np.identity(T)
    space = np.zeros((T, ens))
    space[:, i] = 1.
    return time, space


def select_cond_ind_test(functional_form):
    if functional_form == "nonlinear":
        cond_ind_test = GPDCNew(significance='analytic', gp_params=None)
    else:
        cond_ind_test = ParCorrMultNew(significance='analytic')
    return cond_ind_test


def j_pcmci(data, N, spatial_context_nodes, temporal_context_nodes, tau_max, pc_alpha, functional_form):
    print("### J-PCMCI+ ###")
    ens = len(data)
    T, _ = data[0].shape

    # Define vector-valued variables including dummy variables as well as expressive context vars
    nb_context_nodes = len(spatial_context_nodes) + len(temporal_context_nodes)
    process_vars, time_dummy, space_dummy = define_nodes(T, ens, N, nb_context_nodes)

    data_dict = {}
    for i in range(ens):
        time, space = embed_dummy(T, ens, i)
        data_dict[i] = np.hstack((data[i], time, space))

    vector_vars = {i: [(i, 0)] for i in process_vars + temporal_context_nodes + spatial_context_nodes}
    vector_vars[N + nb_context_nodes] = [(i, 0) for i in time_dummy]
    vector_vars[N + nb_context_nodes + 1] = [(i, 0) for i in space_dummy]

    dataframe = pp.DataFrame(
        data=data_dict,
        vector_vars=vector_vars,
        analysis_mode='multiple',
    )

    cond_ind_test = select_cond_ind_test(functional_form)

    jpcmci = J_PCMCI(dataframe=dataframe,
                     cond_ind_test=cond_ind_test,
                     verbosity=1, time_context_nodes=temporal_context_nodes,
                     space_context_nodes=spatial_context_nodes,
                     time_dummy=N + nb_context_nodes, space_dummy=N + nb_context_nodes + 1)

    res = jpcmci.run_pcmci_jci(tau_min=0,
                               tau_max=tau_max,
                               pc_alpha=pc_alpha)
    return res


def expr_context_only_pcmci(data, N, spatial_context_nodes, temporal_context_nodes, tau_max, pc_alpha, functional_form):
    print("### PCMCI+ with observed contexts ###")
    # Use only expressive context vars
    T, _ = data[0].shape
    process_vars = list(range(N))

    link_assumptions = {}
    for j in process_vars:
        link_assumptions[j] = {(i, -tau): 'o?o' for i in process_vars
                               for tau in range(0, tau_max + 1)}
        link_assumptions[j].update({(i, -tau): '-?>' for i in temporal_context_nodes
                                    for tau in range(0, tau_max + 1)})
        for i in spatial_context_nodes:
            link_assumptions[j][(i, 0)] = '-?>'

    for c in spatial_context_nodes + temporal_context_nodes:
        link_assumptions[c] = {(i, 0): 'o?o' for i in spatial_context_nodes + temporal_context_nodes if i != c}
    for c in temporal_context_nodes:
        link_assumptions[c].update({(i, -tau): 'o?o' for i in temporal_context_nodes if i != c
                                    for tau in range(1, tau_max + 1)})

    dataframe = pp.DataFrame(
        data=data,
        analysis_mode='multiple',
    )

    cond_ind_test = select_cond_ind_test(functional_form)

    pcmci = PCMCI(dataframe=dataframe,
                  cond_ind_test=cond_ind_test,
                  verbosity=0)
    results = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=tau_max,
        link_assumptions=link_assumptions,
        pc_alpha=pc_alpha)

    return results


def expr_context_and_dummy_pcmci(data, N, spatial_context_nodes,
                                 temporal_context_nodes, tau_max, pc_alpha, functional_form):
    print("### PCMCI+ with observed contexts and dummies ###")
    # Use dummy variables as well as expressive context vars
    # essentially same as only including the dummy only, but in practice not because of order-dependence of PC!!
    ens = len(data)
    T, _ = data[0].shape

    nb_context_nodes = len(spatial_context_nodes) + len(temporal_context_nodes)
    process_vars, time_dummy, space_dummy = define_nodes(T, ens, N, nb_context_nodes)

    data_dict = {}
    for i in range(ens):
        time, space = embed_dummy(T, ens, i)
        data_dict[i] = np.hstack((data[i], time, space))

    link_assumptions = {}
    for j in process_vars:
        link_assumptions[j] = {(i, -tau): 'o?o' for i in process_vars
                               for tau in range(0, tau_max + 1)}
        link_assumptions[j].update({(i, -tau): '-?>' for i in temporal_context_nodes
                                    for tau in range(0, tau_max + 1)})
        for i in spatial_context_nodes:
            link_assumptions[j][(i, 0)] = '-?>'
        link_assumptions[j][(N + nb_context_nodes, 0)] = '-?>'
        link_assumptions[j][(N + nb_context_nodes + 1, 0)] = '-?>'

    for c in spatial_context_nodes + temporal_context_nodes:
        link_assumptions[c] = {(i, 0): 'o?o' for i in spatial_context_nodes + temporal_context_nodes if i != c}
        link_assumptions[c][(N + nb_context_nodes, 0)] = '-?>'
        link_assumptions[c][(N + nb_context_nodes + 1, 0)] = '-?>'
    for c in temporal_context_nodes:
        link_assumptions[c].update({(i, -tau): 'o?o' for i in temporal_context_nodes if i != c
                                    for tau in range(1, tau_max + 1)})

    link_assumptions[N + nb_context_nodes] = {}
    link_assumptions[N + nb_context_nodes + 1] = {}

    vector_vars = {i: [(i, 0)] for i in process_vars + temporal_context_nodes + spatial_context_nodes}
    vector_vars[N + nb_context_nodes] = [(i, 0) for i in time_dummy]
    vector_vars[N + nb_context_nodes + 1] = [(i, 0) for i in space_dummy]

    dataframe = pp.DataFrame(
        data=data_dict,
        vector_vars=vector_vars,
        analysis_mode='multiple',
    )

    cond_ind_test = select_cond_ind_test(functional_form)

    pcmci = PCMCI(dataframe=dataframe,
                  cond_ind_test=cond_ind_test,
                  verbosity=0)
    results = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=tau_max,
        link_assumptions=link_assumptions,
        pc_alpha=pc_alpha)

    return results


def dummy_pcmci(data, N, tau_max, pc_alpha, functional_form):
    print("### PCMCI+ with dummies ###")
    # Use only the dummy
    ens = len(data)
    T, _ = data[0].shape

    process_vars, time_dummy, space_dummy = define_nodes(T, ens, N, 0)

    data_dict = {}
    for i in range(ens):
        time, space = embed_dummy(T, ens, i)
        data_dict[i] = np.hstack((data[i][:, :N], time, space))

    link_assumptions = {}
    for j in process_vars:
        link_assumptions[j] = {(i, -tau): 'o?o' for i in process_vars
                               for tau in range(0, tau_max + 1)}
        link_assumptions[j][(N, 0)] = '-?>'
        link_assumptions[j][(N + 1, 0)] = '-?>'

    link_assumptions[N] = {}
    link_assumptions[N + 1] = {}

    vector_vars = {i: [(i, 0)] for i in process_vars}
    vector_vars[N] = [(i, 0) for i in time_dummy]
    vector_vars[N + 1] = [(i, 0) for i in space_dummy]

    dataframe = pp.DataFrame(
        data=data_dict,
        vector_vars=vector_vars,
        analysis_mode='multiple',
    )

    cond_ind_test = select_cond_ind_test(functional_form)

    pcmci = PCMCI(dataframe=dataframe,
                  cond_ind_test=cond_ind_test,
                  verbosity=0)
    results = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=tau_max,
        link_assumptions=link_assumptions,
        pc_alpha=pc_alpha)

    return results


def system_only_pcmci(data, N, tau_max, pc_alpha, functional_form):
    print("### PCMCI+ without contexts (system only) ###")
    # Use only the system variables
    ens = len(data)
    T, _ = data[0].shape

    data_dict = {}
    for i in range(ens):
        data_dict[i] = data[i][:, :N]

    dataframe = pp.DataFrame(
        data=data_dict,
        analysis_mode='multiple',
    )

    cond_ind_test = select_cond_ind_test(functional_form)
    pcmci = PCMCI(dataframe=dataframe,
                  cond_ind_test=cond_ind_test,
                  verbosity=0)
    results = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=pc_alpha)
    return results


def calculate_graph(params_str, seed):
    transient_fraction = 0.2
    model_seed = seed
    further_seeds = model_seed.spawn(1002)
    random_state = np.random.default_rng(model_seed)
    random_state_noise = np.random.default_rng(further_seeds[-1])

    experiment, sample_size, nb_domains, N, L, K_time, K_space, frac_observed, functional_form, ci_test, \
        pc_alpha, tau_max, nb_repeats = unpack_params(params_str)

    # generate model
    true_links, links_tc, links_sc, links_sys, noises = gd.generate_random_contemp_context_model(N, K_space, K_time,
                                                                                                 tau_max, experiment,
                                                                                                 model_seed)
    temporal_context_nodes = list(links_tc.keys())
    spatial_context_nodes = list(links_sc.keys())
    # generate data from model
    data_ens, nonstationary = gd.generate_data(nb_domains, sample_size, transient_fraction, true_links, links_tc,
                                               links_sc, links_sys, noises, random_state_noise)

    i = 0
    while nonstationary and i < 1000:
        # repeat data generation to hopefully find stationary model
        random_state = np.random.default_rng(further_seeds[i])
        random_state_noise = np.random.default_rng(further_seeds[i + 1])
        true_links, links_tc, links_sc, links_sys, noises = gd.generate_random_contemp_context_model(N, K_space, K_time,
                                                                                                     tau_max,
                                                                                                     experiment,
                                                                                                     further_seeds[i])
        temporal_context_nodes = list(links_tc.keys())
        spatial_context_nodes = list(links_sc.keys())
        data_ens, nonstationary = gd.generate_data(nb_domains, sample_size, transient_fraction, true_links, links_tc,
                                                   links_sc, links_sys, noises,
                                                   random_state_noise)
        i = i + 2
    if i >= 1000:
        raise ValueError("No stationary model found")

    # decide which variables should be latent
    observed_indices_time = random_state.choice(temporal_context_nodes,
                                                size=int(frac_observed * len(temporal_context_nodes)),
                                                replace=False).tolist()
    observed_indices_space = random_state.choice(spatial_context_nodes,
                                                 size=int(frac_observed * len(spatial_context_nodes)),
                                                 replace=False).tolist()
    if experiment == 1 or experiment == 2:
        observed_indices_time = [2]
        observed_indices_space = [3]

    observed_indices_time.sort()
    observed_indices_space.sort()
    observed_indices = list(links_sys.keys()) + observed_indices_time + observed_indices_space

    data_observed = {key: data_ens[key][:, observed_indices] for key in data_ens}
    observed_temporal_context_nodes = list(range(N, N + len(observed_indices_time)))
    observed_spatial_context_nodes = list(range(N + len(observed_indices_time), N + len(observed_indices_time)
                                                + len(observed_indices_space)))

    true_graph = toys.links_to_graph(true_links, tau_max)

    # Run the different causal discovery algorithms on the data
    time_start_pcmci_jci = time.time()
    res_jpcmci = j_pcmci(data_observed, N, observed_spatial_context_nodes, observed_temporal_context_nodes,
                              tau_max, pc_alpha, functional_form)
    time_end_pcmci_jci = time.time()

    time_start_expr_context_only = time.time()
    res_expr_context_only = expr_context_only_pcmci(data_observed, N, observed_spatial_context_nodes,
                                                    observed_temporal_context_nodes, tau_max, pc_alpha, functional_form)
    time_end_expr_context_only = time.time()

    time_start_expr_context_and_dummy = time.time()
    res_expr_context_and_dummy = expr_context_and_dummy_pcmci(data_observed, N, observed_spatial_context_nodes,
                                                              observed_temporal_context_nodes,
                                                              tau_max, pc_alpha, functional_form)
    time_end_expr_context_and_dummy = time.time()

    time_start_dummy = time.time()
    res_dummy = dummy_pcmci(data_observed, N, tau_max, pc_alpha, functional_form)
    time_end_dummy = time.time()

    time_start_system_only = time.time()
    res_system_only = system_only_pcmci(data_observed, N, tau_max, pc_alpha, functional_form)
    time_end_system_only = time.time()

    # calculate the computation time for each method
    comptime_jpcmci = time_end_pcmci_jci - time_start_pcmci_jci
    comptime_expr_context_only = time_end_expr_context_only - time_start_expr_context_only
    comptime_expr_context_and_dummy = time_end_expr_context_and_dummy - time_start_expr_context_and_dummy
    comptime_dummy = time_end_dummy - time_start_dummy
    comptime_system_only = time_end_system_only - time_start_system_only

    return {
        'true_links': true_links,
        'true_graph': true_graph,
        'observed_indices_time': observed_indices_time,
        'observed_indices_space': observed_indices_space,
        'temporal_context_nodes': temporal_context_nodes,
        'spatial_context_nodes': spatial_context_nodes,

        # Method results
        'comptime_jpcmci': comptime_jpcmci,
        'comptime_expr_context_only': comptime_expr_context_only,
        'comptime_expr_context_and_dummy': comptime_expr_context_and_dummy,
        'comptime_dummy': comptime_dummy,
        'comptime_system_only': comptime_system_only,

        'res_jpcmci': res_jpcmci,
        'res_expr_context_only': res_expr_context_only,
        'res_expr_context_and_dummy': res_expr_context_and_dummy,
        'res_dummy': res_dummy,
        'res_system_only': res_system_only
    }


def process_chunks(job_list, model_seeds):
    results = {}
    num_here = len(job_list)

    time_start_process = time.time()
    for isam, config_sam in enumerate(job_list):
        results[config_sam] = calculate_graph(config_sam, model_seeds[isam])

        current_runtime = (time.time() - time_start_process) / 3600.
        current_runtime_hr = int(current_runtime)
        current_runtime_min = 60. * (current_runtime % 1.)
        estimated_runtime = current_runtime * num_here / (isam + 1.)
        estimated_runtime_hr = int(estimated_runtime)
        estimated_runtime_min = 60. * (estimated_runtime % 1.)
        print("index %d/%d: %dh %.1fmin / %dh %.1fmin:  %s" % (
            isam + 1, num_here, current_runtime_hr, current_runtime_min,
            estimated_runtime_hr, estimated_runtime_min, config_sam))
    return results


def assemble_experiments(samples, config_list):
    print("Starting to calculate ", config_list)
    time_start = time.time()

    all_configs = dict([(conf, {'results': {},
                                'true_links': {},
                                'true_graph': {},
                                'observed_indices_time': {},
                                'observed_indices_space': {},
                                'temporal_context_nodes': {},
                                'spatial_context_nodes': {},
                                'comptime_jpcmci': {},
                                'comptime_expr_context_only': {},
                                'comptime_expr_context_and_dummy': {},
                                'comptime_dummy': {},
                                'comptime_system_only': {},
                                'res_jpcmci': {},
                                'res_expr_context_only': {},
                                'res_expr_context_and_dummy': {},
                                'res_dummy': {},
                                'res_system_only': {},
                                }) for conf in config_list])

    job_list = [(conf, i) for i in range(samples) for conf in config_list]
    print("num_jobs %s" % len(job_list))

    ss = SeedSequence(12345)
    # Spawn off child SeedSequences to pass to child processes.
    model_seeds = ss.spawn(len(job_list))

    tmp = process_chunks(job_list, model_seeds)
    for conf_sam in list(tmp.keys()):
        conf = conf_sam[0]
        sample = conf_sam[1]
        all_configs[conf]['results'][sample] = tmp[conf_sam]

    print("\nsaving all configs...")

    for conf in list(all_configs.keys()):
        for key in ['res_jpcmci', 'res_system_only', 'res_expr_context_only',
                    'res_expr_context_and_dummy', 'res_dummy']:
            all_configs[conf][key] = {}
        for key in ['comptime_jpcmci', 'comptime_system_only', 'comptime_expr_context_only',
                    'comptime_expr_context_and_dummy', 'comptime_dummy', ]:
            all_configs[conf][key] = []
        all_configs[conf]['true_graphs'] = np.zeros((samples,) + all_configs[conf]['results'][0]['true_graph'].shape,
                                                    dtype='<U3')
        all_configs[conf]['true_links'] = {}

        for i in list(all_configs[conf]['results'].keys()):
            for key in ['res_jpcmci', 'res_system_only', 'res_expr_context_only',
                        'res_expr_context_and_dummy', 'res_dummy']:
                all_configs[conf][key][i] = all_configs[conf]['results'][i][key]

            for key in ['comptime_jpcmci', 'comptime_system_only', 'comptime_expr_context_only',
                        'comptime_expr_context_and_dummy', 'comptime_dummy', ]:
                all_configs[conf][key].append(
                    all_configs[conf]['results'][i][key])

            all_configs[conf]['true_graphs'][i] = all_configs[conf]['results'][i]['true_graph']
            all_configs[conf]['true_links'][i] = all_configs[conf]['results'][i]['true_links']
            for key in ['observed_indices_time', 'observed_indices_space', 'temporal_context_nodes',
                        'spatial_context_nodes']:
                all_configs[conf][key][i] = all_configs[conf]['results'][i][key]

        del all_configs[conf]['results']

        foldername = config.home_path + config.interim_result_path
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        file_name = foldername + '%s' % (conf)
        file_name_cleaned = file_name.replace("'", "").replace('"', '') + '.dat'
        file = open(file_name_cleaned, 'wb')
        pickle.dump(all_configs[conf], file, protocol=-1)
        file.close()

    time_end = time.time()
    print('Run time in hours ', (time_end - time_start) / 3600.)


def parse_arguments(argv):
    arg_experiment = 1
    arg_sample_size = 50
    arg_nb_domains = 20
    arg_n = 2
    arg_l = 1
    arg_k_time = 1
    arg_k_space = 1
    arg_frac_observed = 0.5
    arg_functional_form = "linear"
    arg_ci_test = "parcorr_mult"
    arg_pc_alpha = 0.05
    arg_tau_max = 2
    arg_nb_realisations = 50

    arg_help = "{0} -e <experiment> -ss <sample_size> -d <nb_domains> -n <nb_nodes> -l <nb_links> -kt <k_time> " \
               "-ks <k_space> -o <frac_observed> -f <functional_form> -t <ci_test> -a <pc_alpha> -tm <tau_max> " \
               "-nr <nb_realisations>".format(argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "he:n:a:em:w:s:av:hf:hd:ss:nr:p:k:",
                                   ["help", "experiment=", "sample_size=", "nb_domains=", "nb_nodes=", "nb_links=",
                                    "k_time=", "k_space=", "frac_observed=", "functional_form=", "ci_test=",
                                    "pc_alpha=", "tau_max=", "nb_realisations="])
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-e", "--experiment"):
            arg_experiment = arg
        elif opt in ("-ss", "--sample_size"):
            arg_sample_size = arg
        elif opt in ("-d", "--nb_domains"):
            arg_nb_domains = arg
        elif opt in ("-n", "--nb_nodes"):
            arg_n = arg
        elif opt in ("-l", "--nb_links"):
            arg_l = arg
        elif opt in ("-kt", "--k_time"):
            arg_k_time = arg
        elif opt in ("-ks", "--k_space"):
            arg_k_space = arg
        elif opt in ("-o", "--frac_observed"):
            arg_frac_observed = arg
        elif opt in ("-f", "--functional_form"):
            arg_functional_form = arg
        elif opt in ("-t", "--ci_test"):
            arg_ci_test = arg
        elif opt in ("-a", "--pc_alpha"):
            arg_pc_alpha = arg
        elif opt in ("-tm", "--tau_max"):
            arg_tau_max = arg
        elif opt in ("-nr", "--nb_realisations"):
            arg_nb_realisations = arg

    return int(arg_experiment), int(arg_sample_size), int(arg_nb_domains), int(arg_n), int(arg_l), \
        int(arg_k_time), int(arg_k_space), float(arg_frac_observed), arg_functional_form, arg_ci_test, \
        float(arg_pc_alpha), int(arg_tau_max), int(arg_nb_realisations)


if __name__ == '__main__':
    args = parse_arguments(sys.argv)
    experiment, sample_size, nb_domains, N, L, K_time, K_space, frac_observed, functional_form, ci_test, \
        pc_alpha, tau_max, nb_repeats = args

    name = '%s-' * len(args) % args
    name = name[:-1]
    config_string = str(name)[1:-1].replace(',', '').replace('"', '')
    assemble_experiments(nb_repeats, [config_string])