import os

import numpy as np
import pickle
import config
import sys

import run_experiments as run
import compute as comp


def get_results(para_setup):
    name_string = '%s-' * len(para_setup)  # % para_setup
    name_string = name_string[:-1]
    folder_name = config.home_path + config.interim_result_path
    file_name = folder_name + name_string % tuple(para_setup)
    new_file_name = file_name.replace("'", "").replace('"', '') + '.dat'
    try:
        with open(new_file_name, 'rb') as f:
            results = pickle.load(f)

    except FileNotFoundError:
        print('failed ', new_file_name)
        return None

    return results


def get_counts(para_setup):
    results = get_results(para_setup)

    if results is not None:
        metrics_dict = calculate_metrics(results, para_setup)
        return metrics_dict
    else:
        return None


def _get_match_score(true_link, pred_link):
    if true_link == "" or pred_link == "": return 0
    count = 0
    # If left edgemark is correct add 1
    if true_link[0] == pred_link[0]:
        count += 1
    # If right edgemark is correct add 1
    if true_link[2] == pred_link[2]:
        count += 1
    return count


def get_system_nodes(graphs, n_sys_nodes):
    return graphs[:, :n_sys_nodes, :n_sys_nodes, :]


def calculate_metrics(results, para_setup, boot_samples=200):
    match_func = np.vectorize(_get_match_score, otypes=[int])

    true_graphs = results['true_graphs']
    n_realizations, N, N, max_lag = true_graphs.shape
    experiment, sample_size, nb_domains, n_sys_nodes, L, K_time, K_space, frac_observed, \
        functional_form, ci_test, pc_alpha, taumax, nb_repeats = para_setup

    def get_masks(n_nodes, n_sys_nodes=1, n_obs_context_nodes=0):
        cross_mask = np.repeat(np.identity(n_nodes).reshape((n_nodes, n_nodes, 1)) == False, max_lag, axis=2).astype(
            'bool')
        cross_mask[range(n_nodes), range(n_nodes), 0] = False
        contemp_cross_mask_tril = np.zeros((n_nodes, n_nodes, max_lag)).astype('bool')
        contemp_cross_mask_tril[:, :, 0] = np.tril(np.ones((n_nodes, n_nodes)), k=-1).astype('bool')

        system_mask = np.repeat(np.identity(n_nodes).reshape((n_nodes, n_nodes, 1)) == False, max_lag, axis=2).astype(
            'bool')
        system_mask[:n_sys_nodes, :n_sys_nodes, :max_lag] = False
        if n_nodes > n_sys_nodes + n_obs_context_nodes:
            # dummies exist
            system_mask[n_sys_nodes + n_obs_context_nodes:, :, range(max_lag)] = False
            system_mask[:, n_sys_nodes + n_obs_context_nodes:, range(max_lag)] = False
        system_mask[n_sys_nodes:, n_sys_nodes:, range(max_lag)] = False

        any_mask = np.ones((n_nodes, n_nodes, max_lag)).astype('bool')
        any_mask[:, :, 0] = contemp_cross_mask_tril[:, :, 0]
        contemp_cross_mask_tril = np.repeat(contemp_cross_mask_tril.reshape((1, n_nodes, n_nodes, max_lag)),
                                            n_realizations, axis=0)
        any_mask = np.repeat(any_mask.reshape((1, n_nodes, n_nodes, max_lag)), n_realizations, axis=0)
        system_mask = np.repeat(system_mask.reshape((1, n_nodes, n_nodes, max_lag)), n_realizations, axis=0)
        return cross_mask, any_mask, contemp_cross_mask_tril, system_mask

    metrics_dict = dict()

    for suffix in ['_jpcmci', '_expr_context_only', '_expr_context_and_dummy', '_dummy', '_system_only']:
        suffix2 = '_sys_nodes'
        cross_mask, any_mask, contemp_cross_mask_tril, _ = get_masks(n_sys_nodes)
        comp_graphs = get_system_nodes(true_graphs, n_sys_nodes)
        pred_graphs = np.array([results['res' + suffix][i]['graph'] for i in range(n_realizations)])
        pred_graphs = get_system_nodes(pred_graphs, n_sys_nodes)
        mask = any_mask

        computation_time = results['comptime' + suffix]

        if suffix == '_jpcmci':
            print("comp_graphs",comp_graphs)
            print("pred_graphs", pred_graphs)

        nom = ((comp_graphs == "") * (pred_graphs != "") * mask).sum(axis=(1, 2, 3))
        metrics_dict['adj_fpr' + suffix + suffix2] = (
            nom.reshape(n_realizations),
            ((comp_graphs == "") * mask).sum(axis=(1, 2, 3)).reshape(n_realizations))
        metrics_dict['adj_tpr' + suffix + suffix2] = (
            ((comp_graphs != "") * (pred_graphs != "") * mask).sum(axis=(1, 2, 3)).reshape(n_realizations),
            ((comp_graphs != "") * mask).sum(axis=(1, 2, 3)).reshape(n_realizations))

        metrics_dict['adj_precision' + suffix + suffix2] = (
            ((comp_graphs != "") * (pred_graphs != "") * mask).sum(axis=(1, 2, 3)).reshape(n_realizations),
            ((pred_graphs != "") * mask).sum(axis=(1, 2, 3)).reshape(n_realizations))
        metrics_dict['adj_recall' + suffix + suffix2] = (
            ((comp_graphs != "") * (pred_graphs != "") * mask).sum(axis=(1, 2, 3)).reshape(n_realizations),
            ((comp_graphs != "") * mask).sum(axis=(1, 2, 3)).reshape(n_realizations))

        metrics_dict['edgemarks_precision' + suffix + suffix2] = (
            (match_func(comp_graphs, pred_graphs) * mask).sum(axis=(1, 2, 3)).reshape(n_realizations),
            2. * ((pred_graphs != "") * mask).sum(axis=(1, 2, 3)).reshape(n_realizations))
        metrics_dict['edgemarks_recall' + suffix + suffix2] = (
            (match_func(comp_graphs, pred_graphs) * mask).sum(axis=(1, 2, 3)).reshape(n_realizations),
            2. * ((comp_graphs != "") * mask).sum(axis=(1, 2, 3)).reshape(n_realizations))

        metrics_dict['computation_time' + suffix + suffix2] = (
            np.mean(np.array(computation_time)), np.percentile(np.array(computation_time), [5, 95]))

    for met in metrics_dict.keys():
        if 'computation_time' not in met:
            numerator, denominator = metrics_dict[met]
            metric_boot = np.zeros(boot_samples)
            for b in range(boot_samples):
                # Store the unsampled values in b=0
                rand = np.random.randint(0, n_realizations, n_realizations)
                metric_boot[b] = numerator[rand].sum() / denominator[rand].sum()

            metrics_dict[met] = (numerator.sum() / denominator.sum(), metric_boot.std())

    return metrics_dict


def calculate_all_metrics(all_configurations):
    for conf in all_configurations:
        config_string = str(conf[1]).replace(',', '').replace('"', '')
        foldername = config.home_path + config.interim_result_path
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        file_name = foldername + config_string
        file_name_metrics_cleaned = file_name.replace("'", "").replace('"', '') + '_metrics.dat'

        para_setup = comp.unpack_params((config_string, ""))
        metrics = get_counts(para_setup)
        if metrics is not None:
            for metric in metrics:
                file = open(file_name_metrics_cleaned, 'wb')
                pickle.dump(metrics, file, protocol=-1)
                file.close()


if __name__ == '__main__':
    exp_type = run.parse_arguments(sys.argv)
    all_configurations, mypath = run.generate_configs(exp_type)

    calculate_all_metrics(all_configurations)