import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import config
import run_experiments as run
import compute as comp


def get_metrics_from_file(para_setup):
    name_string = '%s-' * len(para_setup) % para_setup
    name_string = name_string[:-1]
    foldername = config.home_path + config.interim_result_path
    file_name = foldername + name_string
    file_name_cleaned = file_name.replace("'", "").replace('"', '') + '_metrics.dat'

    try:
        results = pickle.load(open(file_name_cleaned, 'rb'),
                              encoding='latin1')
    except:
        print('failed from metrics file ', file_name_cleaned)
        return None

    return results


def get_save_file_name(para_setup, plot_type):
    folder_name_fig = config.home_path + config.figure_path
    if not os.path.exists(folder_name_fig):
        os.makedirs(folder_name_fig)

    filename = '%s-' * len(para_setup) % para_setup
    filename = filename[:-1]
    filename = filename.replace("'", "")

    save_file = folder_name_fig + filename + plot_type + '.png'
    return save_file


def generate_time_conv_plot(para_setup, list_nb_timesteps):
    print(list_nb_timesteps)
    plotting_params, metric_names, metric_clean_names, method_names, linestyles, linewidth_list = get_plotting_setup()

    experiment, nb_domains, N, L, K_time, K_space, frac_observed, functional_form, ci_test, \
        pc_alpha, tau_max, nb_repeats = para_setup

    save_file = get_save_file_name(para_setup, "_time_conv")
    failed_files = []

    plt.rcParams.update(plotting_params)
    fig, axs = plt.subplots(2, 3, figsize=(25, 15))
    plt.subplots_adjust(hspace=0.4)
    axs = axs.ravel()
    j = 0
    suffix = '_sys_nodes'
    method_list = ['_jpcmci', '_system_only', '_expr_context_only', '_dummy', '_system_only']
    metric_list = ['adj_tpr', 'adj_fpr', 'adj_recall', 'adj_precision', 'edgemarks_precision', 'edgemarks_recall']

    for metric in metric_list:
        for index, method in enumerate(method_list):
            key = method + suffix
            mname = method_names[method]
            metric_pp_list = []
            metric_errors_list = []
            for sample_size in list_nb_timesteps:
                para_setup = (experiment, sample_size, nb_domains, N, L, K_time, K_space,
                              frac_observed, functional_form, ci_test, pc_alpha, tau_max, nb_repeats)
                metrics_dict = get_metrics_from_file(tuple(para_setup))

                if metrics_dict is not None:
                    metric_pp_list.append(metrics_dict[metric + key][0])
                    metric_errors_list.append(metrics_dict[metric + key][1])
                else:
                    metric_pp_list.append(0.)
                    metric_errors_list.append(0.)
                    failed_files.append('%s-' * len(para_setup) % tuple(para_setup))


            if np.isnan(metric_pp_list).any():
                print(metric_pp_list)
            else:
                axs[j].plot(list_nb_timesteps, metric_pp_list, label=mname, linestyle=linestyles[index])
                axs[j].errorbar(list_nb_timesteps, metric_pp_list, yerr=metric_errors_list, fmt='.k')
            if metric == 'adj_fpr':
                axs[j].hlines(y=0.05, xmin=list_nb_timesteps[0], xmax=list_nb_timesteps[-1], colors='lightgray',
                              linestyles='--', lw=2)

        axs[j].legend(loc="upper left")
        axs[j].set_title(str(nb_domains) + ' data-sets, \n' + str(N) + ' system nodes, ' + str(
            K_time + K_space) + ' context nodes,\n frac_observed ' + str(frac_observed))
        axs[j].set_xlabel('samplesize', fontsize=20)
        axs[j].set_ylabel(metric_clean_names[metric] + " on " + metric_names[suffix], fontsize=20)
        if metric == 'adj_tpr':
            axs[j].set_ylim([0.6, 1.])
        elif metric == 'adj_fpr':
            axs[j].set_ylim([0.0, 0.16])
        j += 1
    fig.savefig(save_file)

    return failed_files


def generate_plot(para_setup, list_nb_dom):
    plotting_params, metric_names, metric_clean_names, method_names, linestyles, linewidth_list = get_plotting_setup()
    save_file = get_save_file_name(para_setup, "_space_conv")
    failed_files = []

    experiment, sample_size, N, L, K_time, K_space, frac_observed, functional_form, ci_test, \
        pc_alpha, tau_max, nb_repeats = para_setup

    for suffix in ['_sys_nodes']:
        plt.rcParams.update(plotting_params)
        fig, axs = plt.subplots(2, 3, figsize=(25, 15))
        plt.subplots_adjust(hspace=0.4)

        metric_list = ['adj_tpr', 'adj_fpr', 'adj_recall', 'adj_precision', 'edgemarks_precision', 'edgemarks_recall']
        if suffix == '_cnt_nodes':
            method_list = ['_jpcmci', '_expr_context_only']
        else:
            method_list = ['_jpcmci', '_expr_context_only', '_dummy', '_system_only']

        axs = axs.ravel()
        j = 0
        for metric in metric_list:
            for index, method in enumerate(method_list):
                key = method + suffix
                mname = method_names[method]
                metric_pp_list = []
                metric_errors_list = []
                for nb_dom in list_nb_dom:
                    para_setup = (experiment, sample_size, nb_dom, N, L, K_time, K_space, frac_observed,
                                  functional_form, ci_test, pc_alpha, tau_max, nb_repeats)
                    metrics_dict = get_metrics_from_file(tuple(para_setup))

                    if metrics_dict is not None:
                        metric_pp_list.append(metrics_dict[metric + key][0])
                        metric_errors_list.append(metrics_dict[metric + key][1])
                    else:
                        metric_pp_list.append(0.)
                        metric_errors_list.append(0.)
                        failed_files.append('%s-' * len(para_setup) % tuple(para_setup))

                if np.isnan(metric_pp_list).any():
                    print(metric_pp_list)
                else:
                    axs[j].plot(list_nb_dom, metric_pp_list, label=mname, linestyle=linestyles[index])
                    axs[j].errorbar(list_nb_dom, metric_pp_list, yerr=metric_errors_list, fmt='.k')
                if metric == 'adj_fpr':
                    axs[j].hlines(y=pc_alpha, xmin=list_nb_dom[0], xmax=list_nb_dom[-1], colors='lightgray',
                                  linestyles='--', lw=2)

            axs[j].legend(loc='upper left')
            axs[j].set_title(str(sample_size) + ' time steps, \n' + str(N) + ' system nodes, ' + str(
                K_time + K_space) + ' context nodes,\n frac_observed ' + str(frac_observed))
            axs[j].set_xlabel('samplesize', fontsize=20)
            axs[j].set_xlabel('number of data-sets')
            axs[j].set_ylabel(metric_clean_names[metric] + " on " + metric_names[suffix], fontsize=20)
            j += 1

    fig.savefig(save_file)

    return failed_files


def generate_2Dplot(para_setup, list_nb_domains, list_samplesize):
    plotting_params, metric_names, metric_clean_names, method_names, linestyles, linewidth_list = get_plotting_setup()
    save_file = get_save_file_name(para_setup, "_2D")
    failed_files = []

    experiment, N, L, K_time, K_space, frac_observed, functional_form, ci_test, \
        pc_alpha, tau_max, nb_repeats = para_setup

    for suffix in ['_sys_nodes']:
        plt.rcParams.update(plotting_params)
        fig, axs = plt.subplots(2, 3, figsize=(25, 15), subplot_kw=dict(projection='3d'))
        axs = axs.ravel()

        j = 0
        for metric in ['adj_tpr', 'adj_fpr', 'adj_recall', 'adj_precision', 'edgemarks_precision', 'edgemarks_recall']:
            for index, method in enumerate(['_jpcmci']):

                key = method + suffix

                metric_errors_list = []

                metric_pp_list = np.zeros((len(list_samplesize), len(list_nb_domains)))
                for i_dom, nb_dom in enumerate(list_nb_domains):
                    for i_samplesize, sample_size in enumerate(list_samplesize):
                        para_setup = (experiment, sample_size, nb_dom, N, L, K_time, K_space, frac_observed,
                                      functional_form, ci_test, pc_alpha, tau_max, nb_repeats)

                        metrics_dict = get_metrics_from_file(tuple(para_setup))
                        if metrics_dict is not None:
                            metric_pp_list[i_samplesize, i_dom] = metrics_dict[metric + key][0]
                            metric_errors_list.append(metrics_dict[metric + key][1])
                        else:
                            metric_errors_list.append(0.)
                            failed_files.append('%s-' * len(para_setup) % tuple(para_setup))

                        X2, Y2 = np.meshgrid(list_nb_domains, list_samplesize)

                axs[j].plot_wireframe(Y2, X2, metric_pp_list, cmap='binary', linewidth=1.)
                if metric == 'adj_fpr':
                    axs[j].contourf(Y2, X2, metric_pp_list, vmin=0., vmax=0.05, cmap=cm.coolwarm)

                if metric == 'adj_fpr':
                    axs[j].set_zticks([0., 0.02, pc_alpha])

                axs[j].set_ylabel('data-sets M', fontsize=20, labelpad=10)
                axs[j].set_xlabel('samplesize T', fontsize=20, labelpad=10)
                axs[j].set_zlabel(metric_clean_names[metric] + " on " + metric_names[suffix], fontsize=20, labelpad=10)
            j += 1
    fig.savefig(save_file)

    return failed_files


def get_plotting_setup():
    plotting_params = {
        'legend.fontsize': 15,
        'legend.handletextpad': .05,
        'lines.color': 'black',
        'lines.linewidth': 4,
        'lines.markersize': 2,
        'xtick.labelsize': 20,
        'xtick.major.pad': 1,
        'xtick.major.size': 2,
        'ytick.major.pad': 1,
        'ytick.major.size': 2,
        'ytick.labelsize': 20,
        'axes.labelsize': 10,
        'font.size': 20,
        'axes.labelpad': 0,
        'axes.spines.right': False,
        'axes.spines.top': False,
    }

    metric_names = {'_sys_nodes': 'system links', '_all_nodes': 'all links', '_cnt_nodes': 'context links'}
    metric_clean_names = {'adj_tpr': 'adj-TPR', 'adj_fpr': 'adj-FPR',
                          'adj_recall': 'adj-Recall', 'adj_precision': 'adj-Precision',
                          'edgemarks_precision': 'edgemark-Precision', 'edgemarks_recall': 'edgemark-Recall'}
    method_names = {'_jpcmci': 'J-PCMCI+', '_expr_context_only': 'PCMCI+ with C',
                    '_expr_context_and_dummy': 'PCMCI+ with C and D', '_dummy': 'PCMCI+ with D',
                    '_system_only': 'PCMCI+'}
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1))]
    linewidth_list = [4, 2, 2, 2]

    return plotting_params, metric_names, metric_clean_names, method_names, linestyles, linewidth_list


def get_plotting_setups(all_configurations):
    setups_time_conv_plot = set([])
    setups_space_conv_plot = set([])
    setups_2D_plot = set([])

    set_nb_domains = set([])
    set_samplesize = set([])

    for config in all_configurations:
        config_string = str(config[1]).replace(',', '').replace('"', '')
        experiment, sample_size, nb_domains, N, L, K_time, K_space, frac_observed, functional_form, ci_test, \
            pc_alpha, tau_max, nb_repeats = comp.unpack_params((config_string, ""))
        setups_time_conv_plot.add(
            (experiment, nb_domains, N, L, K_time, K_space, frac_observed, functional_form, ci_test,
             pc_alpha, tau_max, nb_repeats))
        setups_space_conv_plot.add(
            (experiment, sample_size, N, L, K_time, K_space, frac_observed, functional_form, ci_test,
             pc_alpha, tau_max, nb_repeats))
        setups_2D_plot.add((experiment, N, L, K_time, K_space, frac_observed, functional_form, ci_test,
                            pc_alpha, tau_max, nb_repeats))
        set_nb_domains.add(nb_domains)
        set_samplesize.add(sample_size)

    list_nb_domains = sorted(set_nb_domains)
    list_samplesize = sorted(set_samplesize)

    return setups_time_conv_plot, setups_space_conv_plot, setups_2D_plot, list_nb_domains, list_samplesize


def generate_all_plots(all_configurations):
    setups_time_conv_plot, setups_space_conv_plot, setups_2D_plot, \
        list_nb_domains, list_samplesize = get_plotting_setups(all_configurations)

    for para_setup in setups_time_conv_plot:
        generate_time_conv_plot(para_setup, list_samplesize)

    for para_setup in setups_space_conv_plot:
        generate_plot(para_setup, list_nb_domains)

    for para_setup in setups_2D_plot:
        generate_2Dplot(para_setup, list_nb_domains, list_samplesize)


if __name__ == "__main__":
    exp_type = run.parse_arguments(sys.argv)
    all_configurations, mypath = run.generate_configs(exp_type)
    generate_all_plots(all_configurations)
