from __future__ import print_function
import numpy as np
from tigramite.pcmci import PCMCI, _nested_to_normal
from tigramite.independence_tests.parcorr_mult import ParCorrMult
from copy import deepcopy
import itertools


# assume context nodes are at the end
class J_PCMCI(PCMCI):
    def __init__(self, time_context_nodes, space_context_nodes, time_dummy=None, space_dummy=None, **kwargs):
        self.time_context_nodes = time_context_nodes
        self.space_context_nodes = space_context_nodes
        self.time_dummy = time_dummy
        self.space_dummy = space_dummy

        self.context_parents = None
        self.observed_context_parents = None
        self.dummy_ci_test = ParCorrMult(significance='analytic')
        self.nb_context_nodes = len(self.time_context_nodes) + len(self.space_context_nodes)

        PCMCI.__init__(self, **kwargs)

        self.dummy_counter = sum([time_dummy is not None, space_dummy is not None])
        self.nb_sysvars = self.N - self.nb_context_nodes - self.dummy_counter

    def run_pcmci_jci(self,
                      link_assumptions=None,
                      tau_min=0,
                      tau_max=2,
                      pc_alpha=0.05,
                      max_conds_dim=None,
                      max_combinations=1):
        # find context - system links, i.e. changing mechanisms
        context_results = self.find_changing_mechanism(link_assumptions, tau_min, max_conds_dim, max_combinations,
                                                       pc_alpha, tau_max)
        self.context_parents = context_results['parents']
        context_parents_values = context_results['values']
        if self.verbosity > 1:
            print("Discovered contextual parents: ", context_results['parents'])

        # Get the parents from run_pc_stable only on the system links
        if self.time_dummy is None:
            dummy_vars = []
        else:
            dummy_vars = [self.time_dummy]

        if self.space_dummy is not None:
            dummy_vars += [self.space_dummy]

        all_context_nodes = self.time_context_nodes + self.space_context_nodes

        if link_assumptions is not None:
            
            system_links = deepcopy(link_assumptions)
            for C in all_context_nodes + dummy_vars:
                system_links[C] = {}
            for j in range(self.nb_sysvars):
                for C in dummy_vars + all_context_nodes:
                    for lag in range(tau_max + 1):
                        if (C, -lag) in system_links[j]:
                            system_links[j].pop((C, -lag), None)
                    system_links[j].update({parent: '-->' for parent in self.context_parents[j]})
        else:
            system_links = {j: {(i, -tau): 'o?o' for i in range(self.nb_sysvars) for tau in range(1, tau_max+1)}
                            for j in range(self.nb_sysvars)}
            for j in system_links:
                print("self.context_parents[j]", self.context_parents[j])
                system_links[j].update({parent: '-->' for parent in self.context_parents[j]})

            for C in all_context_nodes + dummy_vars:
                # we are not interested in links between context variables (thus system_links[C] = {})
                system_links[C] = {}

        print("Discovering system-system links")
        results = self.run_pcmciplus_phase2(
                            lagged_parents=context_results['lagged_context_parents'],
                            tau_min=tau_min, 
                            tau_max=tau_max,
                            link_assumptions=system_links, 
                            pc_alpha=pc_alpha)
        print("Done")
        
        for c in all_context_nodes + dummy_vars:
            for j in list(range(self.nb_sysvars)) + all_context_nodes + dummy_vars:
                for lag in range(tau_max+1):
                    # add context-system links to results, these are needed for orientation
                    results['val_matrix'][c, j, lag] = context_parents_values[c, j, lag]
                    results['val_matrix'][j, c, lag] = context_parents_values[c, j, lag]
       
        # Return the dictionary
        self.results = results
        return results

    def find_changing_mechanism(self, link_assumptions, tau_min, max_conds_dim, max_combinations, pc_alpha=0.05,
                                tau_max=1):
        # Initializing
        parents = {j: [] for j in range(self.nb_sysvars + self.nb_context_nodes + self.dummy_counter)}
        lagged_context_parents = {j: [] for j in range(self.nb_sysvars + self.nb_context_nodes + self.dummy_counter)}
        values = np.zeros((self.nb_sysvars + self.nb_context_nodes + 2, self.nb_sysvars + self.nb_context_nodes + 2, tau_max+1))
        all_context_nodes = self.time_context_nodes + self.space_context_nodes

        # initialize / clean selected_links
        if link_assumptions is not None:
            _link_assumptions = deepcopy(link_assumptions)
        else:
            _link_assumptions = {j: {(i, -tau): 'o?o' for i in range(self.N)
                                     for tau in range(tau_max + 1)} for j in range(self.N)}
        # orient all context-system links such that the context variable is the parent
        for j in _link_assumptions:
            if j in range(self.nb_sysvars):
                for link in _link_assumptions[j]:
                    i, lag = link
                    if i in all_context_nodes + [self.time_dummy, self.space_dummy]: # is context var
                        link_type = _link_assumptions[j][link]
                        _link_assumptions[j][link] = '-' + link_type[1] + '>'
                        
        # remove any links where dummy is the child
        # and any lagged links to dummy, and space_context (not to expressive time context)
        # and system - context links where context is the child
        # and any links between spatial and temporal context
        if self.time_dummy is not None: _link_assumptions[self.time_dummy] = {}
        if self.space_dummy is not None: _link_assumptions[self.space_dummy] = {}

        for j in range(self.nb_sysvars + self.nb_context_nodes):
            for lag in range(1, tau_max + 1):
                for c in [self.time_dummy, self.space_dummy] + self.space_context_nodes:
                    if (c, -lag) in _link_assumptions[j]: _link_assumptions[j].pop((c, -lag), None)
        for c in all_context_nodes:
            for j in range(self.nb_sysvars):
                for lag in range(tau_max + 1):
                    if (j, -lag) in _link_assumptions[c]: _link_assumptions[c].pop((j, -lag), None)
            if (c, 0) in _link_assumptions[c]: _link_assumptions[c].pop((c, 0), None)  # remove self-links
                
        for c in self.space_context_nodes:
            for k in self.time_context_nodes:
                for lag in range(tau_max + 1):
                    if (k, -lag) in _link_assumptions[c]: _link_assumptions[c].pop((k, -lag), None)
        for c in self.time_context_nodes:
            for k in self.space_context_nodes:
                if (k, 0) in _link_assumptions[c]: _link_assumptions[c].pop((k, 0), None)

        # find links in btw expressive context, and btw expressive context and sys_vars
        # here, we exclude any links to dummy
        _link_assumptions_wo_dummy = deepcopy(_link_assumptions)
        for j in range(self.nb_sysvars + self.nb_context_nodes):
            if (self.time_dummy, 0) in _link_assumptions_wo_dummy[j]:
                _link_assumptions_wo_dummy[j].pop((self.time_dummy, 0), None)
            if (self.space_dummy, 0) in _link_assumptions_wo_dummy[j]:
                _link_assumptions_wo_dummy[j].pop((self.space_dummy, 0), None)

        self.mode = "context_search"
        print("discovering context-system links")
        # do PCMCI+ to discover context-system links
        skeleton_results = self.run_pcmciplus(
                        tau_min=0, 
                        tau_max=tau_max,
                        link_assumptions=_link_assumptions_wo_dummy,
                        pc_alpha=pc_alpha)
        skeleton_val = skeleton_results['val_matrix']

        self.mode = "standard"
        skeleton_graph = skeleton_results['graph']

        for j in range(self.nb_sysvars + self.nb_context_nodes):
            for c in all_context_nodes:
                for k in range(tau_max + 1):
                    if skeleton_graph[c, j, k] == 'o-o' or skeleton_graph[c, j, k] == '-->':
                        parents[j].append((c, -k))
                        lagged_context_parents[j].append((c, -k))
                        values[c, j, k] = skeleton_val[c, j, k]
            for i in range(self.nb_sysvars):
                for k in range(tau_max + 1):
                    if skeleton_graph[i, j, k] == 'o-o' or skeleton_graph[i, j, k] == '-->':
                        lagged_context_parents[j].append((i, -k))
        self.observed_context_parents = parents

        print("observed context parents: ", parents)

        if self.time_dummy is not None or self.space_dummy is not None:
            # setup link assumptions without the observed context nodes
            _link_assumptions_wo_obs_context = deepcopy(_link_assumptions)

            for c in all_context_nodes:
                _link_assumptions_wo_obs_context[c] = {}
            for j in range(self.nb_sysvars):
                for c in all_context_nodes:
                    for lag in range(tau_max + 1):
                        if (c, -lag) in _link_assumptions_wo_obs_context[j]:
                            _link_assumptions_wo_obs_context[j].pop((c, -lag), None)

            initial_graph = self._initialize_graph(_link_assumptions_wo_obs_context, tau_max)

            self.mode = "dummy_context_search"
            print("discovering dummy-system links", _link_assumptions_wo_obs_context)
            # run PC algorithm to find links between dummies and system variables
            _int_link_assumptions = self._set_link_assumptions(link_assumptions, tau_min, tau_max)
            links_for_pc = {}
            for j in range(self.N):
                links_for_pc[j] = {}
                for parent in lagged_context_parents[j]:
                    if _int_link_assumptions[j][parent] in ['-?>', '-->']:
                        links_for_pc[j][parent] = _int_link_assumptions[j][parent]
                for link in _int_link_assumptions[j]:
                    i, tau = link
                    link_type = _int_link_assumptions[j][link]
                    if abs(tau) == 0:
                        links_for_pc[j][(i, 0)] = link_type

            skeleton_results_dummy = self._pcalg_skeleton(
                initial_graph=initial_graph,
                lagged_parents=lagged_context_parents,
                pc_alpha=pc_alpha,
                mode='contemp_conds',
                tau_min=0,
                tau_max=tau_max,
                max_conds_dim=np.inf,
                max_combinations=np.inf,
                max_conds_py=None,
                max_conds_px=None,
                max_conds_px_lagged=None,
            )
            skeleton_val_dummy = skeleton_results_dummy['val_matrix']

            skeleton_graph_dummy = skeleton_results_dummy['graph']
            for j in range(self.nb_sysvars):
                for k in range(tau_max + 1):
                    if skeleton_graph_dummy[self.time_dummy, j, k] == 'o-o' or \
                            skeleton_graph_dummy[self.time_dummy, j, k] == '-->':
                        parents[j].append((self.time_dummy, k))
                        values[self.time_dummy, j, k] = skeleton_val_dummy[self.time_dummy, j, k]
                        lagged_context_parents[j].append((self.time_dummy, k))
                    if skeleton_graph_dummy[self.space_dummy, j, k] == 'o-o' or \
                            skeleton_graph_dummy[self.space_dummy, j, k] == '-->':
                        parents[j].append((self.space_dummy, k))
                        lagged_context_parents[j].append((self.space_dummy, k))
                        values[self.space_dummy, j, k] = skeleton_val_dummy[self.space_dummy, j, k]

            self.mode = "standard"

        self.context_parents = parents
        
        # Return the results
        return {'parents': parents,
                'lagged_context_parents': lagged_context_parents,
                'values': values,
                'iterations': 1}

    def _initialize_graph(self, links, tau_max=None):
        N = len(links)
        # Get maximum time lag
        max_lag = 0
        for j in range(N):
            for link in links[j]:
                var, lag = link
                link_type = links[j][link]
                if link_type != "":
                    max_lag = max(max_lag, abs(lag))

        if tau_max is None:
            tau_max = max_lag
        else:
            if tau_max < max_lag:
                raise ValueError("maxlag(links) > tau_max")

        graph = np.zeros((N, N, tau_max + 1), dtype='<U3')
        graph[:] = ""
        for j in range(self.nb_sysvars + self.nb_context_nodes):
            for link in links[j]:
                i, tau = link
                link_type = links[j][link]
                graph[i, j, abs(tau)] = link_type

        return graph

    def _remaining_pairs(self, graph, adjt, tau_min, tau_max, p):
        """Helper function returning the remaining pairs that still need to be
        tested."""
        all_context_nodes = self.time_context_nodes + self.space_context_nodes
        if self.mode == "context_search":
            # during discovery of context-system links we are only
            # interested in context-context and context-system pairs
            N = graph.shape[0]
            pairs = []
            for (i, j) in itertools.product(range(N), range(N)):
                for abstau in range(tau_min, tau_max + 1):
                    if (graph[i, j, abstau] != ""
                            and len(
                                [a for a in adjt[j] if a != (i, -abstau)]) >= p
                            and i in all_context_nodes):
                        pairs.append((i, j, abstau))
            return pairs
        elif self.mode == "dummy_context_search":
            # during discovery of dummy-system links we are only
            # interested in dummy-system pairs
            N = graph.shape[0]
            pairs = []
            for (i, j) in itertools.product(range(N), range(N)):
                for abstau in range(tau_min, tau_max + 1):
                    if (graph[i, j, abstau] != ""
                            and len(
                                [a for a in adjt[j] if a != (i, -abstau)]) >= p
                            and i in [self.time_dummy, self.space_dummy]):
                        pairs.append((i, j, abstau))
            return pairs
        else:
            return PCMCI._remaining_pairs(self, graph, adjt, tau_min, tau_max, p)

    def _run_pcalg_test(self, graph, i, abstau, j, S, lagged_parents, max_conds_py,
                        max_conds_px, max_conds_px_lagged, tau_max):

        if self.mode == 'dummy_context_search':
            # during discovery of dummy-system links we are using the dummy_ci_test
            if lagged_parents is None:
                cond = list(S)
            else:
                cond = list(S)+lagged_parents[j]
            context_parents_j = self.observed_context_parents[j]
            cond = cond + context_parents_j
            cond = list(dict.fromkeys(cond)) 
                
            return self.run_test_dummy([(j, -abstau)], [(i, 0)], cond, tau_max)
        elif self.mode == 'standard':
            # during discovery of system-system links we are conditioning on the found contextual parents
            if self.time_dummy is None:
                dummy_vars = []
            else:
                dummy_vars = [(self.time_dummy,0)]

            if self.space_dummy is not None:
                dummy_vars += [(self.space_dummy,0)]
            
            if lagged_parents is None:
                cond = list(S)
            else:
                cond = list(S)+lagged_parents[j]
            # always add self.obs_context_parents
            context_parents_j = self.context_parents[j]
            cond = cond + context_parents_j
            cond = list(dict.fromkeys(cond)) 
            return PCMCI._run_pcalg_test(self, graph, i, abstau, j, cond, lagged_parents, max_conds_py,
                                         max_conds_px, max_conds_px_lagged, tau_max)
        else:
            return PCMCI._run_pcalg_test(self, graph, i, abstau, j, S, lagged_parents, max_conds_py,
                                         max_conds_px, max_conds_px_lagged, tau_max)

    def run_test_dummy(self, X, Y, Z=None, tau_max=0, cut_off='2xtau_max'):
        # Get the array to test on
        array, xyz, XYZ, _ = self.dataframe.construct_array(X=X, Y=Y, Z=Z, tau_max=tau_max,
                                                            mask_type=self.dummy_ci_test.mask_type,
                                                            return_cleaned_xyz=True,
                                                            do_checks=True,
                                                            remove_overlaps=True,
                                                            cut_off=cut_off,
                                                            verbosity=0)

        # remove the parts of the array within dummy that are constant zero (ones are cut off)
        mask = np.all(array == 0., axis=1) | np.all(array == 1., axis=1)
        xyz = xyz[~mask]
        Y = Y[np.sum(mask) + 1:]
        array = array[~mask]

        # Record the dimensions
        dim, T = array.shape
        # Ensure it is a valid array
        if np.any(np.isnan(array)):
            raise ValueError("nans in the array!")

        #combined_hash = self.cond_ind_test._get_array_hash(array, xyz, XYZ)

        if False:#combined_hash in self.cond_ind_test.cached_ci_results.keys():
            cached = True
            val, pval = self.cond_ind_test.cached_ci_results[combined_hash]
        else:
            cached = False
            # Get the dependence measure, recycling residuals if need be
            val = self.dummy_ci_test._get_dependence_measure_recycle(X, Y, Z, xyz, array)
            
            # Get the p-value
            pval = self.dummy_ci_test.get_significance(val, array, xyz, T, dim)

        if self.verbosity > 1:
            self.dummy_ci_test._print_cond_ind_results(val=val, pval=pval, cached=cached,
                                                       conf=None)
        # Return the value and the p-value
        return val, pval, Z

    def run_pcmciplus_phase2(self, lagged_parents, selected_links=None, link_assumptions=None, tau_min=0, tau_max=1,
                             pc_alpha=0.01,
                             contemp_collider_rule='majority',
                             conflict_resolution=True,
                             reset_lagged_links=False,
                             max_conds_dim=None,
                             max_conds_py=None,
                             max_conds_px=None,
                             max_conds_px_lagged=None,
                             fdr_method='none',
                             ):

        if selected_links is not None:
            raise ValueError("selected_links is DEPRECATED, use link_assumptions instead.")

        # Check if pc_alpha is chosen to optimze over a list
        if pc_alpha is None or isinstance(pc_alpha, (list, tuple, np.ndarray)):
            # Call optimizer wrapper around run_pcmciplus()
            return self._optimize_pcmciplus_alpha(
                        link_assumptions=link_assumptions,
                        tau_min=tau_min,
                        tau_max=tau_max,
                        pc_alpha=pc_alpha,
                        contemp_collider_rule=contemp_collider_rule,
                        conflict_resolution=conflict_resolution,
                        reset_lagged_links=reset_lagged_links,
                        max_conds_dim=max_conds_dim,
                        max_conds_py=max_conds_py,
                        max_conds_px=max_conds_px,
                        max_conds_px_lagged=max_conds_px_lagged,
                        fdr_method=fdr_method)

        # else:
        #     raise ValueError("pc_alpha=None not supported in PCMCIplus, choose"
        #                      " 0 < pc_alpha < 1 (e.g., 0.01)")

        if pc_alpha < 0. or pc_alpha > 1:
            raise ValueError("Choose 0 <= pc_alpha <= 1")

        # For the lagged PC algorithm only the strongest conditions are tested
        max_combinations = 1

        # Check the limits on tau
        self._check_tau_limits(tau_min, tau_max)
        # Set the selected links
        # _int_sel_links = self._set_sel_links(selected_links, tau_min, tau_max)
        _int_link_assumptions = self._set_link_assumptions(link_assumptions, tau_min, tau_max)

        p_matrix = self.p_matrix
        val_matrix = self.val_matrix

        # Step 2+3+4: PC algorithm with contemp. conditions and MCI tests
        if self.verbosity > 0:
            print("\n##\n## Step 2: PC algorithm with contemp. conditions "
                  "and MCI tests\n##"
                  "\n\nParameters:")
            if link_assumptions is not None:
                print("\nlink_assumptions = %s" % str(_int_link_assumptions))
            print("\nindependence test = %s" % self.cond_ind_test.measure
                  + "\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max
                  + "\npc_alpha = %s" % pc_alpha
                  + "\ncontemp_collider_rule = %s" % contemp_collider_rule
                  + "\nconflict_resolution = %s" % conflict_resolution
                  + "\nreset_lagged_links = %s" % reset_lagged_links
                  + "\nmax_conds_dim = %s" % max_conds_dim
                  + "\nmax_conds_py = %s" % max_conds_py
                  + "\nmax_conds_px = %s" % max_conds_px
                  + "\nmax_conds_px_lagged = %s" % max_conds_px_lagged
                  + "\nfdr_method = %s" % fdr_method
                  )

        # Set the maximum condition dimension for Y and X
        max_conds_py = self._set_max_condition_dim(max_conds_py,
                                                   tau_min, tau_max)
        max_conds_px = self._set_max_condition_dim(max_conds_px,
                                                   tau_min, tau_max)

        if reset_lagged_links:
            # Run PCalg on full graph, ignoring that some lagged links
            # were determined as non-significant in PC1 step
            links_for_pc = deepcopy(_int_link_assumptions)
        else:
            # Run PCalg only on lagged parents found with PC1 
            # plus all contemporaneous links
            links_for_pc = {}  #deepcopy(lagged_parents)
            for j in range(self.N):
                links_for_pc[j] = {}
                for parent in lagged_parents[j]:
                    if parent in _int_link_assumptions[j] and _int_link_assumptions[j][parent] in ['-?>', '-->']:
                        links_for_pc[j][parent] = _int_link_assumptions[j][parent]
                # Add Contemporaneous links
                for link in _int_link_assumptions[j]:
                    i, tau = link
                    link_type = _int_link_assumptions[j][link]
                    if abs(tau) == 0:
                        links_for_pc[j][(i, 0)] = link_type

        self.mode = "standard"
        results = self.run_pcalg(
            link_assumptions=links_for_pc,
            pc_alpha=pc_alpha,
            tau_min=tau_min,
            tau_max=tau_max,
            max_conds_dim=max_conds_dim,
            max_combinations=None,
            lagged_parents=lagged_parents,
            max_conds_py=max_conds_py,
            max_conds_px=max_conds_px,
            max_conds_px_lagged=max_conds_px_lagged,
            mode='contemp_conds',
            contemp_collider_rule=contemp_collider_rule,
            conflict_resolution=conflict_resolution)

        graph = results['graph']

        # Update p_matrix and val_matrix with values from links_for_pc
        for j in range(self.N):
            for link in links_for_pc[j]:
                i, tau = link
                if links_for_pc[j][link] not in ['<--', '<?-']:
                    p_matrix[i, j, abs(tau)] = results['p_matrix'][i, j, abs(tau)]
                    val_matrix[i, j, abs(tau)] = results['val_matrix'][i, j, 
                                                                       abs(tau)]

        # Update p_matrix and val_matrix for indices of symmetrical links
        p_matrix[:, :, 0] = results['p_matrix'][:, :, 0]
        val_matrix[:, :, 0] = results['val_matrix'][:, :, 0]

        ambiguous = results['ambiguous_triples']

        conf_matrix = None
        # TODO: implement confidence estimation, but how?
        # if self.cond_ind_test.confidence is not False:
        #     conf_matrix = results['conf_matrix']

        # Correct the p_matrix if there is a fdr_method
        if fdr_method != 'none':
            p_matrix = self.get_corrected_pvalues(p_matrix=p_matrix, tau_min=tau_min, 
                                                  tau_max=tau_max, 
                                                  link_assumptions=_int_link_assumptions,
                                                  fdr_method=fdr_method)

        # Store the parents in the pcmci member
        self.all_lagged_parents = lagged_parents

        # Cache the resulting values in the return dictionary
        return_dict = {'graph': graph,
                       'val_matrix': val_matrix,
                       'p_matrix': p_matrix,
                       'ambiguous_triples': ambiguous,
                       'conf_matrix': conf_matrix}
        # Print the results
        if self.verbosity > 0:
            self.print_results(return_dict, alpha_level=pc_alpha)
        # Return the dictionary
        self.results = return_dict
        return return_dict
