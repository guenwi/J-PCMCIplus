import numpy as np
import scipy
from numpy.random import SeedSequence
import itertools
import math

from tigramite.toymodels import structural_causal_processes as toys


def shift_link_entries(links, const):
    shifted_links = {}
    for key in links.keys():
        shifted_key = key + const
        values = links[key]
        shifted_values = [((item + const, lag), c, f) for ((item, lag), c, f) in values]
        shifted_links[shifted_key] = shifted_values
    return shifted_links


def structural_causal_process(links, T, context_values=None, noises=None,
                              intervention=None, intervention_type='hard',
                              transient_fraction=0.2,
                              random_state=None):
    """Returns a time series generated from a structural causal process.
    Allows lagged and contemporaneous dependencies and includes the option
    to have intervened variables or particular samples.
    The interventional data is in particular useful for generating ground
    truth for the CausalEffects class.
    In more detail, the method implements a generalized additive noise model process of the form
    .. math:: X^j_t = \\eta^j_t + \\sum_{X^i_{t-\\tau}\\in \\mathcal{P}(X^j_t)}
              c^i_{\\tau} f^i_{\\tau}(X^i_{t-\\tau})
    Links have the format ``{0:[((i, -tau), coeff, func),...], 1:[...],
    ...}`` where ``func`` can be an arbitrary (nonlinear) function provided
    as a python callable with one argument and coeff is the multiplication
    factor. The noise distributions of :math:`\\eta^j` can be specified in
    ``noises``.
    Through the parameters ``intervention`` and ``intervention_type`` the model
    can also be generated with intervened variables.
    Parameters
    ----------
    links : dict
        Dictionary of format: {0:[((i, -tau), coeff, func),...], 1:[...],
        ...} for all variables where i must be in [0..N-1] and tau >= 0 with
        number of variables N. coeff must be a float and func a python
        callable of one argument.
    T : int
        Sample size.
    noises : list of callables or array, optional (default: 'np.random.randn')
        Random distribution function that is called with noises[j](T). If an array,
        it must be of shape ((transient_fraction + 1)*T, N).
    intervention : dict
        Dictionary of format: {1:np.array, ...} containing only keys of intervened
        variables with the value being the array of length T with interventional values.
        Set values to np.nan to leave specific time points of a variable un-intervened.
    intervention_type : str or dict
        Dictionary of format: {1:'hard',  3:'soft', ...} to specify whether intervention is 
        hard (set value) or soft (add value) for variable j. If str, all interventions have 
        the same type.
    transient_fraction : float
        Added percentage of T used as a transient. In total a realization of length
        (transient_fraction + 1)*T will be generated, but then transient_fraction*T will be
        cut off.
    seed : int, optional (default: None)
        Random seed.
    Returns
    -------
    data : array-like
        Data generated from this process, shape (T, N).
    nonstationary : bool
        Indicates whether data has NaNs or infinities.
    """

    # random_state = np.random.default_rng(seed)
    # random_state = np.random.RandomState(seed)
    if context_values is None:
        context_values = {}
    N = len(links.keys())
    if noises is None:
        noises = [random_state.standard_normal for j in range(N)]

    if N != max(links.keys()) + 1:
        print(N, max(links.keys()) + 1, links.keys())
        raise ValueError("links keys must match N.")

    if isinstance(noises, np.ndarray):
        if noises.shape != (T + int(math.floor(transient_fraction * T)), N):
            raise ValueError("noises.shape must match ((transient_fraction + 1)*T, N).")
    else:
        if N != len(noises):
            raise ValueError("noises keys must match N.")

    # Check parameters
    max_lag = 0
    contemp_dag = toys._Graph(N)
    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            coeff = link_props[1]
            func = link_props[2]
            if lag == 0: contemp = True
            if var not in range(N):
                raise ValueError("var must be in 0..{}.".format(N - 1))
            if 'float' not in str(type(coeff)):
                raise ValueError("coeff must be float.")
            if lag > 0 or type(lag) != int:
                raise ValueError("lag must be non-positive int.")
            max_lag = max(max_lag, abs(lag))

            # Create contemp DAG
            if var != j and lag == 0:
                contemp_dag.addEdge(var, j)

    if contemp_dag.isCyclic() == 1:
        raise ValueError("Contemporaneous links must not contain cycle.")

    causal_order = contemp_dag.topologicalSort()
    causal_order.reverse()  # exactly wrong direction

    if intervention is not None:
        if intervention_type is None:
            intervention_type = {j: 'hard' for j in intervention}
        elif isinstance(intervention_type, str):
            intervention_type = {j: intervention_type for j in intervention}
        for j in intervention.keys():
            if len(intervention[j]) != T:
                raise ValueError("intervention array for j=%s must be of length T = %d" % (j, T))
            if j not in intervention_type.keys():
                raise ValueError("intervention_type dictionary must contain entry for %s" % (j))

    transient = int(math.floor(transient_fraction * T))

    data = np.zeros((T + transient, N), dtype='float32')
    for j in range(N):
        if isinstance(noises, np.ndarray):
            data[:, j] = noises[:, j]
        else:
            data[:, j] = noises[j](T + transient)
        if context_values is not None and j in context_values:  # use provided values for context variables
            data[:, j] = context_values[j]

    sys_vars = [item for item in causal_order if item not in context_values.keys()]
    for t in range(max_lag, T + transient):
        for j in sys_vars:

            if (intervention is not None and j in intervention and t >= transient
                    and np.isnan(intervention[j][t - transient]) == False):
                if intervention_type[j] == 'hard':
                    data[t, j] = intervention[j][t - transient]
                    # Move to next j and skip link_props-loop from parents below 
                    continue
                else:
                    data[t, j] += intervention[j][t - transient]

            # This loop is only entered if intervention_type != 'hard'
            for link_props in links[j]:
                var, lag = link_props[0]
                coeff = link_props[1]
                func = link_props[2]
                if func == 'linear':
                    func = linear
                elif func == 'nonlinear':
                    func = nonlinear
                elif func == 'nonlinear1':
                    func = nonlinear1
                elif func == 'nonlinear2':
                    func = nonlinear2
                elif func == 'nonlinear3':
                    func = nonlinear3
                elif func == 'nonlinear4':
                    func = nonlinear4
                data[t, j] += coeff * func(data[t + lag, var])

    data = data[transient:]

    nonstationary = (np.any(np.isnan(data)) or np.any(np.isinf(data)) or np.any(np.isinf(np.square(data))))

    return data, nonstationary


def linear(x): return x


def nonlinear(x): return (x + 5. * x ** 2 * np.exp(-x ** 2 / 20.))


def nonlinear1(x): return (x + 5. * x ** 2 * np.exp(-x ** 2 / 20.))


def nonlinear2(x): return (x ** 2 * np.exp(-x ** 2 / 20.))


def nonlinear3(x): return x ** 3


def nonlinear4(x): return x ** 2


def generate_structural_causal_process(
        nodes=list(range(2)),
        L=1,
        dependency_funcs=['linear'],
        dependency_coeffs=[-0.5, 0.5],
        auto_coeffs=[0.5, 0.7],
        contemp_fraction=0.,
        max_lag=1,
        noise_dists=['gaussian'],
        noise_means=[0.],
        noise_sigmas=[0.5, 2.],
        random_state_noise=None,
        random_state=None):
    """"Randomly generates a structural causal process based on input characteristics.
    The process has the form 
    .. math:: X^j_t = \\eta^j_t + a^j X^j_{t-1} + \\sum_{X^i_{t-\\tau}\\in pa(X^j_t)}
              c^i_{\\tau} f^i_{\\tau}(X^i_{t-\\tau})
    where ``j = 1, ..., N``. Here the properties of :math:`\\eta^j_t` are
    randomly frawn from the noise parameters (see below), :math:`pa
    (X^j_t)` are the causal parents drawn randomly such that in total ``L``
    links occur out of which ``contemp_fraction`` are contemporaneous and
    their time lags are drawn from ``[0 or 1..max_lag]``, the
    coefficients :math:`c^i_{\\tau}` are drawn from
    ``dependency_coeffs``, :math:`a^j` are drawn from ``auto_coeffs``,
    and :math:`f^i_{\\tau}` are drawn from ``dependency_funcs``.
    The returned dictionary links has the format 
    ``{0:[((i, -tau), coeff, func),...], 1:[...], ...}`` 
    where ``func`` can be an arbitrary (nonlinear) function provided
    as a python callable with one argument and coeff is the multiplication
    factor. The noise distributions of :math:`\\eta^j` are returned in
    ``noises``, see specifics below.
    The process might be non-stationary. In case of asymptotically linear
    dependency functions and no contemporaneous links this can be checked with
    ``check_stationarity(...)``. Otherwise check by generating a large sample
    and test for np.inf.
    Parameters
    ---------
    N : int
        Number of variables.
    L : int
        Number of cross-links between two different variables.
    dependency_funcs : list
        List of callables or strings 'linear' or 'nonlinear' for a linear and a specific nonlinear function
        that is asymptotically linear.
    dependency_coeffs : list
        List of floats from which the coupling coefficients are randomly drawn.
    auto_coeffs : list
        List of floats from which the lag-1 autodependencies are randomly drawn.
    contemp_fraction : float [0., 1]
        Fraction of the L links that are contemporaneous (lag zero).
    max_lag : int
        Maximum lag from which the time lags of links are drawn.
    noise_dists : list
        List of noise functions. Either in
        {'gaussian', 'weibull', 'uniform'} or user-specified, in which case
        it must be parametrized just by the size parameter. E.g. def beta
        (T): return np.random.beta(a=1, b=0.5, T)
    noise_means : list
        Noise mean. Only used for noise in {'gaussian', 'weibull', 'uniform'}.
    noise_sigmas : list
        Noise standard deviation. Only used for noise in {'gaussian', 'weibull', 'uniform'}.   
    seed : int
        Random seed to draw the above random functions from.
    noise_seed : int
        Random seed for noise function random generator.
    Returns
    -------
    links : dict
        Dictionary of form {0:[((0, -1), coeff, func), ...], 1:[...], ...}.
    noises : list
        List of N noise functions to call by noise(T) where T is the time series length.
    """

    if max_lag == 0:
        contemp_fraction = 1.

    if contemp_fraction > 0.:
        ordered_pairs = list(itertools.combinations(nodes, 2))
        max_poss_links = min(L, len(ordered_pairs))
        L_contemp = int(contemp_fraction * max_poss_links)
        L_lagged = max_poss_links - L_contemp
    else:
        L_lagged = L
        L_contemp = 0

    # Random order
    causal_order = list(random_state.permutation(nodes))

    # Init link dict
    links = dict([(i, []) for i in nodes])

    # Generate auto-dependencies at lag 1
    if max_lag > 0:
        for i in causal_order:
            a = random_state.choice(auto_coeffs)
            if a != 0.:
                links[i].append(((int(i), -1), float(a), linear))

    # Non-cyclic contemp random pairs of links such that
    # index of cause < index of effect
    # Take up to (!) L_contemp links
    ordered_pairs = list(itertools.combinations(range(len(nodes)), 2))
    random_state.shuffle(ordered_pairs)
    contemp_links = [(causal_order[pair[0]], causal_order[pair[1]], None)
                     for pair in ordered_pairs[:L_contemp]]

    # Possibly cyclic lagged random pairs of links 
    # where we remove already chosen contemp links
    # Take up to (!) L_contemp links
    unordered_pairs = list(itertools.permutations(range(len(nodes)), 2))
    unordered_pairs = list(set(unordered_pairs) - set(ordered_pairs[:L_contemp]))
    random_state.shuffle(unordered_pairs)
    lagged_links = [(causal_order[pair[0]], causal_order[pair[1]], None)
                    for pair in unordered_pairs[:L_lagged]]

    chosen_links = lagged_links + contemp_links

    # Populate links
    links = populate_links(links, chosen_links, contemp_links, max_lag, dependency_coeffs, dependency_funcs,
                           random_state)

    # Now generate noise functions
    # Either choose among pre-defined noise types or supply your own

    noises = []
    for j in links:
        noise_dist = random_state.choice(noise_dists)
        noise_mean = random_state.choice(noise_means)
        noise_sigma = random_state.choice(noise_sigmas)

        if noise_dist in ['gaussian', 'weibull', 'uniform']:
            noise = getattr(NoiseModel(mean=noise_mean, sigma=noise_sigma, random_state_noise=random_state_noise),
                            noise_dist)
        else:
            noise = noise_dist

        noises.append(noise)

    return links, noises, causal_order


class NoiseModel:
    def __init__(self, mean=0., sigma=1., random_state_noise=None):
        self.mean = mean
        self.sigma = sigma
        self.random_state_noise = random_state_noise

    def gaussian(self, T):
        # Get zero-mean unit variance gaussian distribution
        return self.mean + self.sigma * self.random_state_noise.standard_normal(T)

    def weibull(self, T):
        # Get zero-mean sigma variance weibull distribution
        a = 2
        mean = scipy.special.gamma(1. / a + 1)
        variance = scipy.special.gamma(2. / a + 1) - scipy.special.gamma(1. / a + 1) ** 2
        return self.mean + self.sigma * (self.random_state_noise.weibull(a=a, size=T) - mean) / np.sqrt(variance)

    def uniform(self, T):
        # Get zero-mean sigma variance uniform distribution
        mean = 0.5
        variance = 1. / 12.
        return self.mean + self.sigma * (self.random_state_noise.uniform(size=T) - mean) / np.sqrt(variance)


def populate_links(links, chosen_links, contemp_links, max_lag, dependency_coeffs, dependency_funcs, random_state):
    for (i, j, tau) in chosen_links:
        # Choose lag
        if tau is None:
            if (i, j) in contemp_links or max_lag == 0:
                tau = 0
            else:
                tau = int(random_state.integers(1, max_lag + 1))

        # Choose dependency
        c = float(random_state.choice(dependency_coeffs))
        if c != 0:
            func = random_state.choice(dependency_funcs)
            if func == 'linear':
                func = linear
            elif func == 'nonlinear':
                func = nonlinear
            elif func == 'nonlinear1':
                func = nonlinear1
            elif func == 'nonlinear2':
                func = nonlinear2
            elif func == 'nonlinear3':
                func = nonlinear3
            elif func == 'nonlinear4':
                func = nonlinear4

            links[j].append(((int(i), -int(tau)), c, func))
    return links


# Utils
def mergeDictionary(dict_1, dict_2):
    dict_3 = dict(dict_1)
    dict_3.update(dict_2)
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = value + dict_1[key]
    return dict_3


def set_links(effect_candidates, L, cause_candidates, tau_max, dependency_funcs, dependency_coeffs, random_state):
    chosen_links = []
    contemp_links = []
    links = {i: [] for i in effect_candidates}
    for l in range(L):
        cause, tau = choose_cause_tau(cause_candidates, tau_max, random_state)
        effect = random_state.choice(effect_candidates)

        while (cause, effect, tau) in chosen_links:
            cause, tau = choose_cause_tau(cause_candidates, tau_max, random_state)
            effect = random_state.choice(effect_candidates)
        if tau == 0:
            contemp_links.append((cause, effect, tau))
            chosen_links.append((cause, effect, tau))
        else:
            chosen_links.append((cause, effect, tau))

        links = populate_links(links, chosen_links, contemp_links, tau_max, dependency_coeffs, dependency_funcs,
                               random_state)

    return links


def choose_cause_tau(candidates, tau_max, random_state):
    if type(candidates[0]) is tuple:
        cause, tau = random_state.choice(candidates)
    else:
        cause = random_state.choice(candidates)
        tau = 0 if tau_max == 0 else int(random_state.integers(1, tau_max + 1))
    return cause, tau


def get_non_parents(links, tau_max):
    parents = sum(links.values(), [])
    parents = [i[0] for i in parents]
    non_parents = [(var, tau) for var in links.keys() for tau in range(tau_max + 1) if (var, -tau) not in parents]
    return non_parents


def generate_random_contemp_context_model(N=3,
                                          K_space=2,
                                          K_time=2,
                                          tau_max=2,
                                          is_simple_exp=0,
                                          model_seed=None):
    child_seeds = model_seed.spawn(7)

    if is_simple_exp == 1: # linear functional relationships, simple fixed model
        links = dict([(i, []) for i in range(2 + 2)])
        links_tc = {2: []}
        links_sc = {3: []}
        links_sys = dict([(i, []) for i in range(2)])

        links[0] = [((1,-1), 0.5, 'linear'), ((2,-1),0.5, 'linear'), ((3, 0), -0.5, 'linear')]
        links[1] = [((1,-1), 0.5, 'linear'), ((2,-1),-0.5, 'linear'), ((3, 0), 0.5, 'linear')]

        links_sys[0] = links[0]
        links_sys[1] = links[1]
        noises = None

    elif is_simple_exp == 2: # nonlinear functional relationships, simple fixed model
        links = dict([(i, []) for i in range(2 + 2)])
        links_tc = {2: []}
        links_sc = {3: []}
        links_sys = dict([(i, []) for i in range(2)])

        links[0] = [((1, -1), 0.3, 'nonlinear4'), ((2, -1), 0.5, 'linear'), ((3, 0), -0.5, 'linear')]
        links[1] = [((1, -1), 0.3, 'linear'), ((2, -1), -0.5, 'linear'), ((3, 0), 0.5, 'linear')]

        links_sys[0] = links[0]
        links_sys[1] = links[1]
        noises = None

    else:
        dependency_funcs = ['linear']
        dependency_coeffs = [-0.5, 0.5]
        auto_coeffs = [0.5, 0.7]
        contemp_fraction = 0.5
        noise_dists = ['gaussian']
        noise_means = [0.]
        noise_sigmas = [0.5, 2.]

        L = 1 if N == 2 else N
        L_space = 1 if K_space == 2 else K_space
        L_time = 1 if K_time == 2 else K_time

        nodes_sc = list(range(N + K_time, N + K_space + K_time))
        nodes_tc = list(range(N, K_time + N))
        nodes_sys = list(range(N))

        links_tc = {}
        links_sc = {}
        noises_tc = []
        noises_sc = []

        # graph for temporal context vars
        if K_time != 0:
            links_tc, noises_tc, causalorder_tc = generate_structural_causal_process(nodes_tc,
                                                                                     L_time,
                                                                                     dependency_funcs,
                                                                                     dependency_coeffs,
                                                                                     auto_coeffs,
                                                                                     contemp_fraction,
                                                                                     tau_max,
                                                                                     noise_dists,
                                                                                     noise_means,
                                                                                     noise_sigmas,
                                                                                     np.random.default_rng(
                                                                                         child_seeds[0]),
                                                                                     np.random.default_rng(
                                                                                         child_seeds[1]))

        if K_space != 0:
            # graph for spatial context vars
            links_sc, noises_sc, causalorder_sc = generate_structural_causal_process(nodes_sc,
                                                                                     L_space,
                                                                                     dependency_funcs,
                                                                                     dependency_coeffs,
                                                                                     auto_coeffs,
                                                                                     1.,
                                                                                     0,
                                                                                     noise_dists,
                                                                                     noise_means,
                                                                                     noise_sigmas,
                                                                                     np.random.default_rng(
                                                                                         child_seeds[2]),
                                                                                     np.random.default_rng(
                                                                                         child_seeds[3]))

        # graph for system vars
        links_sys, noises_sys, causalorder_sys = generate_structural_causal_process(nodes_sys,
                                                                                    L,
                                                                                    dependency_funcs,
                                                                                    dependency_coeffs,
                                                                                    auto_coeffs,
                                                                                    1.,
                                                                                    tau_max,
                                                                                    noise_dists,
                                                                                    noise_means,
                                                                                    noise_sigmas,
                                                                                    np.random.default_rng(
                                                                                        child_seeds[4]),
                                                                                    np.random.default_rng(
                                                                                        child_seeds[5]))

        links = dict(links_tc)
        links.update(links_sc)
        links.update(links_sys)
        noises = noises_sys + noises_tc + noises_sc

        # set context-system links
        non_parent_tc = get_non_parents(links_tc, tau_max)
        non_parent_sc = get_non_parents(links_sc, 0)

        # number of links between system and context
        L_context_sys = 2 * (len(non_parent_sc) + len(non_parent_tc))

        context_sys_links = set_links(nodes_sys, L_context_sys, non_parent_sc + non_parent_tc,
                                      tau_max, dependency_funcs, dependency_coeffs,
                                      np.random.default_rng(child_seeds[6]))

        # join all link-dicts to form graph over context and system nodes
        links = mergeDictionary(links, context_sys_links)

    return links, links_tc, links_sc, links_sys, noises


class ContextModel:
    def __init__(self, links_tc=None, links_sc=None, links_sys=None, noises=None, random_state_context=None):
        if links_sys is None:
            links_sys = {}
        if links_sc is None:
            links_sc = {}
        if links_tc is None:
            links_tc = dict()
        self.N = len(links_sys.keys())
        self.links_tc = links_tc
        self.links_sc = links_sc
        self.links_sys = links_sys
        self.noises = noises
        self.random_state_context = random_state_context

    def temporal_random(self, links_tc, T):
        if self.noises is not None:
            noises_tc = [self.noises[key] for key in links_tc.keys()]
        else:
            noises_tc = None
        shifted_links_tc = shift_link_entries(links_tc, -self.N)
        data_tc, nonstat_tc = structural_causal_process(shifted_links_tc, T=T, random_state=self.random_state_context,
                                                        noises=noises_tc)
        data_tc = {i: data_tc[:, i - self.N] for i in links_tc.keys()}
        return data_tc, nonstat_tc

    def spatial_random(self, links_sc, M, shift):
        shifted_links_sc = shift_link_entries(links_sc, -shift)
        if self.noises is not None:
            noises_sc = [self.noises[key] for key in links_sc.keys()]
        else:
            noises_sc = None
        data_sc, nonstat_sc = structural_causal_process(shifted_links_sc, T=M, random_state=self.random_state_context,
                                                        noises=noises_sc)
        return data_sc, nonstat_sc

    def generate_data(self, M, T, transient_fraction, links):
        transient = int(math.floor(transient_fraction * T))

        K_time = len(self.links_tc.keys())
        K_space = len(self.links_sc.keys())

        data = {}
        data_tc = {}
        data_sc = {}
        nonstat_tc = False
        nonstat_sc = False
        nonstationary = []

        # first generate data for temporal context nodes
        if K_time != 0:
            data_tc, nonstat_tc = self.temporal_random(self.links_tc, T + transient)
        data_tc_list = [data_tc for m in range(M)]

        # generate spatial context data (constant in time)
        if K_space != 0:
            data_sc, nonstat_sc = self.spatial_random(self.links_sc, M, K_time + self.N)

        for m in range(M):  # assume that this is a given order of datasets
            data_sc_m = {i: np.repeat(data_sc[m, i - self.N - K_time], T + transient) for i in self.links_sc.keys()}

            data_context = dict(data_tc_list[m])
            data_context.update(data_sc_m)

            # generate system data that varies over space and time
            data_m, nonstat = structural_causal_process(links, T=T, context_values=data_context,
                                                        random_state=self.random_state_context, noises=self.noises)
            data[m] = data_m
            nonstationary.append(nonstat or nonstat_tc or nonstat_sc)
        return data, np.any(nonstationary)


def generate_data(M, T, transient_fraction, links, links_tc, links_sc, links_sys, noises, random_state):
    # deprecated
    transient = int(math.floor(transient_fraction * T))

    K_time = len(links_tc.keys())
    K_space = len(links_sc.keys())
    N = len(links_sys.keys())

    data = {}
    data_tc = {}
    data_sc = {}
    nonstat_tc = False
    nonstat_sc = False
    nonstationary = []

    # first generate data for temporal context nodes
    contextmodel = ContextModel(links_tc=links_tc, links_sc=links_sc, links_sys=links_sys, noises=noises,
                                random_state_context=random_state)
    if K_time != 0:
        data_tc, nonstat_tc = contextmodel.temporal_random(links_tc, T + transient)

    # generate spatial context data (constant in time)
    if K_space != 0:
        data_sc, nonstat_sc = contextmodel.spatial_random(links_sc, M, K_time + N)

    for m in range(M):
        data_sc_m = {i: np.repeat(data_sc[m, i - N - K_time], T + transient) for i in links_sc.keys()}

        data_context = dict(data_tc)
        data_context.update(data_sc_m)

        # generate system data that varies over space and time
        data_m, nonstat = structural_causal_process(links, T=T, context_values=data_context, random_state=random_state,
                                                    noises=noises)
        data[m] = data_m
        nonstationary.append(nonstat or nonstat_tc or nonstat_sc)

    return data, np.any(nonstationary)