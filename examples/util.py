##################################################################################
# This module contain local utility function defintions that we can reuse 
# in example notebooks to help reduce clutter.
##################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import math
import random 
import numpy as np
import seaborn as sns
import pickle 
import os
#from ws3 import opt
import ws3
##########################################################
# Implement a priority queue heuristic harvest scheduler
##########################################################

def schedule_harvest_areacontrol(fm, max_harvest=1, period=None, acode='harvest', util=0.85, 
                                 target_masks=None, target_areas=None,
                                 target_scalefactors=None,
                                 mask_area_thresh=0.,
                                 verbose=0):
    """
    This function schedules the harvesting using area control in a ForestModel. It takes as input:

    - `fm` (ForestModel object): The forest model to be used.
    - `max_harvest` (float, optional): The maximum amount of resources that can be harvested from any given location. Defaults to 1.
    - `period` (Period object or None, optional): The period for which the schedule should be made. If not specified, it will use all available periods. Defaults to None.
    - `acode` (str, optional): Action codes to use in building the action schedule. Defaults to 'harvest'.
    - `util` (float, optional): A value between 0 and 1 representing the utilization rate of harvesting resources. Defaults to 0.85.
    - `target_masks` (list of str or None, optional): Masks that define the areas to be targeted for harvesting. If not specified, it will default to AU-wise THLB . Defaults to None.
    - `target_areas` (list of float or None, optional): The target area for each masked DT set in square units. If not specified, it will be calculated based on the total volume and area of trees in those areas. Defaults to None.
    - `target_scalefactors` (list of float or None, optional): Scale factors that multiply the target areas. If not specified, no scaling will be applied. Defaults to None.
    - `mask_area_thresh` (float, optional): The minimum area for a masked DT set to be considered in AU-wise THLB . Defaults to 0..
    - `verbose` (int, optional): The level of verbosity (0 = none, 1 = basic information, 2+ = detailed information). Defaults to 0.

    It returns a schedule object representing the harvesting schedule for the specified period(s) and target areas/masks.
    """
    if not target_areas:
        if not target_masks: # default to AU-wise THLB 
            au_vals = []
            au_agg = []
            for au in fm.theme_basecodes(2):
                mask = '? 1 %s ? ?' % au
                masked_area = fm.inventory(0, mask=mask)
                if masked_area > mask_area_thresh:
                    au_vals.append(au)
                else:
                    au_agg.append(au)
                    if verbose > 0:
                        print('adding to au_agg', mask, masked_area)
            if au_agg:
                fm._themes[2]['areacontrol_au_agg'] = au_agg 
                if fm.inventory(0, mask='? ? areacontrol_au_agg ? ?') > mask_area_thresh:
                    au_vals.append('areacontrol_au_agg')
            target_masks = ['? 1 %s ? ?' % au for au in au_vals]
        target_areas = []
        for i, mask in enumerate(target_masks): # compute area-weighted mean CMAI age for each masked DT set
            masked_area = fm.inventory(0, mask=mask, verbose=verbose)
            if not masked_area: continue
            r = sum((fm.dtypes[dtk].ycomp('totvol').mai().ytp().lookup(0) * fm.dtypes[dtk].area(0)) for dtk in fm.unmask(mask))
            r /= masked_area
            asf = 1. if not target_scalefactors else target_scalefactors[i]  
            ta = max_harvest * (1/r) * fm.period_length * masked_area * asf
            target_areas.append(ta)
    periods = fm.periods if not period else [period]
    for period in periods:
        for mask, target_area in zip(target_masks, target_areas):
            if verbose > 0:
                print('calling areaselector', period, acode, target_area, mask)
            fm.areaselector.operate(period, acode, target_area, mask=mask, verbose=verbose)
    sch = fm.compile_schedule()
    return sch



##############################################################
# Implement an LP optimization harvest scheduler
##############################################################

def cmp_c_z(fm, path, expr, mask=None):
    """
    Compile objective function coefficient (given ForestModel instance, 
    leaf-to-root-node path, and expression to evaluate).

     Args:
         fm (ForestModel object): The forest model instance.
         path (str or list of str): Leaf-to-root-node path in a Tree object.
         expr (str): Expression to be evaluated at the given path.
         mask (str, optional): Mask to apply to inventory data before evaluating expression. Defaults to None.

     Returns:
         float: Compiled objective function coefficient.
    """
    result = 0.
    for t, n in enumerate(path, start=1):
        d = n.data()
        if mask and not fm.match_mask(mask, d['dtk']): continue
        if fm.is_harvest(d['acode']):
            result += fm.compile_product(t, expr, d['acode'], [d['dtk']], d['age'], coeff=False)
    return result

def cmp_c_cflw(fm, path, expr, mask=None): # product, all harvest actions
    """
    Compile flow constraint coefficient for product indicator
    (given ForestModel instance, leaf-to-root-node path, expression to evaluate and optional mask).

    Args:
        fm  (ForestModel object): The forest model instance.
        path  (str or list of str): Leaf-to-root-node path in a Tree object.
        expr  (str): Expression to be evaluated at the given path.
        mask  (str, optional): Mask to apply to inventory data before evaluating expression. Defaults to None.

    Returns:
         dict: Dictionary where keys are periods and values are corresponding compiled coefficients.
    """
    result = {}
    for t, n in enumerate(path, start=1):
        d = n.data()
        if mask and not fm.match_mask(mask, d['dtk']): continue
        if fm.is_harvest(d['acode']):
            result[t] = fm.compile_product(t, expr, d['acode'], [d['dtk']], d['age'], coeff=False)
    return result


def cmp_c_caa(fm, path, expr, acodes, mask=None): # product, named actions
    """
    Compile constraint coefficient for product indicator
    (given ForestModel instance, leaf-to-root-node path, expression to evaluate, list of action codes, and optional mask).

    Args:
        fm   (ForestModel object): The forest model instance.
        path  (str or list of str): Leaf-to-root-node path in a Tree object.
        expr  (str): Expression to be evaluated at the given path.
        acodes (list of str): List of action codes for which the constraint coefficient should be compiled.
        mask  (str, optional): Mask to apply to inventory data before evaluating expression. Defaults to None.

    Returns:
         dict: Dictionary where keys are periods and values are corresponding compiled coefficients.
    """
    result = {}
    for t, n in enumerate(path, start=1):
        d = n.data()
        if mask and not fm.match_mask(mask, d['dtk']): continue
        if d['acode'] in acodes:
            result[t] = fm.compile_product(t, expr, d['acode'], [d['dtk']], d['age'], coeff=False)
    return result


def cmp_c_ci(fm, path, yname, mask=None): # product, named actions
    """
    Compile constraint coefficient for inventory indicator
    (given ForestModel instance, leaf-to-root-node path, expression to evaluate, and optional mask).

    Args:
        fm   (ForestModel object): The forest model instance.
        path  (str or list of str): Leaf-to-root-node path in a Tree object.
        yname (str): Name of the inventory indicator to be evaluated at the given path.
        mask  (str, optional): Mask to apply to inventory data before evaluating expression. Defaults to None.

    Returns:
         dict: Dictionary where keys are periods and values are corresponding compiled coefficients for the specified inventory indicator.
    """
    result = {}
    for t, n in enumerate(path, start=1):
        d = n.data()
        if mask and not fm.match_mask(mask, d['_dtk']): continue
        result[t] = fm.inventory(t, yname=yname, age=d['_age'], dtype_keys=[d['_dtk']])
    return result


def compile_scenario(fm):
    """
    Compile a scenario DataFrame for the given ForestModel instance.

    Args:
        fm (ForestModel object): The forest model instance.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'period', 'oha', 'ohv', and 'ogs'. These represent the compiled products for harvest area,
                          their volumes, and overall inventory at each period respectively.
    """
    oha = [fm.compile_product(period, '1.', acode='harvest') for period in fm.periods]
    ohv = [fm.compile_product(period, 'totvol * 0.85', acode='harvest') for period in fm.periods]
    ogs = [fm.inventory(period, 'totvol') for period in fm.periods]
    data = {'period':fm.periods, 
            'oha':oha, 
            'ohv':ohv, 
            'ogs':ogs}
    df = pd.DataFrame(data)
    return df


def plot_scenario(df):
    """
    Plot the scenario DataFrame for a scenario compiled from `compile_scenario`.

    Args:
        df (pandas.DataFrame): The output from compile_scenario function. It should have columns
                                'period', 'oha', 'ohv', and 'ogs'. These represent the compiled products
                                for harvest area, their volumes, and overall inventory at each period respectively.

    Returns:
        matplotlib.figure.Figure: A Figure containing a 3 subplots representing harvested area (ha),
                                   harvested volume (m3), and growing stock (m3) over time.
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].bar(df.period, df.oha)
    ax[0].set_ylim(0, None)
    ax[0].set_title('Harvested area (ha)')
    ax[1].bar(df.period, df.ohv)
    ax[1].set_ylim(0, None)
    ax[1].set_title('Harvested volume (m3)')
    ax[2].bar(df.period, df.ogs)
    ax[2].set_ylim(0, None)
    ax[2].set_title('Growing Stock (m3)')
    return fig, ax

def run_scenario(fm, scenario_name='base', solver=ws3.opt.SOLVER_HIGHS, verbose=False, workers=1, print_df=False):
    """
    Runs a specified scenario on the given ForestModel instance and solves it using an LP solver.

    Args:
        fm (ForestModel object): The forest model on which the scenario is to be run.
        scenario_name (str, optional): Name of the scenario to run. Defaults to 'base'.
        solver: LP Solver to use for solving the generated scenario. Defaults to ws3.opt.SOLVER_HIGHS.
        verbose (bool, optional): If True, will output solver progress and status. Defaults to False.
        workers (int, optional): Number of parallel workers to use. Defaults to 1.
        print_df (bool, optional): If True, will print the resulting DataFrame. Defaults to False.

    Returns:
        tuple: (fig, df, p)
            - fig: Matplotlib Figure object representing scenario plots.
            - df: pandas DataFrame with compiled scenario results.
            - p: Problem instance that was created and solved.
    """
    import sys
    cflw_ha = {}
    cflw_hv = {}
    cgen_ha = {}
    cgen_hv = {}
    cgen_gs = {}
    
    # define harvest area and harvest volume flow constraints
    cflw_ha = ({p:0.05 for p in fm.periods}, 1)
    cflw_hv = ({p:0.05 for p in fm.periods}, 1)

    if scenario_name == 'base': 
        # Base scenario
        print('running base scenario')
    elif scenario_name == 'base-cgen_ha': 
        # Base scenario, plus harvest area general constraints
        print('running base scenario plus harvest area constraints')
        cgen_ha = {'lb':{1:0.}, 'ub':{1:100.}}    
    elif scenario_name == 'base-cgen_hv': 
        # Base scenario, plus harvest volume general constraints
        print('running base scenario plus harvest volume constraints')
        cgen_hv = {'lb':{1:0.}, 'ub':{1:10000.}}    
    elif scenario_name == 'base-cgen_gs': 
        # Base scenario, plus growing stock general constraints
        print('running base scenario plus growing stock constraints')
        cgen_gs = {'lb':{10:120000.}, 'ub':{10:1000000.}}
    else:
        assert False # bad scenario name

    p = gen_scenario(fm=fm, 
                     name=scenario_name, 
                     cflw_ha=cflw_ha, 
                     cflw_hv=cflw_hv,
                     cgen_ha=cgen_ha,
                     cgen_hv=cgen_hv,
                     cgen_gs=cgen_gs,
                     workers=workers)
    p.solver(solver)
    fm.reset()
    p.solve(verbose=verbose)

    if p.status() != ws3.opt.STATUS_OPTIMAL:
        print('Model not optimal.')
        df = None   
    else:
        sch = fm.compile_schedule(p)
        fm.apply_schedule(sch, 
                        force_integral_area=False, 
                        override_operability=False,
                        fuzzy_age=False,
                        recourse_enabled=False,
                        verbose=False,
                        compile_c_ycomps=True)
        df = compile_scenario(fm)
        if print_df:
            print(df)
        fig, ax = plot_scenario(df)
    return fig, df, p


def gen_scenario(
    fm,
    name="base",
    util=0.85,
    harvest_acode="harvest",
    cflw_ha={},
    cflw_hv={},
    cgen_ha={},
    cgen_hv={},
    cgen_gs={},
    tvy_name="totvol",
    obj_mode="max_hv",
    mask=None,
    workers=1,
):
    """
    Generate a linear programming (LP) scenario for a given ForestModel instance.

    Args:
        fm (ForestModel): The forest model instance on which to generate the scenario.
        name (str, optional): Name of the scenario. Defaults to 'base'.
        util (float, optional): Utilization rate of harvesting resources. Defaults to 0.85.
        harvest_acode (str, optional): Action code for harvesting. Defaults to 'harvest'.
        cflw_ha (dict, optional): Even flow constraints for harvest area. Defaults to an empty dictionary.
        cflw_hv (dict, optional): Even flow constraints for harvest volume. Defaults to an empty dictionary.
        cgen_ha (dict, optional): General constraints for harvest area. Defaults to an empty dictionary.
        cgen_hv (dict, optional): General constraints for harvest volume. Defaults to an empty dictionary.
        cgen_gs (dict, optional): General constraints for growing stock. Defaults to an empty dictionary.
        tvy_name (str, optional): Name of the total volume yield. Defaults to 'totvol'.
        obj_mode (str, optional): Objective mode, either 'max_hv' for maximizing harvest volume or 'min_ha' for minimizing harvest area.
            Defaults to 'max_hv'.
        mask (str, optional): Mask to apply to inventory data. Defaults to None.
        workers (int, optional): Number of parallel workers to use. Defaults to 1.

    Returns:
        Problem: A problem instance generated for the given scenario configuration.
    """
    from functools import partial
    import numpy as np

    coeff_funcs = {}
    cflw_e = {}
    cgen_data = {}
    acodes = ["null", harvest_acode]  # define list of action codes
    vexpr = "%s * %0.2f" % (tvy_name, util)  # define volume expression
    if obj_mode == "max_hv":  # maximize harvest volume
        sense = ws3.opt.SENSE_MAXIMIZE 
        zexpr = vexpr
    elif obj_mode == "min_ha":  # minimize harvest area
        sense = opt.SENSE_MINIMIZE 
        zexpr = "1."
    else:
        raise ValueError("Invalid obj_mode: %s" % obj_mode)
    coeff_funcs["z"] = partial(cmp_c_z, expr=zexpr)  # define objective function coefficient function
    T = fm.periods
    if cflw_ha:  # define even flow constraint (on harvest area)
        cname = "cflw_ha"
        coeff_funcs[cname] = partial(cmp_c_caa, expr="1.", acodes=[harvest_acode], mask=None)
        cflw_e[cname] = cflw_ha
    if cflw_hv:  # define even flow constraint (on harvest volume)
        cname = "cflw_hv"
        coeff_funcs[cname] = partial(cmp_c_caa, expr=vexpr, acodes=[harvest_acode], mask=None) 
        cflw_e[cname] = cflw_hv         
    if cgen_ha:  # define general constraint (harvest area)
        cname = "cgen_ha"
        coeff_funcs[cname] = partial(cmp_c_caa, expr="1.", acodes=[harvest_acode], mask=None)
        cgen_data[cname] = cgen_ha
    if cgen_hv:  # define general constraint (harvest volume)
        cname = "cgen_hv"
        coeff_funcs[cname] = partial(cmp_c_caa, expr=vexpr, acodes=[harvest_acode], mask=None) 
        cgen_data[cname] = cgen_hv
    if cgen_gs:  # define general constraint (growing stock)
        cname = "cgen_gs"
        coeff_funcs[cname] = partial(cmp_c_ci, yname=tvy_name, mask=None)
        cgen_data[cname] = cgen_gs
    return fm.add_problem(
        name, coeff_funcs, cflw_e, cgen_data=cgen_data, acodes=acodes, sense=sense, mask=mask, workers=workers
    )


def run_cbm(sit_config, sit_tables, n_steps, plot=True):
    """
    Run the Carbon Budget Model (CBM) from data generated using `ForestModel.to_cbm_sit` method 
    using the libcbm_py open source Python implementation.

    Args:
        sit_config (dict): Configuration settings for the SIT.
        sit_tables (dict): Tables containing SIT data, such as classifiers, disturbance types, etc.
        n_steps (int): Number of timesteps to simulate.
        plot (bool, optional): Whether to plot the annual carbon stocks. Defaults to True.

    Returns:
        CBMOutput: CBMOutput object containing the results of the simulation.
    """
    from libcbm.input.sit import sit_reader
    from libcbm.input.sit import sit_cbm_factory 
    from libcbm.model.cbm.cbm_output import CBMOutput
    from libcbm.storage.backends import BackendType
    from libcbm.model.cbm import cbm_simulator

    sit_data = sit_reader.parse(sit_classifiers=sit_tables['sit_classifiers'],
                                sit_disturbance_types=sit_tables['sit_disturbance_types'],
                                sit_age_classes=sit_tables['sit_age_classes'],
                                sit_inventory=sit_tables['sit_inventory'],
                                sit_yield=sit_tables['sit_yield'],
                                sit_events=sit_tables['sit_events'],
                                sit_transitions=sit_tables['sit_transitions'],
                                sit_eligibilities=None)
    sit = sit_cbm_factory.initialize_sit(sit_data=sit_data, config=sit_config)
    classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
    cbm_output = CBMOutput(classifier_map=sit.classifier_value_names,
                           disturbance_type_map=sit.disturbance_name_map)
    with sit_cbm_factory.initialize_cbm(sit) as cbm:
        # Create a function to apply rule based disturbance events and transition rules based on the SIT input
        rule_based_processor = sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)
        # The following line of code spins up the CBM inventory and runs it through 200 timesteps.
        cbm_simulator.simulate(cbm,
                               n_steps=n_steps,
                               classifiers=classifiers,
                               inventory=inventory,
                               pre_dynamics_func=rule_based_processor.pre_dynamics_func,
                               reporting_func=cbm_output.append_simulation_result,
                               backend_type=BackendType.numpy)
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    biomass_pools = ['SoftwoodMerch','SoftwoodFoliage', 'SoftwoodOther', 'SoftwoodCoarseRoots','SoftwoodFineRoots',                        
                     'HardwoodMerch', 'HardwoodFoliage', 'HardwoodOther', 'HardwoodCoarseRoots', 'HardwoodFineRoots']
    dom_pools = ['AboveGroundVeryFastSoil', 'BelowGroundVeryFastSoil', 'AboveGroundFastSoil', 'BelowGroundFastSoil',
                 'MediumSoil', 'AboveGroundSlowSoil', 'BelowGroundSlowSoil', 'SoftwoodStemSnag', 'SoftwoodBranchSnag',
                 'HardwoodStemSnag', 'HardwoodBranchSnag']
    biomass_result = pi[['timestep']+biomass_pools]
    dom_result = pi[['timestep']+dom_pools]
    total_eco_result = pi[['timestep']+biomass_pools+dom_pools]
    annual_carbon_stocks = pd.DataFrame({'Year':pi['timestep'],
                                         'Biomass':pi[biomass_pools].sum(axis=1),
                                         'DOM':pi[dom_pools].sum(axis=1),
                                         'Total Ecosystem': pi[biomass_pools+dom_pools].sum(axis=1)})
    if plot:
        annual_carbon_stocks.groupby('Year').sum().plot(figsize=(10, 10),xlim=(0, n_steps), ylim=(0, None))
    return cbm_output


def schedule_fire_areacontrol(fm, period=None, acode='fire', util=0.85, 
                                 target_masks=None, target_areas=None,
                                 target_scalefactors=None,
                                 mask_area_thresh=0.,
                                 verbose=0, intensity=None):
    """
    Implement a priority queue heuristic fire scheduler.

    Args:
        fm (ForestModel): The forest model instance to be used for scheduling.
        period (int or None, optional): The period for which the schedule
            should be made. If not specified, it will use all available periods.
            Defaults to None.
        acode (str, optional): Action code to use in building the action
            schedule. Defaults to 'fire'.
        util (float, optional): A value between 0 and 1 representing the
            utilization rate. Defaults to 0.85.
        target_masks (list of str or None, optional): Masks that define the
            areas to be targeted for fire scheduling. Defaults to None.
        target_areas (list of float or None, optional): The target area for
            each masked DT set in square units. If not specified, it will be
            calculated based on the total area of trees in those areas.
            Defaults to None.
        target_scalefactors (list of float or None, optional): Scale factors
            that multiply the target areas. If not specified, no scaling will
            be applied. Defaults to None.
        mask_area_thresh (float, optional): The minimum area for a masked DT
            set to be considered. Defaults to 0.
        verbose (int, optional): The level of verbosity (0 = none, 1 = basic
            information, 2+ = detailed information). Defaults to 0.
        intensity (float, optional): The fire intensity as a value between 0
            and 1. This impacts the target area calculation. Defaults to None.

    Returns:
        list of tuples: A schedule listing which actions to apply to which
        development types and age classes in which periods for the specified
        fire scheduling.
    """
    if not target_areas:
        if not target_masks: # default to AU-wise THLB 
            au_vals_short = []
            au_vals_long = []
            au_vals = []
            au_agg = []
            for au in fm.theme_basecodes(2):
                mask = '? ? %s ? ?' % au
                masked_area = fm.inventory(0, mask=mask)
                if masked_area > mask_area_thresh:
                    if int(au) < 450:
                        au_vals_short.append(au)  
                        au_vals.append(au)
                    else:
                        au_vals_long.append(au)
                        au_vals.append(au)
                else:
                    au_agg.append(au)
                    if verbose > 0:
                        print('adding to au_agg', mask, masked_area)
            if au_agg:
                fm._themes[2]['areacontrol_au_agg'] = au_agg 
                if fm.inventory(0, mask='? ? areacontrol_au_agg ? ?') > mask_area_thresh:
                    au_vals_short.append('areacontrol_au_agg')
            target_masks = ['? ? %s ? ?' % au for au in au_vals]
        target_areas = []
        for i, mask in enumerate(target_masks): # compute area-weighted mean CMAI age for each masked DT set            
            masked_area = fm.inventory(0, mask=mask, verbose=verbose)
            if not masked_area: continue
            if mask in au_vals_short:
                r =  sum((100 * fm.dtypes[dtk].area(0)) for dtk in fm.unmask(mask))
            else: 
                r = sum((200 * fm.dtypes[dtk].area(0)) for dtk in fm.unmask(mask))                
            r /= masked_area
            asf = 1. if not target_scalefactors else target_scalefactors[i]  
            ta = (1-intensity) * (1/r) * fm.period_length * masked_area * asf
            target_areas.append(ta)
    periods = fm.periods if not period else [period]
    for period in periods:
        for mask, target_area in zip(target_masks, target_areas):
            if verbose > 0:
                print('calling areaselector', period, acode, target_area, mask)
            fm.areaselector.operate(period, acode, target_area, mask=mask, verbose=verbose)
    sch = fm.compile_schedule()
    return sch


class RandomAreaSelector:
    """
    Selects areas for treatment from random age classes.

    Attributes:
        parent: The parent object owning this selector.

    Methods:
        operate: Executes actions based on randomly selected operable age classes within a target area.
    """
    def __init__(self, parent):
        """
        Initializes the RandomAreaSelector with a parent object.

        Args:
            parent: The parent object owning this selector.
        """
        self.parent = parent

    def operate(self, period, acode, target_area, mask=None,
                commit_actions=True, verbose=False):
        """
        Operate on random operable age classes.

        Args:
            period (int): The period in which to apply the actions.
            acode (str): The action code indicating the type of operation.
            target_area (float): The target area for which actions need to be applied.
            mask (str or None): Optional mask to filter the operable areas.
            commit_actions (bool): Whether to commit the actions after selection.
                                   Defaults to True.
            verbose (bool): Verbosity flag for printing detailed operation logs.
                            Defaults to False.

        Returns:
            float: The remaining target area that was not operated on.
        """
        key = lambda item: max(item[1])
        odt = sorted(list(self.parent.operable_dtypes(acode, period, mask).items()), key=key)
        if verbose:
            print(' entering selector.operate()', len(odt), 'operable dtypes')
        while target_area > 0 and odt:
            while target_area > 0 and odt:
                popped = odt.pop()
                try:
                    dtk, ages = popped #odt.pop()
                except:
                    print(odt)
                    print(popped)
                    raise
                # age = random.choice(ages)
                upages = ages
                random.shuffle(upages)
                age = upages.pop()
                oa = self.parent.dtypes[dtk].operable_area(acode, period, age)
                if not oa: continue # nothing to operate
                area = min(oa, target_area)
                target_area -= area
                if area < 0:
                    print('negative area', area, oa, target_area, acode, period, age)
                    assert False
                if verbose:
                    print(' selector found area', [' '.join(dtk)], acode, period, age, area)
                self.parent.apply_action(dtk, acode, period, age, area, compile_c_ycomps=True,
                                         fuzzy_age=False, recourse_enabled=False, verbose=verbose)
            odt = sorted(list(self.parent.operable_dtypes(acode, period, mask).items()), key=key)
        self.parent.commit_actions(period, repair_future_actions=True)
        if verbose:
            print('RandomAreaSelector.operate done (remaining target_area: %0.1f)' % target_area)
        return target_area


class GreedyAreaSelector:
    """
    Default AreaSelector implementation. Selects areas for treatment from oldest age classes.

    Attributes:
        parent: The parent object owning this selector.

    Methods:
        operate: Executes actions based on the oldest operable age classes within a target area.
    """
    def __init__(self, parent):
        """
        Initializes the GreedyAreaSelector with a parent object.

        Args:
            parent: The parent object owning this selector.
        """
        self.parent = parent

    def operate(self, period, acode, target_area, mask=None,
                commit_actions=True, verbose=False):
        """
        Greedily operate on oldest operable age classes.

        Args:
            period (int): The period in which to apply the actions.
            acode (str): The action code indicating the type of operation.
            target_area (float): The target area for which actions need to be applied.
            mask (str or None): Optional mask to filter the operable areas.
            commit_actions (bool): Whether to commit the actions after selection. Defaults to True.
            verbose (bool): Verbosity flag for printing detailed operation logs. Defaults to False.

        Returns:
            float: The remaining target area that was not operated on.
        """
        key = lambda item: max(item[1])
        odt = sorted(list(self.parent.operable_dtypes(acode, period, mask).items()), key=key)
        if verbose:
            print(' entering selector.operate()', len(odt), 'operable dtypes')
        while target_area > 0 and odt:
            while target_area > 0 and odt:
                popped = odt.pop()
                try:
                    dtk, ages = popped 
                except:
                    print(odt)
                    print(popped)
                    raise
                age = sorted(ages)[-1]
                oa = self.parent.dtypes[dtk].operable_area(acode, period, age)
                if not oa: continue # nothing to operate
                area = min(oa, target_area)
                target_area -= area
                if area < 0:
                    print('negative area', area, oa, target_area, acode, period, age)
                    assert False
                if verbose:
                    print(' selector found area', [' '.join(dtk)], acode, period, age, area)
                self.parent.apply_action(dtk, acode, period, age, area, compile_c_ycomps=True,
                                         fuzzy_age=False, recourse_enabled=False, verbose=verbose)
            odt = sorted(list(self.parent.operable_dtypes(acode, period, mask).items()), key=key)
        self.parent.commit_actions(period, repair_future_actions=True)
        if verbose:
            print('GreedyAreaSelector.operate done (remaining target_area: %0.1f)' % target_area)
        return target_area


def plot_resultsFuelMitigate_deter_stoch(df_deter_stoch):
    """
    Plots the results in 050_avoid_fire example showing the net emission
    difference between deterministic and stochastic scenarios
    as a function of fuel treatment effectiveness.
    """
    df_plot = df_deter_stoch.melt(id_vars="Fuel_treatment", var_name="Scenario", value_name="Result")
    plt.figure(figsize=(6, 6))
    sns.lineplot(data=df_plot, x="Fuel_treatment", y="Result", hue="Scenario", marker="o")
    plt.title("Net emission difference (deterministic vs stochastic)")
    plt.xlabel("Fuel treatment effectiveness")
    plt.ylabel("Net emission difference between base and alternative scenarios")
    plt.grid(True)
    plt.legend()
    plt.show()


def run_cbm_fire(sit_config, sit_tables, n_steps, plot=True):
    """
    Implement a simple function to run CBM from ws3 export data in case of fire (050_dss_avoid_fire example)
    
    Args:
        sit_config (dict): Configuration settings for the SIT.
        sit_tables (dict): Tables containing SIT data, such as classifiers, disturbance types, etc.
        n_steps (int): Number of timesteps to simulate.
        plot (bool, optional): Whether to plot the annual carbon stocks. Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame representing annual net emission grouped by year.
    """
    from libcbm.input.sit import sit_reader
    from libcbm.input.sit import sit_cbm_factory 
    from libcbm.model.cbm.cbm_output import CBMOutput
    from libcbm.storage.backends import BackendType
    from libcbm.model.cbm import cbm_simulator
    sit_data = sit_reader.parse(sit_classifiers=sit_tables["sit_classifiers"],
                                sit_disturbance_types=sit_tables["sit_disturbance_types"],
                                sit_age_classes=sit_tables["sit_age_classes"],
                                sit_inventory=sit_tables["sit_inventory"],
                                sit_yield=sit_tables["sit_yield"],
                                sit_events=sit_tables["sit_events"],
                                sit_transitions=sit_tables["sit_transitions"],
                                sit_eligibilities=None)
    sit = sit_cbm_factory.initialize_sit(sit_data=sit_data, config=sit_config)
    classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
    cbm_output = CBMOutput(classifier_map=sit.classifier_value_names,
                           disturbance_type_map=sit.disturbance_name_map)
    with sit_cbm_factory.initialize_cbm(sit) as cbm:
        rule_based_processor = sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)
        cbm_simulator.simulate(cbm,
                               n_steps=n_steps,
                               classifiers=classifiers,
                               inventory=inventory,
                               pre_dynamics_func=rule_based_processor.pre_dynamics_func,
                               reporting_func=cbm_output.append_simulation_result,
                               backend_type=BackendType.numpy)
        
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    biomass_pools = ["SoftwoodMerch","SoftwoodFoliage", "SoftwoodOther", "SoftwoodCoarseRoots","SoftwoodFineRoots",
                     "HardwoodMerch", "HardwoodFoliage", "HardwoodOther", "HardwoodCoarseRoots", "HardwoodFineRoots"]
    dom_pools = ["AboveGroundVeryFastSoil", "BelowGroundVeryFastSoil", "AboveGroundFastSoil", "BelowGroundFastSoil",
                 "MediumSoil", "AboveGroundSlowSoil", "BelowGroundSlowSoil", "SoftwoodStemSnag", "SoftwoodBranchSnag",
                 "HardwoodStemSnag", "HardwoodBranchSnag"]
    biomass_result = pi[["timestep"]+biomass_pools]
    dom_result = pi[["timestep"]+dom_pools]
    total_eco_result = pi[["timestep"]+biomass_pools+dom_pools]
    annual_carbon_stocks = pd.DataFrame({"Year":pi["timestep"],
                                         "Biomass":pi[biomass_pools].sum(axis=1),
                                         "DOM":pi[dom_pools].sum(axis=1),
                                         "Total Ecosystem": pi[biomass_pools+dom_pools].sum(axis=1)})
    if plot:
        annual_carbon_stocks.groupby("Year").sum().plot(xlim=(0, n_steps), ylim=(0, None))
        plt.title("Annual carbon stock")
        plt.xlabel("Year")
        plt.ylabel("Tons of carbon")

    fi = cbm_output.classifiers.to_pandas().merge(cbm_output.flux.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    
    ecosystem_decay_emissions_pools = [
        "DecayVFastAGToAir",
        "DecayVFastBGToAir",
        "DecayFastAGToAir",
        "DecayFastBGToAir",
        "DecayMediumToAir",
        "DecaySlowAGToAir",
        "DecaySlowBGToAir",
        "DecaySWStemSnagToAir",
        "DecaySWBranchSnagToAir",
        "DecayHWStemSnagToAir",
        "DecayHWBranchSnagToAir"
    ]
    
    Carbon_Combustion_pools = [
        "DisturbanceCOProduction",
        "DisturbanceCH4Production",
        "DisturbanceCO2Production"
    ]

    GrossGrowth_pools = [
        "DeltaBiomass_AG",
        "TurnoverMerchLitterInput",
        "TurnoverFolLitterInput",
        "TurnoverOthLitterInput",
        "DeltaBiomass_BG",
        "TurnoverCoarseLitterInput",
        "TurnoverFineLitterInput"
    ]
    
    Carbon_Combustion_result = fi[["timestep"]+Carbon_Combustion_pools]
    ecosystem_decay_emissions_result = fi[["timestep"]+ecosystem_decay_emissions_pools]
    GrossGrowth_result = fi[["timestep"]+GrossGrowth_pools]
    net_emission_result = fi[["timestep"]+ecosystem_decay_emissions_pools+GrossGrowth_pools]

    annual_net_emission = pd.DataFrame({"Year": fi["timestep"],
                                        "Total emission": 44/12 * (fi[ecosystem_decay_emissions_pools].sum(axis=1) - fi[GrossGrowth_pools].sum(axis=1) + fi[Carbon_Combustion_pools].sum(axis=1))})
    if plot:
        ax = annual_net_emission.groupby("Year").sum().plot(xlim=(0, n_steps))
        plt.title("Permanent sequestration")
        ax.axhline(y=0, color="red", linestyle="--")
    
    return annual_net_emission.groupby("Year").sum() 

def resultsFuelMitigate_deter_stoch(fm, intensity, n_rep, is_use_pickle = True):
    """
    Compare the net emissions difference between base and alternative scenarios
    under deterministic and stochastic conditions for the 050_dss_avoid_fire example.

    Args:
        fm (ForestModel): The forest model instance.
        intensity (list of float): List of fire treatment intensities to evaluate.
        n_rep (int): Number of repetitions for the stochastic scenario.
        is_use_pickle (bool, optional): Whether to use cached CBM output data from a pickle file. Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame with the net emissions difference.
    """
    cbm_output_rep_pickle_path = "data/cbm_output_rep_pickle.pkl"
    disturbance_type_mapping = [ {"user_dist_type": "fire", "default_dist_type": "Wildfire"}]
    for dtype_key in fm.dtypes:
        fm.dt(dtype_key).last_pass_disturbance = "fire"
    list_scenarios_deter = []
    list_scenarios_stoch = []
    list_scenarios_deter_agg = []
    list_scenarios_stoch_agg = [] 
    if not os.path.isfile(cbm_output_rep_pickle_path) or not is_use_pickle:        
        iter = 1  
        cbm_output_rep = []
        iter_intensity = 1
        for intensity in intensity:            
            while iter <= n_rep: 
                if iter == 1:
                    fm.reset()
                    fm.areaselector = GreedyAreaSelector(fm)
                    sch = schedule_fire_areacontrol(fm, intensity=intensity)
                else: 
                    fm.reset()
                    fm.areaselector = RandomAreaSelector(fm)
                    sch = schedule_fire_areacontrol(fm, intensity=intensity)
                sit_config, sit_tables = fm.to_cbm_sit(softwood_volume_yname="swdvol",
                                                   hardwood_volume_yname="hwdvol",
                                                   admin_boundary="British Columbia",
                                                   eco_boundary="Montane Cordillera",
                                                   disturbance_type_mapping=disturbance_type_mapping)
                n_steps = 100
                cbm_output = run_cbm_fire(sit_config, sit_tables, n_steps, plot=False)
                cbm_output_rep.append(cbm_output)
                iter += 1
            avg_rep_stoch = pd.concat(cbm_output_rep[(iter_intensity-1) * n_rep + 1:iter_intensity * n_rep]).groupby("Year").mean().reset_index()
            list_scenarios_deter.append(cbm_output_rep[(iter_intensity-1) * n_rep])
            list_scenarios_stoch.append(avg_rep_stoch)
            list_scenarios_deter_agg.append(cbm_output_rep[(iter_intensity-1) * n_rep]["Total emission"].sum() / fm.horizon * fm.period_length)
            list_scenarios_stoch_agg.append(avg_rep_stoch["Total emission"].sum() / fm.horizon * fm.period_length)
            iter = 1
            iter_intensity += 1
        pickle.dump(cbm_output_rep, open(cbm_output_rep_pickle_path, "wb"))
    else:
        cbm_output_rep = pickle.load(open(cbm_output_rep_pickle_path, "rb"))
        for i in range(1, len(intensity) + 1):
            avg_rep_stoch = pd.concat(cbm_output_rep[(i-1) * n_rep + 1: i * n_rep]).groupby("Year").mean().reset_index()
            list_scenarios_deter.append(cbm_output_rep[(i-1) * n_rep])
            list_scenarios_stoch.append(avg_rep_stoch)
            list_scenarios_deter_agg.append(cbm_output_rep[(i-1) * n_rep]["Total emission"].sum() / fm.horizon * fm.period_length)
            list_scenarios_stoch_agg.append(avg_rep_stoch["Total emission"].sum() / fm.horizon * fm.period_length)

    list_scenarios_deter_agg = list_scenarios_deter_agg - list_scenarios_deter_agg[0]
    list_scenarios_stoch_agg = list_scenarios_stoch_agg - list_scenarios_stoch_agg[0]

    df_deter_stoch = pd.DataFrame({
        "Fuel_treatment": np.round(np.arange(0, 1.1, 0.1), decimals=1),
        "deterministic": list_scenarios_deter_agg,
        "Stochastic": list_scenarios_stoch_agg
    })
    return df_deter_stoch


def calculate_co2_value_stock(fm, i, product_coefficient, decay_rate, product_percentage):      
    """
    Computes the carbon stock for harvested wood products over a specified period.

    Args:
        fm (ForestModel): The forest model containing inventory data.
        i (int): The time period (in years) for which carbon stock is evaluated.
        product_coefficient (float): The coefficient associated with the product, reflecting its contribution to the stock.
        decay_rate (float): The rate at which the carbon stock decays over time, expressed as a fraction.
        product_percentage (float): The percentage of the product considered for carbon stock calculation.
    Returns:
        float: The computed carbon stock for the products over the defined period.
    """
    period = math.ceil(i / fm.period_length)
    return (
        sum(fm.compile_product(period, f"totvol * {product_coefficient} * {product_percentage}") / 10 * (1 - decay_rate)**(i - j)
            for j in range(1, i + 1)
        ) * 460 * 0.5 * 44 / 12
    )


def calculate_initial_co2_value_stock(fm, i, product_coefficient, product_percentage):
    """
    Computes the initial carbon stock for harvested wood products at the first period.

    Args:
        fm (ForestModel): The forest model containing inventory data.
        i (int): The time period for which the carbon stock is calculated, specifically for the first period.
        product_coefficient (float): The coefficient related to the product, indicating its contribution to the carbon stock.
        product_percentage (float): The percentage of the product considered for carbon stock calculation.

    Returns:
        float: The calculated carbon stock value for the specified product in the first period.
    """
    return fm.compile_product(i, f"totvol * {product_coefficient} * {product_percentage}") * 0.1 * 460 * 0.5 * 44 / 12 / fm.period_length


def hwp_carbon_stock(fm, products, product_coefficients, product_percentages, decay_rates):
    """
    Calculate the carbon stock from harvested wood products over different periods.

    Args:
        fm (ForestModel): The forest model containing inventory data.
        products (list of str): List of product names to consider for carbon stock calculation.
        product_coefficients (dict): Dictionary mapping products to their associated coefficients.
        product_percentages (dict): Dictionary mapping products to the percentage considered for carbon stock.
        decay_rates (dict): Dictionary mapping products to their decay rates.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'period' and 'co2_stock', representing the carbon stock over time.
    """
    from util import calculate_co2_value_stock, calculate_initial_co2_value_stock
    data_carbon_stock = {"period": [], "co2_stock": []}
    for i in range(0, fm.horizon * 10 + 1):
        period_value = i
        co2_values_stock = []
        for product in products:
            product_coefficient = product_coefficients[product]
            product_percentage = product_percentages[product]
            decay_rate = decay_rates[product]            
            if i == 0:
                co2_values_stock.append(0)
            if i == 1:
                co2_values_stock.append(calculate_initial_co2_value_stock(fm, i, product_coefficient, product_percentage))
            else:
                co2_values_stock.append(calculate_co2_value_stock(fm, i, product_coefficient, decay_rate, product_percentage))
        co2_value_stock = sum(co2_values_stock) / 1000
        data_carbon_stock["period"].append(period_value)
        data_carbon_stock["co2_stock"].append(co2_value_stock)
    df_carbon_stock = pd.DataFrame(data_carbon_stock)    
    return df_carbon_stock


def calculate_co2_value_emission(fm, i, product_coefficient, decay_rate, product_percentage):
    """
    Compute the carbon dioxide emissions from harvested wood products over a specified period.

    Args:
        fm (ForestModel): The forest model containing inventory data.
        i (int): The time period (in years) for which emissions are evaluated.
        product_coefficient (float): The product-specific coefficient used in emission calculation.
        decay_rate (float): The rate at which the product decays, expressed as a fraction.
        product_percentage (float): The percentage of the product considered for emission calculation.
    Returns:
        float: The computed carbon dioxide emissions value for the specified product over the defined period.
    """
    period = math.ceil(i / fm.period_length)
    return (
        sum(fm.compile_product(period, f"totvol * {product_coefficient} * {product_percentage}") * 0.1 * (1 - decay_rate)**(i - j)
            for j in range(1, i + 1)
        ) * 460 * 0.5 * 44 / 12 * decay_rate
    )


def calculate_initial_co2_value_emission(fm, i, product_coefficient, decay_rate, product_percentage):
    """
    Compute the initial carbon dioxide emissions for harvested wood products at a specific period.

    Args:
        fm (ForestModel): The forest model containing inventory data.
        i (int): The time period for which emissions are calculated.
        product_coefficient (float): The product-specific coefficient used in emission calculation.
        decay_rate (float): The rate at which the product decays, expressed as a fraction.
        product_percentage (float): The percentage of the product considered for emission calculation.

    Returns:
        float: The calculated initial carbon dioxide emissions for the specified product at the given period.
    """
    return fm.compile_product(i, f"totvol * {product_coefficient} * {product_percentage}") * 0.1 * 460 * 0.5 * 44 / 12 * decay_rate / fm.period_length


def hwp_carbon_emission(fm, products, product_coefficients, product_percentages, decay_rates):
    """
    Calculate the annual carbon emissions from harvested wood products (HWP).

    Args:
        fm (ForestModel): The forest model containing inventory data.
        products (list of str): List of product names for emission calculations.
        product_coefficients (dict): Dictionary mapping products to their associated coefficients.
        product_percentages (dict): Dictionary mapping products to the percentage used for emission calculation.
        decay_rates (dict): Dictionary mapping products to their decay rates.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'period' and 'co2_emission', representing the CO2 emissions over time.
    """
    from util import calculate_co2_value_emission, calculate_initial_co2_value_emission
    data_carbon_emission = {"period": [], "co2_emission": []}
    for i in range(0, fm.horizon * 10  + 1):
        period_value = i
        co2_values_emission = []        
        for product in products:
            product_coefficient = product_coefficients[product]
            product_percentage = product_percentages[product]
            decay_rate = decay_rates[product]            
            if i == 0:
                co2_values_emission.append(0)
            elif i == 1:
                co2_values_emission.append(calculate_initial_co2_value_emission(fm, i, product_coefficient, decay_rate, product_percentage))
            else:
                co2_values_emission.append(calculate_co2_value_emission(fm, i, product_coefficient, decay_rate, product_percentage))
        co2_value_emission = sum(co2_values_emission) / 1000
        data_carbon_emission["period"].append(period_value)
        data_carbon_emission["co2_emission"].append(co2_value_emission)
    df_carbon_emission = pd.DataFrame(data_carbon_emission)
    return df_carbon_emission


def calculate_concrete_volume(fm, i, product_coefficients, clt_percentage, credibility, clt_conversion_rate):            
    """
    Calculate the volume of concrete displaced by using cross-laminated timber (CLT).

    Args:
        fm (ForestModel): The forest model instance.
        i (int): The time period for which concrete volume is calculated.
        product_coefficients (dict): Coefficients for product volumes in the model.
        clt_percentage (float): The percentage of CLT in the products.
        credibility (float): Credibility factor accounting for uncertainties.
        clt_conversion_rate (float): CLT conversion rate factor.

    Returns:
        float: The calculated volume of displaced concrete.
    """
    period = math.ceil(i / fm.period_length)
    return fm.compile_product(period, "totvol") * product_coefficients["plumber"] * clt_percentage * credibility / clt_conversion_rate


def emission_concrete_manu(fm, product_coefficients, clt_percentage, credibility, clt_conversion_rate, co2_concrete_manu_factor):
    """
    Calculate CO2 emissions from concrete manufacturing.

    This function iterates through the specified time periods, calculating
    the CO2 emissions resulting from the manufacturing of concrete based on
    the specified parameters and assumptions.

    Args:
        fm (ForestModel): The forest model instance.
        product_coefficients (dict): Coefficients for product contributions.
        clt_percentage (float): CLT usage percentage.
        credibility (float): Credibility factor in calculations.
        clt_conversion_rate (float): Conversion rate for CLT.
        co2_concrete_manu_factor (float): CO2 emission factor for concrete manufacturing.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'period' and 'co2_concrete_manu', representing
                          the period and corresponding CO2 emissions from concrete manufacturing.
    """
    from util import calculate_concrete_volume
    df_emission_concrete_manu = {"period": [], "co2_concrete_manu": []}
    for i in range(0, fm.horizon * 10 + 1):
        period_value = i
        co2_concrete_manu = []
        if i == 0:
            co2_concrete_manu = 0
        else:
            concrete_volume = calculate_concrete_volume(fm, i, product_coefficients, clt_percentage, credibility, clt_conversion_rate)
            co2_concrete_manu = concrete_volume * co2_concrete_manu_factor * 0.1 / 1000
        df_emission_concrete_manu["period"].append(period_value)
        df_emission_concrete_manu["co2_concrete_manu"].append(co2_concrete_manu)
    df_emission_concrete_manu = pd.DataFrame(df_emission_concrete_manu)
    return df_emission_concrete_manu


def emission_concrete_landfill(
    fm,
    product_coefficients,
    clt_percentage,
    credibility,
    clt_conversion_rate,
    co2_concrete_landfill_factor
):
    """
    Calculate the CO2 emissions from the displacement of concrete landfill using CLT.

    The function iterates through the growth periods and estimates CO2 emissions
    resulting from the displacement of concrete in landfills, considering various
    factors such as conversion rates and credibility levels.

    Args:
        fm (ForestModel): Model object representing the forest model.
        product_coefficients (dict): Coefficients used in calculating product volumes.
        clt_percentage (float): Proportion of cross-laminated timber (CLT) being used.
        credibility (float): Credibility coefficient affecting the conversion calculation.
        clt_conversion_rate (float): Rate at which forest volume converts to CLT volume.
        co2_concrete_landfill_factor (float): CO2 factor representing the emissions per unit of concrete volume displaced.

    Returns:
        pandas.DataFrame: DataFrame containing the periods and the corresponding CO2 emissions
                          due to concrete displacement to landfill.
    """
    from util import calculate_concrete_volume
    df_emission_concrete_landfill = {"period": [], "co2_concrete_landfill": []}
    for i in range(0, fm.horizon * 10 + 1):
        period_value = i
        co2_concrete_landfill = []
        if i == 0:
            co2_concrete_landfill = 0
        else:
            concrete_volume = calculate_concrete_volume(
                fm,
                i,
                product_coefficients,
                clt_percentage,
                credibility,
                clt_conversion_rate,
            )
            co2_concrete_landfill = (
                concrete_volume
                * co2_concrete_landfill_factor
                * 0.1
            )
        df_emission_concrete_landfill["period"].append(period_value)
        df_emission_concrete_landfill["co2_concrete_landfill"].append(co2_concrete_landfill)
    df_emission_concrete_landfill = pd.DataFrame(df_emission_concrete_landfill)
    return df_emission_concrete_landfill


def plot_results(fm):
    """
    Plot the results of the forest management simulation.

    Args:
        fm (ForestModel): The forest model instance, containing periods and the compiled product data.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object for the plotted results.
            - ax (numpy.ndarray): An array of Axes objects corresponding to the subplots.
            - df (pandas.DataFrame): A DataFrame with the harvested area, harvested volume,
              and volume to area ratio for each period.
    """
    pareas = [fm.compile_product(period, "1.") for period in fm.periods]
    pvols = [fm.compile_product(period, "totvol") for period in fm.periods]
    df = pd.DataFrame({"period":fm.periods, "ha":pareas, "hv":pvols})
    fig, ax = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    ax[0].set_ylabel("harvest area")
    ax[0].bar(df.period, df.ha)
    ax[1].set_ylabel("harvest volume")
    ax[1].bar(df.period, df.hv)
    ax[2].set_ylabel("harvest volume:area ratio")
    ax[2].bar(df.period, (df.hv/df.ha).fillna(0))
    ax[2].set_ylim(0, None)
    return fig, ax, df


def run_cbm_avoidharvest(sit_config, sit_tables, n_steps):
    """
    Run the Carbon Budget Model (CBM) for a scenario avoiding harvest actions
    using data generated with `ForestModel.to_cbm_sit`.

    This represents the 060_dss_avoid_harvest example in this context.

    Args:
        sit_config (dict): Configuration settings for the SIT.
        sit_tables (dict): SIT data tables, including classifiers, disturbance types, age classes, inventory, yield, events, and transitions.
        n_steps (int): Number of timesteps for the CBM simulation.

    Returns:
        tuple:
            pandas.DataFrame: Annual carbon stocks over each year including Biomass, DOM, and Total Ecosystem.
            pandas.DataFrame: Annual net emissions over each year including Ecosystem decay emission, Gross growth, and Net emission.
    """
    from libcbm.input.sit import sit_reader
    from libcbm.input.sit import sit_cbm_factory 
    from libcbm.model.cbm.cbm_output import CBMOutput
    from libcbm.storage.backends import BackendType
    from libcbm.model.cbm import cbm_simulator
    sit_data = sit_reader.parse(sit_classifiers=sit_tables["sit_classifiers"],
                                sit_disturbance_types=sit_tables["sit_disturbance_types"],
                                sit_age_classes=sit_tables["sit_age_classes"],
                                sit_inventory=sit_tables["sit_inventory"],
                                sit_yield=sit_tables["sit_yield"],
                                sit_events=sit_tables["sit_events"],
                                sit_transitions=sit_tables["sit_transitions"],
                                sit_eligibilities=None)
    sit = sit_cbm_factory.initialize_sit(sit_data=sit_data, config=sit_config)
    classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
    cbm_output = CBMOutput(classifier_map=sit.classifier_value_names,
                           disturbance_type_map=sit.disturbance_name_map)
    with sit_cbm_factory.initialize_cbm(sit) as cbm:
        # Create a function to apply rule based disturbance events and transition rules based on the SIT input
        rule_based_processor = sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)
        # The following line of code spins up the CBM inventory and runs it through 200 timesteps.
        cbm_simulator.simulate(cbm,
                               n_steps=n_steps,
                               classifiers=classifiers,
                               inventory=inventory,
                               pre_dynamics_func=rule_based_processor.pre_dynamics_func,
                               reporting_func=cbm_output.append_simulation_result,
                               backend_type=BackendType.numpy)
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    biomass_pools = ["SoftwoodMerch","SoftwoodFoliage", "SoftwoodOther", "SoftwoodCoarseRoots","SoftwoodFineRoots",
                     "HardwoodMerch", "HardwoodFoliage", "HardwoodOther", "HardwoodCoarseRoots", "HardwoodFineRoots"]
    dom_pools = ["AboveGroundVeryFastSoil", "BelowGroundVeryFastSoil", "AboveGroundFastSoil", "BelowGroundFastSoil",
                 "MediumSoil", "AboveGroundSlowSoil", "BelowGroundSlowSoil", "SoftwoodStemSnag", "SoftwoodBranchSnag",
                 "HardwoodStemSnag", "HardwoodBranchSnag"]
    biomass_result = pi[["timestep"]+biomass_pools]
    dom_result = pi[["timestep"]+dom_pools]
    total_eco_result = pi[["timestep"]+biomass_pools+dom_pools]
    annual_carbon_stocks = pd.DataFrame({"Year":pi["timestep"],
                                         "Biomass":pi[biomass_pools].sum(axis=1),
                                         "DOM":pi[dom_pools].sum(axis=1),
                                         "Total Ecosystem": pi[biomass_pools+dom_pools].sum(axis=1)})
    annual_carbon_stocks = annual_carbon_stocks.groupby("Year").sum()
    fi = cbm_output.classifiers.to_pandas().merge(cbm_output.flux.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])    
    ecosystem_decay_emissions_pools = [
        "DecayVFastAGToAir",
        "DecayVFastBGToAir",
        "DecayFastAGToAir",
        "DecayFastBGToAir",
        "DecayMediumToAir",
        "DecaySlowAGToAir",
        "DecaySlowBGToAir",
        "DecaySWStemSnagToAir",
        "DecaySWBranchSnagToAir",
        "DecayHWStemSnagToAir",
        "DecayHWBranchSnagToAir"
    ]
    GrossGrowth_pools = [
        "DeltaBiomass_AG",
        "TurnoverMerchLitterInput",
        "TurnoverFolLitterInput",
        "TurnoverOthLitterInput",
        "DeltaBiomass_BG",
        "TurnoverCoarseLitterInput",
        "TurnoverFineLitterInput"
    ]
    ecosystem_decay_emissions_result = fi[["timestep"]+ecosystem_decay_emissions_pools]
    GrossGrowth_result = fi[["timestep"]+GrossGrowth_pools]
    net_emission_result = fi[["timestep"]+ecosystem_decay_emissions_pools+GrossGrowth_pools]
    annual_net_emission = pd.DataFrame({ "Year": fi["timestep"],
                                        "Ecosystem decay emission": 44/12 * fi[ecosystem_decay_emissions_pools].sum(axis=1),
                                        "Gross growth": 44/12 * -1*fi[GrossGrowth_pools].sum(axis=1),
                                        "Net emission": 44/12 * (fi[ecosystem_decay_emissions_pools].sum(axis=1)-fi[GrossGrowth_pools].sum(axis=1))})
    annual_net_emission = annual_net_emission.groupby("Year").sum()
    return annual_carbon_stocks, annual_net_emission


def stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, max_harvest):   
    """
    Evaluate the carbon impact of different forest management scenarios, calculating
    the carbon stock and emission considering harvested wood products and concrete
    displacement through cross-laminated timber (CLT).

    Args:
        fm (ForestModel): The forest model instance being simulated.
        clt_percentage (float): The percent of CLT utilization within wood products.
        credibility (float): Credibility factor for CLT production impacts.
        budget_input (float): Financial budget input for the scenario.
        n_steps (int): Number of simulation steps for the Carbon Budget Model (CBM).
        max_harvest (float): Maximum allowable harvest in the scenario.

    Returns:
        tuple: Contains two elements:
            - cbm_output_1 (pandas.DataFrame): Annual carbon stocks including harvested wood products.
            - cbm_output_2 (pandas.DataFrame): Annual net emission including impacts from concrete
              manufacturing, displacement, and wood products.
    """
    decay_rates = {"plumber":math.log(2.)/35., "ppaper":math.log(2.)/2.}
    product_coefficients = {"plumber":0.9, "ppaper":0.1}
    product_percentages = {"plumber":0.5, "ppaper":1.}
    products = ["plumber", "ppaper"]
    clt_conversion_rate = 1.
    co2_concrete_manu_factor = 298.
    concrete_density = 2.40 #ton/m3
    co2_concrete_landfill_factor = 0.00517 * concrete_density
    sch_base_scenari = schedule_harvest_areacontrol(fm, max_harvest)
    df_carbon_stock = hwp_carbon_stock(fm, products, product_coefficients, product_percentages, decay_rates)
    df_carbon_emission = hwp_carbon_emission(fm, products, product_coefficients, product_percentages, decay_rates)
    df_emission_concrete_manu = emission_concrete_manu(fm, product_coefficients, clt_percentage, credibility, clt_conversion_rate, co2_concrete_manu_factor)
    df_emission_concrete_landfill = emission_concrete_landfill(fm, product_coefficients, clt_percentage, credibility, clt_conversion_rate, co2_concrete_landfill_factor)
    disturbance_type_mapping = [{"user_dist_type": "harvest", "default_dist_type": "Clearcut harvesting without salvage"},
                            {"user_dist_type": "fire", "default_dist_type": "Wildfire"}]
    for dtype_key in fm.dtypes:
        fm.dt(dtype_key).last_pass_disturbance = "fire" if dtype_key[2] == dtype_key[4] else "harvest"
    sit_config, sit_tables = fm.to_cbm_sit(softwood_volume_yname="swdvol",
                                       hardwood_volume_yname="hwdvol",
                                       admin_boundary="British Columbia",
                                       eco_boundary="Montane Cordillera",
                                       disturbance_type_mapping=disturbance_type_mapping)
    annual_carbon_stocks, annual_net_emission = run_cbm_avoidharvest(sit_config, sit_tables, n_steps)
    df_carbon_stock = df_carbon_stock.groupby("period").sum()
    annual_carbon_stocks["HWP"] = df_carbon_stock["co2_stock"]
    annual_carbon_stocks["Total Ecosystem"] += df_carbon_stock["co2_stock"]
    df_carbon_emission =  df_carbon_emission.groupby("period").sum()
    df_emission_concrete_manu = -1 * df_emission_concrete_manu.groupby("period").sum()
    df_emission_concrete_landfill = -1 * df_emission_concrete_landfill.groupby("period").sum()
    annual_net_emission["HWP"] = df_carbon_emission["co2_emission"]
    annual_net_emission["Concrete_manufacturing"] = df_emission_concrete_manu["co2_concrete_manu"]
    annual_net_emission["Concrete_landfill"] = df_emission_concrete_landfill["co2_concrete_landfill"]
    annual_net_emission["Net emission"] += annual_net_emission["HWP"]
    annual_net_emission["Net emission"] += annual_net_emission["Concrete_manufacturing"]
    annual_net_emission["Net emission"] += annual_net_emission["Concrete_landfill"]
    cbm_output_1 = annual_carbon_stocks
    cbm_output_2 = annual_net_emission
    return cbm_output_1, cbm_output_2     


def plot_scenarios(cbm_output_1, cbm_output_2, cbm_output_3, cbm_output_4, n_steps):
    """
    Plot the carbon stocks and emissions for both base and alternative scenarios.

    This function generates a series of plots comparing the carbon stocks and emissions
    over time between a base scenario and an alternative scenario without harvesting.

    Args:
        cbm_output_1 (pandas.DataFrame): Carbon stocks for the base scenario.
        cbm_output_2 (pandas.DataFrame): Carbon emissions for the base scenario.
        cbm_output_3 (pandas.DataFrame): Carbon stocks for the alternative no-harvest scenario.
        cbm_output_4 (pandas.DataFrame): Carbon emissions for the alternative no-harvest scenario.
        n_steps (int): Number of simulation steps over which the scenarios are evaluated.

    Returns:
        None. Displays a set of plots comparing the scenarios.
    """
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 10))   
    cbm_output_1.groupby("Year").sum().plot(ax=axes[0, 0], xlim=(0, n_steps), ylim=(0, None))
    axes[0, 0].set_title("Carbon stocks over years (base scenario)")
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylabel("Carbon stocks")
    cbm_output_2.groupby("Year").sum().plot(ax=axes[1, 0], xlim=(0, n_steps))
    axes[1, 0].axhline(y=0, color="red", linestyle="--")
    axes[1, 0].set_title("Carbon emission over years (base scenario)")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].set_ylabel("Carbon emission")
    cbm_output_3.groupby("Year").sum().plot(ax=axes[0, 1], xlim=(0, n_steps), ylim=(0, None))
    axes[0, 1].set_title("Carbon stocks over years (alternative scenario: no harvesting)")
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].set_ylabel("Carbon stocks")
    cbm_output_4.groupby("Year").sum().plot(ax=axes[1, 1], xlim=(0, n_steps))
    axes[1, 1].axhline(y=0, color="red", linestyle="--")
    axes[1, 1].set_title("Carbon emission over years (alternative scenario: no harvesting)")
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Carbon emission")
    plt.tight_layout()
    plt.show()


def scenario_dif(cbm_output_2, cbm_output_4, budget_input, n_steps):
    """
    Calculate and plot the difference in net emissions between base and alternative scenarios.

    This function computes the net emission difference between the base and alternative scenarios
    over a series of years, plots this difference, and calculates the cost in dollars per ton of carbon.

    Args:
        cbm_output_2 (pandas.DataFrame): The net emissions data for the base scenario.
        cbm_output_4 (pandas.DataFrame): The net emissions data for the alternative scenario.
        budget_input (float): The financial input budget for the scenario in dollars.
        n_steps (int): The number of time steps over which the scenarios are analyzed.

    Returns:
        matplotlib.axes.Axes: The axes object with the plotted net emission differences.
    """
    cbm_output_2.reset_index(drop=False, inplace=True)
    dif_scenario = pd.DataFrame({"Year": cbm_output_2["Year"],
                       "Net emission": cbm_output_4["Net emission"] - cbm_output_2["Net emission"]})
    ax = dif_scenario.groupby("Year").sum().plot(xlim = (0, n_steps))
    ax.axhline(y=0, color="red", linestyle="--")
    ax.set_title("Net emission difference between base and alternative scenarios")
    ax.set_xlabel("Year")
    ax.set_ylabel("Net Carbon emission difference")
    dollar_per_ton = abs(budget_input / dif_scenario.iloc[:25]["Net emission"].sum())
    print( "Net emission difference", dif_scenario.iloc[:25]["Net emission"].sum())
    print( "Net emission base scenario", cbm_output_2.iloc[:25]["Net emission"].sum())
    print( "Net emission alternative scenario", cbm_output_4.iloc[:25]["Net emission"].sum())
    print("dollar_per_ton is: ", dollar_per_ton)
    return ax


def results_scenarios(fm, clt_percentage, credibility, budget_input, n_steps, max_harvest):
    """
    Evaluate and compare carbon stocks and emissions for two forest management scenarios.

    This function computes the carbon stocks and emissions for both a base scenario (with harvesting)
    and an alternative scenario (without harvesting) using the specified parameters. It then plots
    the results to visualize the differences between these scenarios and displays the net emission
    differences over the specified number of steps.

    Args:
        fm (ForestModel): The forest model instance being simulated.
        clt_percentage (float): The percentage of cross-laminated timber (CLT) utilization within wood products.
        credibility (float): Credibility factor for CLT production impacts.
        budget_input (float): Financial budget input for the scenario.
        n_steps (int): Number of simulation steps for the Carbon Budget Model (CBM).
        max_harvest (float): Maximum allowable harvest in the base scenario.

    Returns:
        None. The function directly plots the comparison between the base and alternative scenarios
        and provides financial metrics related to the carbon emissions difference.
    """
    from util import stock_emission_scenario, plot_scenarios, scenario_dif
    cbm_output_1, cbm_output_2 = stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, max_harvest)
    fm.reset()
    cbm_output_3, cbm_output_4 = stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, 0)
    plot_scenarios(cbm_output_1, cbm_output_2, cbm_output_3, cbm_output_4, n_steps)
    dif_plot = scenario_dif(cbm_output_2, cbm_output_4, budget_input, n_steps)
