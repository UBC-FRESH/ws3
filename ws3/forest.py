###################################################################################
# MIT License

# Copyright (c) 2015-2017 Gregory Paradis

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###################################################################################

"""
This module implements functions for building and running the wood supply simulation
models.

The ``ForestModel`` and ``DevelopmentType`` classes constitute the core functional units of this module, and of the ``ws3`` package in general.
"""

import math
import sys
import re
import copy
import operator
import random
import itertools
from itertools import chain
from functools import reduce
_cfi = chain.from_iterable
from collections import defaultdict as dd
import pandas as pd


try:
    from ws3 import common
    from ws3 import core
    from ws3 import opt
except: # "__main__" case
    from ws3 import common
    from ws3 import core
    from ws3 import opt
from ws3.common import timed

from pdb import set_trace
    
#_mad = common.MAX_AGE_DEFAULT


class GreedyAreaSelector:
    """
    Default AreaSelector implementation. Selects areas for treatment from oldest age classes.
    """
    def __init__(self, parent):
        self.parent = parent
        
    def operate(self, period, acode, target_area, mask=None,
                commit_actions=True, verbose=False):
        """
        Greedily operate on oldest operable age classes.
        Returns missing area (i.e., difference between target and operated areas).

        :param int period: The time period for the operation.
        :param str acode: The action code to specify the action.
        :param float target_area: The desired area to be achieved through operation.
        :param tuple mask: Tuple of values for development types.
        :param bool commit_actions: Flag indicating whether to commit actions. Defaults to True.
        :param bool verbose: Verbosity flag. Defaults to False.
         
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
    
class Action:
    """
    Encapsulates data for an action.
    
    """
    def __init__(self,
                 code,
                 targetage=None,
                 descr='',
                 lockexempt=False,
                 components=None,
                 partial=None,
                 is_harvest=0,
                 is_sticky=0):
        self.code = code
        self.targetage = targetage
        self.descr = descr
        self.lockexempt = lockexempt
        self.oper_a = None 
        self.oper_p = None
        self.components = components or []
        self.partial = partial or []
        self.is_compiled = False
        self.is_harvest = is_harvest
        self.is_sticky = is_sticky
        self.treatment_type = None
    
class DevelopmentType:
    """
    Encapsulates Forest development type data (curves, age, area), and provides methods to operate on the data.
    """
    _bo = {'AND':operator.and_, '&':operator.and_, 'OR':operator.or_, '|':operator.or_}
    
    def __init__(self,
                 key,
                 parent):
        """
        The key is basically the fully expanded mask (expressed as a tuple of values). 
        The parent is a reference to the ForestModel object in which self is embedded.
        """
        self.key = key
        self.parent = parent
        self._rc = parent.register_curve # shorthand
        self._max_age = parent.max_age
        self._ycomps = {}
        self._complex_ycomps = {}
        self._zero_curve = parent.common_curves['zero']
        self._unit_curve = parent.common_curves['unit']
        self._ages_curve = parent.common_curves['ages']                           
        self._resolvers = {'MULTIPLY':self._resolver_multiply,
                           'DIVIDE':self._resolver_divide,
                           'SUM':self._resolver_sum,
                           'CAI':self._resolver_cai,
                           'MAI':self._resolver_mai,
                           'YTP':self._resolver_ytp,
                           'RANGE':self._resolver_range}
        self.transitions = {} # keys are (acode, age) tuples
        #######################################################################
        # Use period 0 slot to store starting inventory.
        self._areas = {p:dd(float) for p in range(0, self.parent.horizon+1)}
        #######################################################################
        self.oper_expr = dd(list)
        self.operability = {}        

    def operable_ages(self, acode, period):
        """
        Finds list of ages at which self is operable, given an action code and period index.
        """
        if acode not in self.oper_expr: # action not defined for this development type
            return None
        if acode not in self.operability: # action not compiled yet...
            if self.compile_action(acode) == -1: return None # never operable
        #print ' '.join(self.key), acode, period, self.operability, self.oper_expr
        if period not in self.operability[acode]:
            return None
        else:
            lo, hi = self.operability[acode][period]
            return list(set(range(lo, hi+1)).intersection(list(self._areas[period].keys())))        
    
    def is_operable(self, acode, period, age=None, verbose=False):
        """
        Test hypothetical operability.
        Does not imply that there is any operable area in current inventory.

        :param str acode: The action code to test operability for.
        :param int period: The period to test operability for.
        :param int age: The age to test operability for. If None, only checks operability for the period.
        :param bool verbose: If True, prints additional information for debugging purposes. Default is False.
        """
        if acode not in self.oper_expr: # action not defined for this development type
            if verbose: print('acode operability undefined', acode, self.oper_expr)
            return False
        if acode not in self.operability: # action not compiled yet...
            if self.compile_action(acode) == -1:
                if verbose: print('never operable', acode)
                return False # never operable
        if period not in self.operability[acode]:
            return False
        else:
            lo, hi = self.operability[acode][period]
            if age is not None:
                return age >= lo and age <= hi
            else:
                return lo, hi
            
    def operable_area(self, acode, period, age=None, cleanup=True):
        """
        Returns 0 if inoperable or no current inventory, operable area given action code and period 
        (and optionally age) index otherwise.

        :param str acode: The action code to determine operability.
        :param int period: The period to determine operability for.
        :param int age: The age to determine operability for. If None, only checks operability for the period.
        :param bool cleanup: If True (default), removes the age class from the inventory dict if operable area is less than                                 self.parent.area_epsilon.

        """
        if acode not in self.oper_expr: # action not defined for this development type
            return 0.
        if acode not in self.operability: # action not xf yet...
            if self.compile_action(acode) == -1: return 0. # never operable
        if age is None: # return total operable area
            return sum(self.operable_area(acode, period, a) for a in list(self._areas[period].keys()))
        if age not in self._areas[period]:
            # age class not in inventory
            return 0.
        elif abs(self._areas[period][age]) < self.parent.area_epsilon:
            # negligible area
            #print 'no area', acode, period, age, self._areas[period][age]
            #print ' '.join(self.key)
            #print self._areas[period].keys()
            if cleanup: # remove ageclass from dict (frees up memory)
                del self._areas[period][age]
            return 0.
        elif self.is_operable(acode, period, age):
            #print 'operable', acode, period, age #, self.operability[acode]
            return self._areas[period][age]
        else:
            return 0.
        assert False


    #def reset_areas(self, period):
    #    for age in self._areas[period].keys():
    #        self._areas[period][age] = 0.
            
        
    def area(self, period, age=None, area=None, delta=True):
        """
        If area not specified, returns area inventory for period (optionally age), else sets area for period and age. 
        If delta switch active (default True), area value is interpreted as an increment on current inventory.
        
        :param int period: The period for which the area is being retrieved or set.
        :param int age: The age for which the area is being retrieved or set. If None, returns total area.
        :param float area:  The area value to set. If None, returns the area inventory.
        :param bool delta:  If True (default), interprets the area value as an increment on the current inventory. 
                            If False, sets the area value directly.       
        """
        #if area is not None:
        #    print area
        #    assert area > 0
        if area is None: # return area for period and age
            if age is not None:
                try:
                    return self._areas[period][age]
                except Exception as e:
                    print(e)
                    return 0.
            else: # return total area
                return sum(self._areas[period][a] for a in self._areas[period])
        else: 
            if delta:
                self._areas[period][age] += area
            else:
                self._areas[period][age] = area
        
    def resolve_condition(self, yname, lo, hi):
        """
        Find lower and upper ages that correspond to lo and hi values of yname (interpreted as first occurence of yield value, reading curve from left and right, respectively).
        """
        return [x for x, y in enumerate(self.ycomp(yname)) if y >= lo and y <= hi]
       
    def reset_areas(self, period=None):
        """
        Reset areas dictionary.
        """
        periods = self.parent.periods if period is None else [period]
        for period in periods:
            self._areas[period] = dd(float)

    def ycomps(self):
        """
        Returns list of yield component keys.
        """
        return list(self._ycomps.keys())
            
            
    def ycomp(self, yname, silent_fail=True):
        """
        Returns the yield components associated with the given yield name. Returns None if the yield name is not found and silent_fail is True.

        :param str yname: The name of the yield to retrieve components for.
        :param bool silent_fail: If True (default), returns None if the yield name is not found. If False, raises a KeyError                                    that yield name is not found.        
        """
        if yname in self._ycomps:
            if not self._ycomps[yname]: # complex ycomp not compiled yet
                self._compile_complex_ycomp(yname)
            return self._ycomps[yname]
        else: # not a valid yname
            if silent_fail:
                return None 
            else: 
                 raise KeyError("ycomp '%s' not in development type '%s'" % (yname, ' '.join(self.key)))
                    
    def _o(self, s, default_ycomp=None): # resolve string operands
        if not default_ycomp: default_ycomp = self._zero_curve
        if common.is_num(s):
            return float(s)
        elif s.startswith('#'):
            return self.parent.constants[s[1:]]
        else:
            s = s.lower() # just to be safe
            ycomp = self.ycomp(s)
            return ycomp if ycomp else default_ycomp
        
    def _resolver_multiply(self, yname, d):
        args = [self._o(s.lower()) for s in re.split('\s?,\s?', re.search('(?<=\().*(?=\))', d).group(0))]
        ##################################################################################################
        # NOTE: Not consistent with Remsoft documentation on 'complex-compound yields' (fix me)...
        ytype_set = set(a.type for a in args if isinstance(a, core.Curve))
        return ytype_set.pop() if len(ytype_set) == 1 else 'c', self._rc(reduce(lambda x, y: x*y, args))
        ##################################################################################################

    def _resolver_divide(self, yname, d):
        _tmp = list(zip(re.split('\s?,\s?', re.search('(?<=\().*(?=\))', d).group(0)),
                   (self._zero_curve, self._unit_curve)))
        args = [self._o(s, default_ycomp) for s, default_ycomp in _tmp]
        return args[0].type if not args[0].is_special else args[1].type, self._rc(args[0] / args[1])
        
    def _resolver_sum(self, yname, d):
        #print 'resolving SUM', yname, d
        args = [self._o(s.lower()) for s in re.split('\s?,\s?', re.search('(?<=\().*(?=\))', d).group(0))] 
        ytype_set = set(a.type for a in args if isinstance(a, core.Curve))
        #print [a.label, a.type for a in args if isinstance(a, core.Curve)]
        return ytype_set.pop() if len(ytype_set) == 1 else 'c', self._rc(reduce(lambda x, y: x+y, [a for a in args]))
        
    def _resolver_cai(self, yname, d):
        arg = self._o(re.split('\s?,\s?', re.search('(?<=\().*(?=\))', d).group(0))[0])
        return arg.type, self._rc(arg.mai())
        
    def _resolver_mai(self, yname, d):
        arg = self._o(re.split('\s?,\s?', re.search('(?<=\().*(?=\))', d).group(0))[0])
        return arg.type, self._rc(arg.mai())
        
    def _resolver_ytp(self, yname, d):
        arg = self._o(re.search('(?<=\().*(?=\))', d).group(0).lower())
        return arg.type, self._rc(arg.ytp())
        
    def _resolver_range(self, yname, d):
        args = [self._o(s.lower()) for s in re.split('\s?,\s?', re.search('(?<=\().*(?=\))', d).group(0))] 
        arg_triplets = [args[i:i+3] for i in range(0, len(args), 3)]
        range_curve = self._rc(reduce(lambda x, y: x*y, [t[0].range(t[1], t[2]) for t in arg_triplets]))
        #print ' '.join(self.key), yname, range_curve.points()
        return args[0].type, self._rc(reduce(lambda x, y: x*y, [t[0].range(t[1], t[2]) for t in arg_triplets]))

    def _compile_complex_ycomp(self, yname):
        expression = self._complex_ycomps[yname]
        keyword = re.search('(?<=_)[A-Z]+(?=\()', expression).group(0)
        #print('compiling complex', yname, keyword, expression)
        try:
            ytype, ycomp = self._resolvers[keyword](yname, expression)
            ycomp.label = yname
            ycomp.type = ytype
            self._ycomps[yname] = ycomp
            #assert False
        except KeyError:
                raise ValueError('Problem compiling complex yield: %s, %s' % (yname, expression))
            
    def compile_actions(self, verbose=False):
        """
        Compile all actions.

        :param bool verbose: Verbosity flag. Defaults to False.
        
        """
        for acode in self.oper_expr:
            self.compile_action(acode, verbose)

    def compile_action(self, acode, verbose=False):
        """
        Compile action, given action code. 
        This mostly involves resolving operability expression strings into
        lower and upper operability limits, defined as (alo, ahi) age pair for each period.
        Deletes action from self if not operable in any period.
        
        :param str acode: The action code.
        :param bool verbose: Verbosity flag. Defaults to False.
        """
        self.operability[acode] = {}
        for expr in self.oper_expr[acode]:
            self._compile_oper_expr(acode, expr, verbose)
        is_operable = False
        for p in self.operability[acode]:
            if self.operability[acode][p] is not None:
                #print 'compile_action', expr, acode, p, self.operability[acode][p]
                is_operable = True
        if not is_operable:
            if verbose: print('not operable (deleting):', acode)
            del self.operability[acode]
            del self.oper_expr[acode]
            return -1
        else:
            if verbose: print('operable:', acode) #, self.operability[acode]
        return 0

    def _compile_oper_expr(self, acode, expr, verbose=False):
        expr = expr.replace('&', 'and').replace('|', 'or')
        oper = None
        plo, phi = 1, self.parent.horizon # count periods from 1, as in Forest...
        alo, ahi = 0, self._max_age 
        if 'and' in expr:
            oper = 'and'
        elif 'or' in expr:
            oper = 'or'
            alo, ahi = self._max_age+1, -1
        cond_comps = expr.split(' %s ' % oper)
        lhs, rel_operators, rhs = list(zip(*[cc.split(' ') for cc in cond_comps]))
        rhs = list(map(float, rhs))
        _plo, _phi, _alo, _ahi = None, None, None, None
        for i, o in enumerate(lhs):
            if o == '_cp':
                #print 'rhs', rhs
                period = int(rhs[i])
                assert period <= self.parent.horizon # sanity check
                #################################################################
                # Nonsense to relate time-based and age-based conditions with OR?
                # Recondider if this actually ever comes up...
                assert oper != 'or'  
                #################################################################
                if rel_operators[i] == '=':
                    _plo, _phi = period, period
                elif rel_operators[i] == '>=':
                    _plo = period
                elif rel_opertors[i] == '<=':
                    _phi = period
                else:
                    raise ValueError('Bad relational operator.')
                plo, phi = max(_plo, plo), min(_phi, phi)
            elif o == '_age':
                age = int(rhs[i])
                if rel_operators[i] == '=':
                    _alo, _ahi = age, age
                elif rel_operators[i] == '>=':
                    _alo = age
                elif rel_operators[i] == '<=':
                    _ahi = age
                else:
                    raise ValueError('Bad relational operator.')                    
            else: # must be yname
                ycomp = self.ycomp(o)
                if rel_operators[i] == '=':
                    _alo = _ahi = ycomp.lookup(rhs[i])
                elif rel_operators[i] == '>=':
                    #print ' ge', o, ycomp[45], ycomp.lookup(0)  
                    _alo = ycomp.lookup(rhs[i])
                elif rel_operators[i] == '<=':
                    #print ' le', o 
                    _ahi = ycomp.lookup(rhs[i])
                else:
                    raise ValueError('Bad relational operator.')
                #print ' ', o, (alo, _alo), (ahi, _ahi)
        if oper == 'and' or not oper:
            if _alo is not None: alo = max(_alo, alo)
            if _ahi is not None: ahi = min(_ahi, ahi)
        else: # or
            if _alo is not None: alo = min(_alo, alo)
            if _ahi is not None: ahi = max(_ahi, ahi)
        #if plo >= phi:
        #    print(plo, phi)
        assert plo <= phi # should never explicitly declare infeasible period range...
        for p in range(plo, phi+1):
            assert alo <= ahi
            #print self.key, acode, p, alo, ahi
            self.operability[acode][p] = (alo, ahi) if alo <= ahi else None
            #print acode, p, (alo, ahi), expr
            
                
    def add_ycomp(self, ytype, yname, ycomp, first_match=True):
        
        """
        
        Adds a yield component to the yield components.
        
        :param str ytype: The type of yield component to add ('c' for complex).
        :param str yname: The name of the yield component.
        :param str ycomp: The yield component  to add.
        :param bool first_match: Flag indicating whether to only add the component if it doesn't already exist. Defaults to True.
    
        """
       
        if first_match and yname in self._ycomps: return # already exists (reject)
        if ytype == 'c':
            self._complex_ycomps[yname] = ycomp
            self._ycomps[yname] = None
        if isinstance(ycomp, core.Curve):
            self._ycomps[yname] = ycomp
    
    def grow(self, start_period=1, cascade=True):
        """
        Grow self (default starting period 1, and cascading to end of planning horizon).

        :param int start_period: The starting period for growth (default is 1).
        :param bool cascade: If True, growth cascades to the end of the planning horizon. Default is True.

        """
        
        end_period = start_period + 1 if not cascade else self.parent.horizon
        for p in range(start_period, end_period):
            self.reset_areas(p+1) #, self._areas[p], self._areas[p+1] # WTF?
            #for age, area in list(self._areas[p].items()): self._areas[p+1][age+1] = area
            for age, area in list(self._areas[p].items()): self._areas[p+1][age+self.parent.period_length] = area

    def overwrite_initial_areas(self, period):

        """
        Overwrites the initial areas for a specified period.   
        """
        self._areas[0] = copy.copy(self._areas[period])
        self.initialize_areas()
            
    def initialize_areas(self):
        """
        Copy initial inventory to period-1 inventory.
        """
        self._areas[1] = copy.copy(self._areas[0])
        
class Output:
    """
    Encapsulates data and methods to operate on aggregate outputs from the model.
    Emulates behaviour of Forest outputs.
    
    .. warning:: 
       Behaviour of Forest outputs is quite complex. This class needs more work before it is used in a production setting (i.e., resolution of some complex output cases is buggy).
  
    """
    def __init__(self,
                 parent,
                 code=None,
                 expression=None,
                 factor=(1., 1),
                 description='',
                 theme_index=-1,
                 is_basic=False,
                 is_level=False):
        self.parent = parent
        self.code = code
        self.expression = expression
        self._factor = factor
        self.description = description
        self.theme_index = theme_index
        self.is_themed = True if theme_index > -1 else False 
        self.is_basic = is_basic
        if is_basic:
            self._compile_basic(expression) # shortcut
        elif not is_level:
            self._compile(expression) # will detect is_basic
        self.is_level = is_level

    def _lval(self, s):
        """
        Resolve left operand in sub-expression.
        """
        if s.lower() in self.parent.outputs:
            return self.parent.outputs[s.lower()]
        else: # expression
            return s.lower()

    def _rval(self, s): 
        """
        Resolve right operand in sub-expression.
        """
        if common.is_num(s):
            return float(s)
        elif s.startswith('#'):
            return self.parent.constants[s[1:].lower()]
        else: # time-based ycomp code
            return s.lower()
            
    def _compile(self, expression):
        """
        Resolve operands in expression to the extent possible.
        Can be basic or summary.
        Assuming operand pattern:
          lval_1 [*|/ rval_1] +|- .. +|- lval_n [*|/ rval_n]
        where
          lval := ocode or expression
          rval := number or #constant or ycomp
        """
        t = re.split(r'\s+(\+|-)\s+', expression)
        ocomps = t[::2]  # output component sub-expressions
        signs = [1.] # implied + in front of expression
        signs.extend(1. if s == '+' else -1 for s in t[1::2]) 
        factors = [(1., 1) for i in ocomps]
        for i, s in enumerate(ocomps):
            tt = re.split(r'\s+(\*|/)\s+', s) # split on */ operator
            lval = self._lval(tt[0])
            if len(tt) > 1:
                factors[i] = self._rval(tt[2]), 1 if tt[1] == '*' else -1
            if not isinstance(lval, Output):     
                if len(ocomps) == 1: # simple basic output (special case)
                    self.is_basic = True
                    self._factor = factors[0]
                    self._compile_basic(lval)
                    return
                else: # compound basic output
                    ocomps[i] = Output(parent=self.parent,
                                       expression=lval,
                                       factor=factors[i],
                                       is_basic=True)
            else: # summary output
                ocomps[i] = lval #self.parent.outputs[lval]
        self._ocomps = ocomps
        self._signs = signs
        self._factors = factors

    def _compile_basic(self, expression):
        # clean up (makes parsing easier)
        s = re.sub('\s+', ' ', expression) # separate tokens by single space
        s = s.replace(' (', '(')  # remove space to left of left parentheses
        t = s.lower().split(' ')
        # filter dtypes, if starts with mask
        mask = None
        if not (t[0] == '@' or t[0] == '_' or t[0] in self.parent.actions):
            mask = tuple(t[:self.parent.nthemes()])
            t = t[self.parent.nthemes():] # pop
        #try:
        #print expression
        #self._dtype_keys = self.parent.unmask(mask) if mask else self.parent.dtypes.keys()
        #except:
        #    print expression
        #    assert False
        # extract @AGE or @YLD condition, if present
        self._ages = None
        self._condition = None
        if t[0].startswith('@age'):
            lo, hi = [int(a)+i for i, a in enumerate(t[0][5:-1].split('..'))]
            hi = min(hi, self.parent.max_age+1) # they get carried away with range bounds...
            self._ages = list(range(lo, hi))
            t = t[1:] # pop
        elif t[0].startswith('@yld'):
            ycomp, args = t[0][5:-1].split(',')
            self._condition = tuple([ycomp] + [float(a) for a in args.split('..')])
            self._ages = None
            t = t[1:] # pop
        if not self._ages and not self._condition: self._ages = self.parent.ages
        # extract _INVENT or acode
        if t[0].startswith('_'): # _INVENT
            self._is_invent = True
            self._invent_acodes = t[0][8:-1].split(',') if len(t[0]) > 7 else None
            self._acode = None
        else: # acode
            self._is_invent = False
            self._invent_acodes = None
            self._acode = t[0]
        t = t[1:] # pop
        # extract _AREA or ycomp
        if t[0].startswith('_'): # _AREA
            self._is_area = True
            self._ycomp = None
        else: # acode
            self._is_area = False
            self._ycomp = t[0]
        t = t[1:] # pop

    def _evaluate_basic(self, period, factors, verbose=0, cut_corners=True):
        result = 0.
        if self._invent_acodes:
            acodes = [acode for acode in self._invent_acodes if parent.applied_actions[period][acode]]
            if cut_corners and not acodes:
                return 0. # area will be 0...
        for k in list(self.parent.dtypes.keys()):
            dt = self.parent.dtypes[k]
            if cut_corners and not self._is_invent and k not in self.parent.applied_actions[period][self._acode]:
                if verbose: print('bailing on', period, self._acode, ' '.join(k))
                continue # area will be 0...
            if isinstance(self._factor[0], float):
                f = pow(*self._factor)
            else:
                f = pow(dt.ycomp(self._factor[0])[period], self._factor[1])
            for factor in factors:
                if isinstance(factor[0], float):
                    f *= pow(*factor)
                else:
                    f *= pow(dt.ycomp(factor[0])[period], factor[0])
            if cut_corners and not f:
                if verbose: print('f is null', f)
                continue # one of the factors is 0, no point calculating area...
            ages = self._ages if not self._condition else dt.resolve_condition(*self._condition)
            for age in ages:
                area = 0.
                if self._is_invent:
                    if cut_corners and not dt.area(period, age):
                        continue
                    if self._invent_acodes:
                        any_operable = False
                        for acode in acodes:
                            if acode not in dt.operability: continue
                            if dt.is_operable(acode, period, age):
                                any_operable = True
                        if any_operable:
                            area += dt.area(period, age)
                    else:
                        area += dt.area(period, age)
                else:
                    assert False # not implemented yet...
                y = 1. if self._is_area else dt.ycomp(self._ycomp)[age]
                result += y * area * f
        return result

    def _evaluate_summary(self, period, factors):
        result = 0.
        for i, ocomp in enumerate(self._ocomps):
            result += ocomp(period, [self._factors[i]] + factors)
        return result

    def _evaluate_basic_themed(self, period):
        pass

    def _evaluate_summed_themed(self, period):
        pass
            
    def __call__(self, period, factors=[(1., 1)]):
        if self.is_basic:
            return self._evaluate_basic(period, factors)
        else:
            return self._evaluate_summary(period, factors)

    def __add__(self, other):
        # assume Output + Output
        if self.is_themed:
            return [i + j for i, j in zip(self(), other())]
        else:
            return self() + other()

    def __sub__(self, other):
        # assume Output - Output 
        if self.is_themed:
            return [i - j for i, j in zip(self(), other())]
        else:
            return self() - other()

class ForestModel:
    """
    This is the core class of the ws3 package.
    Includes methods import data from various sources, simulate growth and apply actions.
    The model can be used in either a (prescriptive) simulation-based approach or a (descriptive) 
    optimization-based approach.

    This class encapsulates all the information used to simulate scenarios from a given dataset (i.e., 
    stratified intial inventory, growth and yield functions, action eligibility, transition matrix, action 
    schedule, etc.), as well as a large collection of functions to import and export data, generate activity 
    schedules, and simulate application of these schedules  (i.e., run scenarios).

    At the heart of the ``ForestModel`` class is a list of ``DevelopentType`` instances. Each ``DevelopmentType``
    instance encapsulates information about one development type (i.e., a forest stratum, which is an aggregate of 
    smaller *stands* that make up the raw forest inventory input data). The ``DevelopmentType`` class also 
    stores a list of operable *actions*, maps *state variable transitions* to these actions, stores growth and 
    yield functions, and knows how to *grow itself* when time is incremented during a simulation.

    A typical use case starts with creating an instance of the ``ForestModel`` class. Then, we need to load data 
    into this instance, define one or more scenarios (using a mix of heuristic and optimization approaches), 
    run the scenarios, and export output data to a format suitable for analysis (or link to the next model in 
    a larger modelling pipeline).  
    """
    _ytypes = {'*Y':'a', '*YT':'t', '*YC':'c'}
    tree = (lambda f: f(f))(lambda a: (lambda: dd(a(a))))
    #_vp_ratio_default = 1.
    #_piece_size_yname_default = 'yd3s'
    #_piece_size_factor_default = 0.001 # convert cubic decimeters to cubic meters
    #_total_volume_yname_default = 'yv_s'

            
    def __init__(self,
                 model_name,
                 model_path,
                 base_year,
                 horizon=common.HORIZON_DEFAULT,
                 period_length=common.PERIOD_LENGTH_DEFAULT,
                 #aggr_period_length=common.PERIOD_LENGTH_DEFAULT,
                 max_age=common.MAX_AGE_DEFAULT,
                 #species_groups=common.SPECIES_GROUPS_FOREST_QC, # not used (DELETE) [commenting out]
                 area_epsilon=common.AREA_EPSILON_DEFAULT,
                 curve_epsilon=common.CURVE_EPSILON_DEFAULT):
                 #vp_ratio=_vp_ratio_default,
                 #piece_size_yname=_piece_size_yname_default,
                 #piece_size_factor=_piece_size_factor_default,
                 #total_volume_yname=_total_volume_yname_default):
        """
         Initializes the ForestModel with the provided parameters.

         :param str model_name: The name of model.
         :param str model_path: The path to input data of model.
         :param int base_year: The base year of teh model.
         :param int horizon: The simulation horizon of the model.
         :param int horizon: The length of the simulation period.
         :param int max_age: The maximum age considered in the model.
         :param int area_epsilon: 
         :param int curve_epsilon: 
                           
        """       
        self.model_name = model_name
        self.model_path = model_path
        self.base_year = base_year
        self.set_horizon(horizon)
        self.period_length = period_length
        #self.aggr_period_length = aggr_period_length
        #self._period_coeff = float(period_length) / float(aggr_period_length)
        #assert self._period_coeff <= 1.
        self.max_age = max_age
        self.ages = list(range(max_age+1))
        #self._species_groups = species_groups # Not used (DELETE) [commenting out]
        self.yields = []
        self.ynames = set()
        self.actions = {}
        self.transitions = {}
        self.oper_expr = {}
        self._themes = []
        self._theme_basecodes = []
        self.dtypes = {}
        self.constants = {}
        self.output_groups = {}
        self.outputs = {}
        self.applied_actions = {p:{acode:{} for acode in list(self.actions.keys())} for p in self.periods}
        self.reset_actions()
        self.curves = {}
        self.problems = {}
        c_zero = self.register_curve(core.Curve('zero',
                                                is_special=True,
                                                type=''))
        c_unit = self.register_curve(core.Curve('unit',
                                                points=[(0, 1)],
                                                is_special=True,
                                                type=''))
        c_ages = self.register_curve(core.Curve('ages',
                                                points=[(0, 0), (max_age, max_age)],
                                                is_special=True,
                                                type='')) 
        self.common_curves = {'zero':c_zero,
                              'unit':c_unit,
                              'ages':c_ages}
        self.area_epsilon = area_epsilon
        self.curve_epsilon = curve_epsilon
        self.areaselector = GreedyAreaSelector(self)
        self.inoperable_dtypes = []
        #self._vp_ratio = vp_ratio
        #self.piece_size_yname = piece_size_yname
        #self.piece_size_factor = piece_size_factor
        #self.total_volume_yname = total_volume_yname
        self._problems = {}
        #self.nthemes = 0

    def nthemes(self):
        """
        Returns number of themes
        """
        return len(self._themes)
        
    def reset(self):
        """
        Resets the forest model by clearing applied actions and reinitializing areas.
        """
        self.reset_actions()
        self.initialize_areas()
        
        
    def set_horizon(self, horizon):
        """
        Sets the horizon of the model.

        This method updates the horizon of the model to the specified value and adjusts the list of periods accordingly.
        """
        self.horizon = int(horizon)
        self.periods = list(range(1, horizon+1))
        
        
    def compile_actions(self, mask=None, verbose=False):
        """
        Compile actions for the development types filtered by mask.
        """
        dtype_keys = self.unmask(mask) if mask else list(self.dtypes.keys())
        for dtk in dtype_keys:
            dt = self.dtypes[dtk]
            dt.compile_actions(verbose=verbose)            
        
    def _compile_schedule_from_problem(self, problem, formulation=1, skip_null='null'):
        """
        Compiles a ``ws3``-compatible schedule data object from a solved ``ws3.opt.Problem`` instance. 
        This is just a dispatcher function---the actual compilation is done by a formulation-specific function 
        (assumes *Model I* formulation if not specified).
        """
        cmp_sch_dsp = {1:self._cmp_sch_m1, 2:self._cmp_sch_m2}
        return cmp_sch_dsp[formulation](problem, skip_null)

    def _cmp_sch_m1(self, problem, skip_null):
        _sch = [[] for t in self.periods]
        sln = problem.solution()
        for i, tree in list(problem.trees.items()):
            for path in tree.paths():
                x = 'x_%i' % hash((i, tuple(n.data('acode') for n in path)))
                if not sln[x]: continue
                for t, n in enumerate(path):
                    d = n.data()
                    if skip_null and d['acode'] == skip_null: continue
                    #print 'sch', i, d['dtk'], d['period'], d['acode'], d['age'], d['area'], '%0.1f' % (d['area'] * sln[x])
                    etype = '_existing' if self.dt(i[0]).area(0) else '_future'
                    _sch[t].append((d['dtk'], d['age'], d['area'] * sln[x], d['acode'], d['period'], etype))
                    #_sch[t].append((d['dtk'], d['age'], d['area'] * 1., d['acode'], d['period'], etype))
        return list(itertools.chain.from_iterable(_sch))
                
    def _cmp_sch_m2(self, problem):
        pass

    def add_problem(self, name, coeff_funcs, cflw_e=None, cgen_data=None,
                    solver=opt.SOLVR_GUROBI, formulation=1, z_coeff_key='z', acodes=None,
                    sense=opt.SENSE_MAXIMIZE, mask=None):
        """
        Add an optimization problem to the model.

        :param str name: Used as key to store ``ws3.opt.Problem`` instances in a dict in the ``ws3.forest.ForestModel`` instanace, so make 
            sure it is unique within a given model or you will overwrite dict values (assuming you want to stuff multiple 
            problems, and their solutions, into your model at the same time).                   
    
        :param dict coeff_funcs: Dict of function references, keyed on _row name_ strings. These are the functions that generate 
            the LP optimization problem matrix coefficients (for the objective function and constraint rows). This one gets
            complicated, and is a likely source of bugs. Make sure the row name key strings are all unique or you will make 
            a mess. You can name the constraint rows anything you want, but the objective function row has to be named 'z'. 
            All coefficient functions must accept exactly two args, in this order: a ``ws3.forest.ForestModel`` instance and a
            ``ws3.common.Path`` instance. The 'z' coefficient function is special in that it must return a single float value. 
            All other (i.e., constraint) coefficient functions just return a dict of floats, keyed on period ints (can be 
            sparse, i.e., not necessary to include key:value pairs in output dict if value is 0.0). It is useful (but not 
            necessary) to use ``functools.partial`` to specialize a smaller number of more general function definitions (with 
            more args, that get "locked down" and hidden by ``partial``) as we have done in the example in this notebook. 
           

        :param dict cflw_e: Dict of (dict, int) tuples, keyed on _row name_ strings (must match _row name_ key values used to 
            define coefficient functions for flow constraints in coeff_func dict), where the int:float dict embedded in the 
            tuple defines epsilon values keyed on periods (must include all periods, even if epsilon value is always the same). 
            See example below. 

            ``{'foo':({1:0.01, ..., 10:0.01}, 1), 'bar':({1:0.05, ..., 10:0.05}, 1)}``            


        :param dict cgen_data: Dict of dict of dicts. The outer-level dict is keyed on _row name_ strings (must match row names used 
            in coeff_funcs. The middle second level of dicts always has keys 'lb' and 'ub', and the inner level of dicts 
            specifies lower- and upper-bound general constraint RHS (float) values, keyed on period (int). See example below.
            
            ``{'foo':{'lb':{1:1., ..., 10:1.}, 'ub':{1:2., ..., 10:2.}}, 'bar':{{'lb':{1:1., ..., 10:1.}, 'ub':{1:2., ..., 10:4.}}}}``
            
        :param int acodes: List of strings. Action codes to be included in optimization problem formulation (actions must defined 
            in the `ForestModel` instance, but can be only a subset). 
            
        :param int sense: Must be one of ``ws3.opt.SENSE_MAXIMIZE`` or ``ws3.opt.SENSE_MINIMIZE``, or equivalent int values (just use the
            constants to keep code more legible).            

        :param tuple mask: Tuple of strings constituting a valid mask for your `ForestModel` instance. Can be `None` if you do not want
            to filter `DevelopmentType` instances.

        :return: ws3.opt.Problem. Reference to a new Problem instance that was created. Also stored in the ForestModel instance (problems attribute,
            keyed on problem name). 
            
        """
        self.reset()
        bld_p_dsp = {1:self._bld_p_m1, 2:self._bld_p_m2}
        cmp_cflw_dsp = {1:self._cmp_cflw_m1, 2:self._cmp_cflw_m2}
        cmp_cgen_dsp = {1:self._cmp_cgen_m1, 2:self._cmp_cgen_m2}
        assert formulation == 1 # only support Model I formulations for now
        p = bld_p_dsp[formulation](name, coeff_funcs, solver, z_coeff_key, acodes, sense, mask) # build problem
        cmp_cflw_dsp[formulation](p, cflw_e) # compile flow constraints
        cmp_cgen_dsp[formulation](p, cgen_data) # compile general constraints
        self.problems[name] = p
        return p
    
    def _bld_p_m1(self, name, coeff_funcs, solver, z_coeff_key='z', acodes=None, sense=opt.SENSE_MAXIMIZE, mask=None):
        """
        Builds optimization problem, using Model I (m1) formulation.
        Each column (variable) of the matrix represents a "prescription"
        (i.e., a feasible sequence of actions, one per period, including the null action).
        Variables x_ij are linear, bounded by 0 and 1, and represent proportion of a zone i
        on which prescription j is applied.
        Coverage constraints ensure that each zone i is fully covered by one or more prescriptions.
        """
        p = opt.Problem(name, sense=sense, solver=solver)
        p.coeff_funcs = coeff_funcs
        p.formulation = 1
        self._problems[name] = p
        p.trees, p._vars = self._gen_vars_m1(coeff_funcs, acodes=acodes, mask=mask)
        for i, tree in list(p.trees.items()):
            cname = 'cov_%i' % hash(i)
            coeffs = {'x_%i' % hash((i, tuple(n.data('acode') for n in path))):1. for path in tree.paths()}
            p.add_constraint(name=cname, coeffs=coeffs, sense=opt.SENSE_EQ, rhs=1.)
            for path in tree.paths():
                try:
                    p._z['x_%i' % hash((i, tuple(n.data('acode') for n in path)))] = path[-1].data(z_coeff_key)
                except Exception as e:
                    print('error processing tree', i)
                    print(e)
                    assert False
        return p
            
    def _bld_p_m2(self, problem):
        pass # not implemented
        
    def _cmp_cgen_m1(self, problem, cgen_data):
        if not cgen_data: return
        mu = {t:{o:{} for o in list(cgen_data.keys())} for t in self.periods}
        for i, tree in list(problem.trees.items()):
            for path in tree.paths():
                j = tuple(n.data('acode') for n in path)
                for o in list(cgen_data.keys()):
                    _mu = path[-1].data(o) 
                    for t in self.periods:
                        mu[t][o][i, j] = _mu[t] if t in _mu else 0. 
        for o, b in list(cgen_data.items()):
            for t in self.periods:
                _mu = {'x_%i' % hash((i, j)):mu[t][o][i, j] for i, j in mu[t][o]}
                if b['lb'] is not None and t in b['lb']:
                    problem.add_constraint(name='gen-lb_%03d_%s' % (t, o), coeffs=_mu, sense=opt.SENSE_GEQ, rhs=b['lb'][t])
                if b['ub'] is not None and t in b['ub']:
                    problem.add_constraint(name='gen-ub_%03d_%s' % (t, o), coeffs=_mu, sense=opt.SENSE_LEQ, rhs=b['ub'][t])
                
        

    def _cmp_cgen_m2(self):
        pass # not implemented

    
    def _cmp_cflw_m1(self, problem, cflw_e):
        """
        Compiles flow constraints (lb and ub, per targeted output, per targeted period).
        """
        if not cflw_e: return
        mu = {t:{o:{} for o in list(cflw_e.keys())} for t in self.periods}
        for i, tree in list(problem.trees.items()):
            for path in tree.paths():
                j = tuple(n.data('acode') for n in path)
                for o in list(cflw_e.keys()):
                    _mu = path[-1].data(o)
                    for t in self.periods:
                        mu[t][o][i, j] = _mu[t] if t in _mu else 0.
        for t in self.periods:
            for o, e in list(cflw_e.items()):
                if t in e[0]:
                    mu_lb = {'x_%i' % hash((i, j)):(mu[t][o][i, j] - (1 - e[0][t]) * mu[e[1]][o][i, j]) for i, j in mu[t][o]}
                    mu_ub = {'x_%i' % hash((i, j)):(mu[t][o][i, j] - (1 + e[0][t]) * mu[e[1]][o][i, j]) for i, j in mu[t][o]}
                    problem.add_constraint(name='flw-lb_%03d_%s' % (t, o), coeffs=mu_lb, sense=opt.SENSE_GEQ, rhs=0.)
                    problem.add_constraint(name='flw-ub_%03d_%s' % (t, o), coeffs=mu_ub, sense=opt.SENSE_LEQ, rhs=0.)

    def _cmp_cflw_m2(self):
        pass # not implemented

    def _bld_tree_m1(self, area, dtk, age, coeff_funcs, tree=None, period=1, acodes=None, compile_c_ycomps=True):
        if not tree:
            self.reset_areas()
            self.dtypes[dtk]._areas[1][age] = area
            self.reset_actions()
            tree = common.Tree()
        acodes = list(self.actions.keys()) if not acodes else acodes
        for acode in acodes:
            if self.dt(dtk).is_operable(acode, period, age):
                self.reset_actions(period)
                if period > 1:
                    self.dt(dtk).grow(period-1, False)
                errorcode, missingarea, tstate = self.apply_action(dtk, acode, period, age, area,
                                                                   compile_c_ycomps=compile_c_ycomps,
                                                                   override_operability=False,
                                                                   fuzzy_age=False,
                                                                   recourse_enabled=False)
                if errorcode:
                    print('apply_action error', dtk, acode, period, age, area, errorcode, missingarea, tstate)
                _dtk, tprop, _age = tstate[0]
                
                assert tprop == 1. # cannot handle 'split' case yet...
                products = None
                tree.grow({'dtk':dtk, '_dtk':_dtk, 'acode':acode, 'period':period, 
                           'age':age, '_age':_age, 'products':products, 'area':area})
                if period < self.periods[-1]: # dive deeper (dfs)
                    self.dt(_dtk).grow(period, False)
                    self._bld_tree_m1(area, _dtk, _age+self.period_length, coeff_funcs, tree, period+1, acodes)
                elif period == self.periods[-1]: # found leaf
                    path = tree.path()
                    leaf = path[-1]
                    assert leaf.is_leaf()
                    leaf._data.update({k:coeff_funcs[k](self, path) for k in coeff_funcs})
                tree.ungrow()
        return tree
    
    def _gen_vars_m1(self, coeff_funcs, acodes=None, mask=None):
        trees, vars = {}, {}
        dtype_keys = self.dtypes.keys() if not mask else self.unmask(mask)
        for dtk in list(dtype_keys):
            self.reset()
            dt = self.dtypes[dtk]
            for age in list(dt._areas[1].keys()):
                self.reset()
                if not dt.area(1, age): continue
                i = (dt.key, age)
                t = trees[i] = self._bld_tree_m1(dt.area(1, age), dt.key, age, coeff_funcs, acodes=acodes)
                for path in t.paths():
                    j = tuple(n.data('acode') for n in path)
                    vname = 'x_%i' % hash((i, j))
                    vtype = opt.VTYPE_CONTINUOUS
                    lb, ub = 0., 1.
                    vars[vname] = opt.Variable(vname, vtype, lb, ub)
        return trees, vars
    
    def _gen_vars_m2(self):
        pass

    def add_null_action(self, acode='null', minage=None, maxage=None):
        """
        Adds a null action with the specified action code, minimum age (default is None), and maximum age (default is None).
        
        """  
        mask = tuple(['?' for _ in range(self.nthemes())])
        oe = '_age >= 0 and _age <= %i' % self.max_age
        target = [(mask, 1.0, None, None, None, None, None)]
        self.actions[acode] = Action(acode)
        self.oper_expr[acode] = {mask:oe}
        self.transitions[acode] = {mask:{'':target}}
        for dtk in self.dtypes:
            self.dtypes[dtk].oper_expr[acode] = [oe]
            self.dtypes[dtk].transitions[acode, -1] = target
            #for age in range(self.dtypes[dtk]._max_age):
            #    self.dtypes[dtk].transitions[acode, age] = target
        for p in self.applied_actions:
            self.applied_actions[p][acode] = {}
    
    def is_harvest(self, acode):
        """
        Returns True if acode corresponds to a harvesting action.
        """
        return self.actions[acode].is_harvest
        
    def piece_size(self, dtype_key, age):
        """
        Returns piece size, given development type key and age.
        """
        return self.dtypes[dtype_key].ycomp(self.piece_size_yname)[age] * self.piece_size_factor

    def dt(self, dtype_key):
        """
        Returns development type, given key (returns None on invalid key).
        """
        try:
            return self.dtypes[dtype_key]
        except:
            return None

    def age_class_distribution(self, period, mask=None, omit_null=False):
        """
        Returns age class distribution (dict of areas, keys on age).

        :param int period: The period for which to retrieve the age class distribution. 
        :param tuple mask: A mask to filter development types. Default is None.
        :param bool omit_null: If True, omits null areas from the distribution. Default is False.  

        :return: A dictionary where keys are ages and values are the corresponding area distributions.
        """
        result = {age:0. for age in self.ages}
        dtype_keys = self.unmask(mask) if mask else list(self.dtypes.keys())
        for dtk in dtype_keys:
            dt = self.dtypes[dtk]
            for age in dt._areas[period]:
                result[age] += dt._areas[period][age]
        if omit_null:
            result = {k:v for k, v in result.items() if v}
        return result
           
    def operable_dtypes(self, acode, period, mask=None):
        """
        Returns dict (keyed on development type key, values are lists of operable ages).
        """
        result = {}
        dtype_keys = self.unmask(mask) if mask else list(self.dtypes.keys())
        for dtk in dtype_keys:
            dt = self.dtypes[dtk]
            operable_ages = dt.operable_ages(acode, period)
            if operable_ages:
                result[dt.key] = operable_ages
        return result

    def inventory(self, period, yname=None, age=None, mask=None, dtype_keys=None, verbose=0):
        """
        Flexible method that compiles inventory at given period.
        Unit of return data defaults to area if yname not given, 
        but takes on unit of yield component otherwise. 
        Can be constrained by age and development type mask.
        """
        result = 0.
        assert not (mask and dtype_keys) # too confusing to allow both to be specified...
        if mask:
            _dtype_keys = self.unmask(mask, verbose=verbose)
        elif dtype_keys:
            _dtype_keys = dtype_keys
        else:
            _dtype_keys = list(self.dtypes.keys())
        #print len(_dtype_keys)
        for dtk in _dtype_keys:
            dt = self.dtypes[dtk]
            ycomp = dt.ycomp(yname) if yname else {a:1. for a in dt._areas[period]}
            if age is not None:
                #print dtk
                #print dt._areas[period]
                #print age, yname, ycomp
                result += dt.area(period, age) * ycomp[age] if age in dt._areas[period] else 0. 
            else:
                result += sum(dt.area(period, a) * ycomp[a] for a in dt._areas[period]) if ycomp else 0.
        return result
        
    def operable_area(self, acode, period, age=None, mask=None):
        """
        Returns total operable area, given action code and period (and optionally age).
        """
        dtype_keys = list(self.dtypes.keys()) if not mask else self.unmask(mask)
        return sum(self.dtypes[dtk].operable_area(acode, period, age) for dtk in dtype_keys)

    def overwrite_initial_areas(self, period):
        """
        Overwrites the initial areas for all development types for the specified period.
        """
        for dt in list(self.dtypes.values()): dt.overwrite_initial_areas(period)
    
    def initialize_areas(self, reset_areas=True):
        """
        Copies areas from period 0 to period 1.
        """
        if reset_areas: self.reset_areas()
        #for dt in list(self.dtypes.values()): dt.initialize_areas()
        for dtk in self.dtypes: self.dtypes[dtk].initialize_areas()
        
    def reset_areas(self, period=None):
        """
        Reset areas for all development types.        
        """
        for dtk in self.dtypes: self.dtypes[dtk].reset_areas(period)
        
    def register_curve(self, curve):
        """
        Add curve to global curve hash map (uses result of Curve.points() to construct hash key). 
        """
        key = tuple(curve.points())
        if key not in self.curves:
            # new curve (lock and register)
            curve.is_locked = True # points list must not change, else not valid key
            self.curves[key] = curve
        return self.curves[key]
            
    # def _rdd(self):
    #     """
    #     Recursive defaultdict (i.e., tree)
    #     """
    #     return dd(self._rdd)   
    
    def reset_actions(self, period=None, acode=None, override_sticky=False):
        """
        Resets actions.
        By default resets, all actions in all periods (except for sticky actions, unless overridden),
        unless period or acode specified. 
        """
        periods = [period] if period else self.periods
        acodes = [acode] if acode else list(self.actions.keys())
        for p in periods:
            if p not in self.applied_actions: self.applied_actions[p] = {}
            for a in acodes:
                if a in self.actions and self.actions[a].is_sticky and not override_sticky: continue
                self.applied_actions[p][a] = {} 

    # def reset_actions(self, period=None, acode=None):
    #     """
    #     Resets actions (default resets all periods, all actions, unless period or acode specified).
    #     """
    #     if period is None:
    #         print("resetting actions")
    #         self.applied_actions = {p:{acode:{} for acode in list(self.actions.keys())} for p in self.periods}
    #     else:
    #         if acode is None:
    #             # NOTE: This DOES NOT deal with consequences in future periods...
    #             self.applied_actions[period] = {acode:{} for acode in list(self.actions.keys())}
    #         else:
    #             assert period is not None
    #             self.applied_actions[period][acode] = {}

    # def reset_actions(self, period=None, acode=None):
    #     if period is None:
    #         self.applied_actions = {p:self._rdd() for p in self.periods}
    #     else:
    #         if acode is None:
    #             # NOTE: This DOES NOT deal with consequences in future periods...
    #             self.applied_actions[period] = self._rdd()
    #         else:
    #             assert period is not None
    #             self.applied_actions[period][acode] = self._rdd()

    def compile_product(self,
                        period,
                        expr,
                        acode=None,
                        dtype_keys=None,
                        age=None,
                        coeff=False,
                        verbose=False):
        """
        Compiles products from applied actions in given period. Parses string expression, which resolves to a single coefficient. 
        Operated area can be filtered on action code, development type key list, and age. Result is product of sum of filtered 
        area and coefficient. 
 
        """
        aa = self.applied_actions
        if acode is None:
            acodes = list(self.actions.keys())
        #elif type(acode) == list: # assume list of acode strings
        #    pass
        else:# elif type(acode) == str: 
            acodes = [acode] if not self.actions[acode].components else self.actions[acode].components
        tokens = expr.split(' ')
        result = 0.
        for _acode in acodes:
            #if not aa[period][_acode]: continue # acode not in solution
            if _acode not in list(aa[period].keys()): continue # acode not in solution
            _dtype_keys = list(aa[period][_acode].keys()) if dtype_keys is None else dtype_keys
            #print 'compile_product len(dtype_keys)', len(dtype_keys)
            #keep = 0
            #skip = 0
            for dtk in _dtype_keys:
                #print dtk
                if dtk not in list(aa[period][_acode].keys()):
                    #skip += 1
                    #if verbose: print len(aa[period][_acode].keys()), dtk 
                    continue
                #keep += 1
                ages = list(aa[period][_acode][dtk].keys()) if age is None else [age]
                for _age in ages:
                    aaa = aa[period][_acode][dtk][_age]
                    #print aaa
                    _tokens = []
                    for token in tokens:
                        if token in self.ynames: # found reference to ycomp
                            if token in aaa[1]: # token is yname in products (replace with value)
                                _tokens.append(str(aaa[1][token]))
                            else: # assume null value if ycomp exists but not stored in solution
                                _tokens.append('0.')
                        else:
                            _tokens.append(token)
                    _expr = ' '.join(_tokens)
                    area = aaa[0] if not coeff else 1.
                    try:
                        result += eval(_expr) * area
                    except ZeroDivisionError:
                        pass # let this one go...
                    except:
                        print(("Unexpected error:", sys.exc_info()[0]))
                        print("evaluating expression '%s' for case:" % ' '.join(_tokens), period, [' '.join(dtk)], _acode, _age)
                        raise

            #print _acode, 'keep', keep, 'skip', skip
        return result
        
    def operated_area(self, acode, period, dtype_key=None, age=None):
        """
        Compiles operated area, given action code and period (and optionally list of development type keys or age).
        """
        aa = self.applied_actions
        acodes = [acode] if not self.actions[acode].components else self.actions[acode].components
        result = 0.
        for _acode in acodes:
            if not aa[period][_acode]: continue # acode not in solution
            dtype_keys = list(aa[period][_acode].keys()) if dtype_key is None else [dtype_key]
            for _dtype_key in dtype_keys:
                ages = list(aa[period][_acode][_dtype_key].keys()) if age is None else [age]
                for _age in ages:
                    result += aa[period][_acode][_dtype_key][_age][0]
        return result

    def repair_actions(self, period, areaselector=None, verbose=False):
        """
        Attempts to repair the action schedule for given period, using an AreaSelector object 
        (defaults to class-default areaselector, which is a simple greedy oldest-first selector).
        """
        if areaselector is None: # use default (greedy) selector
            areaselector = self.areaselector
        aa = copy.copy(self.applied_actions[period])
        self.reset_actions(period)
        for acode in aa:
            if not aa[acode]: continue # null solution, move along...
            old_area = 0.
            new_area = 0.
            # start by re-applying as much of the old solution as possible
            for dtype_key in aa[acode]:
                for age in aa[acode][dtype_key]:
                    aaa = aa[acode][dtype_key][age][0]
                    old_area += aaa
                    oa = self.dtypes[dtype_key].operable_area(acode, period, age)
                    if not oa: continue
                    applied_area = min(aaa, oa)
                    #print ' applying old area', applied_area
                    new_area += applied_area
                    self.apply_action(dtype_key, acode, period, age, applied_area)
            # try to make up for missing area...
            target_area = old_area - new_area
            if verbose:
                print(' patched %i of %i solution hectares, missing' % (int(new_area), int(old_area)), target_area)
            if areaselector is None: # use default area selector
                areaselector = self.areaselector
            areaselector.operate(period, acode, target_area)
                     
        
    def commit_actions(self, period=1, repair_future_actions=False, verbose=False):
        """
        Commits applied actions (i.e., apply transitions and grow, default starting at period 1).
        By default, will attempt to repair broken (infeasible) future actions, attempting to replace infeasiblea operated area using default AreaSelector.  
        """
        while period < self.horizon:
            if verbose: print('growing period', period)
            self.grow(period, cascade=False)
            period += 1
            if repair_future_actions:
                if verbose: print('repairing actions in period', period)
                self.repair_actions(period)
            else:
                self.reset_actions(period)

    def resolve_replace(self, dtk, expr):
        """
        Enables the creation of new development types by replacing an existing attribute code with a new value for a specific theme, 
        instead of directly coding the attribute change in transition.        
        """
        # HACK ####################################################################
        # Too lazy to implement all the use cases.
        # This should work OK for BFEC models (TO DO: confirm).
        tokens = re.split('\s+', expr)
        i = int(tokens[0][3]) - 1
        try:
            return str(eval(expr.replace(tokens[0], dtk[i])))
        except:
            print('source', ' '.join(dtype_key))
            print('target', ' '.join(tmask), tprop, tage, tlock, treplace, tappend)
            print('dtk', ' '.join(dtk))
            raise
        
    ###########################################################################
    # HACK ####################################################################
    # Too lazy to implement.
    # Not used in BFEC models (TO DO: confirm).
    def resolve_append(self, dtk, expr):
        """
        This method has not been implemented yet.
        """
        assert False # brick wall (deal with this case later, as needed)

    def resolve_targetage(self, dtk, tyield, sage, tage, acode, verbose=False):
        """
        This method determines the target age for a transition based on different
        parameters such as the development type key (dtk), yield information (tyield),
        stand age (sage), target age override (tage), and action code (acode).
        """
        action = self.actions[acode]
        if tyield is not None: # yield-based age definition
            if verbose:
                print('yield-based age definition', tyield, self.dt(dtk).ycomp(tyield[0]).lookup(tyield[1], roundx=True))
            try:
                targetage = self.dt(dtk).ycomp(tyield[0]).lookup(tyield[1], roundx=True)
            except:
                print(' '.join(dtk), tyield[0], self.dt(dtk).ycomps())
                assert False
        elif tage is not None: # target age override specifed in transition
            if verbose: print('_AGE override', tage)
            targetage = tage
        elif action.targetage is None: # use source age
            if verbose: print('source age', age)
            targetage = sage
        else: # default: age reset to 0
            if verbose: print('default age reset to 0')
            targetage = 0
        return targetage
                                            
    def apply_action(self,
                     dtype_key,
                     acode,
                     period,
                     age,
                     area,
                     override_operability=False,
                     fuzzy_age=True,
                     recourse_enabled=True,
                     areaselector=None,
                     compile_t_ycomps=False,
                     compile_c_ycomps=False,
                     verbose=False):
        """
        Applies action, given action code, development type, period, age, area.
        Can optionally override operability limits, optionally use fuzzy age (i.e., attempt 
        to apply action to proximal age class if specified age is not operable), optionally use 
        default AreaSelector to patch missing area (if recourse enabled). Applying an action is 
        a rather complex process, involving testing for operability (JIT-compiling operability 
        expression as required), checking that valid transitions are defined, checking that area
        is available (possibly using fuzzy age and area selector functions to find missing area),
        generate list of target development types (from source development type and  transition
        expressions [which may need to be JIT-compiled]), creating new development types 
        (as needed), doing the area accounting correctly (without creating or destroying any area)
        and compiling the products from the action (which gets a bit complicated in the case of 
        partial cuts...).

        :param tuple dtype_key: The key identifying the development type.
        :param str acode: The action code to apply.
        :param int period: The period in which to apply the action.
        :param int age:  The age at which to apply the action.
        :param float area: The area to apply the action on.
        :param bool override_operability: If True, overrides operability limits. Default is False.
        :param bool fuzzy_age: If True, attempts to apply action to proximal age class if specified age is not operable.
        :param bool recourse_enabled: If True, uses default AreaSelector to patch missing area. Default is True.
        :param bool areaselector: The AreaSelector object to use for patching missing area. Default is None.
        :param bool compile_t_ycomps: If True, compiles time-indexed yield components. Default is False.
        :param bool compile_c_ycomps: If True, compiles complex yield components. Default is False.
        :param bool verbose: If True, prints additional information for debugging purposes. Default is False.
 
        Returns (errorcode, missing_area, target_dt) triplet, where errorcode is an error code, 
        missing_area is the missing area, and target_dt is a list of (dtk, tprop, targetage) 
        triplets (one triplet per target development type).

        :return: A tuple containing errorcode, missing_area, and target_dt.

        **Error codes:**
        1. invalid area argument
        2. requested action not defined for development type
        3. requested action defined, but never operable
        4. action not operable
        5. transitions not defined for action
        """
        if area <= 0.: return 1, None, None 
        if verbose > 1:
            print('applying action', [' '.join(dtype_key)], acode, period, age, area)
        dt = self.dtypes[dtype_key]
        ############################################
        # TO DO: better error handling... ##########
        #print dt.oper_expr
        if acode not in dt.oper_expr:
            print('requested action not defined for development type...')
            print(' ', [' '.join(dtype_key)], acode, period, age, area)
            return 2, None, None
        if acode not in dt.operability: # action not compiled yet...
            if dt.compile_action(acode) == -1:
                print('requested action is defined, but never not operable...')
                print(' ', [' '.join(dtype_key)], acode, period, age, area)
                return 3, None, None
        if not dt.is_operable(acode, period, age) and not override_operability:
            print('not operable')
            print(' '.join(dt.key), acode, period, age)
            print(dt.operability[acode][period])
            #assert False # dt.is_operable(acode, period, age)
            return 4, None, None
        if not any((acode, __age) in dt.transitions for __age in (age, -1)): # sanity check...
            print('transitions not defined...')
            print(' ', [' '.join(dtype_key)], acode, period, age, area)
            print(dt.oper_expr)
            print(dt.operability)
            #print dt.transitions
            #assert False 
            return 5, None, None
        if dt.area(period, age) - area < self.area_epsilon:
            # tweak area if slightly over or under, so we don't get any accounting drift...
            #print 'foobar', dt.key, period, age, dt.area(period, age), area
            area = dt.area(period, age)
        missing_area = 0.
        if dt.area(period, age) < area:
            # insufficient area in dt to operate (infeasible)
            # apply action to operable area, then look for missing area in adjacent ageclasses
            if dt.area(period, age) > 0: # operate available area before applying recourse
                print('insufficient area in dt to operate (infeasible)', dtype_key, period, age)
                self.apply_action(dtype_key, acode, period, age, dt.area(period, age),
                                  False, False, False, None, True)
            missing_area = area - dt.area(period, age)
            if fuzzy_age and missing_area:
                for age_delta in [+1, -1, +2, -2]:
                    _age = age + age_delta
                    if dt.area(period, _age) > 0 and any((acode, __age) in dt.transitions for __age in (_age, -1)):
                        _area = min(missing_area, dt.area(period, _age))
                        self.apply_action(dtype_key, acode, period, _age, _area,
                                          False, False, False, None, True)
                        missing_area -= _area
                        if missing_area < self.area_epsilon:
                            missing_area = 0.
                            break 
            if recourse_enabled and missing_area:
                areaselector = self.areaselector if areaselector is None else areaselector
                missing_area = areaselector.operate(period, acode, missing_area)
                if missing_area < self.area_epsilon:
                    missing_area = 0.
        action = self.actions[acode]
        #if not dt.actions[acode].is_compiled: dt.compile_action(acode)
        ###########################################################################
        dt.area(period, age, -area)
        target_dt = []
        __age = age if (acode, age) in dt.transitions else -1
        for target in dt.transitions[acode, __age]:
            tmask, tprop, tyield, tage, tlock, treplace, tappend = target # unpack tuple
            dtk = list(dtype_key) # start with source key
            ###########################################################################
            # DO TO: Confirm correct order for evaluating mask, _APPEND and _REPLACE...
            dtk = [t if tmask[i] == '?' else tmask[i] for i, t in enumerate(dtk)] 
            if treplace: dtk[treplace[0]] = self.resolve_replace(dtk, treplace[1])
            if tappend: dtk[tappend[0]] = self.resolve_append(dtk, tappend[1])
            dtk = tuple(dtk)
            ###########################################################################
            _dt = self.create_dtype_fromkey(dtk) if dtk not in self.dtypes else self.dtypes[dtk]
            targetage = self.resolve_targetage(dtk, tyield, age, tage, acode)
            _dt.area(period, targetage, area*tprop)
            target_dt.append([dtk, tprop, targetage])
        aa = self.applied_actions[period][acode]
        if dtype_key not in aa: aa[dtype_key] = {}
        if age not in aa[dtype_key]: aa[dtype_key][age] = [0., {}]
        aa[dtype_key][age][0] += area
        for yname in dt.ycomps():
            ycomp = dt.ycomp(yname)
            if ycomp.type == 't' and not compile_t_ycomps: continue # skip time-indexed ycomps
            if ycomp.type == 'c' and not compile_c_ycomps: continue # skip complex ycomps
            if yname in action.partial:
                value = 0.
                for dtk, tprop, targetage in target_dt:
                    _dt = self.dtypes[dtk]
                    _value = 0.
                    if yname in dt.ycomps():
                        if yname in _dt.ycomps():
                            _value = (dt.ycomp(yname)[age] - _dt.ycomp(yname)[targetage])
                        else:
                            _value = dt.ycomp(yname)[age]
                    if _value > 0.:
                        value += _value * tprop
                    else:
                        if verbose:
                            if _value < 0:
                                print('negative partial value', acode, yname, tprop, _value)
                                print(' ', ''.join(dtype_key), age)
                                print(' ', ''.join(dtk), targetage)
                                print()
            else: # not partial
                value = dt.ycomp(yname)[age]
            if value != 0.:
                aa[dtype_key][age][1][yname] = value
        return 0, missing_area, target_dt

    
    def sylv_cred_formula(self, treatment_type, cover_type):
        """
        Calculate Sylviculture Credits based on treatment type and cover type.
        """
        if treatment_type == 'ec':
            return 1 if cover_type.lower() in ['r', 'm'] else 2
        if treatment_type == 'cj':
            return 4
        if treatment_type == 'cprog':
            return 7 if cover_type.lower() in ['r', 'm'] else 4        
        return 0

            
    def create_dtype_fromkey(self, key):
        """
        Creates a new development type, given a key (checks for existing, auto-assigns yield compompontents, 
        auto-assign actions and transitions, checks for operability (filed under inoperable if applicable).
        """        
        assert key not in self.dtypes # should not be creating new dtypes from existing key
        dt = DevelopmentType(key, self)
        self.dtypes[key] = dt
        # assign yields
        for mask, t, ycomps in self.yields:
            if self.match_mask(mask, key):
                for yname, ycomp in ycomps:
                    dt.add_ycomp(t, yname, ycomp)
        # assign actions and transitions
        for acode in self.oper_expr:
            for mask in self.oper_expr[acode]:
                if self.match_mask(mask, key):
                    dt.oper_expr[acode].append(self.oper_expr[acode][mask]) 
            for mask in self.transitions[acode]:
                if self.match_mask(mask, key):
                    for scond in self.transitions[acode][mask]:
                        for x in self.resolve_condition(scond, key): 
                            dt.transitions[acode, x] = self.transitions[acode][mask][scond]
        if not dt.transitions:
            self.inoperable_dtypes.append(key)
        return dt
    
    def _resolve_outputs_buffer(self, s, for_flag=None):
        n = self.nthemes()
        group = 'no_group' # outputs declared at top of file assigned to 'no_group'
        self.output_groups[group] = set()
        ocode = ''
        buffering_for = False
        s = re.sub(r'\{.*?\}', '', s, flags=re.M|re.S) # remove curly-bracket comments
        for l in re.split(r'[\r\n]+', s, flags=re.M|re.S):
            if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
            matches = re.findall('#[A-Za-z0-9_]*', l)
            for m in matches: # replace CONSTANTS variables with value
                try:
                    l = l.replace(m, str(self.constants[m[1:].lower()]))
                except:
                    import sys
                    print(sys.exc_info()[0])
                    print(l)
                    print(matches, m)
                    assert False

            if buffering_for:
                if l.strip().startswith('ENDFOR'):
                    for i in range(for_lo, for_hi+1):
                        ss = '\n'.join(for_buffer).replace(for_var, str(i))
                        self._resolve_outputs_buffer(ss, for_flag=i)
                    buffering_for = False
                    continue
                else:
                    for_buffer.append(l)
                    continue
            l = re.sub('\s+', ' ', l) # separate tokens by single space
            l = l.strip().partition(';')[0].strip()
            l = l.replace(' (', '(')  # remove space to left of left parentheses
            t = l.lower().split(' ')
            ##################################################
            # HACK ###########################################
            # substitute ugly symbols have in ocodes...
            l = l.replace(r'%', 'p')
            l = l.replace(r'$', 's')
            ##################################################
            tokens = l.lower().split(' ')
            if l.startswith('*GROUP'):
                keyword = 'group'
                group = tokens[1].lower()
                self.output_groups[group] = set()
            elif l.startswith('FOR'):
                # pattern matching may not be very robust, but works for now with:
                # 'FOR XX := 1 to 99'
                # TO DO: implement DOWNTO, etc.
                for_var = re.search(r'(?<=FOR\s).+(?=:=)', l).group(0).strip()
                for_lo = int(re.search(r'(?<=:=).+(?=to)', l).group(0))
                for_hi = int(re.search(r'(?<=to).+', l).group(0))
                for_buffer = []
                buffering_for = True
                continue
            if l.startswith('*OUTPUT') or l.startswith('*LEVEL'):
                keyword = 'output' if l.startswith('*OUTPUT') else 'level'
                if ocode: # flush data collected from previous lines
                    self.outputs[ocode] = Output(parent=self,
                                                 code=ocode,
                                                 expression=expression,
                                                 description=description,
                                                 theme_index=theme_index)
                tt = tokens[1].split('(')
                ocode = tt[0]
                theme_index = tt[1][3:-1] if len(tt) > 1 else None
                description = ' '.join(tokens[2:])
                expression = ''
                self.output_groups[group].add(ocode)
                if keyword == 'level':
                    self.outputs[ocode] = Output(parent=self,
                                                 code=ocode,
                                                 expression=expression,
                                                 description=description,
                                                 theme_index=theme_index,
                                                 is_level=True)
                    ocode = ''
            elif l.startswith('*SOURCE'):
                keyword = 'source'
                expression += l[8:]
            elif keyword == 'source': # continuation line of SOURCE expression
                expression += ' '
                expression += l       
        
    #@timed
    def import_outputs_section(self, filename_suffix='out'):
        """
        Imports OUTPUTS section from a Forest model.
        """
        with open('%s/%s.%s' % (self.model_path, self.model_name, filename_suffix)) as f:
            s = f.read()
        self._resolve_outputs_buffer(s)

    def add_theme(self, name, basecodes=[], aggs={}, description=''):
        """
        Adds a theme to the model.

        
        :param str name: The name of theme.
        :param list basecodes: List of base codes for the theme.
        :param dict aggs: Dictionary containing aggregated values for the theme.
        :param str description: Description of the theme.       
        """



        
        self._themes.append({})
        #self.nthemes +- 1
        self._themes[-1]['__name__'] = name
        self._themes[-1]['__description__'] = description
        if basecodes: self._theme_basecodes.append([])
        for c in basecodes:
            self._themes[-1][c] = c
            self._theme_basecodes[-1].append(c)
        for c in aggs:            
            self._themes[-1][c] = aggs[c]
        
    #@timed
    def import_landscape_section(self, filename_suffix='lan', ti_offset=0):
        """
        Imports LANDSCAPE section from a Forest model.
        """
        with open('%s/%s.%s' % (self.model_path, self.model_name, filename_suffix)) as f:
            data = f.read()
        _data = re.search(r'\*THEME.*', data, re.M|re.S).group(0) # strip leading junk
        t_data = re.split(r'\*THEME.*\n', _data)[1:] # split into theme-wise chunks
        for ti, t in enumerate(t_data, start=ti_offset):
            self._themes.append({})
            self._themes[-1]['__name__'] = 'theme%i' % ti
            self._themes[-1]['__description__'] = '' # TO DO: extract comments in theme declaration
            self._theme_basecodes.append([])
            defining_aggregates = False
            for l in [l for l in t.split('\n') if not re.match('^\s*(;|{|$)', l)]: 
                if re.match('^\s*\*AGGREGATE', l): # aggregate theme attribute code
                    tac = re.split('\s+', l.strip())[1].lower()
                    self._themes[ti][tac] = []
                    defining_aggregates = True
                    continue
                if not defining_aggregates: # line defines basic theme attribute code
                    tac = re.search('\S+', l.strip()).group(0).lower()
                    self._themes[ti][tac] = tac
                    self._theme_basecodes[ti].append(tac)
                else: # line defines aggregate values (parse out multiple values before comment)
                    _tacs = [_tac.lower() for _tac in re.split('\s+', l.strip().partition(';')[0].strip())]
                    self._themes[ti][tac].extend(_tacs)
        #self.nthemes = len(self._themes)

    def theme_basecodes(self, theme_index):
        """
        Return list of base codes, given theme index.
        """
        return self._theme_basecodes[theme_index]
        #return self._themes[theme_index]
        
    #@timed    
    def import_areas_section(self, model_path=None, model_name=None, filename_suffix='are', import_empty=False):
        """
        Imports AREAS section from a Forest model.

        .. note::
            - This method parses the AREAS section from a Forest model file.
            - Each line in the section represents an area for a specific development type and age class.
            - The section should contain information about development type keys, ages, and corresponding areas.
            - Empty areas (with values less than the area_epsilon) can be skipped if import_empty is False.
        """
        n = self.nthemes()
        model_path = self.model_path if not model_path else model_path
        model_name = self.model_name if not model_name else model_name
        with open('%s/%s.%s' % (model_path, model_name, filename_suffix)) as f:
            for l in f:
                try:
                    if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
                    l = l.lower().strip().partition(';')[0] # strip leading whitespace and trailing comments
                    t = re.split('\s+', l)
                    key = tuple(_t for _t in t[1:n+1])
                    age = int(t[n+1])
                    area = float(t[n+2].replace(',', ''))
                    if area < self.area_epsilon and not import_empty: continue
                    if key not in self.dtypes: self.dtypes[key] = DevelopmentType(key, self)
                    self.dtypes[key].area(0, age, area)
                except Exception as e:
                    print('Failed AREAS import on line: \n%s' % l)
                    return 1
        return 0

                    
    def _expand_action(self, c):
        self._actions = t
        return [c] if t[c] == c else list(_cfi(self._expand_action(t, c) for c in t[c]))
                
    def _expand_theme(self, t, c, verbose=False): # depth-first search recursive aggregate theme code expansion
        if verbose > 1:
            print('ws3.forest.ForestModel._expand_theme', t, c)
            print(c)
        return [c] if t[c] == c else list(_cfi(self._expand_theme(t, c) for c in t[c]))

                
    def match_mask(self, mask, key):
        """
        Returns True if key matches mask.
        """
        #dt = self.dtypes[key]
        for ti, tac in enumerate(mask):
            if tac == '?': continue # wildcard matches all keys
            tacs = self._expand_theme(self._themes[ti], tac)
            if key[ti] not in tacs: return False # reject key
        return True # key matches
        
    def unmask(self, mask, verbose=0):
        """
        Iteratively filter list of development type keys using mask values.
        Accepts Woodstock-style string masks to facilitate cut-and-paste testing.
        """
        if isinstance(mask, str): # Woodstock-style string mask format
            mask = tuple(re.sub('\s+', ' ', mask).lower().split(' '))
            assert len(mask) == self.nthemes() # must be bad mask if wrong theme count
        else:
            try:
                assert isinstance(mask, tuple) and len(mask) == self.nthemes()
            except:
                print(len(mask), type(mask), mask)
                assert False
        dtype_keys = copy.copy(list(self.dtypes.keys())) # filter this
        for ti, tac in enumerate(mask):
            if tac == '?': continue # wildcard matches all
            tacs = self._expand_theme(self._themes[ti], tac, verbose=verbose) if tac in self._themes[ti] else []
            dtype_keys = [dtk for dtk in dtype_keys if dtk[ti] in tacs] # exclude bad matches
        return dtype_keys

    #@timed                            
    def import_constants_section(self, filename_suffix='con'):
        """
        Imports CONSTANTS section from a Forest model.

        .. note::
            - This method parses the CONSTANTS section from a Forest model file.
            - Each line in the section represents a constant and its value.
            - Constants are stored in a dictionary where the keys are the constant names and the values are their respective values.
            - The section should contain information about various constants used in the model.
        """
        with open('%s/%s.%s' % (self.model_path, self.model_name, filename_suffix)) as f:
            for lnum, l in enumerate(f):
                if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
                l = l.strip().partition(';')[0].strip() # strip leading whitespace, trailing comments
                t = re.split('\s+', l)
                self.constants[t[0].lower()] = float(t[1])

    #@timed        
    def import_yields_section(self, filename_suffix='yld', mask_func=None, verbose=False):
        """
        Imports YIELDS section from a Forest model.
        """
        ###################################################
        # local utility functions #########################
        def flush_ycomps(t, m, n, c):
            if t == 'a': # age-based ycomps
                _c = lambda y: self.register_curve(core.Curve(y,
                                                              points=c[y],
                                                              type='a',
                                                              period_length=self.period_length))
                ycomps = [(y, _c(y)) for y in n]
            elif t == 't': # time-based ycomps (skimp on x range)
                _c = lambda y: self.register_curve(core.Curve(y,
                                                              points=c[y],
                                                              type='t',
                                                              xmax=self.horizon,
                                                              period_length=self.period_length))
                ycomps = [(y, _c(y)) for y in n]
            else: # complex ycomps
                ycomps = [(y, c[y]) for y in n]
            self.yields.append((m, t, ycomps)) # stash for creating new dtypes at runtime...
            self.ynames.update(n)
            for k in self.unmask(m):
                for yname, ycomp in ycomps:
                    self.dtypes[k].add_ycomp(t, yname, ycomp)
        ###################################################
        n = self.nthemes()
        ytype = ''
        mask = ('?',) * self.nthemes()
        ynames = []
        data = None
        with open('%s/%s.%s' % (self.model_path, self.model_name, filename_suffix)) as f:
            for lnum, l in enumerate(f):
                if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
                l = l.strip().partition(';')[0].strip() # strip leading whitespace and trailing comments
                t = re.split('\s+', l)
                if t[0].startswith('*Y'): # new yield definition
                    newyield = True
                    flush_ycomps(ytype, mask, ynames, data) # apply yield from previous block
                    ytype = self._ytypes[t[0]]
                    mask = tuple(_t.lower() for _t in t[1:])
                    mask = mask_func(mask) if mask_func else mask
                    if verbose: print(lnum, ' '.join(mask))
                    continue
                if newyield:
                    if t[0] == '_AGE':
                        is_tabular = True
                        ynames = [_t.lower() for _t in t[1:]]
                        data = {yname:[] for yname in ynames}
                        newyield = False
                        continue
                    else:
                        is_tabular = False
                        ynames = []
                        data = {}
                        newyield = False
                else:
                    if t[0] == '_AGE': # same yield block, new table
                        flush_ycomps(ytype, mask, ynames, data) # apply yield from previous block
                        is_tabular = True
                        ynames = [_t.lower() for _t in t[1:]]
                        data = {yname:[] for yname in ynames}
                        newyield = False
                        continue
                if is_tabular:
                    try:
                        x = int(t[0])
                    except:
                        print(lnum, l)
                    for i, yname in enumerate(ynames):
                        data[yname].append((x, float(t[i+1])))
                else:
                    if ytype in 'at': # standard or time-based yield (extract xy values)
                        if not common.is_num(t[0]): # first line of row-based yield component
                            yname = t[0].lower()
                            ynames.append(yname)
                            data[yname] = [((int(t[1])*self.period_length)+(i*self.period_length),
                                             float(t[i+2])) 
                                           for i in range(len(t)-2)]
                        else: # continuation of row-based yield compontent
                            x_last = data[yname][-1][0]
                            data[yname].extend([(i+x_last+1, float(t[i])) for i in range(len(t))])
                    else:
                        yname = t[0].lower()
                        ynames.append(yname)
                        data[yname] = ' '.join(t[1:]) # complex yield (defer interpretation)
        flush_ycomps(ytype, mask, ynames, data)

                    
            
    #@timed        
    def import_actions_section(self, filename_suffix='act', mask_func=None, nthemes=None):
        """
        Imports ACTIONS section from a Forest model.

        .. note:: 
            - This method parses the ACTIONS section from a Forest model file.
            - The section should contain information about actions, operability, aggregates, and partials.
            - Each action is represented by a code, description, and operability condition.
            - Operability conditions are defined for different masks.
            - Aggregates combine multiple actions.
            - Partials represent components of an action.        
        """
        nthemes = nthemes if nthemes else self.nthemes()
        actions = {}
        #oper = {}
        aggregates = {}
        partials = {}
        keyword = ''
        with open('%s/%s.%s' % (self.model_path, self.model_name, filename_suffix)) as f: s = f.read().lower()
        s = re.sub(r'\{.*?\}', '', s, flags=re.M|re.S) # remove curly-bracket comments
        for l in re.split(r'[\r\n]+', s, flags=re.M|re.S):
            if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
            l = l.strip().partition(';')[0].strip() # strip leading whitespace and trailing comments
            l = re.sub('\s+', ' ', l) # separate tokens by single space
            tokens = l.split(' ')
            if l.startswith('*action'): 
                keyword = 'action'
                acode = tokens[1]
                targetage = 0 if tokens[2] == 'y' else None
                descr = ' '.join(tokens[3:])
                lockexempt = '_lockexempt' in tokens
                self.actions[acode] = Action(acode, targetage, descr, lockexempt)
                self.oper_expr[acode] = {}
            elif l.startswith('*operable'):
                keyword = 'operable'
                acode = tokens[1]
            elif l.startswith('*aggregate'):
                keyword = 'aggregate'
                acode = tokens[1]
                self.actions[acode] = Action(acode)
            elif l.startswith('*partial'): 
                keyword = 'partial'
                acode = tokens[1]
                partials[acode] = []
            else: # continuation of OPERABLE, AGGREGATE, or PARTIAL block
                if keyword == 'operable':
                    mask = tuple(tokens[:nthemes])
                    mask = mask_func(mask) if mask_func else mask
                    self.oper_expr[acode][mask] = ' '.join(tokens[nthemes:])
                elif keyword == 'aggregate':
                    self.actions[acode].components.extend(tokens)
                elif keyword == 'partial':
                    self.actions[acode].partial.extend(tokens)
        for acode, a in list(self.actions.items()):
            if a.components: continue # aggregate action, skip
            for mask, expression in list(self.oper_expr[acode].items()):
                for k in self.unmask(mask):
                    #if acode == 'act1': print ' '.join(k), acode, expression
                    self.dtypes[k].oper_expr[acode].append(expression)

    def resolve_treplace(self, dt, treplace):
        if '_TH' in treplace: # assume incrementing integer theme value
            i = int(re.search('(?<=_TH)\w+', treplace).group(0))
            return eval(re.sub('_TH%i'%i, str(dt.key[i-1]), treplace))
        else:
            assert False # many other possible arguments (see Forest documentation)

    def resolve_tappend(self, dt, tappend):
        """
        This feature has not been implemented yet.  
        """
        assert False # brick wall (not implemented yet)

    def resolve_tmask(self, dt, tmask, treplace, tappend):
        """
        Returns new developement type key (tuple of values, one per theme), given developement type and (treplace, tappend) expressions.
        """
        key = list(dt.key)
        if treplace:
            key[treplace[0]] = resolve_treplace(dt, treplace[1])
        if tappend:
            key[tappend[0]] = resolve_tappend(dt, tappend[1])
        for i, val in enumerate(tmask):
            if theme == '?': continue # wildcard (skip it)
            key[i] = val
        return tuple(key)

    def resolve_condition(self, condition, dtype_key=None):
        """
        Evaluate @AGE or @YLD condition.
        Returns list of ages.
        """
        if not condition:
            #return self.ages
            return [-1]
        elif condition.startswith('@AGE'):
            lo, hi = [int(a) for a in condition[5:-1].split('..')]
            return list(range(lo, hi+1))
        elif condition.startswith('@YLD'):
            args = re.split('\s?,\s?', condition[5:-1])
            yname = args[0].lower()
            lo, hi = [float(y) for y in args[1].split('..')]
            dt = self.dtypes[dtype_key]
            lo_age, hi_age = dt.ycomp(yname).range(lo, hi, as_bounds=True)
            return list(range(lo_age, hi_age+1))
            #return self.dtypes[dtype_key].resolve_condition(yname, hi, lo)
        
    #@timed                        
    def import_transitions_section(self, filename_suffix='trn', mask_func=None, nthemes=None):
        """
        Imports TRANSITIONS section from a Forest model.
        """
        nthemes = nthemes if nthemes else self.nthemes()
        # local utility function ####################################
        def flush_transitions(acode, sources):
            if not acode: return # nothing to flush on first loop
            self.transitions[acode] = {}
            for smask, scond in sources:
                #if acode in ['acp']:
                #    print [' '.join(smask)], scond, sources[smask, scond]
                # store transition data for future dtypes creation 
                if smask not in self.transitions[acode]:
                    self.transitions[acode][smask] = {}
                #if scond not in self.transitions[acode][smask]:
                #    self.transitions[acode][smask][scond] = []
                self.transitions[acode][smask][scond] = sources[smask, scond]
                # assign transitions to existing dtypes
                for k in self.unmask(smask):
                    dt = self.dtypes[k]
                    for x in self.resolve_condition(scond, k): # store targets
                        dt.transitions[acode, x] = sources[smask, scond] 
        # def flush_transitions(acode, sources):
        #     if not acode: return # nothing to flush on first loop
        #     for smask, scond in sources:
        #         for k in self.unmask(smask):
        #             dt = self.dtypes[k]
        #             for x in self.resolve_condition(scond, k): # store targets
        #                 dt.transitions[acode, x] = sources[smask, scond] 
        #############################################################                    
        acode = None
        with open('%s/%s.%s' % (self.model_path, self.model_name, filename_suffix)) as f:
            s = f.read()
        s = re.sub(r'\{.*?\}', '', s, flags=re.M|re.S) # remove curly-bracket comments
        for l in re.split(r'[\r\n]+', s, flags=re.M|re.S):
            if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
            l = l.strip().partition(';')[0].strip() # strip leading whitespace, trailing comments
            tokens = re.split('\s+', l)
            if l.startswith('*CASE'):
                if acode: flush_transitions(acode, sources)
                acode = tokens[1].lower()
                sources = {}
            elif l.startswith('*SOURCE'):
                smask = tuple(t.lower() for t in tokens[1:nthemes+1])
                smask = mask_func(smask) if mask_func else smask
                match = re.search(r'@.+\)', l)
                scond = match.group(0) if match else ''
                sources[(smask, scond)] = []
            elif l.startswith('*TARGET'):
                tmask = tuple(t.lower() for t in tokens[1:nthemes+1])
                tmask = mask_func(tmask) if mask_func else tmask
                tprop = float(tokens[nthemes+1]) * 0.01
                tyield = None
                if len(tokens) > nthemes+2 and tokens[nthemes+2].lower() in self.ynames:
                    tyield = (tokens[nthemes+2].lower(), float(tokens[nthemes+3]))
                try: # _AGE keyword
                    tage = int(tokens[tokens.index('_AGE')+1])
                except:
                    tage = None
                try: # _LOCK keyword
                    tlock = int(tokens[tokens.index('_LOCK')+1])
                except:
                    tlock = None
                try: # _REPLACE keyword (TO DO: implement other cases)
                    args = re.split('\s?,\s?', re.search('(?<=_REPLACE\().*(?=\))', l).group(0))
                    theme_index = int(args[0][3]) - 1
                    treplace = theme_index, args[1]
                except:
                    treplace = None
                try: # _APPEND keyword (TO DO: implement other cases)
                    args = re.split('\s?,\s?', re.search('(?<=_APPEND\().*(?=\))', l).group(0))
                    theme_index = int(args[0][3]) - 1
                    tappend = theme_index, args[1]
                except:
                    tappend = None
                sources[(smask, scond)].append((tmask, tprop, tyield, tage, tlock, treplace, tappend))
        flush_transitions(acode, sources)

    
    def import_optimize_section(self, filename_suffix='opt'):
        """
        Imports OPTIMIZE section from a Forest model.
        
        .. warning:: 
            Not implemented yet.
        """
        pass

    def import_graphics_section(self, filename_suffix='gra'):
        """
        Imports GRAPHICS section from a Forest model.
        
        .. warning:: 
            Not implemented yet.

        """
        pass

    def import_lifespan_section(self, filename_suffix='lif'):
        """
        Imports LIFESPAN section from a Forest model.
        
        .. warning:: 
            Not implemented yet.

        """
        pass


    def import_schedule_section(self, filename_suffix='seq', replace_commas=True, filename_prefix=None):
        """
        Imports SCHEDULE section from a Forest model.
        """
        filename_prefix = self.model_name if filename_prefix is None else filename_prefix
        schedule = []
        n = self.nthemes()
        with open('%s/%s.%s' % (self.model_path, filename_prefix, filename_suffix)) as f:
            for lnum, l in enumerate(f):
                if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
                l = l.lower().strip().partition(';')[0].strip() # strip leading whitespace and trailing comments
                t = re.split('\s+', l)
                if len(t) != n + 5: break
                dtype_key = tuple(t[:n])
                age = int(t[n])
                area = float(t[n+1].replace(',', '')) if replace_commas else float(t[n+1])
                acode = t[n+2]
                period = int(t[n+3])
                etype = t[n+4] if len(t) >= n+4 else ''
                schedule.append((dtype_key, age, area, acode, period, etype))
                if area <= 0: print('area <= 0', l)
        return schedule

    
    def compile_schedule(self, problem=None):
        """
        Compiles the schedule of actions.
        If a problem is specified, compiles the schedule from the given problem data. 
        Otherwise, compiles the schedule from the data in `self.applied_actions`.        
        """
        if problem is not None:
            return self._compile_schedule_from_problem(problem)
        else: # use data in self.applied_actions
            return self._compile_schedule_from_actions()

        
    def _compile_schedule_from_actions(self):
        result = []
        for period in self.periods:
            aa = self.applied_actions[period]
            for acode in aa.keys():
                for dtk in aa[acode].keys():
                    etype = '_existing' if self.dt(dtk).area(0) else '_future'
                    for age in aa[acode][dtk].keys():
                        area = aa[acode][dtk][age][0]
                        result.append((dtk, age, area, acode, period, etype))
        return result
           
    
    def apply_schedule(self, schedule, max_period=None, verbose=False,
                       fail_on_missingarea=False, force_integral_area=False,
                       override_operability=False, fuzzy_age=True,
                       recourse_enabled=True, areaselector=None,
                       compile_t_ycomps=False, compile_c_ycomps=False,
                       rounding_bias=0.15, scale_area=None, reset=True):
        """
        Assumes schedule in format returned by import_schedule_section().
        That is: list of (dtype_key, age, area, acode, period, etype) tuples.
        Also assumes that actions in list are sorted by applied period.

        :param list schedule: The schedule of actions to apply.
        :param int max_period: The maximum period to apply actions for. If None, defaults to the horizon.
        :param bool verbose: If True, prints additional information for debugging purposes. Default is False.
        :param bool fail_on_missingarea: If True, raises an exception if missing area is encountered. Default is False.
        :param bool force_integral_area: If True, forces the area to be integral. Default is False.
        :param bool override_operability:  If True, overrides operability limits. Default is False.
        :param bool fuzzy_age: If True, attempts to apply action to proximal age class if specified age is not operable.
                               Default is True.
        :param bool recourse_enabled: If True, uses default AreaSelector to patch missing area. Default is True.
        :param object areaselector: The AreaSelector object to use for patching missing area. Default is None.
        :param bool compile_t_ycomps: If True, compiles time-indexed yield components. Default is False.
        :param bool compile_c_ycomps: If True, compiles complex yield components. Default is False.
        :param float rounding_bias:  The rounding bias to use when forcing integral area. Default is 0.15.
        :param scale_area: The scaling factor to apply to the area. Default is None. 
        :param bool reset: If True, resets the model before applying the schedule. Default is True.


        :return: The missing area (float) after applying the schedule. 
        
        """
        if max_period is None: max_period = self.horizon
        if reset: self.reset()
        missing_area = 0.
        for _period in self.periods:
            for dtype_key, age, area, acode, period, etype in schedule:
                if period != _period: continue
                if scale_area: area = area * scale_area
                if period > _period:
                    if verbose: print('apply_schedule: committing actions for period', _period, '(missing area %0.1f)' % missing_area)
                    self.commit_actions(_period)
                if period > max_period: return
                if force_integral_area:
                    area = round(area+rounding_bias)
                    if not area: continue
                    assert not area % 1.
                e, _aa, _ = self.apply_action(dtype_key,
                                              acode,
                                              period,
                                              age,
                                              area,
                                              override_operability=override_operability,
                                              fuzzy_age=fuzzy_age,
                                              recourse_enabled=recourse_enabled,
                                              areaselector=areaselector,
                                              compile_t_ycomps=compile_t_ycomps,
                                              compile_c_ycomps=compile_c_ycomps,
                                              verbose=verbose)
                crash_on_apply_action_error = False # hack... put in method signature later
                if crash_on_apply_action_error:
                    assert not e # crash on error (TO DO: better error handling)
                else:
                    if e:
                        print('apply action error', e, dtype_key, acode, period, age, area) 
                if isinstance(_aa, float): 
                    if fail_on_missingarea and missing_area:
                        raise
                    else:
                        missing_area += _aa
                if verbose: print('missing area %0.1f (%0.2f)' % (_aa, _aa/area))
                _period = period
            self.commit_actions(_period)
        return missing_area

    def import_control_section(self, filename_suffix='run'):
        """
        Imports CONTROL section from a Forest model.
        
        .. warning:: 
            Not implemented yet.
        """
        pass

    def grow(self, start_period=1, cascade=True):
        """
        Simulates growth (default startint at period 1 and cascading to the end of the planning horizon).
        """
        for dt in list(self.dtypes.values()): dt.grow(start_period, cascade)
    
    def _cbm_sit_classifiers(self):
        """
        Compile sit_classifiers dataframe.
        """
        data = {'classifier_id':[], 'name':[], 'description':[]}
        for i, theme in enumerate(self._themes):
            data['classifier_id'].append(i+1)
            data['name'].append('_CLASSIFIER')
            data['description'].append(theme['__name__'])
            for v in self.theme_basecodes(i):
                data['classifier_id'].append(i+1)
                data['name'].append(v)
                data['description'].append(v) # not great descriptions, but no effect on CBM output
        data['classifier_id'].append(self.nthemes()+1)
        data['name'].append('_CLASSIFIER')
        data['description'].append('species') 
        data['classifier_id'].append(self.nthemes()+1)
        data['name'].append('softwood')
        data['description'].append('softwood') 
        data['classifier_id'].append(self.nthemes()+1)
        data['name'].append('hardwood')
        data['description'].append('hardwood')
        result = pd.DataFrame(data)
        return result
    
    def _cbm_sit_disturbance_types(self):
        """
        Compile sit_disturbance_types dataframe.
        """
        acodes = [acode for acode in self.actions.keys() if acode != 'null'] + ['fire']
        data = {'id':acodes, 'name':acodes}
        result = pd.DataFrame(data)
        return result
    
    def _cbm_sit_age_classes(self):
        """
        Compile sit_age_classes dataframe.
        """
        data = {'name':['age_0'],
                'class_size':[0],
                'start_year':[0],
                'end_year':[0]}
        for i, ac in enumerate(range(self.period_length, 
                                     self.max_age+self.period_length, 
                                     self.period_length)):
            data['name'].append('age_%i' % (i+1))
            data['class_size'].append(self.period_length)
            data['start_year'].append(ac - self.period_length + 1)
            data['end_year'].append(ac)
        result = pd.DataFrame(data)
        return result
    
    def _cbm_sit_inventory(self, softwood_volume_yname, hardwood_volume_yname, default_last_pass_disturbance='fire'):
        """
        Compile sit_inventory dataframe.
        """
        def leading_species(r):
            """
            Determine if softwood or hardwood leading species by comparing softwood and hardwood
            volume at peak MAI age.
            """
            dt = self.dtypes[tuple(r.tolist())]
            svol_curve, hvol_curve = dt.ycomp(softwood_volume_yname), dt.ycomp(hardwood_volume_yname)
            tvol_curve = svol_curve + hvol_curve
            x_cmai = tvol_curve.mai().ytp().lookup(0)
            return 'softwood' if svol_curve[x_cmai] > hvol_curve[x_cmai] else 'hardwood'
        def landclass(r):
            """
            The landclass column values should contain an integer in the range [0, 22], which CBM 
            maps to one of 23 UNFCCC land classes (see Table 3-1 in the the CBM-CFS3 user guide. 
            Use the value of the 'landclass' attribute of the corresponding development type (if defined), 
            otherwise default to 0 ('forest land remaining forest land').
            """
            dt = self.dtypes[tuple(r.tolist())]
            if hasattr(dt, 'landclass'):
                return dt.landclass
            else:
                return '0'            
        def last_pass_disturbance(r):
            """
            We use the value of the 'last_pass_disturbance' attribute of the corresponding development 
            type (if defined), otherwise default to default_last_pass_disturbance.

            """
            dt = self.dtypes[tuple(r.tolist())]
            if hasattr(dt, 'last_pass_disturbance'):
                return dt.last_pass_disturbance
            else:
                return default_last_pass_disturbance
        theme_cols = [theme['__name__'] for theme in self._themes]
        other_cols = ['species',
                      'using_age_class', 
                      'age', 
                      'area', 
                      'delay', 
                      'landclass', 
                      'historic_disturbance', 
                      'last_pass_disturbance']

        data = {**{c:[] for c in theme_cols}, **{c:[] for c in ['age', 'area']}}
        for dtype_key in self.dtypes:
            dt = self.dtypes[dtype_key]
            if not dt._areas[0]: continue # developement type not in initial inventory
            for age, area in dt._areas[0].items():
                for i, c in enumerate(theme_cols): data[c].append(dtype_key[i])
                data['age'].append(age)
                data['area'].append(area)
        result = pd.DataFrame(data)
        age_series, area_series = result['age'], result['area']
        result.drop(['age', 'area'], axis=1, inplace=True) # wrong column order                    
        result['species'] = result[theme_cols].apply(leading_species, axis=1)
        result['using_age_class'] = 'FALSE'
        result['age'] = age_series
        result['area'] = area_series
        result['delay'] = 0
        result['landclass'] = result[theme_cols].apply(landclass, axis=1)
        result['historic_disturbance'] = 'fire'
        result['last_pass_disturbance'] = result[theme_cols].apply(last_pass_disturbance, axis=1)
        return result
    
    def _cbm_sit_yield(self, softwood_volume_yname, hardwood_volume_yname, n_yield_vals):
        """
        Compile sit_yields dataframe.
        
        This is where the ws3 and CBM data models diverge and things get a bit messy.
        The hack: 
            Define a bogus Model I optimization problem in ws3.
            A side effect of generating the Model I problem matrix is to dynamically create 
            all possible DevelopmentType cases and add them to the ForestModel. 
            Then we can can use ForestModel.unmask to get at least one valid
            DevelopmentType instance for each yield mask in ForestModel.yields
            (there could be multiple, but just grap the first one in the list).
            From there, use the same species-grokking logic we used to compile
            the species column in sit_inventory. Ugly, but should work
            as long as ForestModel.yields includes full-wildcard softwood and 
            hardwood complex yield curves defined with the _SUM(...) function (because
            this way all DevelopmentType instances, initially present or dynamically
            created, will automatically have softwood and hardwood yield curves defined).
            The bogus Model I problem can just use a bogus z coefficient 
            function (that alwasy returns 0) and no flow or general constraints.
            Maybe there is a better way?
        """
        def leading_species(dt):
            """
            Determine if softwood or hardwood leading species by comparing softwood and hardwood
            volume at peak MAI age.
            """
            svol_curve, hvol_curve = dt.ycomp(softwood_volume_yname), dt.ycomp(hardwood_volume_yname)
            tvol_curve = svol_curve + hvol_curve
            x_cmai = tvol_curve.mai().ytp().lookup(0)
            return 'softwood' if svol_curve[x_cmai] > hvol_curve[x_cmai] else 'hardwood'
        schedule = self.compile_schedule()
        p = self.add_problem('__cbm_sit_bogus', {'z':(lambda forestmodel, path: 0.)})
        theme_cols = [theme['__name__'] for theme in self._themes]
        data = {**{c:[] for c in theme_cols}, 
                'species':[], 'leading_species':[],        
                **{'v%i' % i:[] for i in range(n_yield_vals + 1)}}
        for dtype_key in self.dtypes:
            dt = self.dt(dtype_key)
            dt.leading_species = leading_species(dt)
            for species, yname in zip(('softwood', 'hardwood'), (softwood_volume_yname, hardwood_volume_yname)):
                for i, c in enumerate(theme_cols): data[c].append(dtype_key[i])
                data['species'].append(species)
                data['leading_species'].append(dt.leading_species)
                for i in range(n_yield_vals + 1): data['v%i' % i].append(dt.ycomp(yname)[i * self.period_length])
        result = pd.DataFrame(data)
        self.apply_schedule(schedule) # running add_problem above broke the schedule so restore from backup we stashed
        return result

    def _cbm_sit_events(self):
        theme_cols = [theme['__name__'] for theme in self._themes]
        columns = theme_cols.copy()
        columns += ['species',
                    'using_age_class',
                    'min_softwood_age',
                    'max_softwood_age',
                    'min_hardwood_age',
                    'max_hardwood_age',
                    'MinYearsSinceDist',
                    'MaxYearsSinceDist',
                    'LastDistTypeID',
                    'MinTotBiomassC',
                    'MaxTotBiomassC',
                    'MinSWMerchBiomassC',
                    'MaxSWMerchBiomassC',
                    'MinHWMerchBiomassC',
                    'MaxHWMerchBiomassC',
                    'MinTotalStemSnagC',
                    'MaxTotalStemSnagC',	
                    'MinSWStemSnagC',
                    'MaxSWStemSnagC',
                    'MinHWStemSnagC',
                    'MaxHWStemSnagC',
                    'MinTotalStemSnagMerchC',
                    'MaxTotalStemSnagMerchC',
                    'MinSWMerchStemSnagC',
                    'MaxSWMerchStemSnagC',
                    'MinHWMerchStemSnagC',
                    'MaxHWMerchStemSnagC',
                    'efficiency',
                    'sort_type',
                    'target_type',
                    'target',
                    'disturbance_type',
                    'disturbance_year']
        data = {c:[] for c in columns}
        for dtype_key, age, area, acode, period, _ in self.compile_schedule():
            #set_trace()
            for i, c in enumerate(theme_cols): data[c].append(dtype_key[i])
            data['species'].append(self.dt(dtype_key).leading_species)
            data['using_age_class'].append('FALSE')
            #############################################################################
            # might need to be more flexible with age range, to avoid OBO bugs and such?
            data['min_softwood_age'].append(-1)
            data['max_softwood_age'].append(-1)
            data['min_hardwood_age'].append(-1)
            data['max_hardwood_age'].append(-1)
            # data['min_softwood_age'].append(age)
            # data['max_softwood_age'].append(age)
            # data['min_hardwood_age'].append(age)
            # data['max_hardwood_age'].append(age)
            #############################################################################
            for c in columns[11:-6]: data[c].append(-1)
            data['efficiency'].append(1)
            data['sort_type'].append(3) # oldest first (see Table 3-3 in the CBM-CFS3 user guide)
            data['target_type'].append('A') # area target
            data['target'].append(area)
            data['disturbance_type'].append(acode)
            data['disturbance_year'].append(period*self.period_length)
        result = pd.DataFrame(data)         
        return result
    
    def _cbm_sit_transitions(self, null_acode='null'):
        def resolve_target(dtype_key, target, sage):
            tmask, tprop, tyield, tage, tlock, treplace, tappend = target # unpack tuple
            dtk = list(dtype_key) # start with source key
            dtk = [t if tmask[i] == '?' else tmask[i] for i, t in enumerate(dtk)] 
            if treplace: dtk[treplace[0]] = self.resolve_replace(dtk, treplace[1])
            if tappend: dtk[tappend[0]] = self.resolve_append(dtk, tappend[1])
            dtk = tuple(dtk)
            targetage = self.resolve_targetage(dtk, tyield, sage, tage, acode)
            return dtk, targetage
        theme_cols = colunmns = [theme['__name__'] for theme in self._themes]
        columns = theme_cols.copy()
        columns += ['species',
                    'using_age_class',
                    'min_softwood_age',
                    'max_softwood_age',
                    'min_hardwood_age',
                    'max_hardwood_age',
                    'disturbance_type']
        columns += ['to_%s' % c for c in theme_cols]
        columns += ['to_species',
                    'regen_delay',
                    'reset_age',
                    'percent']
        data = {c:[] for c in columns}
        for dtype_key in self.dtypes:
            dt = self.dt(dtype_key)
            for acode, sage in dt.transitions:
                if acode == null_acode: continue
                for target in dt.transitions[acode, sage]:
                    #tmask, tprop, tyield, tage, tlock, treplace, tappend = target
                    for i, c in enumerate(theme_cols): data[c].append(dtype_key[i])
                    data['species'].append(dt.leading_species)
                    data['using_age_class'].append('FALSE')
                    data['min_softwood_age'].append(sage)
                    data['max_softwood_age'].append(sage)
                    data['min_hardwood_age'].append(sage)
                    data['max_hardwood_age'].append(sage)
                    data['disturbance_type'].append(acode)
                    to_dtype_key, to_age = resolve_target(dtype_key, target, sage)
                    for i in range(5): data['to_theme%i' % i].append(to_dtype_key[i])
                    data['to_species'].append(self.dt(to_dtype_key).leading_species)
                    data['regen_delay'].append(0)
                    data['reset_age'].append(to_age)
                    data['percent'].append(int(target[1] * 100))
        result = pd.DataFrame(data)
        return result

        for acode in fm.transitions:
            if acode == null_acode: continue
            for smask in self.transitions[acode]:
                tmask, tprop, _, _, _, _, _ = self.transitions[acode][smask][''][0]
                for i, c in enumerate(theme_cols): data[c].append(smask[i])
                data['species'].append('softwood' if au_table1.loc[int(smask[2])].canfi_species < 1200 else 'hardwood')
                data['using_age_class'].append('FALSE')
                data['min_softwood_age'].append(1)
                data['max_softwood_age'].append(999)
                data['min_hardwood_age'].append(1)
                data['max_hardwood_age'].append(999)
                data['disturbance_type'].append('harvest')
                for i in range(5): data['to_theme%i' % i].append(tmask[i])
                data['to_%s' % species_classifier_colname].append('softwood' if au_table2.loc[int(tmask[4])].canfi_species < 1200 else 'hardwood')
                data['regen_delay'].append(0)
                data['reset_age'].append(0)
                data['percent'].append(100)
        result = pd.DataFrame(data)
        return result
        
    def to_cbm_sit(self, softwood_volume_yname, hardwood_volume_yname, admin_boundary, eco_boundary, 
                   disturbance_type_mapping, 
                   export_csv=False, sit_data_path='', default_last_pass_disturbance='fire', n_yield_vals=100):
        """
        Exports model data in a CBM standard import tool (SIT) data exchange format.
        Returns sit_config (JSON-like dict namespace) and sit_tables (dict of pandas.DataFrame objects).

        :param str softwood_volume_yname: The yield component name for softwood volume.
        :param str hardwood_volume_yname: The yield component name for hardwood volume.
        :param str admin_boundary: The administrative boundary for spatial units mapping.
        :param str eco_boundary: The ecological boundary for spatial units mapping.
        :param dict disturbance_type_mapping: A dictionary containing disturbance type mapping information.
        :param bool export_csv: Flag indicating whether to export data to CSV files. Default is False.
        :param str sit_data_path: The path to export CSV files. Default is empty string.
        :param str default_last_pass_disturbance:  The default last pass disturbance type. Default is 'fire'.
        :param int n_yield_vals: The number of yield values. Default is 100.
        
        """
        sit_config = {
                        'mapping_config': {
                            'nonforest': None,
                            'species': {
                                'species_classifier': 'species',
                                'species_mapping': [
                                    {'user_species': 'softwood', 'default_species': 'Softwood forest type'},
                                    {'user_species': 'hardwood', 'default_species': 'Hardwood forest type'}
                                ]
                            },
                            'spatial_units': {
                                'mapping_mode': 'SingleDefaultSpatialUnit',
                                'admin_boundary': admin_boundary,
                                'eco_boundary': eco_boundary},
                            'disturbance_types': {
                                'disturbance_type_mapping': disturbance_type_mapping
                            }
                        }
                    }
        sit_tables = {'sit_classifiers':self._cbm_sit_classifiers(),
                      'sit_disturbance_types':self._cbm_sit_disturbance_types(),
                      'sit_age_classes':self._cbm_sit_age_classes(),
                      'sit_inventory':self._cbm_sit_inventory(softwood_volume_yname='swdvol', 
                                                              hardwood_volume_yname='hwdvol'),
                      'sit_yield':self._cbm_sit_yield(softwood_volume_yname='swdvol', 
                                                      hardwood_volume_yname='hwdvol', n_yield_vals=100),
                      'sit_events':self._cbm_sit_events(),
                      'sit_transitions':self._cbm_sit_transitions()}
        return sit_config, sit_tables

                      
        
                
        
        
if __name__ == '__main__':
    pass