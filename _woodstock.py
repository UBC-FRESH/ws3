import re
import copy
import operator
from itertools import chain
_cfi = chain.from_iterable
from collections import deque

try:
    from . import common
    from . import core
except: # "__main__" case
    import common
    import core
from common import timed
    
#_mad = common.MAX_AGE_DEFAULT
    
class Action:
    #_oper_a_default = [None, None]
    #_oper_p_default = [None, None]
    def __init__(self,
                 code,
                 targetage=None,
                 descr='',
                 lockexempt=False,
                 oper_expr='',
                 components={}):
        self.code = code
        self.targetage = targetage
        self.descr = descr
        self.lockexempt = lockexempt
        self.oper_a = None #self._oper_a_default 
        self.oper_p = None #self._oper_p_default
        self.oper_expr = oper_expr
        self.components = components

    
class DevelopmentType:
    """
    Encapsulates Woodstock development type (curves, age, area).
    """
    
    def __init__(self,
                 key,
                 parent):
        self.key = key
        self.parent = parent
        self.areas = deque(0. for x in range(parent.max_age))
        self.actions = {}
        self._complex_ycomps = [] # ordered list for deferred evaluation
        self._zero_curve = core.Curve('zero', points=[(0, 0)], is_special=True, type='')
        self._unit_curve = core.Curve('unit', points=[(0, 1)], is_special=True, type='')
        self._ages_curve = core.Curve('ages',
                                      points=[(0, 0), (parent.max_age, parent.max_age)],
                                      is_special=True, type='')        
        self.ycomps = {}
        self._resolvers = {'MULTIPLY':self._resolver_multiply,
                           'DIVIDE':self._resolver_divide,
                           'SUM':self._resolver_sum,
                           'CAI':self._resolver_cai,
                           'MAI':self._resolver_mai,
                           'YTP':self._resolver_ytp,
                           'RANGE':self._resolver_range}
        self.transitions = {} # keys are (acode, age) tuples

    def resolve_condition(self, ycomp, lo, hi):
        return [x for x, y in enumerate(self.ycomps[ycomp]) if y >= lo and y <= hi]
       
    def reset_areas(self):
        self.areas = deque(0. for x in range(parent.max_age))
                
    def _o(self, s, default_ycomp=None): # resolve string operands
        if not default_ycomp: default_ycomp = self._zero_curve
        if common.is_num(s):
            return float(s)
        elif s.startswith('#'):
            return self.parent.constants[s[1:]]
        else:
            s = s.lower() # just to be safe
            try:
                return self.ycomps[s]
            except KeyError: # BFEC model bug workaround
                ## HACK ###################################################
                if s not in self.ycomps: return default_ycomp
                ## HACK ####################################################
        
    def _resolver_multiply(self, yname, d):
        args = [self._o(s.lower()) for s in re.split('\s?,\s?', re.search('(?<=\().*(?=\))', d).group(0))]
        ytype_set = set(a.type for a in args if isinstance(a, core.Curve))
        return ytype_set.pop() if len(ytype_set) == 1 else 'c', reduce(lambda x, y: x*y, args)

    def _resolver_divide(self, yname, d):
        _tmp = zip(re.split('\s?,\s?', re.search('(?<=\().*(?=\))', d).group(0)),
                   (self._zero_curve, self._unit_curve))
        args = [self._o(s, default_ycomp) for s, default_ycomp in _tmp]
        return args[0].type if not args[0].is_special else args[1].type, args[0] / args[1]
        
    def _resolver_sum(self, yname, d):
        args = [self._o(s.lower()) for s in re.split('\s?,\s?', re.search('(?<=\().*(?=\))', d).group(0))] 
        ytype_set = set(a.type for a in args if isinstance(a, core.Curve))
        return ytype_set.pop() if len(ytype_set) == 1 else 'c', reduce(lambda x, y: x+y, [a for a in args])
        
    def _resolver_cai(self, yname, d):
        arg = self._o(re.split('\s?,\s?', re.search('(?<=\().*(?=\))', d).group(0))[0])
        return arg.type, arg.mai()
        
    def _resolver_mai(self, yname, d):
        arg = self._o(re.split('\s?,\s?', re.search('(?<=\().*(?=\))', d).group(0))[0])
        return arg.type, arg.mai()
        
    def _resolver_ytp(self, yname, d):
        arg = self._o(re.search('(?<=\().*(?=\))', d).group(0).lower())
        return arg.type, arg.ytp()
        
    def _resolver_range(self, yname, d):
        args = [self._o(s.lower()) for s in re.split('\s?,\s?', re.search('(?<=\().*(?=\))', d).group(0))] 
        arg_triplets = [args[i:i+3] for i in xrange(0, len(args), 3)]
        return args[0].type, reduce(lambda x, y: x*y, [t[0].range(t[1], t[2]) for t in arg_triplets])
        
    def compile_complex_ycomps(self):
        for yname in self._complex_ycomps:
            keyword = re.search('(?<=_)[A-Z]+(?=\()', self.ycomps[yname]).group(0)
            try:
                ytype, ycomp = self._resolvers[keyword](yname, self.ycomps[yname])
                ycomp.label = yname
                ycomp.type = ytype
                self.ycomps[yname] = ycomp 
            except KeyError:
                raise ValueError('Unsupported keyword: %s' % keyword)
        
    def compile_actions(self, verbose=False):
        bo = {'AND':operator.and_, '&':operator.and_, 'OR':operator.or_, '|':operator.or_}
        for a in self.actions.values():
            if verbose: print
            cond_comps = re.split(r'\s+&\s+|\s+\|\s+|\s+AND\s+|\s+OR\s+', a.oper_expr)
            bool_operators = re.findall(r'&|\||AND|OR', a.oper_expr)
            bool_operators.insert(0, '&')
            lhs, rel_operators, rhs = zip(*[re.split('\s+', cc) for cc in cond_comps])
            rhs = map(lambda x: float(x), rhs)
            r = self._unit_curve
            for i, o in enumerate(lhs):
                if o == '_CP':
                    if rel_operators[i] == '=':
                        a.oper_p = [int(rhs[i])]
                    elif rel_operators[i] == '>=':
                        a.oper_p = [p for p in range(int(rhs[i]), parent.horizon)]
                    elif rel_opertors[i] == '<=':
                        a.oper_p = [p for p in range(0, int(rhs[i])+1)]
                    else:
                        raise ValueError('Bad relational operator.')
                elif o == '_AGE':
                    if rel_operators[i] == '=':
                        r = bo[bool_operators[i]](r, self._ages_curve.range(rhs[i], rhs[i]))
                    elif rel_operators[i] == '>=':
                        r = bo[bool_operators[i]](r, self._ages_curve.range(rhs[i], None))
                    elif rel_operators[i] == '<=':
                        r = bo[bool_operators[i]](r, self._ages_curve.range(None, rhs[i]))
                    else:
                        raise ValueError('Bad relational operator.')
                else: # must be yname
                    ycomp = self.ycomps[o.lower()]
                    verbose = 0
                    if rel_operators[i] == '=':
                        r = bo[bool_operators[i]](r, ycomp.range(rhs[i], rhs[i], verbose))
                        print ''.join(str(int(y)) for y in ycomp.range(rhs[i], rhs[i])), o, '=', rhs[i]
                    elif rel_operators[i] == '>=':
                        r = bo[bool_operators[i]](r, ycomp.range(rhs[i], None, verbose))
                        if verbose: print ''.join(str(int(y)) for y in ycomp.range(rhs[i], None)), o, '>=', rhs[i]
                    elif rel_operators[i] == '<=':
                        r = bo[bool_operators[i]](r, ycomp.range(None, rhs[i], verbose))
                        if verbose: print ''.join(str(int(y)) for y in ycomp.range(None, rhs[i])), o, '<=', rhs[i]
                    else:
                        raise ValueError('Bad relational operator.')
            a.oper_a = [x for x, y in enumerate(r) if y]
            #if not a.oper_a: del self.actions[a.code] # not operable at any age
                
    def add_ycomp(self, ytype, yname, ycomp):
        self.ycomps[yname] = ycomp
        if ytype == 'c': self._complex_ycomps.append(yname)
                           
    def ynames(self, ytypes='at'):
        return [yname for yname in self.ycomps if self._ytype[yname] in ytypes]
                 
    def grow(self, periods=1):
        self.areas.rotate(periods)

        
class Output:
    def __init__(self,
                 parent,
                 code=None,
                 expression=None,
                 factor=(1., 1),
                 description='',
                 theme=-1,
                 is_basic=False,
                 is_level=False):
        self.parent = parent
        self.code = code
        self.expression = expression
        self.factor = factor
        self.description = description
        if theme > -1:
            self.theme = theme
            self.is_themed = True
        self.is_basic = is_basic
        if is_basic:
            self._compile_basic(expression) # shortcut
        else:
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
            #if self.code == 'OSUPREALREGAREBECON'.lower():
            #    print self.parent.outputs.keys()
            if len(tt) > 1:
                factors[i] = self._rval(tt[2]), 1 if tt[1] == '*' else -1
            if not isinstance(lval, Output):     
                if len(ocomps) == 1: # simple basic output (special case)
                    self.is_basic = True
                    self.factor = factors[i]
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
            mask = tuple(t[:self.parent.nthemes])
            t = t[self.parent.nthemes:] # pop
        self._dtype_keys = self.parent.unmask(mask) if mask else self.parent.dtypes.keys()
        # extract @AGE or @YLD condition, if present
        self._ages = None
        self._condition = None
        if t[0].startswith('@age'):
            lo, hi = [int(a)+i for i, a in enumerate(t[0][5:-1].split('..'))]
            hi = min(hi, self.parent.max_age+1) # they get carried away with range bounds...
            self._ages = range(lo, hi)
            t = t[1:] # pop
        elif t[0].startswith('@yld'):
            ycomp, args = t[0][5:-1].split(',')
            self._condition = ycomp, tuple(float(a) for a in args.split('..'))
            self._ages = None
            t = t[1:] # pop
        if not self._ages and not self._condition: self._ages = range(self.parent.max_age+1)
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

        # DEBUG
        print
        print s
        print 'mask', ' '.join(mask)
        print 'n dtypes', len(self._dtype_keys)
        print 'ages', (self._ages[0], self._ages[-1]) if self._ages else None
        print 'condition', self._condition 
        print 'is_invent', self._is_invent
        print 'invent_acodes', self._invent_acodes
        print 'acode', self._acode 
        print 'is_area', self._is_area
        print 'ycomp', self._ycomp 

    def _evaluate_basic(self, period, factors):
        result = 0.
        for k in self._dtype_keys:
            dt = self.parent.dtypes[k]
            ycomp = None if self._is_area else dt.ycomps[self._ycomp]
            if isinstance(self._factor[0], float):
                f = pow(*self._factor)
            else:
                f = pow(dt.ycomps[self._factor[0]][period], self._factor[1])
            for factor in factors:
                if isinstance(factor[0], float):
                    f *= pow(*factor)
                else:
                    f *= pow(dt.ycomps[factor[0]][period], factor[0])
            for age in ages:
                area = 0.
                y = ycomp[age] if ycomp else 1.
                if self._is_invent:
                    if self._invent_acodes: 
                        any_operable = False
                        for acode in self._invent_acodes:
                            a = dt.actions[acode]
                            if age in a.oper_a and period in a.oper_p:
                                any_operable = True
                        if any_operable:
                            area += dt.areas[age]
                    else:
                        area += dt.areas[age]
                else: 
                    aa = parent.applied_actions[period]
                    key = k, self._acode, age
                    if key in aa: area += aa[key]
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

            
    def __call__(self, period, factors, exponents):
        if self.is_basic:
            return self._evaluate_basic(period, factor, exponent)
        else:
            return self._evaluate_summary(period, factor, exponent)

        # NOTE: could speed up dispatch with dict of function references
        #if self.is_basic:
        #    if not self.is_themed:
        #        return self._evaluate_basic(period)
        #    else:
        #        return self._evaluate_basic_themed(period)
        #else: # summary output
        #    return self._evaluate_summary(period)

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

    # def __mul__(self, other):
    #     # assume an Output * (float OR time-based ycomp)
    #     if not self.is_themed:
    #         return [i * other for i self()]
    #     else:
    #         return self() * other

    # def __div__(self, other):
    #     # assume an Output / (float OR time-based ycomp)
    #     if not self.is_themed:
    #         return [i / other for i self()]
    #     else:
    #         return self() / other

         
class WoodstockModel:
    """
    Interface to import Woodstock models.
    """
    _ytypes = {'*Y':'a', '*YT':'t', '*YC':'c'}
        
    def __init__(self,
                 model_name,
                 model_path,
                 horizon=common.HORIZON_DEFAULT,
                 period=common.PERIOD_DEFAULT,
                 max_age=common.MAX_AGE_DEFAULT,
                 species_groups=common.SPECIES_GROUPS_WOODSTOCK_QC):
        self.model_name = model_name
        self.model_path = model_path
        self.horizon = horizon
        self.period = period
        self.current_period = 0
        self.max_age = max_age
        self.ages = range(max_age+1)
        self._species_groups = species_groups
        self.actions = {}
        self._themes = []
        self._theme_basecodes = []
        self.dtypes = {}
        self.constants = {}
        self.output_groups = {}
        self.outputs = {}
        self.applied_actions = {i:{} for i in range(self.horizon)}
        self.ycomps = set()
    
    def apply_action(dtype_key, acode, age, area):
        def resolve_replace(dtk, expr):
            # HACK ####################################################################
            # Too lazy to implement all the use cases.
            # This should work OK for BFEC models (TO DO: confirm).
            tokens = re.split('\s+', expr)
            i = int(tokens[0][3]) - 1
            return str(eval(expr.replace(tokens[0], dtk[i])))
            ###########################################################################
        def resolve_append(dtk, expr):
            # HACK ####################################################################
            # Too lazy to implement.
            # Not used in BFEC models (TO DO: confirm).
            assert False # brick wall (deal with this case later, as needed)
            ###########################################################################
        dt = self.dtypes[dtype_key]
        assert acode in dt.actions
        assert dt.areas[age] >= area
        dt.areas[age] -= area
        for target in dt.transitions[acode, age]:
            tmask, tprop, tage, tlock, treplace, tappend = target # unpack tuple
            dtk = list(dtype_key) # start with source key
            ###########################################################################
            # DO TO: Confirm correct order for evaluating mask, _APPEND and _REPLACE...
            dtk = [t if tmask[i] == '?' else tmask[i] for t in dtk] 
            if treplace: dtk[treplace[0]] = resolve_replace(dtk, treplace[1])
            if tappend: dtk[tappend[0]] = resolve_append(dtk, tappend[1])
            ###########################################################################
            if dtk not in self.dtypes: # new development type (clone source type)
                _dt = copy.copy(dt)
                _dt.areas[age] = area * tprop
                self.dtypes[dtk] = _dt
            else:
                self.dtypes[dtk].areas += area * tprop
        self.applied_actions[self.current_period][dtype_key, acode, age] = area

    def _resolve_outputs_buffer(self, s, for_flag=None):
        n = self.nthemes
        group = 'no_group' # outputs declared at top of file assigned to 'no_group'
        self.output_groups[group] = set()
        ocode = ''
        buffering_for = False 
        s = re.sub(r'\{.*?\}', '', s, flags=re.M|re.S) # remove curly-bracket comments
        #ocode_pattern = re.compile(r'(?<=\*OUTPUT)\s+(\w|%)+((?=\s)|(?=\())|(?=$)')
        ocode_pattern1 = re.compile(r'(?<=\*OUTPUT)\s+(\w)+((?=\s)|(?=\())|(?=$)')
        ocode_pattern2 = re.compile(r'(?<=\*OUTPUT)\s+(\w)+')
        theme_pattern = re.compile(r'(?<=\(_TH)[0-9]+')
        for l in re.split(r'[\r\n]+', s, flags=re.M|re.S):
            if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
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
            l = l.strip().partition(';')[0].strip()
                    s = re.sub('\s+', ' ', expression) # separate tokens by single space
        s = s.replace(' (', '(')  # remove space to left of left parentheses
        t = s.lower().split(' ')
        # filter dtypes, if starts with mask
        mask = None
        if not (t[0] == '@' or t[0] == '_' or t[0] in self.parent.actions):
            mask = tuple(t[:self.parent.nthemes])
            t = t[self.parent.nthemes:] # pop
        self._dtype_keys = self.parent.unmask(mask) if mask else self.parent.dtypes.keys()
        # extract @AGE or @YLD condition, if present
        self._ages = None
        self._condition = None
        if t[0].startswith('@age'):
            lo, hi = [int(a)+i for i, a in enumerate(t[0][5:-1].split('..'))]
            hi = min(hi, self.parent.max_age+1) # they get carried away with range bounds...
            self._ages = range(lo, hi)
            t = t[1:] # pop
        elif t[0].startswith('@yld'):
            ycomp, args = t[0][5:-1].split(',')
            self._condition = ycomp, tuple(float(a) for a in args.split('..'))
            self._ages = None
            t = t[1:] # pop
        if not self._ages and not self._condition: self._ages = range(self.parent.max_age+1)
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

        # DEBUG
        print
        print s
        print 'mask', ' '.join(mask)
        print 'n dtypes', len(self._dtype_keys)
        print 'ages', (self._ages[0], self._ages[-1]) if self._ages else None
        print 'condition', self._condition 
        print 'is_invent', self._is_invent
        print 'invent_acodes', self._invent_acodes
        print 'acode', self._acode 
        print 'is_area', self._is_area
        print 'ycomp', self._ycomp 

    def _evaluate_basic(self, period, factors):
        result = 0.
        for k in self._dtype_keys:
            dt = self.parent.dtypes[k]
            ycomp = None if self._is_area else dt.ycomps[self._ycomp]
            if isinstance(self._factor[0], float):
                f = pow(*self._factor)
            else:
                f = pow(dt.ycomps[self._factor[0]][period], self._factor[1])
            for factor in factors:
                if isinstance(factor[0], float):
                    f *= pow(*factor)
                else:
                    f *= pow(dt.ycomps[factor[0]][period], factor[0])
            for age in ages:
                area = 0.
                y = ycomp[age] if ycomp else 1.
                if self._is_invent:
                    if self._invent_acodes: 
                        any_operable = False
                        for acode in self._invent_acodes:
                            a = dt.actions[acode]
                            if age in a.oper_a and period in a.oper_p:
                                any_operable = True
                        if any_operable:
                            area += dt.areas[age]
                    else:
                        area += dt.areas[age]
                else: 
                    aa = parent.applied_actions[period]
                    key = k, self._acode, age
                    if key in aa: area += aa[key]
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

            
    def __call__(self, period, factors, exponents):
        if self.is_basic:
            return self._evaluate_basic(period, factor, exponent)
        else:
            return self._evaluate_summary(period, factor, exponent)

        # NOTE: could speed up dispatch with dict of function references
        #if self.is_basic:
        #    if not self.is_themed:
        #        return self._evaluate_basic(period)
        #    else:
        #        return self._evaluate_basic_themed(period)
        #else: # summary output
        #    return self._evaluate_summary(period)

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

    # def __mul__(self, other):
    #     # assume an Output * (float OR time-based ycomp)
    #     if not self.is_themed:
    #         return [i * other for i self()]
    #     else:
    #         return self() * other

    # def __div__(self, other):
    #     # assume an Output / (float OR time-based ycomp)
    #     if not self.is_themed:
    #         return [i / other for i self()]
    #     else:
    #         return self() / other

         
class WoodstockModel:
    """
    Interface to import Woodstock models.
    """
    _ytypes = {'*Y':'a', '*YT':'t', '*YC':'c'}
        
    def __init__(self,
                 model_name,
                 model_path,
                 horizon=common.HORIZON_DEFAULT,
                 period=common.PERIOD_DEFAULT,
                 max_age=common.MAX_AGE_DEFAULT,
                 species_groups=common.SPECIES_GROUPS_WOODSTOCK_QC):
        self.model_name = model_name
        self.model_path = model_path
        self.horizon = horizon
        self.period = period
        self.current_period = 0
        self.max_age = max_age
        self.ages = range(max_age+1)
        self._species_groups = species_groups
        self.actions = {}
        self._themes = []
        self._theme_basecodes = []
        self.dtypes = {}
        self.constants = {}
        self.output_groups = {}
        self.outputs = {}
        self.applied_actions = {i:{} for i in range(self.horizon)}
        self.ycomps = set()
    
    def apply_action(dtype_key, acode, age, area):
        def resolve_replace(dtk, expr):
            # HACK ####################################################################
            # Too lazy to implement all the use cases.
            # This should work OK for BFEC models (TO DO: confirm).
            tokens = re.split('\s+', expr)
            i = int(tokens[0][3]) - 1
            return str(eval(expr.replace(tokens[0], dtk[i])))
            ###########################################################################
        def resolve_append(dtk, expr):
            # HACK ####################################################################
            # Too lazy to implement.
            # Not used in BFEC models (TO DO: confirm).
            assert False # brick wall (deal with this case later, as needed)
            ###########################################################################
        dt = self.dtypes[dtype_key]
        assert acode in dt.actions
        assert dt.areas[age] >= area
        dt.areas[age] -= area
        for target in dt.transitions[acode, age]:
            tmask, tprop, tage, tlock, treplace, tappend = target # unpack tuple
            dtk = list(dtype_key) # start with source key
            ###########################################################################
            # DO TO: Confirm correct order for evaluating mask, _APPEND and _REPLACE...
            dtk = [t if tmask[i] == '?' else tmask[i] for t in dtk] 
            if treplace: dtk[treplace[0]] = resolve_replace(dtk, treplace[1])
            if tappend: dtk[tappend[0]] = resolve_append(dtk, tappend[1])
            ###########################################################################
            if dtk not in self.dtypes: # new development type (clone source type)
                _dt = copy.copy(dt)
                _dt.areas[age] = area * tprop
                self.dtypes[dtk] = _dt
            else:
                self.dtypes[dtk].areas += area * tprop
        self.applied_actions[self.current_period][dtype_key, acode, age] = area

    def _resolve_outputs_buffer(self, s, for_flag=None):
        n = self.nthemes
        group = 'no_group' # outputs declared at top of file assigned to 'no_group'
        self.output_groups[group] = set()
        ocode = ''
        buffering_for = False 
        s = re.sub(r'\{.*?\}', '', s, flags=re.M|re.S) # remove curly-bracket comments
        #ocode_pattern = re.compile(r'(?<=\*OUTPUT)\s+(\w|%)+((?=\s)|(?=\())|(?=$)')
        ocode_pattern1 = re.compile(r'(?<=\*OUTPUT)\s+(\w)+((?=\s)|(?=\())|(?=$)')
        ocode_pattern2 = re.compile(r'(?<=\*OUTPUT)\s+(\w)+')
        theme_pattern = re.compile(r'(?<=\(_TH)[0-9]+')
        for l in re.split(r'[\r\n]+', s, flags=re.M|re.S):
            if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
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
            l = l.strip().partition(';')[0].strip()
                    s = re.sub('\s+', ' ', expression) # separate tokens by single space
        s = s.replace(' (', '(')  # remove space to left of left parentheses
        t = s.lower().split(' ')
        # filter dtypes, if starts with mask
        mask = None
        if not (t[0] == '@' or t[0] == '_' or t[0] in self.parent.actions):
            mask = tuple(t[:self.parent.nthemes])
            t = t[self.parent.nthemes:] # pop
        self._dtype_keys = self.parent.unmask(mask) if mask else self.parent.dtypes.keys()
        # extract @AGE or @YLD condition, if present
        self._ages = None
        self._condition = None
        if t[0].startswith('@age'):
            lo, hi = [int(a)+i for i, a in enumerate(t[0][5:-1].split('..'))]
            hi = min(hi, self.parent.max_age+1) # they get carried away with range bounds...
            self._ages = range(lo, hi)
            t = t[1:] # pop
        elif t[0].startswith('@yld'):
            ycomp, args = t[0][5:-1].split(',')
            self._condition = ycomp, tuple(float(a) for a in args.split('..'))
            self._ages = None
            t = t[1:] # pop
        if not self._ages and not self._condition: self._ages = range(self.parent.max_age+1)
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

        # DEBUG
        print
        print s
        print 'mask', ' '.join(mask)
        print 'n dtypes', len(self._dtype_keys)
        print 'ages', (self._ages[0], self._ages[-1]) if self._ages else None
        print 'condition', self._condition 
        print 'is_invent', self._is_invent
        print 'invent_acodes', self._invent_acodes
        print 'acode', self._acode 
        print 'is_area', self._is_area
        print 'ycomp', self._ycomp 

    def _evaluate_basic(self, period, factors):
        result = 0.
        for k in self._dtype_keys:
            dt = self.parent.dtypes[k]
            ycomp = None if self._is_area else dt.ycomps[self._ycomp]
            if isinstance(self._factor[0], float):
                f = pow(*self._factor)
            else:
                f = pow(dt.ycomps[self._factor[0]][period], self._factor[1])
            for factor in factors:
                if isinstance(factor[0], float):
                    f *= pow(*factor)
                else:
                    f *= pow(dt.ycomps[factor[0]][period], factor[0])
            for age in ages:
                area = 0.
                y = ycomp[age] if ycomp else 1.
                if self._is_invent:
                    if self._invent_acodes: 
                        any_operable = False
                        for acode in self._invent_acodes:
                            a = dt.actions[acode]
                            if age in a.oper_a and period in a.oper_p:
                                any_operable = True
                        if any_operable:
                            area += dt.areas[age]
                    else:
                        area += dt.areas[age]
                else: 
                    aa = parent.applied_actions[period]
                    key = k, self._acode, age
                    if key in aa: area += aa[key]
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

            
    def __call__(self, period, factors, exponents):
        if self.is_basic:
            return self._evaluate_basic(period, factor, exponent)
        else:
            return self._evaluate_summary(period, factor, exponent)

        # NOTE: could speed up dispatch with dict of function references
        #if self.is_basic:
        #    if not self.is_themed:
        #        return self._evaluate_basic(period)
        #    else:
        #        return self._evaluate_basic_themed(period)
        #else: # summary output
        #    return self._evaluate_summary(period)

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

    # def __mul__(self, other):
    #     # assume an Output * (float OR time-based ycomp)
    #     if not self.is_themed:
    #         return [i * other for i self()]
    #     else:
    #         return self() * other

    # def __div__(self, other):
    #     # assume an Output / (float OR time-based ycomp)
    #     if not self.is_themed:
    #         return [i / other for i self()]
    #     else:
    #         return self() / other

         
class WoodstockModel:
    """
    Interface to import Woodstock models.
    """
    _ytypes = {'*Y':'a', '*YT':'t', '*YC':'c'}
        
    def __init__(self,
                 model_name,
                 model_path,
                 horizon=common.HORIZON_DEFAULT,
                 period=common.PERIOD_DEFAULT,
                 max_age=common.MAX_AGE_DEFAULT,
                 species_groups=common.SPECIES_GROUPS_WOODSTOCK_QC):
        self.model_name = model_name
        self.model_path = model_path
        self.horizon = horizon
        self.period = period
        self.current_period = 0
        self.max_age = max_age
        self.ages = range(max_age+1)
        self._species_groups = species_groups
        self.actions = {}
        self._themes = []
        self._theme_basecodes = []
        self.dtypes = {}
        self.constants = {}
        self.output_groups = {}
        self.outputs = {}
        self.applied_actions = {i:{} for i in range(self.horizon)}
        self.ycomps = set()
    
    def apply_action(dtype_key, acode, age, area):
        def resolve_replace(dtk, expr):
            # HACK ####################################################################
            # Too lazy to implement all the use cases.
            # This should work OK for BFEC models (TO DO: confirm).
            tokens = re.split('\s+', expr)
            i = int(tokens[0][3]) - 1
            return str(eval(expr.replace(tokens[0], dtk[i])))
            ###########################################################################
        def resolve_append(dtk, expr):
            # HACK ####################################################################
            # Too lazy to implement.
            # Not used in BFEC models (TO DO: confirm).
            assert False # brick wall (deal with this case later, as needed)
            ###########################################################################
        dt = self.dtypes[dtype_key]
        assert acode in dt.actions
        assert dt.areas[age] >= area
        dt.areas[age] -= area
        for target in dt.transitions[acode, age]:
            tmask, tprop, tage, tlock, treplace, tappend = target # unpack tuple
            dtk = list(dtype_key) # start with source key
            ###########################################################################
            # DO TO: Confirm correct order for evaluating mask, _APPEND and _REPLACE...
            dtk = [t if tmask[i] == '?' else tmask[i] for t in dtk] 
            if treplace: dtk[treplace[0]] = resolve_replace(dtk, treplace[1])
            if tappend: dtk[tappend[0]] = resolve_append(dtk, tappend[1])
            ###########################################################################
            if dtk not in self.dtypes: # new development type (clone source type)
                _dt = copy.copy(dt)
                _dt.areas[age] = area * tprop
                self.dtypes[dtk] = _dt
            else:
                self.dtypes[dtk].areas += area * tprop
        self.applied_actions[self.current_period][dtype_key, acode, age] = area

    def _resolve_outputs_buffer(self, s, for_flag=None):
        n = self.nthemes
        group = 'no_group' # outputs declared at top of file assigned to 'no_group'
        self.output_groups[group] = set()
        ocode = ''
        buffering_for = False 
        s = re.sub(r'\{.*?\}', '', s, flags=re.M|re.S) # remove curly-bracket comments
        #ocode_pattern = re.compile(r'(?<=\*OUTPUT)\s+(\w|%)+((?=\s)|(?=\())|(?=$)')
        ocode_pattern1 = re.compile(r'(?<=\*OUTPUT)\s+(\w)+((?=\s)|(?=\())|(?=$)')
        ocode_pattern2 = re.compile(r'(?<=\*OUTPUT)\s+(\w)+')
        theme_pattern = re.compile(r'(?<=\(_TH)[0-9]+')
        for l in re.split(r'[\r\n]+', s, flags=re.M|re.S):
            if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
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
            l = l.strip().partition(';')[0].strip()
                    s = re.sub('\s+', ' ', expression) # separate tokens by single space
        s = s.replace(' (', '(')  # remove space to left of left parentheses
        t = s.lower().split(' ')
        # filter dtypes, if starts with mask
        mask = None
        if not (t[0] == '@' or t[0] == '_' or t[0] in self.parent.actions):
            mask = tuple(t[:self.parent.nthemes])
            t = t[self.parent.nthemes:] # pop
        self._dtype_keys = self.parent.unmask(mask) if mask else self.parent.dtypes.keys()
        # extract @AGE or @YLD condition, if present
        self._ages = None
        self._condition = None
        if t[0].startswith('@age'):
            lo, hi = [int(a)+i for i, a in enumerate(t[0][5:-1].split('..'))]
            hi = min(hi, self.parent.max_age+1) # they get carried away with range bounds...
            self._ages = range(lo, hi)
            t = t[1:] # pop
        elif t[0].startswith('@yld'):
            ycomp, args = t[0][5:-1].split(',')
            self._condition = ycomp, tuple(float(a) for a in args.split('..'))
            self._ages = None
            t = t[1:] # pop
        if not self._ages and not self._condition: self._ages = range(self.parent.max_age+1)
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

        # DEBUG
        print
        print s
        print 'mask', ' '.join(mask)
        print 'n dtypes', len(self._dtype_keys)
        print 'ages', (self._ages[0], self._ages[-1]) if self._ages else None
        print 'condition', self._condition 
        print 'is_invent', self._is_invent
        print 'invent_acodes', self._invent_acodes
        print 'acode', self._acode 
        print 'is_area', self._is_area
        print 'ycomp', self._ycomp 

    def _evaluate_basic(self, period, factors):
        result = 0.
        for k in self._dtype_keys:
            dt = self.parent.dtypes[k]
            ycomp = None if self._is_area else dt.ycomps[self._ycomp]
            if isinstance(self._factor[0], float):
                f = pow(*self._factor)
            else:
                f = pow(dt.ycomps[self._factor[0]][period], self._factor[1])
            for factor in factors:
                if isinstance(factor[0], float):
                    f *= pow(*factor)
                else:
                    f *= pow(dt.ycomps[factor[0]][period], factor[0])
            for age in ages:
                area = 0.
                y = ycomp[age] if ycomp else 1.
                if self._is_invent:
                    if self._invent_acodes: 
                        any_operable = False
                        for acode in self._invent_acodes:
                            a = dt.actions[acode]
                            if age in a.oper_a and period in a.oper_p:
                                any_operable = True
                        if any_operable:
                            area += dt.areas[age]
                    else:
                        area += dt.areas[age]
                else: 
                    aa = parent.applied_actions[period]
                    key = k, self._acode, age
                    if key in aa: area += aa[key]
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

            
    def __call__(self, period, factors, exponents):
        if self.is_basic:
            return self._evaluate_basic(period, factor, exponent)
        else:
            return self._evaluate_summary(period, factor, exponent)

        # NOTE: could speed up dispatch with dict of function references
        #if self.is_basic:
        #    if not self.is_themed:
        #        return self._evaluate_basic(period)
        #    else:
        #        return self._evaluate_basic_themed(period)
        #else: # summary output
        #    return self._evaluate_summary(period)

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

    # def __mul__(self, other):
    #     # assume an Output * (float OR time-based ycomp)
    #     if not self.is_themed:
    #         return [i * other for i self()]
    #     else:
    #         return self() * other

    # def __div__(self, other):
    #     # assume an Output / (float OR time-based ycomp)
    #     if not self.is_themed:
    #         return [i / other for i self()]
    #     else:
    #         return self() / other

         
class WoodstockModel:
    """
    Interface to import Woodstock models.
    """
    _ytypes = {'*Y':'a', '*YT':'t', '*YC':'c'}
        
    def __init__(self,
                 model_name,
                 model_path,
                 horizon=common.HORIZON_DEFAULT,
                 period=common.PERIOD_DEFAULT,
                 max_age=common.MAX_AGE_DEFAULT,
                 species_groups=common.SPECIES_GROUPS_WOODSTOCK_QC):
        self.model_name = model_name
        self.model_path = model_path
        self.horizon = horizon
        self.period = period
        self.current_period = 0
        self.max_age = max_age
        self.ages = range(max_age+1)
        self._species_groups = species_groups
        self.actions = {}
        self._themes = []
        self._theme_basecodes = []
        self.dtypes = {}
        self.constants = {}
        self.output_groups = {}
        self.outputs = {}
        self.applied_actions = {i:{} for i in range(self.horizon)}
        self.ycomps = set()
    
    def apply_action(dtype_key, acode, age, area):
        def resolve_replace(dtk, expr):
            # HACK ####################################################################
            # Too lazy to implement all the use cases.
            # This should work OK for BFEC models (TO DO: confirm).
            tokens = re.split('\s+', expr)
            i = int(tokens[0][3]) - 1
            return str(eval(expr.replace(tokens[0], dtk[i])))
            ###########################################################################
        def resolve_append(dtk, expr):
            # HACK ####################################################################
            # Too lazy to implement.
            # Not used in BFEC models (TO DO: confirm).
            assert False # brick wall (deal with this case later, as needed)
            ###########################################################################
        dt = self.dtypes[dtype_key]
        assert acode in dt.actions
        assert dt.areas[age] >= area
        dt.areas[age] -= area
        for target in dt.transitions[acode, age]:
            tmask, tprop, tage, tlock, treplace, tappend = target # unpack tuple
            dtk = list(dtype_key) # start with source key
            ###########################################################################
            # DO TO: Confirm correct order for evaluating mask, _APPEND and _REPLACE...
            dtk = [t if tmask[i] == '?' else tmask[i] for t in dtk] 
            if treplace: dtk[treplace[0]] = resolve_replace(dtk, treplace[1])
            if tappend: dtk[tappend[0]] = resolve_append(dtk, tappend[1])
            ###########################################################################
            if dtk not in self.dtypes: # new development type (clone source type)
                _dt = copy.copy(dt)
                _dt.areas[age] = area * tprop
                self.dtypes[dtk] = _dt
            else:
                self.dtypes[dtk].areas += area * tprop
        self.applied_actions[self.current_period][dtype_key, acode, age] = area

    def _resolve_outputs_buffer(self, s, for_flag=None):
        n = self.nthemes
        group = 'no_group' # outputs declared at top of file assigned to 'no_group'
        self.output_groups[group] = set()
        ocode = ''
        buffering_for = False 
        s = re.sub(r'\{.*?\}', '', s, flags=re.M|re.S) # remove curly-bracket comments
        #ocode_pattern = re.compile(r'(?<=\*OUTPUT)\s+(\w|%)+((?=\s)|(?=\())|(?=$)')
        #ocode_pattern1 = re.compile(r'(?<=\*OUTPUT)\s+(\w)+((?=\s)|(?=\())|(?=$)')
        #ocode_pattern2 = re.compile(r'(?<=\*OUTPUT)\s+(\w)+')
        #theme_pattern = re.compile(r'(?<=\(_TH)[0-9]+')
        for l in re.split(r'[\r\n]+', s, flags=re.M|re.S):
            if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
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
            l = l.strip().partition(';')[0].strip()
            s = re.sub('\s+', ' ', expression) # separate tokens by single space
            s = s.replace(' (', '(')  # remove space to left of left parentheses
            t = s.lower().split(' ')

            ##################################################
            # HACK ###########################################
            # substitute ugly symbols have in ocodes...
            l = l.replace(r'%', 'p')
            l = l.replace(r'$', 's')
            ##################################################
            tokens = l.split(' ')
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
                if ocode: self.outputs[ocode] = Output(self, ocode, expr, descr, ti)
                tt = tokens[1].split('(')
                ocode = tt[0]
                ti = tt[1][4:-1] if len(tt) > 1 else None
                descr = ' '.join(tokens[2:])
                expr = ''
                self.output_groups[group].add(ocode)
                if keyword == 'level':
                    self.outputs[ocode] = Output(self, ocode, expr, descr, ti, is_level=True)
                    ocode = ''
            elif l.startswith('*SOURCE'):
                keyword = 'source'
                expr += ' '.join(tokens[1:]) # clean up: separate args by single space
            elif keyword == 'source': # continuation line of SOURCE expression
                expr += ' ' + ' '.join(tokens)       
        
    @timed
    def import_outputs_section(self, filename_suffix='out'):
        #keyword = ''
        with open('%s/%s.%s' % (self.model_path, self.model_name, filename_suffix)) as f:
            s = f.read()
        self._resolve_outputs_buffer(s)
            
    @timed
    def import_landscape_section(self, filename_suffix='lan'):
        with open('%s/%s.%s' % (self.model_path, self.model_name, filename_suffix)) as f:
            data = f.read()
        _data = re.search(r'\*THEME.*', data, re.M|re.S).group(0) # strip leading junk
        t_data = re.split(r'\*THEME.*\n', _data)[1:] # split into theme-wise chunks
        for ti, t in enumerate(t_data):
            self._themes.append({})
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
        self.nthemes = len(self._themes)

    def theme_basecodes(self, theme_index):
        return self._themes[theme_index]
        
    @timed    
    def import_areas_section(self, filename_suffix='are'):
        n = self.nthemes
        with open('%s/%s.%s' % (self.model_path, self.model_name, filename_suffix)) as f:
            for l in f:
                if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
                l = l.strip().partition(';')[0] # strip leading whitespace and trailing comments
                t = re.split('\s+', l)
                key = tuple(_t.lower() for _t in t[1:n+1])
                age = int(t[n+1])
                area = float(t[n+2].replace(',', ''))
                if key not in self.dtypes: self.dtypes[key] = DevelopmentType(key, self)
                self.dtypes[key].areas[age] += area
                    
    def _expand_action(self, c):
        self._actions = t
        return [c] if t[c] == c else list(_cfi(self._expand_action(t, c) for c in t[c]))
                
    def _expand_theme(self, t, c): # depth-first search recursive aggregate theme code expansion
        return [c] if t[c] == c else list(_cfi(self._expand_theme(t, c) for c in t[c]))
                
    def unmask(self, mask):
        """
        Iteratively filter list of development type keys using mask values.
        """
        dtype_keys = copy.copy(self.dtypes.keys()) # filter this
        for ti, tac in enumerate(mask):
            if tac == '?': continue # wildcard matches all
            tacs = self._expand_theme(self._themes[ti], tac)
            dtype_keys = [dt for dt in dtype_keys if dt[ti] in tacs] # exclude bad matches
        return dtype_keys

    @timed                            
    def import_constants_section(self, filename_suffix='con'):
        with open('%s/%s.%s' % (self.model_path, self.model_name, filename_suffix)) as f:
            for lnum, l in enumerate(f):
                if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
                l = l.strip().partition(';')[0].strip() # strip leading whitespace, trailing comments
                t = re.split('\s+', l)
                self.constants[t[0].lower()] = float(t[1])

    @timed        
    def import_yields_section(self, filename_suffix='yld', verbose=True):
        ###################################################
        # local utility functions #########################
        def flush_yields(t, m, n, c):
            self.ycomps.update(n)
            if t in 'at': c = {y:core.Curve(y, points=c[y], type=t) for y in n}
            for k in self.unmask(m):
                dt = self.dtypes[k]
                dtk = dt.ycomps.keys()
                ynames = [y for y in n if y not in dtk]
                for y in ynames: dt.add_ycomp(t, y, c[y])
        ###################################################
        n = self.nthemes
        ytype = ''
        mask = ('?',) * self.nthemes
        ynames = []
        data = None
        with open('%s/%s.%s' % (self.model_path, self.model_name, filename_suffix)) as f:
            for lnum, l in enumerate(f):
                if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
                l = l.strip().partition(';')[0].strip() # strip leading whitespace and trailing comments
                t = re.split('\s+', l)
                if t[0].startswith('*Y'): # new yield definition
                    newyield = True
                    flush_yields(ytype, mask, ynames, data) # apply yield from previous block
                    ytype = self._ytypes[t[0]]
                    mask = tuple(_t.lower() for _t in t[1:])
                    if verbose: print lnum, ' '.join(mask)
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
                if is_tabular:
                    x = int(t[0])
                    for i, yname in enumerate(ynames):
                        data[yname].append((x, float(t[i+1])))
                else:
                    if ytype in 'at': # standard or time-based yield (extract xy values)
                        if not common.is_num(t[0]): # first line of row-based yield component
                            yname = t[0].lower()
                            ynames.append(yname)
                            data[yname] = [(i+int(t[1]), float(t[i+2])) for i in range(len(t)-2)]
                        else: # continuation of row-based yield compontent
                            x_last = data[yname][-1][0]
                            data[yname].extend([(i+x_last+1, float(t[i])) for i in range(len(t))])
                    else:
                        yname = t[0].lower()
                        ynames.append(yname)
                        data[yname] = t[1] # complex yield (defer interpretation) 
        flush_yields(ytype, mask, ynames, data)
        i_max = 10 # debug only
        for i, key in enumerate(self.dtypes):
            if verbose: print 'compiling complex ycomps for dtype', i, 'of', len(self.dtypes), ' '.join(key)
            if i > i_max: break # debug only
            self.dtypes[key].compile_complex_ycomps()

    @timed        
    def import_actions_section(self, filename_suffix='act'):
        n = self.nthemes
        actions = {}
        oper = {}
        aggregates = {}
        partials = {}
        keyword = ''
        with open('%s/%s.%s' % (self.model_path, self.model_name, filename_suffix)) as f: s = f.read()
        s = re.sub(r'\{.*?\}', '', s, flags=re.M|re.S) # remove curly-bracket comments
        for l in re.split(r'[\r\n]+', s, flags=re.M|re.S):
            if re.match('^\s*(;|$)', l): continue # skip comments and blank lines
            l = l.strip().partition(';')[0].strip() # strip leading whitespace and trailing comments
            tokens = re.split('\s+', l)
            if l.startswith('*ACTION'): 
                keyword = 'action'
                acode = tokens[1].lower()
                targetage = 0 if tokens[2] == 'Y' else None
                descr = ' '.join(tokens[3:])
                lockexempt = '_LOCKEXEMPT' in tokens
                self.actions[acode] = Action(acode, targetage, descr, lockexempt)
                oper[acode] = {}
                #self._actions[acode] = acode
            elif l.startswith('*OPERABLE'):
                keyword = 'operable'
                acode = tokens[1].lower()
                action = self.actions[acode]
            elif l.startswith('*AGGREGATE'):
                keyword = 'aggregate'
                acode = tokens[1].lower()
                self.actions[acode] = Action(acode)
            elif l.startswith('*PARTIAL'): 
                keyword = 'partial'
                acode = tokens[1].lower()
                partials[acode] = []
            else: # continuation of OPERABLE, AGGREGATE, or PARTIAL block
                if keyword == 'operable':
                    oper[acode][tuple(t.lower() for t in tokens[:n])] =  ' '.join(tokens[n:])
                elif keyword == 'aggregate':
                    components = {acode:self.actions[acode.lower()] for acode in tokens}
                    self.actions[acode].components.update(components)
                elif keyword == 'partial':
                    partials[acode].extend(tokens)
        for acode, a in self.actions.items():
            if a.components: continue # aggregate action, skip
            if acode in partials: a.partial = partials[acode]
            for mask, expression in oper[acode].items():
                for k in self.unmask(mask):
                    _a = copy.copy(a)
                    _a.oper_expr = expression
                    self.dtypes[k].actions[acode] = _a
        for dt in self.dtypes.values(): dt.compile_actions()

    def resolve_treplace(self, dt, treplace):
        if '_TH' in treplace: # assume incrementing integer theme value
            i = int(re.search('(?<=_TH)\w+', treplace).group(0))
            return eval(re.sub('_TH%i'%i, str(dt.key[i-1]), treplace))
        else:
            assert False # many other possible arguments (see Woodstock documentation)

    def resolve_tappend(self, dt, tappend):
        assert False # brick wall (not implemented yet)

    def resolve_tmask(self, dt, tmask, treplace, tappend):
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
            return self.ages
        elif condition.startswith('@AGE'):
            lo, hi = [int(a) for a in condition[5:-1].split('..')]
            return range(lo, hi+1)
        elif condition.startswith('@YLD'):
            args = re.split('\s?,\s?', condition[5:-1])
            ycomp = args[0].lower()
            lo, hi = [float(y) for y in args[1].split('..')]
            return self.dtypes[dtype_key].resolve_condition(ycomp, hi, lo)
        
    @timed                        
    def import_transitions_section(self, filename_suffix='trn'):
        # local utility function #########################
        def flush_transitions(acode, sources):
            if not acode: return
            for smask, scond in sources:
                for k in self.unmask(smask):
                    dt = self.dtypes[k]
                    for x in self.resolve_condition(scond, k):
                        dt.transitions[acode, x] = sources[smask, scond] # resolve later
        ##################################################                    
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
                smask = tuple(t.lower() for t in tokens[1:self.nthemes+1])
                match = re.search(r'@.+\)', l)
                scond = match.group(0) if match else ''
                #scond = l.split('@')[1] if l.find('@') >= 0 else ''
                sources[(smask, scond)] = []
            elif l.startswith('*TARGET'):
                tmask = tuple(t.lower() for t in tokens[1:self.nthemes+1])
                #print self.nthemes, n, tokens, tokens[self.nthemes+1]
                tprop = float(tokens[self.nthemes+1]) * 0.01
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
                sources[(smask, scond)].append((tmask, tprop, tage, tlock, treplace, tappend))
        flush_transitions(acode, sources)

    
    def import_optimize_section(self, filename_suffix='opt'):
        pass

    def import_graphics_section(self, filename_suffix='gra'):
        pass

    def import_lifespan_section(self, filename_suffix='lif'):
        pass

    def import_lifespan_section(self, filename_suffix='lif'):
        pass

    def import_schedule_section(self, filename_suffix='seq'):
        pass

    def import_control_section(self, filename_suffix='run'):
        pass

    def grow(self, periods=1):
        self.current_period += periods
        for dt in self.dtypes.values(): dt.grow(periods)

if __name__ == '__main__':
    pass
