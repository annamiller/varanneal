"""
Paul Rozdeba (prozdeba@physics.ucsd.edu)
Department of Physics
University of California, San Diego
May 23, 2017

Functions and base class definitions common to all system types using 
variational annealing.
"""

import numpy as np
import adolc
import pyipopt
import scipy.optimize as opt
import time

class ADmin(object):
    """
    ADmin is an object type for using AD ad implemented in ADOL-C to minimize
    arbitrary scalar functions, i.e. functions f s.t. f: R^N --> R.
    """
    def __init__(self):
        """
        These routines are the same for all system types and their variables
        are set in the Annealer objects which inherit ADmin, so nothing special
        to do here really.
        """
        pass

    ############################################################################
    # AD taping & derivatives
    ############################################################################
    def tape_A(self, xtrace):
        """
        Tape the objective function.
        """
        print('Taping action evaluation...')
        tstart = time.time()
        
        """
        trace objective function
        """
        adolc.trace_on(self.adolcID)
        # set the active independent variables
        ax = adolc.adouble(xtrace)
        adolc.independent(ax)
        # set the dependent variable (or vector of dependent variables)
        af = self.A(ax)
        adolc.dependent(af)
        adolc.trace_off()


        #IPOPT needs the lagrangian functions to be traced
        if self.method == 'IPOPT':
            """
            trace lagrangian unconstrained
            """
            #This adolc numbering could cause problems in the future
            adolc.trace_on(self.adolcID + 1000)
            ax = adolc.adouble(xtrace)
            aobj_factor = adolc.adouble(1.)
            adolc.independent(ax)
            adolc.independent(aobj_factor)
            ay = eval_lagrangian(ax,aobj_factor)
            adolc.trace_off()
               
            
        self.taped = True
        print('Done!')
        print('Time = {0} s\n'.format(time.time()-tstart))


    """
    IPOPT Function for unconstrained optimization
    """
    def eval_jac_g(x, flag, user_data=None):
        rows = numpy.array([], dtype=int)
        cols = numpy.array([], dtype=int)
        if flag:
            return (rows, cols)
        else:
            raise Exception('this should not be called for unconstrained optimization')

    """
    Function added for IPOPT
    Add extra input, lagrange and dot(lagrange,eval_g) for constrained optimizations
    """
    def eval_lagrangian(x,obj_factor,user_data = None):
        return obj_factor*eval_f(x)

    """
    Creating this function for IPOPT. Assume Unconstrained.
    """
    def eval_g(x, user_data=None):
        return np.array([],dtype=float)

    """
    Class for Hessian of Lagrangian
    """
    class Eval_h_adolc:
    
        def __init__(self, x):
            options = numpy.array([0,0],dtype=int)
            result = adolc.colpack.sparse_hess_no_repeat(self.adolcID+1000,x,options)
        
            self.rind = numpy.asarray(result[1],dtype=int)
            self.cind = numpy.asarray(result[2],dtype=int)
            self.values = numpy.asarray(result[3],dtype=float)
            self.mask = numpy.where(self.cind < len(x))
            self.nnz = len(numpy.asarray(self.mask).flatten())

        
        def __call__(self, x, lagrange, obj_factor, flag, user_data = None):

            if flag:
                return (self.rind[self.mask], self.cind[self.mask])
            else:
                x = numpy.hstack([x,lagrange,obj_factor])
                result = adolc.colpack.sparse_hess_repeat(self.adolcID+1000, x, self.rind, self.cind, self.values)
                return result[3][self.mask]

    #user_data=None added for IPOPT, needed?
    def A_taped(self, XP, user_data=None):
        return adolc.function(self.adolcID, XP)
    
    def gradA_taped(self, XP,user_data=None):
        return adolc.gradient(self.adolcID, XP)

    def A_gradA_taped(self, XP):
        return adolc.function(self.adolcID, XP), adolc.gradient(self.adolcID, XP)

    def jacA_taped(self, XP):
        return adolc.jacobian(self.adolcID, XP)

    def A_jacaA_taped(self, XP):
        return adolc.function(self.adolcID, XP), adolc.jacobian(self.adolcID, XP)

    #This calculation will likely lead to memory errors
    def hessianA_taped(self, XP):
        return adolc.hessian(self.adolcID, XP)

    ################################################################################
    # Minimization functions
    ################################################################################
    def min_lbfgs_scipy(self, XP0, xtrace=None):
        """
        Minimize f starting from XP0 using L-BFGS-B method in scipy.
        This method supports the use of bounds.
        Returns the minimizing state, the minimum function value, and the L-BFGS
        termination information.
        """
        if self.taped == False:
            self.tape_A(xtrace)

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        res = opt.minimize(self.A_gradA_taped, XP0, method='L-BFGS-B', jac=True,
                           options=self.opt_args, bounds=self.bounds)
        XPmin,status,Amin = res.x, res.status, res.fun

        print("Optimization complete!")
        print("Time = {0} s".format(time.time()-tstart))
        print("Exit flag = {0}".format(status))
        print("Exit message: {0}".format(res.message))
        print("Iterations = {0}".format(res.nit))
        print("Obj. function value = {0}\n".format(Amin))
        return XPmin, Amin, status

    def min_cg_scipy(self, XP0, xtrace=None):
        """
        Minimize f starting from XP0 using nonlinear CG method in scipy.
        Returns the minimizing state, the minimum function value, and the CG
        termination information.
        """
        if self.taped == False:
            self.tape_A(xtrace)

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        res = opt.minimize(self.A_gradA_taped, XP0, method='CG', jac=True,
                           options=self.opt_args)
        XPmin,status,Amin = res.x, res.status, res.fun

        print("Optimization complete!")
        print("Time = {0} s".format(time.time()-tstart))
        print("Exit flag = {0}".format(status))
        print("Exit message: {0}".format(res.message))
        print("Iterations = {0}".format(res.nit))
        print("Obj. function value = {0}\n".format(Amin))
        return XPmin, Amin, status

    def min_tnc_scipy(self, XP0, xtrace=None):
        """
        Minimize f starting from XP0 using Newton-CG method in scipy.
        Returns the minimizing state, the minimum function value, and the CG
        termination information.
        """
        if self.taped == False:
            self.tape_A(xtrace)

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        res = opt.minimize(self.A_gradA_taped, XP0, method='TNC', jac=True,
                           options=self.opt_args, bounds=self.bounds)
        XPmin,status,Amin = res.x, res.status, res.fun

        print("Optimization complete!")
        print("Time = {0} s".format(time.time()-tstart))
        print("Exit flag = {0}".format(status))
        print("Exit message: {0}".format(res.message))
        print("Iterations = {0}".format(res.nit))
        print("Obj. function value = {0}\n".format(Amin))
        return XPmin, Amin, status

    def min_lm_scipy(self, XP0, xtrace=None):
        """
        Minimize f starting from XP0 using Levenberg-Marquardt in scipy.
        Returns the minimizing state, the minimum function value, and the CG
        termination information.
        """
        if self.taped == False:
            self.tape_A(xtrace)

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        res = opt.root(self.A_jacA_taped, XP0, method='lm', jac=True,
                       options=self.opt_args)

        XPmin,status,Amin = res.x, res.status, res.fun

        print("Optimization complete!")
        print("Time = {0} s".format(time.time()-tstart))
        print("Exit flag = {0}".format(status))
        print("Exit message: {0}".format(res.message))
        print("Iterations = {0}".format(res.nit))
        print("Obj. function value = {0}\n".format(Amin))
        return XPmin, Amin, status

    #Other inputs TBD
    def min_ipopt(self, XP0, bounds, xtrace=None):
        """
        Minimize f starting from XP0 using IPOPT.
        Returns the minimizing state, the minimum function value, and the
        termination information.
        """
        if self.taped == False:
            self.tape_A(xtrace)

        #Is this time consuming - creates a new instance of the class at each call
        #Need to move outside this function
        eval_h_adolc = Eval_h_adolc(XP0)
        nnzj = 0
        nnzh = eval_h_adolc.nnz
        #Includes NPest
        nvar = len(XP0)

        #Use bounds - Includes parameters. Does this work for time dependent parameters?
        x_L = np.asarray(bounds)[:,0]
        x_U = np.asarray(bounds)[:,1]
     
        ncon = 0
        g_L = np.array([])
        g_U = np.array([])


        nlp_adolc = pyipopt.create(nvar,x_L,x_U, ncon, g_L, g_U, nnzj, nnzh, A_taped, gradA_taped, eval_g, eval_jac_g, eval_h_adolc)

        #Setting default settings to match minAone with adjustments        
        nlp_adolc.num_option('tol',1e-6)
        nlp_adolc.str_option('mu_strategy','adaptive')
        nlp_adolc.str_option('adaptive_mu_globalization','never-monotone-mode')
        nlp_adolc.int_option('max_iter',1000)
        nlp_adolc.str_option('linear_solver','ma97')
        nlp_adolc.num_option('bound_relax_factor',0)


        #IPOPT distinguishes between num, int, and str options
        #Only supports 3 options atm, could for loop through values with if statements...
        if self.opt_args is not None:
            if 'max_iter' in boundz:
                nlp_adolc.int_option('max_iter',boundz.get('max_iter'))

            if 'tol' in boundz:
                nlp_adolc.num_option('tol',boundz.get('tol'))

            if 'linear_solver' in boundz:
                nlp_adolc.str_option('linear_solver',boundz.get('linear_solver'))
      


        
        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()

        XPmin, _, _, _, Amin, status = nlp_adolc.solve(XP0)
        nlp_adolc.close()

        print("Optimization complete!")
        print("Time = {0} s".format(time.time()-tstart))
        #print("Exit flag = {0}".format(status))
        #print("Exit message: {0}".format(res.message))
        #print("Iterations = {0}".format(res.nit))
        print("Obj. function value = {0}\n".format(Amin))

        return XPmin, Amin, status

    #def min_lm_scipy(self, XP0):
    #    """
    #    Minimize f starting from XP0 using Levenberg-Marquardt in scipy.
    #    Returns the minimizing state, the minimum function value, and the CG
    #    termination information.
    #    """
    #    if self.taped == False:
    #        self.tape_A()
    #
    #    # start the optimization
    #    print("Beginning optimization...")
    #    tstart = time.time()
    #    res = opt.root(self.A_jacA_taped, XP0, method='lm', jac=True,
    #                   options=self.opt_args)
    #
    #    XPmin,status,Amin = res.x, res.status, res.fun
    #
    #    print("Optimization complete!")
    #    print("Time = {0} s".format(time.time()-tstart))
    #    print("Exit flag = {0}".format(status))
    #    print("Exit message: {0}".format(res.message))
    #    print("Iterations = {0}".format(res.nit))
    #    print("Obj. function value = {0}\n".format(Amin))
    #    return XPmin, Amin, status
