import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

def template_mpc(model, silence_solver = False):

    mpc = do_mpc.controller.MPC(model)

    mpc.settings.n_horizon =  40
    mpc.settings.n_robust =  0
    mpc.settings.open_loop =  0
    mpc.settings.t_step =  0.04
    mpc.settings.state_discretization =  'collocation'
    mpc.settings.collocation_type =  'radau'
    mpc.settings.collocation_deg =  3
    mpc.settings.collocation_ni =  1
    mpc.settings.store_full_solution =  True

    if silence_solver:
        mpc.settings.supress_ipopt_output()

    mterm = model.aux['E_kin'] - model.aux['E_pot']
    lterm = -model.aux['E_pot'] + 10 * sum1((model.x['theta'] - model.tvp['theta_set'])**2)

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(torque=0.01)

    mpc.bounds['lower', '_u', 'torque', 0] = -30
    mpc.bounds['upper', '_u', 'torque', 0] = 30
    mpc.bounds['lower', '_u', 'torque', 1] = -60
    mpc.bounds['upper', '_u', 'torque', 1] = 60

    tvp_template = mpc.get_tvp_template()
    tvp_template['_tvp', 0, 'theta_set'] = 0
    tvp_template['_tvp', 1, 'theta_set'] = 0
    mpc.set_tvp_fun(lambda t_ind: tvp_template)  # 時変パラメータとして固定値を返す

    mpc.setup()

    return mpc
