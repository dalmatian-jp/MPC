#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

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
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    mpc.settings.n_horizon =  100
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
    mpc.set_rterm(torque=0.1)


    mpc.bounds['lower', '_u', 'torque'] = -10
    mpc.bounds['upper', '_u', 'torque'] = 10



    # Values for the masses (for robust MPC)
    m1_var = 0.2*np.array([1, 0.95, 1.05])
    m2_var = 0.2*np.array([1, 0.95, 1.05])
    mpc.set_uncertainty_values(m1=m1_var, m2=m2_var)


    tvp_template = mpc.get_tvp_template()

    # When to switch setpoint:
    t_switch = 4    # seconds
    ind_switch = t_switch // mpc.settings.t_step

    def tvp_fun(t_ind):
        ind = t_ind // mpc.settings.t_step
        if ind <= ind_switch:
            tvp_template['_tvp', 0, 'theta_set'] = -0.8
            tvp_template['_tvp', 1, 'theta_set'] =  0.8
        else:
            tvp_template['_tvp', 0, 'theta_set'] = 0.0
            tvp_template['_tvp', 1, 'theta_set'] = 0.0
        return tvp_template



    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    return mpc
