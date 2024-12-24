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


def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Certain parameters

    L1 = 0.9  #m, length of the first rod
    L2 = 0.88  #m, length of the second rod
    r1 = 0.58
    r2 = 0.32
    I1 = 9.21
    I2 = 5.35

    #jafari
    L1 = 0.78  #m, length of the first rod
    L2 = 0.73  #m, length of the second rod
    r1 = L1/2
    r2 = L2/2
    I1 = 0.35
    I2 = 0.25

    m1 = 11.41  # kg, mass of the first rod
    m2 = 50.14  # kg, mass of the second rod

    g = 9.8 # m/s^2, gravity

    h1 = I1 + I2 + m1 * r1**2 + m2 * (L1**2 + r2**2)
    h2 = m2 * L1 * r2
    h3 = I2 + m2 * r2

    # Setpoint angles (target positions):
    theta_set = model.set_variable('_tvp', 'theta_set', (2, 1))
    dtheta_set = model.set_variable('_tvp', 'dtheta_set', (2, 1))

    # States struct (optimization variables):
    theta = model.set_variable('_x', 'theta', (2, 1))  # Angles for the two links
    dtheta = model.set_variable('_x', 'dtheta', (2, 1))  # Angular velocities for the two links

    # Algebraic states:
    ddtheta = model.set_variable('_z', 'ddtheta', (2, 1))  # Angular accelerations for the two links

    # Input struct (optimization variables):
    torque_0 = model.set_variable('_u', 'torque_0')  # torque[0]の定義
    torque_1 = model.set_variable('_u', 'torque_1')  # torque[1]の定義


    # Differential equations
    model.set_rhs('theta', dtheta)
    model.set_rhs('dtheta', ddtheta)



    # Euler Lagrange equations for the DIP system (in the form f(x,u,z) = 0)
    euler_lagrange = vertcat(
        # 1
        (h1 + 2 * h2 * cos(theta[1])) * ddtheta[0] + (h2 + h3 * cos(theta[1])) * ddtheta[1] - 
        h2 * sin(theta[1]) * dtheta[0] * dtheta[1] - h2 * sin(theta[1]) * (dtheta[0] + dtheta[1]) * dtheta[1]
        - g * ((m1 * r1 + m2 * L1) * sin(theta[0]) + m2 * r2 * sin(theta[0]+theta[1])) - torque_0,
        # 2
        (h2 + h3 * cos(theta[1])) * ddtheta[0] + 
        h3 * ddtheta[1] + 
        h2 * sin(theta[1]) * dtheta[0] * dtheta[0] - 
        g * m2 * r2 * sin(theta[0]+theta[1]) - torque_1
    )

    model.set_alg('euler_lagrange', euler_lagrange)

    # Expressions for kinetic and potential energy
    E_kin_p1 = 1/2 * I1 * dtheta[0] ** 2 + 1/2 * m1 * r1 ** 2 * dtheta[0] ** 2
    E_kin_p2 = 1/2 * I2 * (dtheta[0] + dtheta[1]) ** 2 + 1/2 * m2 * (L1 ** 2 * dtheta[0] ** 2 + 2 * L1 * r2 * cos(theta[1]) * dtheta[0] * (dtheta[0] + dtheta[1]) + r2 ** 2 * (dtheta[0] + dtheta[1]) ** 2)

    E_kin =  E_kin_p1 + E_kin_p2

    E_pot = m1 * g * r1 * cos(theta[0]) + m2 * g * (L1 * cos(theta[0]) + r2 * cos(theta[0] + theta[1]))

    model.set_expression('E_kin', E_kin)
    model.set_expression('E_pot', E_pot)

    # Build the model
    model.setup()

    return model

