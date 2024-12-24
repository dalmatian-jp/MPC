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

    # L1 = 0.9  #m, length of the first rod
    # L2 = 0.88  #m, length of the second rod
    # r1 = 0.58
    # r2 = 0.32
    # I1 = 9.21
    # I2 = 5.35

    #jafari
    L1 = 0.78  #m, length of the first rod
    L2 = 0.73  #m, length of the second rod
    r1 = L1/2
    r2 = L2/2
    I1 = 0.35
    I2 = 0.25

    m1 = 0.41  # kg, mass of the first rod
    m2 = 0.14  # kg, mass of the second rod

    g = 9.8 # m/s^2, gravity

    # Setpoint angles (target positions):
    theta_set = model.set_variable('_tvp', 'theta_set', (2, 1))
    dtheta_set = model.set_variable('_tvp', 'dtheta_set', (2, 1))
    torque_set_0 = model.set_variable('_tvp', 'torque_set_0')  # 目標トルク (リンク1)
    torque_set_1 = model.set_variable('_tvp', 'torque_set_1')  # 目標トルク (リンク2)


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

    M11 = I1 + I2 + m1 * r1 **2 + m2 * (L1 ** 2 + r2 ** 2) + L1 * L2 * m2 * cos(theta[1])
    M12 = m2 * r2 ** 2 + m2 * L1 * r2 * cos(theta[1]) + I2
    M21 = M12
    M22 = I2 + m2 * r2 ** 2

    C1 = -1/2 * L1 * L2 * m2 * sin(theta[1]) * dtheta[1] ** 2 - L1 * L2 * m2 * sin(theta[1]) * dtheta[0] * dtheta[1]
    C2 = 1/2 * L1 * dtheta[0] ** 2 * sin(theta[1])

    F1 = 1/2 * g * (-L2 * m2 * sin(theta[0] + theta[1]) - L1 * m1 * sin(theta[0]) + 2 * L1 + m2 + sin(theta[0])) 
    F2 = -1/2 * g * L2 * m2 * sin(theta[0] + theta[1])

    # Euler Lagrange equations for the DIP system (in the form f(x,u,z) = 0)
    euler_lagrange = vertcat(
        # 1
        M11 * ddtheta[0] + M12 * ddtheta[1] + C1 + F1 - torque_0,
        # 2
        M21 * ddtheta[0] + M22 * ddtheta[1] + C2 + F2 - torque_1
    )

    model.set_alg('euler_lagrange', euler_lagrange)

    # Build the model
    model.setup()

    return model

