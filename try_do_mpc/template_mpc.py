import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

def template_mpc(model, silence_solver=False):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    # MPC設定
    mpc.settings.n_horizon = 10
    mpc.settings.n_robust = 0
    mpc.settings.open_loop = 0
    mpc.settings.t_step = 0.01
    mpc.settings.state_discretization = 'collocation'
    mpc.settings.collocation_type = 'radau'
    mpc.settings.collocation_deg = 3
    mpc.settings.collocation_ni = 1
    mpc.settings.store_full_solution = True

    if silence_solver:
        mpc.settings.supress_ipopt_output()

    # 状態変数と入力変数の取得
    theta = model.x['theta']
    dtheta = model.x['dtheta']
    torque_0 = model.u['torque_0']
    torque_1 = model.u['torque_1']
    theta_set = model.tvp['theta_set']
    dtheta_set = model.tvp['dtheta_set']
    torque_set_0 = model.tvp['torque_set_0']
    torque_set_1 = model.tvp['torque_set_1']

    # 目的関数の定義
    lterm = (
        5000 * (theta[0] - theta_set[0])**2 +
        8000 * (theta[1] - theta_set[1])**2 +
        7000 * (dtheta[0] - dtheta_set[0])**2 +
        8500 * (dtheta[1] - dtheta_set[1])**2 +
        0.02 * (torque_0 - torque_set_0)**2 +
        0.01 * (torque_1 - torque_set_1)**2
    )

    mpc.set_objective(mterm=SX(0), lterm=lterm)

    # 定数ペナルティを設定
    mpc.set_rterm(
        torque_0=0.02,
        torque_1=0.01
    )

    # トルク制約の設定
    mpc.bounds['lower', '_u', 'torque_0'] = -20
    mpc.bounds['upper', '_u', 'torque_0'] = 20
    mpc.bounds['lower', '_u', 'torque_1'] = -40
    mpc.bounds['upper', '_u', 'torque_1'] = 40

    # 角度制約の設定
    mpc.bounds['lower', '_x', 'theta', 0] = -0.35
    mpc.bounds['upper', '_x', 'theta', 0] = 0.53
    mpc.bounds['lower', '_x', 'theta', 1] = -0.53
    mpc.bounds['upper', '_x', 'theta', 1] = 0.87

    # 時間変化パラメータテンプレートの取得
    tvp_template = mpc.get_tvp_template()

    # 時間変化パラメータ関数
    t_switch = 4  # seconds
    ind_switch = t_switch // mpc.settings.t_step

    def tvp_fun(t_ind):
        ind = t_ind // mpc.settings.t_step
        if ind <= ind_switch:
            tvp_template['_tvp', 0, 'theta_set'] = -0.8
            tvp_template['_tvp', 1, 'theta_set'] = 0.8
            tvp_template['_tvp', 0, 'dtheta_set'] = 0.0
            tvp_template['_tvp', 1, 'dtheta_set'] = 0.0
            tvp_template['_tvp', 0, 'torque_set_0'] = 5.0
            tvp_template['_tvp', 1, 'torque_set_1'] = -5.0
        else:
            tvp_template['_tvp', 0, 'theta_set'] = 0.0
            tvp_template['_tvp', 1, 'theta_set'] = 0.0
            tvp_template['_tvp', 0, 'dtheta_set'] = 0.0
            tvp_template['_tvp', 1, 'dtheta_set'] = 0.0
            tvp_template['_tvp', 0, 'torque_set_0'] = 0.0
            tvp_template['_tvp', 1, 'torque_set_1'] = 0.0
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    return mpc
