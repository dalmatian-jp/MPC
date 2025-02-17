import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
# Plot settings
rcParams['text.usetex'] = False
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'xx-large'
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'

import time

from template_mpc import template_mpc
from template_simulator import template_simulator
from template_model import template_model

""" User settings: """
show_animation = True
store_animation = False
store_results = False

"""
Get configured do-mpc modules:
"""

model = template_model()
simulator = template_simulator(model)
mpc = template_mpc(model)
estimator = do_mpc.estimator.StateFeedback(model)

"""
Set initial state
"""

simulator.x0['theta'] = [0.2618, 0]  # 振り子1と振り子2の初期角度

x0 = simulator.x0.cat.full()

mpc.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()
z0 = simulator.init_algebraic_variables()

"""
Setup graphic:
"""

# Function to create lines:
L1 = 0.5  # m, length of the first rod
L2 = 0.5  # m, length of the second rod
def pendulum_bars(x):
    x = x.flatten()
    # Get the x, y coordinates of the two bars for the given state x.
    line_1_x = np.array([
        0,  # 振り子1の基点を原点と仮定
        L1 * np.sin(x[0])
    ])

    line_1_y = np.array([
        0,
        L1 * np.cos(x[0])
    ])

    line_2_x = np.array([
        line_1_x[1],
        line_1_x[1] + L2 * np.sin(x[1])
    ])

    line_2_y = np.array([
        line_1_y[1],
        line_1_y[1] + L2 * np.cos(x[1])
    ])

    line_1 = np.stack((line_1_x, line_1_y))
    line_2 = np.stack((line_2_x, line_2_y))

    return line_1, line_2

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)

fig = plt.figure(figsize=(16,9))
plt.ion()

ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=4)
ax2 = plt.subplot2grid((4, 2), (0, 1))
ax3 = plt.subplot2grid((4, 2), (1, 1))
ax5 = plt.subplot2grid((4, 2), (2, 1))

ax2.set_ylabel('$E_{kin}$ [J]')
ax3.set_ylabel('$E_{pot}$ [J]')
ax5.set_ylabel('Input torque [Nm]')

mpc_graphics.add_line(var_type='_aux', var_name='E_kin', axis=ax2)
mpc_graphics.add_line(var_type='_aux', var_name='E_pot', axis=ax3)
mpc_graphics.add_line(var_type='_u', var_name='torque', axis=ax5)

ax1.axhline(0, color='black')

# Axis on the right.
for ax in [ax2, ax3, ax5]:
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    if ax != ax5:
        ax.xaxis.set_ticklabels([])

ax5.set_xlabel('time [s]')

bar1 = ax1.plot([], [], '-o', linewidth=5, markersize=10)
bar2 = ax1.plot([], [], '-o', linewidth=5, markersize=10)

ax1.set_xlim(-1.8, 1.8)
ax1.set_ylim(-1.2, 1.2)
ax1.set_axis_off()

fig.align_ylabels()
fig.tight_layout()

"""
Run MPC main loop:
"""
time_list = []

n_steps = 120
for k in range(n_steps):
    tic = time.time()
    u0 = mpc.make_step(x0)
    toc = time.time()
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    time_list.append(toc-tic)

    if show_animation:
        line1, line2 = pendulum_bars(x0)
        bar1[0].set_data(line1[0], line1[1])
        bar2[0].set_data(line2[0], line2[1])
        mpc_graphics.plot_results()
        mpc_graphics.plot_predictions()
        mpc_graphics.reset_axes()
        plt.show()
        plt.pause(0.04)

time_arr = np.array(time_list)
mean = np.round(np.mean(time_arr[1:])*1000)
var = np.round(np.std(time_arr[1:])*1000)
print('mean runtime:{}ms +- {}ms for MPC step'.format(mean, var))

# The function describing the gif:
if store_animation:
    x_arr = mpc.data['_x']
    def update(t_ind):
        line1, line2 = pendulum_bars(x_arr[t_ind])
        bar1[0].set_data(line1[0], line1[1])
        bar2[0].set_data(line2[0], line2[1])
        mpc_graphics.plot_results(t_ind)
        mpc_graphics.plot_predictions(t_ind)
        mpc_graphics.reset_axes()

    anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)
    gif_writer = ImageMagickWriter(fps=20)
    anim.save('anim_dip.gif', writer=gif_writer)

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'dip_mpc')

input('Press any key to exit.')
