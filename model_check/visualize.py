import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

class MinimalVisualization:
    def __init__(self, states, estimated_states, time, L1, L2):
        self.states = states
        self.estimated_states = estimated_states
        self.time = time
        self.L1 = L1
        self.L2 = L2

    def animate(self, save_path=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect("equal")
        ax.grid()

        (line,) = ax.plot([], [], "o-", lw=2, c="cornflowerblue", label="Actual")
        (line_est,) = ax.plot([], [], "x--", lw=2, c="orange", label="Estimated")
        time_template = "time = %.1fs"
        time_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=12)

        ax.legend()

        def init():
            line.set_data([], [])
            line_est.set_data([], [])
            time_text.set_text("")
            return line, line_est, time_text

        def update(i):
            # Actual positions
            x1 = self.L1 * np.sin(self.states[i, 0])
            y1 = self.L1 * np.cos(self.states[i, 0])
            x2 = x1 + self.L2 * np.sin(self.states[i, 0] + self.states[i, 1])
            y2 = y1 + self.L2 * np.cos(self.states[i, 0] + self.states[i, 1])

            # Estimated positions
            x1_est = self.L1 * np.sin(self.estimated_states[i, 0])
            y1_est = self.L1 * np.cos(self.estimated_states[i, 0])
            x2_est = x1_est + self.L2 * np.sin(self.estimated_states[i, 0] + self.estimated_states[i, 1])
            y2_est = y1_est + self.L2 * np.cos(self.estimated_states[i, 0] + self.estimated_states[i, 1])

            line.set_data([0, x1, x2], [0, y1, y2])
            line_est.set_data([0, x1_est, x2_est], [0, y1_est, y2_est])
            time_text.set_text(time_template % self.time[i])
            return line, line_est, time_text

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.time),
            interval=(self.time[1] - self.time[0]) * 1000,
            init_func=init,
            blit=True,
        )

        if save_path:
            # 保存処理
            ani.save(save_path, writer="ffmpeg", fps=30)
            print(f"Animation saved to {save_path}")

        plt.show()

# 使用例
# states, estimated_states, time, L1, L2 は適切なデータを渡してください。
# vis = MinimalVisualization(states, estimated_states, time, L1, L2)
# vis.animate()
