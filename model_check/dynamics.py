import numpy as np


class dynamics:
    def model_dynamics(self, x, u):
        # 非線形な動的モデルを用いた更新
            L1 = 0.78
            L2 = 0.73
            M1=11.41
            M2=50.14
            I1=0.35
            I2=0.25
            g = 9.8
            alpha1 = 1/4 * L1**2 * M1 + I1
            alpha2 = 1/4 * L2**2 * M2 + I2
            alpha3 = 1/2 * L1 * L2 * M2 * np.cos(x[1])
            alpha4 = L1**2 * M2

            theta1, theta2,theta1_dot, theta2_dot = x
            
            sin_theta1 = np.sin(theta1)
            sin_theta2 = np.sin(theta2)
            sin_theta12 = np.sin(theta1 + theta2)


            det_M = alpha2 * (alpha1 + alpha4) - alpha3**2
            M_inv = np.array([
                [ alpha2 / det_M, -(alpha2 + alpha3) / det_M],
                [-(alpha2 + alpha3) / det_M, (alpha1 + alpha2 + 2 * alpha3 + alpha4) / det_M]
            ])
            
            f_1 = (
                u[0] + 1/2 * L1 * L2 * M2 * sin_theta2 * theta2_dot ** 2 +
                L1 * L2 * M2 * sin_theta2 * theta1_dot * theta2_dot +
                1/2 * g * L2 * M2 * sin_theta12 +
                1/2 * g * L1 * M1 * sin_theta1 -
                g * L1 * M2 * sin_theta1
            )

            f_2 = (
                u[1] - 1/2 * L1 *theta1_dot ** 2 * sin_theta2 +
                1/2 * L2 * M2 * g * sin_theta12
            )

            f = np.array([
                f_1,
                f_2
            ])
            ddot_theta = M_inv @ f

            dxdt = np.zeros_like(x)
            dxdt[0] = theta1_dot
            dxdt[1] = theta2_dot
            dxdt[2:] = ddot_theta  # theta1_ddot, theta2_ddot を一度に更新

            return dxdt
    
    def runge_kutta_step(self, x, dt, u=np.zeros(2)):
        func = self.model_dynamics
        k1 = func(x, u)
        k2 = func(x + dt*0.5*k1, u)
        k3 = func(x + dt*0.5*k2, u)
        k4 = func(x + dt*k3, u)
        return x + dt*(k1+2*k2+2*k3+k4)/6
    
    def next_state(self, x, u, dt):
        return self.runge_kutta_step(x, dt, u)
    
    def observation_function(self, x):
        return x
    
    def observation_jacobian(self, x):
        return np.eye(len(x))
    
    def numerical_jacobian(self, x, u, h=1e-5):
        f = self.next_state
        n = len(x)
        J = np.zeros((4, n))
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = h
            J[:, i] = (f(x + dx,u,dx) - f(x - dx,u,dx)) / (2 * h)
        return J