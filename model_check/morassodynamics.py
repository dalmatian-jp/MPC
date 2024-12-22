import numpy as np


class dynamics:
    def model_dynamics(self, x, u):
        # 非線形な動的モデルを用いた更新
            L1 = 0.90
            L2 = 0.88
            M1=28.047
            M2=56.093
            I1=9.305
            I2=5.681
            g = 9.8
            r1 = 0.576
            r2 = 0.318
            alpha1 = I1 + I2 + M1 * r1**2 + M2 * (L1**2 + r2**2)
            alpha2 = M2 * L1 * r2
            alpha4 = I2 + M2 * r2**2

            theta1, theta2,theta1_dot, theta2_dot = x
            
            sin_theta1 = np.sin(theta1)
            sin_theta2 = np.sin(theta2)
            sin_theta12 = np.sin(theta1 + theta2)
            cos_theta2 = np.cos(theta2)
            
            det_M = alpha1 *alpha4 - alpha4**2 - alpha2**2 * cos_theta2**2
            M_inv = np.array([
                [ alpha4 / det_M, -(alpha4 + alpha2*cos_theta2) / det_M],
                [-(alpha4 + alpha2*cos_theta2) / det_M, (alpha1 + 2* alpha2* cos_theta2) / det_M]
            ])
            
            f_1 = (
                u[0] + alpha2 * sin_theta2 * theta2_dot * theta1_dot +
                alpha2 * sin_theta2 * (theta2_dot + theta1_dot) * theta2_dot +
                g * (M1 * r1 + M2 * L1) * sin_theta1 +
                g * M2 * r2 * sin_theta12 
            )

            f_2 = (
                u[1] - alpha2 * sin_theta2 * theta1_dot * theta2_dot +
                g * M2 * r2 * sin_theta12 
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