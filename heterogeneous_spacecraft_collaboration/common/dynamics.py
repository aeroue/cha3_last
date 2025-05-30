# common/dynamics.py
import numpy as np
from scipy.linalg import expm

class CWEquation:
    """
    Clohessy-Wiltshire (CW) 方程的二维实现。
    描述了一个航天器相对于目标（或参考点）的相对运动。
    状态向量 state = [x, y, vx, vy]^T
        x: 径向相对位置 (m)
        y: 沿迹向相对位置 (m)
        vx: 径向相对速度 (m/s)
        vy: 沿迹向相对速度 (m/s)
    控制输入 u = [ux, uy]^T
        ux: 径向加速度 (m/s^2)
        uy: 沿迹向加速度 (m/s^2)
    """
    def __init__(self, n: float = 0.0011, dt: float = 1.0):
        """
        初始化CW方程参数。

        参数:
            n (float): 目标轨道的平均角速度 (rad/s)。
                       默认值对应于典型的LEO轨道周期约90分钟。
            dt (float): 离散时间步长 (s)。
        """
        if n <= 0:
            raise ValueError("平均角速度 n 必须为正。")
        if dt <= 0:
            raise ValueError("时间步长 dt 必须为正。")

        self.n = n  # 平均角速度
        self.dt = dt  # 时间步长

        # 连续时间状态矩阵 A_c (4x4)
        self.A_c = np.array([
            [0,    0,     1, 0],
            [0,    0,     0, 1],
            [3*n*n,0,     0, 2*n],
            [0,    0,  -2*n, 0]
        ])

        # 连续时间输入矩阵 B_c (4x2)
        self.B_c = np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ])

        # 离散时间状态矩阵 A_d 和输入矩阵 B_d
        # 使用矩阵指数法进行离散化: Phi = [[A_c, B_c], [0, 0]] * dt
        # A_d = Phi_11, B_d = Phi_12
        M_rows = self.A_c.shape[0]
        N_cols = self.B_c.shape[1]

        # 构建增广矩阵 M = [[A_c, B_c], [0, I]] (这里用0代替I，因为我们只需要A_d, B_d部分)
        # 标准方法是 M = [[A_c, B_c], [0, 0]] * dt，然后 exp(M*dt)
        # Phi = expm([[A_c, B_c], [np.zeros((N_cols, M_rows)), np.zeros((N_cols, N_cols))]] * self.dt)
        # 另一种常见方法是构造块矩阵
        zero_block_NB_MA = np.zeros((N_cols, M_rows)) # B_c的行数是状态数，列数是输入数
        zero_block_NB_NB = np.zeros((N_cols, N_cols))
        
        phi_matrix_argument = np.block([
            [self.A_c, self.B_c],
            [zero_block_NB_MA, zero_block_NB_NB]
        ]) * self.dt

        phi_matrix = expm(phi_matrix_argument)

        self.A_d = phi_matrix[:M_rows, :M_rows]
        self.B_d = phi_matrix[:M_rows, M_rows:]
        
        self.state_size = M_rows
        self.control_size = N_cols

    def step(self, current_state: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        """
        根据当前状态和控制输入，计算下一个时间步的状态。

        参数:
            current_state (np.ndarray): 当前状态向量 [x, y, vx, vy]^T。
            control_input (np.ndarray): 控制输入向量 [ux, uy]^T。

        返回:
            np.ndarray: 下一个时间步的状态向量。
        """
        if not isinstance(current_state, np.ndarray):
            current_state = np.array(current_state)
        if not isinstance(control_input, np.ndarray):
            control_input = np.array(control_input)

        if current_state.shape != (self.state_size,):
            current_state = current_state.reshape(self.state_size,) # 尝试修正形状
            if current_state.shape != (self.state_size,):
                 raise ValueError(f"当前状态向量的形状应为 ({self.state_size},)，得到 {current_state.shape}")
        if control_input.shape != (self.control_size,):
            control_input = control_input.reshape(self.control_size,) # 尝试修正形状
            if control_input.shape != (self.control_size,):
                raise ValueError(f"控制输入向量的形状应为 ({self.control_size},)，得到 {control_input.shape}")

        next_state = self.A_d @ current_state + self.B_d @ control_input
        return next_state

    def get_discrete_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        返回离散时间的状态矩阵 A_d 和输入矩阵 B_d。

        返回:
            tuple[np.ndarray, np.ndarray]: (A_d, B_d)
        """
        return self.A_d, self.B_d

    def get_continuous_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        返回连续时间的状态矩阵 A_c 和输入矩阵 B_c。
        返回:
            tuple[np.ndarray, np.ndarray]: (A_c, B_c)
        """
        return self.A_c, self.B_c

# --- 模块自测试与可视化 ---
def test_dynamics():
    import matplotlib.pyplot as plt
    import matplotlib
    # 设置 matplotlib 支持中文显示
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei（黑体）
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    # 初始化动力学模型
    n_orbit = 2 * np.pi / (90 * 60)  # 90分钟轨道周期对应的平均角速度
    dt_sim = 10.0  # 仿真时间步长
    cw_model = CWEquation(n=n_orbit, dt=dt_sim)

    # 初始状态: [x, y, vx, vy]
    initial_state = np.array([1000.0, 1000.0, 1.0, -1.0]) # 1km, 1km, 1m/s, -1m/s
    
    # 仿真参数
    total_time = 3600  # 总仿真时长 1小时
    num_steps = int(total_time / dt_sim)

    # 存储历史状态
    states_history_no_control = [initial_state]
    states_history_with_control = [initial_state]

    current_state_no_control = initial_state.copy()
    current_state_with_control = initial_state.copy()

    # 零控制输入
    zero_control = np.array([0.0, 0.0])
    
    # 简单比例控制 (示例，非最优)
    # 目标是原点 [0,0,0,0]
    Kp_pos = 0.000005 
    Kp_vel = 0.0005

    for _ in range(num_steps):
        # 无控制
        current_state_no_control = cw_model.step(current_state_no_control, zero_control)
        states_history_no_control.append(current_state_no_control)

        # 有简单比例控制
        # 目标状态为原点，所以误差 = 0 - current_state
        control_input_p = -Kp_pos * current_state_with_control[:2] # 位置误差
        control_input_v = -Kp_vel * current_state_with_control[2:] # 速度误差
        # 限制控制量大小 (示例)
        control_input_p = np.clip(control_input_p, -0.01, 0.01)
        control_input_v = np.clip(control_input_v, -0.01, 0.01)
        
        # 简单叠加，这里简化控制逻辑，实际应由MPC计算
        # 这里仅用于演示动力学模型，控制输入是人为给定的
        # ux = -Kp_pos * x - Kp_vel * vx
        # uy = -Kp_pos * y - Kp_vel * vy (CW方程的控制是加速度)
        # 我们假设控制直接作用于加速度，因此控制输入是加速度
        ux = -Kp_pos * current_state_with_control[0] - Kp_vel * current_state_with_control[2]
        uy = -Kp_pos * current_state_with_control[1] - Kp_vel * current_state_with_control[3]
        
        # 限制控制加速度大小
        control_force = np.clip(np.array([ux, uy]), -0.001, 0.001) # m/s^2

        current_state_with_control = cw_model.step(current_state_with_control, control_force)
        states_history_with_control.append(current_state_with_control)


    states_history_no_control = np.array(states_history_no_control)
    states_history_with_control = np.array(states_history_with_control)

    # 可视化
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("CW方程动力学仿真 (二维)", fontsize=16)

    # 子图1: 无控制下的轨迹
    axs[0, 0].plot(states_history_no_control[:, 1], states_history_no_control[:, 0], label="无控制轨迹")
    axs[0, 0].scatter(states_history_no_control[0, 1], states_history_no_control[0, 0], color='red', marker='o', label="初始位置")
    axs[0, 0].scatter(0, 0, color='green', marker='x', label="参考点 (0,0)")
    axs[0, 0].set_xlabel("沿迹向相对位置 Y (m)")
    axs[0, 0].set_ylabel("径向相对位置 X (m)")
    axs[0, 0].set_title("无控制下的相对轨迹")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].axis('equal')

    # 子图2: 有简单比例控制下的轨迹
    axs[0, 1].plot(states_history_with_control[:, 1], states_history_with_control[:, 0], label="简单比例控制轨迹")
    axs[0, 1].scatter(states_history_with_control[0, 1], states_history_with_control[0, 0], color='red', marker='o', label="初始位置")
    axs[0, 1].scatter(0, 0, color='green', marker='x', label="参考点 (0,0)")
    axs[0, 1].set_xlabel("沿迹向相对位置 Y (m)")
    axs[0, 1].set_ylabel("径向相对位置 X (m)")
    axs[0, 1].set_title("简单比例控制下的相对轨迹")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].axis('equal')

    # 子图3: 速度分量 (无控制 vs 有控制)
    time_axis = np.arange(num_steps + 1) * dt_sim
    axs[1, 0].plot(time_axis, states_history_no_control[:, 2], label="Vx (无控)")
    axs[1, 0].plot(time_axis, states_history_no_control[:, 3], label="Vy (无控)")
    axs[1, 0].plot(time_axis, states_history_with_control[:, 2], '--', label="Vx (有控)")
    axs[1, 0].plot(time_axis, states_history_with_control[:, 3], '--', label="Vy (有控)")
    axs[1, 0].set_xlabel("时间 (s)")
    axs[1, 0].set_ylabel("相对速度 (m/s)")
    axs[1, 0].set_title("相对速度变化")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # 子图4: 位置分量 (无控制 vs 有控制)
    axs[1, 1].plot(time_axis, states_history_no_control[:, 0], label="X (无控)")
    axs[1, 1].plot(time_axis, states_history_no_control[:, 1], label="Y (无控)")
    axs[1, 1].plot(time_axis, states_history_with_control[:, 0], '--', label="X (有控)")
    axs[1, 1].plot(time_axis, states_history_with_control[:, 1], '--', label="Y (有控)")
    axs[1, 1].set_xlabel("时间 (s)")
    axs[1, 1].set_ylabel("相对位置 (m)")
    axs[1, 1].set_title("相对位置变化")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("A_d (离散状态矩阵):\n", cw_model.A_d)
    print("B_d (离散输入矩阵):\n", cw_model.B_d)

if __name__ == '__main__':
    test_dynamics()