# common/delta_v_estimator.py
import numpy as np
import casadi as ca
import sys
import os
import matplotlib.pyplot as plt
import matplotlib

# --- 临时路径处理 ---
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
_project_root_directory = os.path.dirname(_current_file_directory)
if _project_root_directory not in sys.path:
    sys.path.insert(0, _project_root_directory)
# --- 结束 ---

from common.dynamics import CWEquation
# 注意：DeltaVEstimatorMPC 本质上是一个简化的MPC，专注于最小化控制并到达终端。
# 我们可以选择复用 MPCController 或像您现有代码一样独立构建 CasADi 问题。
# 为保持与您现有结构的相似性，我们独立构建。

class DeltaVEstimatorMPC:
    """
    使用简化的模型预测控制来估算到达目标状态所需的 Delta-V。
    目标是最小化总控制输入，同时确保在预测时域末端到达目标状态。
    """
    def __init__(self,
                 dynamics_model: CWEquation,
                 prediction_horizon: int,
                 R_control_effort: np.ndarray, # 权重，用于惩罚控制量，间接最小化Delta-V
                 Q_terminal_state: np.ndarray, # 权重，用于确保到达终端目标
                 control_constraints: dict = None):
        """
        初始化 Delta-V 估算器。

        参数:
            dynamics_model (CWEquation): 航天器动力学模型。
            prediction_horizon (int): MPC的预测时域 N。
            R_control_effort (np.ndarray): 控制输入代价的权重矩阵 R。形状 (control_size, control_size)。
                                          较小的 R 值意味着更积极地使用控制以快速到达。
                                          较大的 R 值则更节省燃料。
            Q_terminal_state (np.ndarray): 终端状态偏差的权重矩阵 Qf。形状 (state_size, state_size)。
                                           应足够大以确保目标被满足。
            control_constraints (dict, optional): 控制输入的边界约束。
                                                  格式: {'min': [u_min_x, u_min_y], 'max': [u_max_x, u_max_y]}
        """
        self.dynamics_model = dynamics_model
        self.N = prediction_horizon
        self.dt = dynamics_model.dt

        self.state_size = dynamics_model.state_size
        self.control_size = dynamics_model.control_size

        if not isinstance(R_control_effort, np.ndarray) or R_control_effort.shape != (self.control_size, self.control_size):
            raise ValueError(f"R_control_effort 矩阵形状应为 ({self.control_size}, {self.control_size})")
        if not isinstance(Q_terminal_state, np.ndarray) or Q_terminal_state.shape != (self.state_size, self.state_size):
            raise ValueError(f"Q_terminal_state 矩阵形状应为 ({self.state_size}, {self.state_size})")

        self.R = R_control_effort
        self.Qf = Q_terminal_state
        self.control_constraints = control_constraints

        # CasADi 优化问题将在每次调用 estimate_delta_v 时构建和求解
        # 这样做是为了处理可能变化的权重或约束（尽管当前实现中它们是固定的）
        # 或者可以像MPCController那样预构建求解器框架

    def estimate_delta_v(self, current_state: np.ndarray, target_state: np.ndarray) -> tuple[float, np.ndarray, bool]:
        """
        估算从当前状态到达目标状态所需的 Delta-V。

        参数:
            current_state (np.ndarray): 当前航天器状态 [x, y, vx, vy]^T。
            target_state (np.ndarray): 期望的终端目标状态 [x_N, y_N, vx_N, vy_N]^T。
                                       (也可以是部分状态，例如只关心位置，速度自由)

        返回:
            tuple[float, np.ndarray, bool]:
                - estimated_delta_v (float): 估算的 Delta-V 值。如果求解失败则为 np.inf。
                - optimal_trajectory (np.ndarray): 计算得到的最优轨迹。如果求解失败则为 None。
                - success (bool): 求解是否成功。
        """
        if not isinstance(current_state, np.ndarray): current_state = np.array(current_state)
        if not isinstance(target_state, np.ndarray): target_state = np.array(target_state)

        # 确保形状正确
        current_state = current_state.reshape(self.state_size,)
        target_state = target_state.reshape(self.state_size,)

        opti = ca.Opti()

        # 优化变量
        X_opt = opti.variable(self.state_size, self.N + 1)  # 状态轨迹
        U_opt = opti.variable(self.control_size, self.N)    # 控制序列

        # 目标函数
        obj = 0
        for k in range(self.N):
            # 主要目标：最小化控制能量/力度 (近似于最小化 Delta-V)
            obj += ca.mtimes([U_opt[:, k].T, self.R, U_opt[:, k]])
            # 这里不添加中间状态的惩罚，除非为了避障等特定需求

        # 确保到达终端目标
        terminal_error = X_opt[:, self.N] - target_state
        obj += ca.mtimes([terminal_error.T, self.Qf, terminal_error])
        
        opti.minimize(obj)

        # 动力学约束
        A_d, B_d = self.dynamics_model.get_discrete_matrices()
        for k in range(self.N):
            opti.subject_to(X_opt[:, k+1] == ca.mtimes(A_d, X_opt[:, k]) + ca.mtimes(B_d, U_opt[:, k]))

        # 初始状态约束
        opti.subject_to(X_opt[:, 0] == current_state)

        # 控制约束
        if self.control_constraints:
            u_min = self.control_constraints['min']
            u_max = self.control_constraints['max']
            for k in range(self.N):
                for j in range(self.control_size):
                    opti.subject_to(opti.bounded(u_min[j], U_opt[j, k], u_max[j]))
        
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'} # 静默模式
        opti.solver('ipopt', opts)

        try:
            solution = opti.solve()
            optimal_control_sequence = solution.value(U_opt) # (control_size, N)
            optimal_trajectory = solution.value(X_opt)     # (state_size, N+1)
            
            # 计算 Delta-V: sum_k (sum_j |u_jk|) * dt
            # optimal_control_sequence 的形状是 (control_size, N)
            # np.sum(np.abs(optimal_control_sequence), axis=0) -> sum over controls for each time step (shape N,)
            # np.sum(np.sum(np.abs(optimal_control_sequence), axis=0)) -> sum over time steps (scalar)
            # 或者更直接：
            estimated_delta_v = np.sum(np.sum(np.abs(optimal_control_sequence), axis=0)) * self.dt # 与您现有代码一致
            # 或者 L2 范数形式： estimated_delta_v = np.sum(np.linalg.norm(optimal_control_sequence, axis=0)) * self.dt
            
            return float(estimated_delta_v), optimal_trajectory, True
        except RuntimeError:
            # print(f"Delta-V Estimator求解失败")
            return np.inf, np.tile(current_state.reshape(-1,1), (1, self.N+1)), False

# --- 模块自测试与可视化 ---
def test_delta_v_estimator():
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 动力学模型参数
    n_orbit = 2 * np.pi / (90 * 60)
    dt_sim = 20.0 # Delta-V估算的每一步时长
    dynamics = CWEquation(n=n_orbit, dt=dt_sim)

    # Delta-V 估算器参数
    horizon_dv = 15 # 估算Delta-V的预测时域
    
    # 权重矩阵
    # R_dv 应该相对较小，以便让控制器“愿意”使用控制力来到达目标
    # 但如果太小，可能会导致数值问题或不切实际的控制。
    # 如果目标是严格最小化 sum |u|*dt, R_dv 可以设为单位阵或非常小的对角阵，
    # 主要靠约束和Qf来驱动。
    # 您的原始代码使用 R_delta_v = np.diag([0.01, 0.01, 0.01]) for 3D
    R_dv = np.diag([0.01, 0.01]) # 二维，惩罚控制力度
    # Qf_dv 应该较大，以强迫到达终端目标
    # 您的原始代码使用 Qf_delta_v = np.diag([100, 100, 100, 1, 1, 1]) for 3D
    Qf_dv = np.diag([100.0, 100.0, 10.0, 10.0]) # 位置权重远大于速度权重

    # 控制约束 (加速度 m/s^2)
    u_abs_max_dv = 0.05 # Delta-V估算时允许的最大加速度
    control_con_dv = {'min': [-u_abs_max_dv, -u_abs_max_dv], 'max': [u_abs_max_dv, u_abs_max_dv]}

    estimator = DeltaVEstimatorMPC(dynamics_model=dynamics,
                                   prediction_horizon=horizon_dv,
                                   R_control_effort=R_dv,
                                   Q_terminal_state=Qf_dv,
                                   control_constraints=control_con_dv)

    # 测试场景
    initial_state_dv = np.array([1000.0, -500.0, 0.0, 0.0]) # 初始位置，静止相对速度
    # 目标状态：到达原点，并具有一定相对速度（例如，对接前准备）
    # target_state_dv = np.array([0.0, 0.0, 0.0, -0.1]) # 目标位置(0,0), y方向速度-0.1m/s
    # 或者简单地只关心位置，速度自由（通过Qf中速度权重为0实现，但这里不为0）
    target_state_dv = np.array([0.0, 0.0, 0.0, 0.0]) # 到达原点并静止

    print(f"从状态: {initial_state_dv}")
    print(f"到目标状态: {target_state_dv}")
    print(f"在 {horizon_dv} 步内，每步 {dt_sim} 秒")

    estimated_dv, optimal_traj_dv, success_dv = estimator.estimate_delta_v(initial_state_dv, target_state_dv)

    if success_dv:
        print(f"估算成功！")
        print(f"  估算的 Delta-V: {estimated_dv:.4f} m/s")
        print(f"  机动总时长: {horizon_dv * dt_sim:.0f} s")
        print(f"  实际终端状态: {optimal_traj_dv[:, -1]}")

        # 可视化
        fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=False)
        fig.suptitle(f"Delta-V 估算轨迹 (估算 $\Delta V=${estimated_dv:.2f} m/s)", fontsize=16)

        # 子图1: 估算轨迹 X-Y
        axs[0].plot(optimal_traj_dv[1, :], optimal_traj_dv[0, :], 'g-o', label="估算轨迹", markersize=4)
        axs[0].scatter(initial_state_dv[1], initial_state_dv[0], color='blue', marker='s', s=100, label="初始位置")
        axs[0].scatter(target_state_dv[1], target_state_dv[0], color='red', marker='x', s=100, label="目标位置")
        axs[0].set_xlabel("沿迹向相对位置 Y (m)")
        axs[0].set_ylabel("径向相对位置 X (m)")
        axs[0].set_title("估算的机动轨迹")
        axs[0].legend()
        axs[0].grid(True)
        axs[0].axis('equal')

        # 子图2: 状态分量随时间变化 (在估算的轨迹上)
        time_axis_dv = np.arange(horizon_dv + 1) * dt_sim
        axs[1].plot(time_axis_dv, optimal_traj_dv[0, :], label="X")
        axs[1].plot(time_axis_dv, optimal_traj_dv[1, :], label="Y")
        axs[1].plot(time_axis_dv, optimal_traj_dv[2, :], label="Vx")
        axs[1].plot(time_axis_dv, optimal_traj_dv[3, :], label="Vy")
        axs[1].set_xlabel("时间 (s)")
        axs[1].set_ylabel("状态值")
        axs[1].set_title("估算轨迹的状态演化")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        print("Delta-V 估算失败。")

if __name__ == '__main__':
    test_delta_v_estimator()