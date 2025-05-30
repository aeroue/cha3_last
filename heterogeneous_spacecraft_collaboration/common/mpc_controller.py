# # # # common/mpc_controller.py
# # # import casadi as ca
# # # import numpy as np
# # # import sys
# # # import os
# # # import matplotlib.pyplot as plt # 移到这里，因为测试函数会用到
# # # import matplotlib # 移到这里

# # # # --- 临时路径处理，确保直接运行时能找到 dynamics ---
# # # # 获取当前文件 (mpc_controller.py) 的目录 (common/)
# # # _current_file_directory = os.path.dirname(os.path.abspath(__file__))
# # # # 获取 common/ 目录的父目录 (heterogeneous_spacecraft_collaboration/)
# # # _project_root_directory = os.path.dirname(_current_file_directory)

# # # # 将父目录（项目根目录）添加到 sys.path
# # # if _project_root_directory not in sys.path:
# # #     sys.path.insert(0, _project_root_directory)
# # # # --- 结束临时路径处理 ---

# # # from common.dynamics import CWEquation # 现在可以这样导入，因为 common 的父目录在 sys.path 中

# # # class MPCController:
# # #     """
# # #     模型预测控制器 (MPC) 用于航天器的轨迹控制。
# # #     该MPC旨在将航天器驱动到一个期望的终端目标状态，同时最小化控制能量。
# # #     """
# # #     def __init__(self,
# # #                  dynamics_model: CWEquation,
# # #                  prediction_horizon: int,
# # #                  Q_terminal: np.ndarray, # 终端状态权重
# # #                  R_control: np.ndarray,  # 控制输入权重
# # #                  Q_state: np.ndarray = None, # 中间状态权重 (可选)
# # #                  control_constraints: dict = None, # 控制约束 e.g., {'min': [-u_max, -u_max], 'max': [u_max, u_max]}
# # #                  state_constraints: dict = None):  # 状态约束 (可选，通常用于安全边界)
# # #         """
# # #         初始化MPC控制器。

# # #         参数:
# # #             dynamics_model (CWEquation): 航天器动力学模型。
# # #             prediction_horizon (int): MPC的预测时域 N。
# # #             Q_terminal (np.ndarray): 终端状态偏差的权重矩阵 (Qf)。形状 (state_size, state_size)。
# # #             R_control (np.ndarray): 控制输入的权重矩阵 (R)。形状 (control_size, control_size)。
# # #             Q_state (np.ndarray, optional): 中间状态偏差的权重矩阵 (Q)。形状 (state_size, state_size)。
# # #                                             如果为None，则不惩罚中间状态与特定参考的偏差。
# # #             control_constraints (dict, optional): 控制输入的边界约束。
# # #                                                   格式: {'min': [u_min_x, u_min_y], 'max': [u_max_x, u_max_y]}
# # #             state_constraints (dict, optional): 状态变量的边界约束。
# # #                                                 格式: {'min': [s_min_0,...], 'max': [s_max_0,...]}
# # #         """
# # #         self.dynamics_model = dynamics_model
# # #         self.N = prediction_horizon
# # #         self.dt = dynamics_model.dt # MPC的控制步长与动力学模型离散步长一致

# # #         self.state_size = dynamics_model.state_size
# # #         self.control_size = dynamics_model.control_size

# # #         if not isinstance(Q_terminal, np.ndarray) or Q_terminal.shape != (self.state_size, self.state_size):
# # #             raise ValueError(f"Q_terminal 矩阵形状应为 ({self.state_size}, {self.state_size})")
# # #         if not isinstance(R_control, np.ndarray) or R_control.shape != (self.control_size, self.control_size):
# # #             raise ValueError(f"R_control 矩阵形状应为 ({self.control_size}, {self.control_size})")
# # #         if Q_state is not None and (not isinstance(Q_state, np.ndarray) or Q_state.shape != (self.state_size, self.state_size)):
# # #             raise ValueError(f"Q_state 矩阵形状应为 ({self.state_size}, {self.state_size})")

# # #         self.Qf = Q_terminal
# # #         self.R = R_control
# # #         self.Q = Q_state if Q_state is not None else np.zeros((self.state_size, self.state_size))

# # #         self.control_constraints = control_constraints
# # #         self.state_constraints = state_constraints

# # #         self.opti = None
# # #         self.X_opt = None # 优化变量：状态序列
# # #         self.U_opt = None # 优化变量：控制序列
# # #         self._build_solver()

# # #     def _build_solver(self):
# # #         """
# # #         构建CasADi优化问题。
# # #         """
# # #         self.opti = ca.Opti()

# # #         # 优化变量
# # #         self.X_opt = self.opti.variable(self.state_size, self.N + 1)  # 状态轨迹 (x0, x1, ..., xN)
# # #         self.U_opt = self.opti.variable(self.control_size, self.N)    # 控制序列 (u0, u1, ..., uN-1)

# # #         # 参数 (在求解时设置)
# # #         self.x0_param = self.opti.parameter(self.state_size, 1)       # 初始状态
# # #         self.xtarget_N_param = self.opti.parameter(self.state_size, 1) # 终端目标状态

# # #         # 目标函数
# # #         obj = 0
# # #         for k in range(self.N):
# # #             # 最小化控制输入
# # #             obj += ca.mtimes([self.U_opt[:, k].T, self.R, self.U_opt[:, k]])
# # #             # (可选) 最小化中间状态与某个参考（如果提供）的偏差
# # #             # 如果Q非零，这里可以定义一个中间参考状态，或者惩罚与0的偏差（如果希望状态保持小）
# # #             # 为简单起见，如果Q非零且无特定中间参考，我们这里不添加中间状态惩罚，
# # #             # 除非用户明确要求跟踪一条轨迹。当前要求是只关心终端状态。
# # #             # 如果需要对中间状态进行非常宽松的约束或小的惩罚以平滑轨迹，
# # #             # 可以引入一个非常稀疏的或对角线元素很小的Q矩阵，并惩罚 X_opt[:, k] 本身的大小
# # #             # 或者与一个非常粗略的路径点的偏差。
# # #             # 例如，如果Q非零，可以添加： obj += ca.mtimes([self.X_opt[:, k].T, self.Q, self.X_opt[:, k]])
# # #             # 但这会试图将中间状态拉向0，可能不是期望的。
# # #             # 我们这里只严格按照用户的两个主要代价项。
# # #             # 如果提供了Q，并且希望对中间状态进行约束，则需要一个中间的x_ref_intermediate_param
# # #             # 如果只是为了平滑，可以 obj += ca.mtimes([self.X_opt[:, k].T, self.Q, self.X_opt[:, k]])
# # #             if not np.all(self.Q == 0): # 如果Q不是零矩阵
# # #                  # 这里可以是对状态本身的惩罚，或者是一个变化的中间目标（需要参数化）
# # #                  # 按照最简化的要求，这里可以先不加，或者只惩罚状态大小
# # #                  # obj += ca.mtimes([self.X_opt[:, k].T, self.Q, self.X_opt[:, k]])
# # #                  pass # 暂时不添加中间状态对固定参考的惩罚

# # #         # 最小化终端状态与期望终端状态的偏差
# # #         terminal_error = self.X_opt[:, self.N] - self.xtarget_N_param
# # #         obj += ca.mtimes([terminal_error.T, self.Qf, terminal_error])
        
# # #         self.opti.minimize(obj)

# # #         # 动力学约束
# # #         A_d, B_d = self.dynamics_model.get_discrete_matrices()
# # #         for k in range(self.N):
# # #             self.opti.subject_to(self.X_opt[:, k+1] == ca.mtimes(A_d, self.X_opt[:, k]) + ca.mtimes(B_d, self.U_opt[:, k]))

# # #         # 初始状态约束
# # #         self.opti.subject_to(self.X_opt[:, 0] == self.x0_param)

# # #         # 控制约束
# # #         if self.control_constraints:
# # #             u_min = self.control_constraints['min']
# # #             u_max = self.control_constraints['max']
# # #             for k in range(self.N):
# # #                 for j in range(self.control_size):
# # #                     self.opti.subject_to(self.opti.bounded(u_min[j], self.U_opt[j, k], u_max[j]))
        
# # #         # 状态约束 (可选)
# # #         if self.state_constraints:
# # #             s_min = self.state_constraints['min']
# # #             s_max = self.state_constraints['max']
# # #             for k in range(self.N + 1): # 包括终端状态
# # #                 for j in range(self.state_size):
# # #                     self.opti.subject_to(self.opti.bounded(s_min[j], self.X_opt[j, k], s_max[j]))
        
# # #         # 设置求解器选项 (使用ipopt)
# # #         # opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'} # 静默模式
# # #         opts = {'ipopt.print_level': 3, 'print_time': 1} # 调试模式
# # #         self.opti.solver('ipopt', opts)

# # #     def solve(self, current_state: np.ndarray, target_terminal_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
# # #         """
# # #         求解MPC优化问题。

# # #         参数:
# # #             current_state (np.ndarray): 当前航天器状态 [x, y, vx, vy]^T。
# # #             target_terminal_state (np.ndarray): 期望的终端状态 [x_N, y_N, vx_N, vy_N]^T。

# # #         返回:
# # #             tuple[np.ndarray, np.ndarray, bool]:
# # #                 - optimal_control (np.ndarray): 计算得到的第一个最优控制输入 [ux, uy]^T。如果求解失败则为None。
# # #                 - predicted_trajectory (np.ndarray): 预测的状态轨迹。如果求解失败则为None。
# # #                 - success (bool): 求解是否成功。
# # #         """
# # #         if not isinstance(current_state, np.ndarray):
# # #             current_state = np.array(current_state)
# # #         if not isinstance(target_terminal_state, np.ndarray):
# # #             target_terminal_state = np.array(target_terminal_state)

# # #         if current_state.shape != (self.state_size,):
# # #              current_state = current_state.reshape(self.state_size,)
# # #              if current_state.shape != (self.state_size,):
# # #                  raise ValueError(f"当前状态向量的形状应为 ({self.state_size},)，得到 {current_state.shape}")
# # #         if target_terminal_state.shape != (self.state_size,):
# # #             target_terminal_state = target_terminal_state.reshape(self.state_size,)
# # #             if target_terminal_state.shape != (self.state_size,):
# # #                 raise ValueError(f"目标终端状态向量的形状应为 ({self.state_size},)，得到 {target_terminal_state.shape}")

# # #         # 设置参数值
# # #         self.opti.set_value(self.x0_param, current_state)
# # #         self.opti.set_value(self.xtarget_N_param, target_terminal_state)

# # #         try:
# # #             solution = self.opti.solve()
# # #             optimal_control_sequence = solution.value(self.U_opt)
# # #             predicted_trajectory = solution.value(self.X_opt)
            
# # #             # 返回第一个控制输入和整个预测轨迹
# # #             return optimal_control_sequence[:, 0], predicted_trajectory, True
# # #         except RuntimeError as e:
# # #             print(f"MPC求解失败: {e}")
# # #             # 可以尝试返回一个安全的默认控制，或者None
# # #             # print("尝试使用上次成功的解或默认值（如果实现）")
# # #             # For now, return None
# # #             return np.zeros(self.control_size), np.tile(current_state.reshape(-1,1), (1, self.N+1)), False

# # # # --- 模块自测试与可视化 ---
# # # def test_mpc_controller():
# # #     # 设置 matplotlib 支持中文显示
# # #     matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# # #     matplotlib.rcParams['axes.unicode_minus'] = False

# # #     # 动力学模型参数
# # #     n_orbit = 2 * np.pi / (90 * 60)  # 90分钟轨道周期
# # #     dt_sim = 20.0  # 仿真时间步长，MPC的控制步长也与之对应
# # #     dynamics = CWEquation(n=n_orbit, dt=dt_sim)

# # #     # MPC参数
# # #     horizon = 20  # 预测时域
# # #     state_size = dynamics.state_size
# # #     control_size = dynamics.control_size

# # #     # 权重矩阵
# # #     Q_f = np.diag([10.0, 10.0, 1.0, 1.0]) # 终端位置权重 > 终端速度权重
# # #     R = np.diag([0.1, 0.1])             # 控制输入权重
# # #     # Q_s = np.diag([0.01, 0.01, 0.01, 0.01]) # (可选) 中间状态惩罚，暂时不用

# # #     # 控制约束 (加速度 m/s^2)
# # #     u_max = 0.05 # 最大加速度
# # #     control_con = {'min': [-u_max, -u_max], 'max': [u_max, u_max]}

# # #     # 实例化MPC控制器
# # #     mpc = MPCController(dynamics_model=dynamics,
# # #                         prediction_horizon=horizon,
# # #                         Q_terminal=Q_f,
# # #                         R_control=R,
# # #                         control_constraints=control_con)

# # #     # 仿真设置
# # #     initial_state = np.array([1000.0, -500.0, 2.0, 1.0]) # 初始状态 [x, y, vx, vy]
# # #     # 目标是移动到原点附近的某个相对状态，例如，x方向移动-1000, y方向移动+500
# # #     # 这个目标状态是相对于当前状态的“期望终端状态”
# # #     # 为了简单测试，我们让目标终端状态固定为原点。
# # #     # 在实际应用中，target_terminal_state 将由上层决策模块（如谱分析+AIF）给出
# # #     # 且这个 target_terminal_state 是指 N*dt 时间后的期望状态
# # #     target_state_at_N = np.array([0.0, 0.0, 0.0, 0.0]) # 期望的N步后的状态

# # #     num_simulation_steps = 100 # 模拟多少个MPC控制周期
# # #     sim_time_total = num_simulation_steps * dt_sim

# # #     # 记录历史
# # #     actual_states_history = [initial_state]
# # #     control_inputs_history = []
# # #     predicted_trajectories_history = [] # 记录每次MPC求解的预测轨迹

# # #     current_actual_state = initial_state.copy()

# # #     print("开始MPC仿真...")
# # #     for i in range(num_simulation_steps):
# # #         print(f"\n仿真步: {i+1}/{num_simulation_steps}")
# # #         print(f"当前实际状态: {current_actual_state}")

# # #         # 求解MPC，得到第一个控制输入和预测轨迹
# # #         # 注意：target_terminal_state 是 N 步之后的期望状态
# # #         optimal_u0, predicted_traj, success = mpc.solve(current_actual_state, target_state_at_N)

# # #         if not success:
# # #             print("MPC求解失败，终止仿真。")
# # #             break
        
# # #         print(f"计算得到的最优控制 u0: {optimal_u0}")
# # #         control_inputs_history.append(optimal_u0)
# # #         predicted_trajectories_history.append(predicted_traj) # 存储整个预测

# # #         # 应用第一个控制输入到实际系统中
# # #         current_actual_state = dynamics.step(current_actual_state, optimal_u0)
# # #         actual_states_history.append(current_actual_state)

# # #         # 检查是否接近目标 (简单检查)
# # #         if np.linalg.norm(current_actual_state - target_state_at_N) < 50: # 50m 容差
# # #             print(f"在第 {i+1} 步达到目标附近。")
# # #             break
    
# # #     print("仿真结束。")

# # #     actual_states_history = np.array(actual_states_history)
# # #     control_inputs_history = np.array(control_inputs_history)

# # #     # 可视化
# # #     time_axis_actual = np.arange(len(actual_states_history)) * dt_sim
# # #     time_axis_control = np.arange(len(control_inputs_history)) * dt_sim

# # #     fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=False)
# # #     fig.suptitle("MPC 控制器性能仿真 (二维CW模型)", fontsize=16)

# # #     # 子图1: 实际轨迹 X-Y
# # #     axs[0].plot(actual_states_history[:, 1], actual_states_history[:, 0], 'b-o', label="实际轨迹", markersize=3)
# # #     axs[0].scatter(initial_state[1], initial_state[0], color='red', marker='s', s=100, label="初始位置")
# # #     axs[0].scatter(target_state_at_N[1], target_state_at_N[0], color='green', marker='x', s=100, label="终端目标状态")
    
# # #     plot_preds_indices = []
# # #     if len(predicted_trajectories_history) > 0:
# # #         plot_preds_indices.append(0) # 绘制第一次的预测
# # #     if len(predicted_trajectories_history) > 1: # 如果至少有两次预测
# # #         plot_preds_indices.append(len(predicted_trajectories_history) // 2) # 中间的
# # #     if len(predicted_trajectories_history) > 2: # 如果至少有三次预测
# # #          plot_preds_indices.append(len(predicted_trajectories_history) - 1) # 最后一次的

# # #     colors = ['gray', 'lightcoral', 'lightblue']
# # #     # 移除重复的索引，以防列表很短时重复添加
# # #     plot_preds_indices = sorted(list(set(plot_preds_indices)))

# # #     for k_idx, pred_idx in enumerate(plot_preds_indices):
# # #         # pred_idx 已经是有效索引了
# # #         pred_traj_to_plot = predicted_trajectories_history[pred_idx]
# # #         axs[0].plot(pred_traj_to_plot[1, :], pred_traj_to_plot[0, :], linestyle='--', color=colors[k_idx % len(colors)],
# # #                     label=f"预测轨迹 (MPC迭代 {pred_idx+1})") # 标签改为 pred_idx+1 更直观

# # #     axs[0].set_xlabel("沿迹向相对位置 Y (m)")
# # #     axs[0].set_ylabel("径向相对位置 X (m)")
# # #     axs[0].set_title("航天器轨迹")
# # #     axs[0].legend()
# # #     axs[0].grid(True)
# # #     axs[0].axis('equal')

# # #     # 子图2: 状态分量随时间变化
# # #     axs[1].plot(time_axis_actual, actual_states_history[:, 0], label="X (实际)")
# # #     axs[1].plot(time_axis_actual, actual_states_history[:, 1], label="Y (实际)")
# # #     axs[1].plot(time_axis_actual, actual_states_history[:, 2], label="Vx (实际)")
# # #     axs[1].plot(time_axis_actual, actual_states_history[:, 3], label="Vy (实际)")
# # #     axs[1].set_xlabel("时间 (s)")
# # #     axs[1].set_ylabel("状态值")
# # #     axs[1].set_title("状态分量演化")
# # #     axs[1].legend()
# # #     axs[1].grid(True)

# # #     # 子图3: 控制输入随时间变化
# # #     if control_inputs_history.shape[0] > 0:
# # #         axs[2].plot(time_axis_control, control_inputs_history[:, 0], label="Ux (控制输入)")
# # #         axs[2].plot(time_axis_control, control_inputs_history[:, 1], label="Uy (控制输入)")
# # #         if control_con:
# # #             axs[2].axhline(y=control_con['min'][0], color='r', linestyle='--', label="控制下限")
# # #             axs[2].axhline(y=control_con['max'][0], color='r', linestyle='--', label="控制上限")
# # #     axs[2].set_xlabel("时间 (s)")
# # #     axs[2].set_ylabel("控制输入 (m/s^2)")
# # #     axs[2].set_title("控制输入历史")
# # #     axs[2].legend()
# # #     axs[2].grid(True)

# # #     plt.tight_layout(rect=[0, 0, 1, 0.96])
# # #     plt.show()

# # # if __name__ == '__main__':
# # #     test_mpc_controller()
# # # common/mpc_controller.py
# # import casadi as ca
# # import numpy as np
# # import sys
# # import os
# # from typing import Tuple
# # # ... (路径管理和导入保持不变) ...
# # _current_file_directory = os.path.dirname(os.path.abspath(__file__))
# # _project_root_directory = os.path.dirname(_current_file_directory)
# # if _project_root_directory not in sys.path:
# #     sys.path.insert(0, _project_root_directory)
# # from common.dynamics import CWEquation

# # class MPCController:
# #     def __init__(self,
# #                  dynamics_model: CWEquation,
# #                  prediction_horizon: int,
# #                  Q_terminal: np.ndarray,
# #                  R_control: np.ndarray,
# #                  Q_state: np.ndarray = None,
# #                  control_constraints: dict = None,
# #                  state_constraints: dict = None,
# #                  solver_print_level: int = 0): # 新增参数控制打印级别
# #         self.dynamics_model = dynamics_model
# #         self.Ad = dynamics_model.A_d
# #         self.Bd = dynamics_model.B_d
# #         self.N = prediction_horizon
# #         self.Qf = Q_terminal
# #         self.R = R_control
# #         self.Q = Q_state if Q_state is not None else np.zeros((dynamics_model.state_size, dynamics_model.state_size))
# #         self.control_constraints = control_constraints
# #         self.state_constraints = state_constraints
# #         self.state_size = dynamics_model.state_size
# #         self.control_size = dynamics_model.control_size
# #         self.solver_print_level = solver_print_level # 保存打印级别

# #         self._build_solver()

# #     def _build_solver(self):
# #         self.opti = ca.Opti()
# #         self.X_opt = self.opti.variable(self.state_size, self.N + 1)
# #         self.U_opt = self.opti.variable(self.control_size, self.N)
# #         self.x0_param = self.opti.parameter(self.state_size, 1)
# #         self.xtarget_N_param = self.opti.parameter(self.state_size, 1)
# #         obj = 0
# #         for k in range(self.N):
# #             obj += ca.mtimes([self.U_opt[:, k].T, self.R, self.U_opt[:, k]])
# #             state_error_k = self.X_opt[:, k] - self.xtarget_N_param # 惩罚中间状态与最终目标的偏差 (可选)
# #             if not np.all(self.Q == 0): # 仅当Q非全零时才加入中间状态惩罚
# #                  obj += ca.mtimes([state_error_k.T, self.Q, state_error_k])
# #         terminal_error = self.X_opt[:, self.N] - self.xtarget_N_param
# #         obj += ca.mtimes([terminal_error.T, self.Qf, terminal_error])
# #         self.opti.minimize(obj)
# #         self.opti.subject_to(self.X_opt[:, 0] == self.x0_param)
# #         for k in range(self.N):
# #             self.opti.subject_to(self.X_opt[:, k+1] == ca.mtimes(self.Ad, self.X_opt[:, k]) + ca.mtimes(self.Bd, self.U_opt[:, k]))
# #         if self.control_constraints:
# #             u_min = self.control_constraints.get('min', -ca.inf)
# #             u_max = self.control_constraints.get('max', ca.inf)
# #             for k in range(self.N):
# #                 self.opti.subject_to(self.opti.bounded(u_min, self.U_opt[:, k], u_max))
# #         if self.state_constraints:
# #             s_min = self.state_constraints.get('min', -ca.inf)
# #             s_max = self.state_constraints.get('max', ca.inf)
# #             for k in range(1, self.N + 1): # 状态约束通常从x1开始
# #                 self.opti.subject_to(self.opti.bounded(s_min, self.X_opt[:, k], s_max))
        
# #         # 使用传入的 solver_print_level
# #         # ipopt_opts = {'ipopt.print_level': self.solver_print_level, 'print_time': 0, 'ipopt.sb': 'yes'} # sb (suppress banner)
# #         ipopt_opts = {'ipopt.print_level': self.solver_print_level, 'print_time': 0 if self.solver_print_level == 0 else 1, 'ipopt.sb': 'yes'}
# #         if self.solver_print_level == 0: # 如果不打印，可以尝试更快的线性求解器
# #             ipopt_opts['ipopt.mu_strategy'] = 'adaptive'
# #             ipopt_opts['ipopt.linear_solver'] = 'mumps' # 或者 'ma27', 'ma57' 等，需要安装

# #         self.opti.solver('ipopt', ipopt_opts)
# #         # print("[MPC DEBUG _build_solver] Solver built.")


# #     def solve(self, current_state: np.ndarray, target_terminal_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
# #         self.opti.set_value(self.x0_param, current_state)
# #         self.opti.set_value(self.xtarget_N_param, target_terminal_state)

# #         # print(f"      [MPC DEBUG solve] Target (abs pos): {np.round(target_terminal_state[:2],1)}, Current (abs pos): {np.round(current_state[:2],1)}")
# #         # state_diff = target_terminal_state - current_state
# #         # print(f"        State Diff (pos): {np.round(state_diff[:2],1)}, (vel): {np.round(state_diff[2:],2)}")


# #         try:
# #             solution = self.opti.solve()
# #             optimal_control_sequence = solution.value(self.U_opt)
# #             predicted_trajectory = solution.value(self.X_opt)
# #             # if self.solver_print_level > 0 or np.linalg.norm(optimal_control_sequence[:,0]) < 1e-4 : # 如果控制量小也打印
# #             #     print(f"        [MPC DEBUG solve] Solved. Optimal u0: {np.round(optimal_control_sequence[:,0],4)}")
# #             #     print(f"          Solver status: {solution.stats().get('return_status', 'N/A')}, Success: {solution.stats().get('success', False)}")
# #             #     print(f"          Predicted final state (pos): {np.round(predicted_trajectory[:2,-1],1)}")
# #             return optimal_control_sequence[:, 0], predicted_trajectory, solution.stats().get('success', False)
# #         except RuntimeError as e:
# #             print(f"        [MPC DEBUG solve] MPC SOLVE FAILED: {e}")
# #             # 返回零控制和从当前状态开始的无控轨迹作为预测
# #             zero_control_traj = np.zeros((self.state_size, self.N + 1))
# #             zero_control_traj[:, 0] = current_state
# #             temp_state = current_state.copy()
# #             for k_fail in range(self.N):
# #                 temp_state = self.Ad @ temp_state # 假设零控制
# #                 zero_control_traj[:, k_fail+1] = temp_state
# #             return np.zeros(self.control_size), zero_control_traj, False

# # # --- 模块自测试 (保持不变) ---
# # # (test_mpc_controller 函数)
# # if __name__ == '__main__':
# #     # ... (自测试代码保持不变，但现在可以用 solver_print_level)
# #     print("--- 测试 MPCController ---")
# #     dt_sim_mpc = 1.0; n_sim_mpc = 0.0011
# #     dynamics_mpc = CWEquation(n=n_sim_mpc, dt=dt_sim_mpc)
# #     qf_diag_mpc = [100, 100, 10, 10]; r_diag_mpc = [0.01, 0.01]
# #     horizon_mpc = 15; u_max_mpc = 0.2
# #     mpc_instance = MPCController(
# #         dynamics_model=dynamics_mpc, prediction_horizon=horizon_mpc,
# #         Q_terminal=np.diag(qf_diag_mpc), R_control=np.diag(r_diag_mpc),
# #         control_constraints={'min': [-u_max_mpc]*2, 'max': [u_max_mpc]*2},
# #         solver_print_level=0 # 测试时可以设为0或3
# #     )
# #     initial_state_mpc = np.array([100.0, 50.0, 0.1, -0.2]) # 远离原点，有初始速度
# #     target_state_mpc = np.array([0.0, 0.0, 0.0, 0.0])    # 目标是回到原点并停止
# #     print(f"MPC Test: Initial state {initial_state_mpc}, Target state {target_state_mpc}")
# #     num_sim_steps_mpc = 60
# #     actual_states_hist_mpc = [initial_state_mpc]
# #     control_inputs_hist_mpc = []
# #     predicted_trajs_hist_mpc = []
# #     current_s_mpc = initial_state_mpc.copy()
# #     for i_mpc in range(num_sim_steps_mpc):
# #         print(f"\nMPC Test Step {i_mpc+1}")
# #         u_opt_0, pred_traj, success_mpc = mpc_instance.solve(current_s_mpc, target_state_mpc)
# #         if not success_mpc: print("  MPC solve failed in this step!"); break
# #         print(f"  Optimal u0: {u_opt_0}")
# #         current_s_mpc = dynamics_mpc.step(current_s_mpc, u_opt_0)
# #         actual_states_hist_mpc.append(current_s_mpc.copy())
# #         control_inputs_hist_mpc.append(u_opt_0.copy())
# #         predicted_trajs_hist_mpc.append(pred_traj.copy())
# #     # (可视化部分保持不变)
# #     import matplotlib
# #     import matplotlib.pyplot as plt # 移到测试函数内部
# #     matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# #     matplotlib.rcParams['axes.unicode_minus'] = False
# #     actual_states_hist_mpc = np.array(actual_states_hist_mpc)
# #     control_inputs_hist_mpc = np.array(control_inputs_hist_mpc) if control_inputs_hist_mpc else np.array([]).reshape(0,dynamics_mpc.control_size)
# #     time_axis_actual_mpc = np.arange(actual_states_hist_mpc.shape[0]) * dt_sim_mpc
# #     time_axis_control_mpc = np.arange(control_inputs_hist_mpc.shape[0]) * dt_sim_mpc + dt_sim_mpc
# #     fig_mpc, axs_mpc = plt.subplots(3,1,figsize=(10,15), sharex=False)
# #     axs_mpc[0].plot(actual_states_hist_mpc[:,1], actual_states_hist_mpc[:,0], 'b-o', label="实际轨迹", markersize=3)
# #     axs_mpc[0].scatter(initial_state_mpc[1],initial_state_mpc[0],color='g',marker='s',s=100,label="初始")
# #     axs_mpc[0].scatter(target_state_mpc[1],target_state_mpc[0],color='r',marker='x',s=100,label="目标")
# #     for k_pred_viz, pred_traj_viz in enumerate(predicted_trajs_hist_mpc):
# #         if k_pred_viz % 5 == 0: # 每隔5次预测画一次
# #              axs_mpc[0].plot(pred_traj_viz[1,:], pred_traj_viz[0,:], 'k--', alpha=0.3, label=f"预测@{k_pred_viz*dt_sim_mpc:.0f}s" if k_pred_viz % 10 == 0 else None)
# #     axs_mpc[0].set_xlabel("Y (m)"); axs_mpc[0].set_ylabel("X (m)"); axs_mpc[0].set_title("MPC轨迹"); axs_mpc[0].legend(); axs_mpc[0].grid(True); axs_mpc[0].axis('equal');
# #     axs_mpc[1].plot(time_axis_actual_mpc, actual_states_hist_mpc[:,0], label="X"); axs_mpc[1].plot(time_axis_actual_mpc, actual_states_hist_mpc[:,1], label="Y")
# #     axs_mpc[1].plot(time_axis_actual_mpc, actual_states_hist_mpc[:,2], label="Vx"); axs_mpc[1].plot(time_axis_actual_mpc, actual_states_hist_mpc[:,3], label="Vy")
# #     axs_mpc[1].set_xlabel("Time (s)"); axs_mpc[1].set_ylabel("状态"); axs_mpc[1].set_title("状态演化"); axs_mpc[1].legend(); axs_mpc[1].grid(True);
# #     if control_inputs_hist_mpc.shape[0] > 0:
# #         axs_mpc[2].plot(time_axis_control_mpc, control_inputs_hist_mpc[:,0], label="ux"); axs_mpc[2].plot(time_axis_control_mpc, control_inputs_hist_mpc[:,1], label="uy")
# #     axs_mpc[2].set_xlabel("Time (s)"); axs_mpc[2].set_ylabel("控制 ($m/s^2$)"); axs_mpc[2].set_title("控制输入"); axs_mpc[2].legend(); axs_mpc[2].grid(True);
# #     plt.tight_layout(); plt.show()
# # common/mpc_controller.py
# import casadi as ca
# import numpy as np
# import sys
# import os
# from typing import Tuple
# # ... (路径管理和导入保持不变) ...
# _current_file_directory = os.path.dirname(os.path.abspath(__file__))
# _project_root_directory = os.path.dirname(os.path.dirname(_current_file_directory)) # mpc_controller.py 在 common 下
# if _project_root_directory not in sys.path:
#     sys.path.insert(0, _project_root_directory)
# from common.dynamics import CWEquation

# class MPCController:
#     def __init__(self,
#                  dynamics_model: CWEquation,
#                  prediction_horizon: int,
#                  Q_terminal: np.ndarray,
#                  R_control: np.ndarray,
#                  Q_state: np.ndarray = None, # 您代码中有此参数
#                  control_constraints: dict = None,
#                  state_constraints: dict = None, # 您代码中有此参数
#                  solver_print_level: int = 0): 
#         self.dynamics_model = dynamics_model
#         self.Ad = dynamics_model.A_d
#         self.Bd = dynamics_model.B_d
#         self.N = prediction_horizon
#         self.Qf = Q_terminal
#         self.R = R_control
#         self.Q = Q_state if Q_state is not None else np.zeros((dynamics_model.state_size, dynamics_model.state_size))
#         self.control_constraints = control_constraints
#         self.state_constraints = state_constraints # 您代码中有此属性
#         self.state_size = dynamics_model.state_size
#         self.control_size = dynamics_model.control_size
#         self.solver_print_level = solver_print_level

#         self._build_solver()

#     def _build_solver(self):
#         self.opti = ca.Opti()
#         self.X_opt = self.opti.variable(self.state_size, self.N + 1)
#         self.U_opt = self.opti.variable(self.control_size, self.N)
#         self.x0_param = self.opti.parameter(self.state_size, 1)
#         self.xtarget_N_param = self.opti.parameter(self.state_size, 1) # 终端目标参数
        
#         obj = 0
#         # 控制代价
#         for k in range(self.N):
#             obj += ca.mtimes([self.U_opt[:, k].T, self.R, self.U_opt[:, k]])
#         # 终端状态代价
#         terminal_error = self.X_opt[:, self.N] - self.xtarget_N_param
#         obj += ca.mtimes([terminal_error.T, self.Qf, terminal_error])
#         # 中间状态代价 (如果Q_state提供)
#         if not np.all(self.Q == 0): # 您代码中的条件
#             for k in range(self.N): # 对预测时域内的每个中间状态（除了初始状态x0）
#                 state_error_k = self.X_opt[:, k+1] - self.xtarget_N_param # 也可以是到目标点的误差
#                 # 或者，如果Q_state的目的是跟踪某个参考轨迹（这里没有），则误差项会不同
#                 # 如果Q_state是惩罚状态偏离0（例如在调节器问题中），则是 self.X_opt[:,k+1].T @ self.Q @ self.X_opt[:,k+1]
#                 # 这里假设Q也是惩罚与最终目标的偏差，与您的代码一致
#                 obj += ca.mtimes([state_error_k.T, self.Q, state_error_k])

#         self.opti.minimize(obj)
        
#         # 初始状态约束
#         self.opti.subject_to(self.X_opt[:, 0] == self.x0_param)
#         # 动力学约束
#         for k in range(self.N):
#             self.opti.subject_to(self.X_opt[:, k+1] == ca.mtimes(self.Ad, self.X_opt[:, k]) + ca.mtimes(self.Bd, self.U_opt[:, k]))
        
#         # 控制约束
#         if self.control_constraints:
#             u_min = self.control_constraints.get('min', -ca.inf)
#             u_max = self.control_constraints.get('max', ca.inf)
#             for k in range(self.N): # 对每个控制输入施加约束
#                 self.opti.subject_to(self.opti.bounded(u_min, self.U_opt[:, k], u_max))
        
#         # 状态约束 (如果提供)
#         if self.state_constraints: # 您代码中有此部分
#             s_min = self.state_constraints.get('min', -ca.inf)
#             s_max = self.state_constraints.get('max', ca.inf)
#             for k in range(1, self.N + 1): # 从 x1 到 xN
#                 self.opti.subject_to(self.opti.bounded(s_min, self.X_opt[:, k], s_max))
        
#         ipopt_opts = {'ipopt.print_level': self.solver_print_level, 'print_time': 0 if self.solver_print_level == 0 else 1, 'ipopt.sb': 'yes'}
#         if self.solver_print_level == 0:
#             ipopt_opts['ipopt.mu_strategy'] = 'adaptive'
#             # ipopt_opts['ipopt.linear_solver'] = 'mumps' # 尝试默认，如果mumps有问题
#         self.opti.solver('ipopt', ipopt_opts)
#         # print(f"[MPC DEBUG _build_solver] Solver built with print_level: {self.solver_print_level}.")


#     def solve(self, current_state: np.ndarray, target_terminal_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
#         self.opti.set_value(self.x0_param, current_state.reshape(-1,1)) # 确保是列向量
#         self.opti.set_value(self.xtarget_N_param, target_terminal_state.reshape(-1,1)) # 确保是列向量

#         # print(f"      [MPC DEBUG solve] Target (abs pos): {np.round(target_terminal_state[:2],1)}, Vel: {np.round(target_terminal_state[2:],2)}")
#         # print(f"        Current (abs pos): {np.round(current_state[:2],1)}, Vel: {np.round(current_state[2:],2)}")
#         # state_diff = target_terminal_state - current_state
#         # print(f"        State Diff (pos): {np.round(state_diff[:2],1)}, (vel): {np.round(state_diff[2:],2)}")
#         # print(f"        Qf diag: {np.diag(self.Qf)}, R diag: {np.diag(self.R)}")

#         try:
#             # 初始猜测 (可选，但有时有帮助)
#             # self.opti.set_initial(self.X_opt, np.tile(current_state.reshape(-1,1), (1, self.N + 1)))
#             # self.opti.set_initial(self.U_opt, np.zeros((self.control_size, self.N)))

#             solution = self.opti.solve()
#             optimal_control_sequence = solution.value(self.U_opt)
#             predicted_trajectory = solution.value(self.X_opt)
#             success = solution.stats().get('success', False) # IPOPT的成功标志

#             # if self.solver_print_level > 0 or np.linalg.norm(optimal_control_sequence[:,0]) < 1e-3 or not success:
#             #     print(f"        [MPC DEBUG solve] Solved. Optimal u0: {np.round(optimal_control_sequence[:,0],4)}")
#             #     print(f"          Solver status: {solution.stats().get('return_status', 'N/A')}, Success flag: {success}")
#             #     print(f"          Predicted final state (pos): {np.round(predicted_trajectory[:2,-1],1)}")
#             return optimal_control_sequence[:, 0], predicted_trajectory, success
#         except RuntimeError as e:
#             print(f"        [MPC DEBUG solve] MPC SOLVE FAILED for state: {np.round(current_state[:2],1)} target: {np.round(target_terminal_state[:2],1)}. Error: {e}")
#             zero_control_traj = np.zeros((self.state_size, self.N + 1))
#             zero_control_traj[:, 0] = current_state
#             temp_state = current_state.copy()
#             for k_fail in range(self.N):
#                 temp_state = self.Ad @ temp_state
#                 zero_control_traj[:, k_fail+1] = temp_state
#             return np.zeros(self.control_size), zero_control_traj, False

# # --- 模块自测试 (保持不变) ---
# # (test_mpc_controller 函数)
# common/mpc_controller.py
# common/mpc_controller.py
import casadi as ca
import numpy as np
import sys
import os
from typing import Tuple
# ... (路径管理和导入保持不变) ...
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
_project_root_directory = os.path.dirname(os.path.dirname(_current_file_directory)) # mpc_controller.py 在 common 下
if _project_root_directory not in sys.path:
    sys.path.insert(0, _project_root_directory)
from common.dynamics import CWEquation

class MPCController:
    def __init__(self,
                 dynamics_model: CWEquation,
                 prediction_horizon: int,
                 Q_terminal: np.ndarray,
                 R_control: np.ndarray,
                 Q_state: np.ndarray = None, # 您代码中有此参数
                 control_constraints: dict = None,
                 state_constraints: dict = None, # 您代码中有此参数
                 solver_print_level: int = 0): 
        self.dynamics_model = dynamics_model
        self.Ad = dynamics_model.A_d
        self.Bd = dynamics_model.B_d
        self.N = prediction_horizon
        self.Qf = Q_terminal
        self.R = R_control
        self.Q = Q_state if Q_state is not None else np.zeros((dynamics_model.state_size, dynamics_model.state_size))
        self.control_constraints = control_constraints
        self.state_constraints = state_constraints # 您代码中有此属性
        self.state_size = dynamics_model.state_size
        self.control_size = dynamics_model.control_size
        self.solver_print_level = solver_print_level

        self._build_solver()

    def _build_solver(self):
        self.opti = ca.Opti()
        self.X_opt = self.opti.variable(self.state_size, self.N + 1)
        self.U_opt = self.opti.variable(self.control_size, self.N)
        self.x0_param = self.opti.parameter(self.state_size, 1)
        self.xtarget_N_param = self.opti.parameter(self.state_size, 1) # 终端目标参数
        
        obj = 0
        # 控制代价
        for k in range(self.N):
            obj += ca.mtimes([self.U_opt[:, k].T, self.R, self.U_opt[:, k]])
        # 终端状态代价
        terminal_error = self.X_opt[:, self.N] - self.xtarget_N_param
        obj += ca.mtimes([terminal_error.T, self.Qf, terminal_error])
        # 中间状态代价 (如果Q_state提供)
        if not np.all(self.Q == 0): # 您代码中的条件
            for k in range(self.N): # 对预测时域内的每个中间状态（除了初始状态x0）
                state_error_k = self.X_opt[:, k+1] - self.xtarget_N_param # 也可以是到目标点的误差
                # 或者，如果Q_state的目的是跟踪某个参考轨迹（这里没有），则误差项会不同
                # 如果Q_state是惩罚状态偏离0（例如在调节器问题中），则是 self.X_opt[:,k+1].T @ self.Q @ self.X_opt[:,k+1]
                # 这里假设Q也是惩罚与最终目标的偏差，与您的代码一致
                obj += ca.mtimes([state_error_k.T, self.Q, state_error_k])

        self.opti.minimize(obj)
        
        # 初始状态约束
        self.opti.subject_to(self.X_opt[:, 0] == self.x0_param)
        # 动力学约束
        for k in range(self.N):
            self.opti.subject_to(self.X_opt[:, k+1] == ca.mtimes(self.Ad, self.X_opt[:, k]) + ca.mtimes(self.Bd, self.U_opt[:, k]))
        
        # 控制约束
        if self.control_constraints:
            u_min = self.control_constraints.get('min', -ca.inf)
            u_max = self.control_constraints.get('max', ca.inf)
            for k in range(self.N): # 对每个控制输入施加约束
                self.opti.subject_to(self.opti.bounded(u_min, self.U_opt[:, k], u_max))
        
        # 状态约束 (如果提供)
        if self.state_constraints: # 您代码中有此部分
            s_min = self.state_constraints.get('min', -ca.inf)
            s_max = self.state_constraints.get('max', ca.inf)
            for k in range(1, self.N + 1): # 从 x1 到 xN
                self.opti.subject_to(self.opti.bounded(s_min, self.X_opt[:, k], s_max))
        
        ipopt_opts = {'ipopt.print_level': self.solver_print_level, 'print_time': 0 if self.solver_print_level == 0 else 1, 'ipopt.sb': 'yes'}
        if self.solver_print_level == 0:
            ipopt_opts['ipopt.mu_strategy'] = 'adaptive'
            # ipopt_opts['ipopt.linear_solver'] = 'mumps' # 尝试默认，如果mumps有问题
        self.opti.solver('ipopt', ipopt_opts)
        # print(f"[MPC DEBUG _build_solver] Solver built with print_level: {self.solver_print_level}.")


    def solve(self, current_state: np.ndarray, target_terminal_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        self.opti.set_value(self.x0_param, current_state.reshape(-1,1)) # 确保是列向量
        self.opti.set_value(self.xtarget_N_param, target_terminal_state.reshape(-1,1)) # 确保是列向量

        # print(f"      [MPC DEBUG solve] Target (abs pos): {np.round(target_terminal_state[:2],1)}, Vel: {np.round(target_terminal_state[2:],2)}")
        # print(f"        Current (abs pos): {np.round(current_state[:2],1)}, Vel: {np.round(current_state[2:],2)}")
        # state_diff = target_terminal_state - current_state
        # print(f"        State Diff (pos): {np.round(state_diff[:2],1)}, (vel): {np.round(state_diff[2:],2)}")
        # print(f"        Qf diag: {np.diag(self.Qf)}, R diag: {np.diag(self.R)}")

        try:
            # 初始猜测 (可选，但有时有帮助)
            # self.opti.set_initial(self.X_opt, np.tile(current_state.reshape(-1,1), (1, self.N + 1)))
            # self.opti.set_initial(self.U_opt, np.zeros((self.control_size, self.N)))

            solution = self.opti.solve()
            optimal_control_sequence = solution.value(self.U_opt)
            predicted_trajectory = solution.value(self.X_opt)
            success = solution.stats().get('success', False) # IPOPT的成功标志

            # if self.solver_print_level > 0 or np.linalg.norm(optimal_control_sequence[:,0]) < 1e-3 or not success:
            #     print(f"        [MPC DEBUG solve] Solved. Optimal u0: {np.round(optimal_control_sequence[:,0],4)}")
            #     print(f"          Solver status: {solution.stats().get('return_status', 'N/A')}, Success flag: {success}")
            #     print(f"          Predicted final state (pos): {np.round(predicted_trajectory[:2,-1],1)}")
            return optimal_control_sequence[:, 0], predicted_trajectory, success
        except RuntimeError as e:
            print(f"        [MPC DEBUG solve] MPC SOLVE FAILED for state: {np.round(current_state[:2],1)} target: {np.round(target_terminal_state[:2],1)}. Error: {e}")
            zero_control_traj = np.zeros((self.state_size, self.N + 1))
            zero_control_traj[:, 0] = current_state
            temp_state = current_state.copy()
            for k_fail in range(self.N):
                temp_state = self.Ad @ temp_state
                zero_control_traj[:, k_fail+1] = temp_state
            return np.zeros(self.control_size), zero_control_traj, False

# --- 模块自测试 (保持不变) ---
# (test_mpc_controller 函数)