# common/spectral_analysis.py
import numpy as np
from scipy.linalg import eigh # 使用 eigh 因为格拉姆矩阵是对称的
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# --- 临时路径处理 ---
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
_project_root_directory = os.path.dirname(_current_file_directory)
if _project_root_directory not in sys.path:
    sys.path.insert(0, _project_root_directory)
# --- 结束 ---

from common.dynamics import CWEquation

class SpectralAnalyzer:
    """
    通过分析系统可控性格拉姆矩阵的谱特性，生成候选的目标状态偏差。
    这些偏差代表了系统在预测时域 H 内，通过单位能量的归一化控制输入，
    能够达到的、相对于零输入响应的主要状态空间方向。
    """
    def __init__(self,
                 dynamics_model: CWEquation,
                 prediction_horizon_H: int,
                 control_normalization_matrix: np.ndarray = None,
                 alpha_scale: float = 1.0):
        """
        初始化谱分析器。

        参数:
            dynamics_model (CWEquation): 航天器动力学模型。
            prediction_horizon_H (int): 用于构建可控性矩阵的预测时域 H。
            control_normalization_matrix (np.ndarray, optional): 控制输入的归一化/缩放矩阵 S。
                                                                形状 (control_size, control_size)。
                                                                如果为None，则默认为单位阵。
                                                                S 的对角元素通常是 2 / (u_max - u_min)。
            alpha_scale (float): 特征模态的缩放因子，用于调整目标偏差的大小。
                                 在SETS中，这个因子可能与MCTS的探索策略有关，这里设为1.0，
                                 表示偏差的大小直接由 sqrt(lambda_j) 决定。
        """
        if prediction_horizon_H < 0:
            raise ValueError("预测时域 H 不能为负。")

        self.dynamics_model = dynamics_model
        self.H = prediction_horizon_H # 这是谱分析向前看的步数
        self.alpha_scale = alpha_scale

        self.Ad, self.Bd = dynamics_model.get_discrete_matrices()
        self.state_size = dynamics_model.state_size
        self.control_size = dynamics_model.control_size

        if control_normalization_matrix is None:
            self.S_norm = np.eye(self.control_size)
        elif control_normalization_matrix.shape == (self.control_size, self.control_size):
            self.S_norm = control_normalization_matrix
        else:
            raise ValueError(f"控制归一化矩阵S的形状应为 ({self.control_size}, {self.control_size})")

        # CH_norm_strict 将是 (state_size, H * control_size)
        # WH 将是 (state_size, state_size)
        self.CH_norm_strict, self.WH = self._compute_controllability_matrices_strict()
        self.eigenvalues, self.eigenvectors = self._compute_spectrum()

    def _compute_controllability_matrices_strict(self) -> tuple[np.ndarray, np.ndarray]:
        """
        计算H步归一化可控性矩阵 CH_norm_strict 和可控性格拉姆矩阵 WH。
        CH_norm_strict = [Ad^(H-1)*B_norm, Ad^(H-2)*B_norm, ..., Ad*B_norm, B_norm]
        where B_norm = Bd*S_norm.
        This matrix maps the control sequence [u_0', u_1', ..., u_{H-1}']^T (stacked)
        to the state deviation delta_x_H = CH_norm_strict * U_prime_stacked.
        WH = CH_norm_strict * CH_norm_strict^T
        """
        if self.H == 0:
            return np.zeros((self.state_size, 0)), np.zeros((self.state_size, self.state_size))

        B_norm = self.Bd @ self.S_norm # shape (state_size, control_size)
        
        CH_terms = []
        # Ad_pow_i_minus_1_B_norm stores A_d^(i-1) * B_norm
        # For k from 0 to H-1, we need Ad^(H-1-k)*B_norm
        # k=0: Ad^(H-1)*B_norm
        # k=1: Ad^(H-2)*B_norm
        # ...
        # k=H-1: Ad^0*B_norm = B_norm
        
        current_Ad_power = np.eye(self.state_size) # Ad^0
        # First compute Ad^(H-1)
        Ad_pow_H_minus_1 = np.linalg.matrix_power(self.Ad, self.H - 1) if self.H > 0 else np.eye(self.state_size)

        for k in range(self.H):
            # For u_k, the term is Ad^(H-1-k) * B_norm
            if self.H - 1 - k >= 0:
                term = np.linalg.matrix_power(self.Ad, self.H - 1 - k) @ B_norm
            else: # Should not happen if k ranges from 0 to H-1
                term = B_norm # This case is for k = H-1, power is 0
            CH_terms.append(term)
        
        # CH_norm_strict 的形状是 (state_size, H * control_size)
        # np.hstack stacks horizontally, so each term must be (state_size, control_size)
        CH_norm_strict = np.hstack(CH_terms)
        
        WH = CH_norm_strict @ CH_norm_strict.T
        return CH_norm_strict, WH

    def _compute_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """
        计算可控性格拉姆矩阵 WH 的特征值和特征向量。
        返回按特征值降序排列的结果。
        """
        if self.WH.shape[0] == 0 or self.WH.shape[1] == 0 :
            return np.array([]), np.array([])

        eigenvalues, eigenvectors = eigh(self.WH)
        eigenvalues = np.maximum(eigenvalues, 1e-12) 
        
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        return sorted_eigenvalues, sorted_eigenvectors

    def calculate_target_deviation_options(self, num_modes_to_consider: int = None) -> list[np.ndarray]:
        """
        根据谱分析结果生成候选的目标状态偏差选项。
        每个选项代表沿一个主要可控方向的状态偏差。
        这个偏差是 delta_s_H = x_H - Ad^H * x_0，即H步后状态相对于零输入响应的偏差。

        参数:
            num_modes_to_consider (int, optional): 要考虑的特征模态数量（对应最大的特征值）。
                                                如果为None，则考虑所有特征值大于阈值的模态。
        返回:
            list[np.ndarray]: 一个包含多个目标状态偏差向量的列表。
                              每个向量形状为 (state_size,)。
        """
        options = []
        if self.eigenvalues.size == 0:
            return [np.zeros(self.state_size)]

        if num_modes_to_consider is None:
            effective_modes_indices = np.where(self.eigenvalues > 1e-9)[0]
        else:
            num_to_take = min(num_modes_to_consider, len(self.eigenvalues))
            potential_indices = np.arange(num_to_take)
            effective_modes_indices = [idx for idx in potential_indices if self.eigenvalues[idx] > 1e-9]
        
        if not effective_modes_indices and len(effective_modes_indices) == 0 : # 修正判断条件
             return [np.zeros(self.state_size)]

        for i in effective_modes_indices:
            eigval = self.eigenvalues[i]
            eigvec = self.eigenvectors[:, i] # eigvec 是 WH 的特征向量 v_j
            
            # delta_s_option = alpha_scale * sqrt(lambda_j) * v_j
            # sqrt(lambda_j) 是在归一化控制能量下，沿v_j方向能达到的最大偏差的某种度量。
            # v_j 本身定义了一个方向。
            magnitude = self.alpha_scale * np.sqrt(eigval) 
            
            delta_s_option_pos = magnitude * eigvec
            delta_s_option_neg = -magnitude * eigvec
            
            options.append(delta_s_option_pos)
            options.append(delta_s_option_neg)
            
        unique_options_tuples = {tuple(opt.flatten()) for opt in options}
        unique_options = [np.array(t).reshape(self.state_size,) for t in unique_options_tuples]

        if not unique_options:
            unique_options.append(np.zeros(self.state_size))
            
        return unique_options

    def get_zero_input_response(self, current_state: np.ndarray) -> np.ndarray:
        """
        计算当前状态在零控制输入下，经过 H 步后的状态 (漂移状态)。
        x_H_drift = Ad^H * current_state
        """
        if self.H == 0:
            return current_state.copy()
        Ad_pow_H = np.linalg.matrix_power(self.Ad, self.H)
        return Ad_pow_H @ current_state

    def get_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        return self.eigenvalues, self.eigenvectors

# --- 模块自测试与可视化 ---
def test_spectral_analyzer():
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    n_orbit = 2 * np.pi / (90 * 60)
    dt_sim = 20.0
    dynamics = CWEquation(n=n_orbit, dt=dt_sim)

    H_spectral = 10
    # 假设控制输入 u_x, u_y 的范围都是 [-0.01, 0.01] m/s^2
    u_abs_max = 0.01
    # S_jj = 2 / (u_max_j - u_min_j) = 2 / (2 * u_abs_max_j) = 1 / u_abs_max_j
    # 如果所有控制输入的 u_abs_max 都相同
    s_diag_elements = [1.0 / u_abs_max] * dynamics.control_size
    S_norm_matrix = np.diag(s_diag_elements) 
    
    alpha_s = 1.0

    analyzer = SpectralAnalyzer(dynamics_model=dynamics,
                                prediction_horizon_H=H_spectral,
                                control_normalization_matrix=S_norm_matrix,
                                alpha_scale=alpha_s)

    eigvals, eigvecs = analyzer.get_spectrum()
    # ... (打印特征值和特征向量的代码) ...
    print("可控性格拉姆矩阵的特征值 (降序):")
    for i, val in enumerate(eigvals):
        print(f"  λ_{i+1} = {val:.4e}")
    
    print("\n对应的特征向量 (主方向):")
    for i in range(min(3, len(eigvals))): # 只打印前3个
        print(f"  v_{i+1} = {np.array2string(eigvecs[:, i], precision=3, floatmode='fixed')}")

    # 计算相对于零输入响应的目标状态偏差选项
    deviation_options = analyzer.calculate_target_deviation_options(num_modes_to_consider=dynamics.state_size) # 考虑所有状态维度个模态
    print(f"\n生成的目标状态偏差选项 (delta_s_H) ({len(deviation_options)} 个):")
    for i, opt in enumerate(deviation_options):
        print(f"  选项 {i+1}: [x_dev, y_dev, vx_dev, vy_dev] = {np.array2string(opt, precision=3, floatmode='fixed')}")

    # 示例：如何将偏差转换为绝对目标状态 (用于后续MPC)
    current_sim_state = np.array([100.0, 200.0, 0.5, -0.5]) # 假设一个当前状态
    s_H_drift = analyzer.get_zero_input_response(current_sim_state)
    print(f"\n假设当前状态: {current_sim_state}")
    print(f"在H={H_spectral}步后的零输入响应 (漂移状态 s_H_drift): {s_H_drift}")

    if deviation_options:
        selected_deviation = deviation_options[0] # 选择第一个偏差选项作为示例
        # 目标绝对状态 s_target_H = s_H_drift + selected_deviation
        # 这是在H步之后期望到达的绝对状态
        absolute_target_state_H = s_H_drift + selected_deviation
        print(f"选择第一个偏差选项: {selected_deviation}")
        print(f"对应的绝对目标状态 (s_target_H): {absolute_target_state_H}")
        
        # 如果直接将偏差作为相对于当前状态的目标（即MPC的N=H_spectral）
        # target_for_mpc = current_sim_state + selected_deviation
        # print(f"如果偏差直接作为相对于当前状态的目标 (s_current + delta_s): {target_for_mpc}")
        print("\n重要：以上偏差选项是相对于H步后的 *零输入响应状态* 的。")
        print("在AIF决策并传递给MPC时，通常会将此偏差转换为相对于 *当前状态* 的目标，")
        print("或者计算出绝对目标位置供MPC使用。")
        print("例如，若AIF选择偏差 delta_s_chosen, 则MPC的终端目标可以是 current_state + delta_s_chosen (如果谱分析的H和MPC的N匹配)")


    # --- 可视化 ---
    fig_main, axes = plt.subplots(1, 2, figsize=(19, 7))
    fig_main.suptitle("谱分析结果可视化", fontsize=16)

    # 子图1: 目标状态偏差选项 (位置分量投影)
    ax_dev = axes[0]
    ax_dev.set_title(f"目标状态偏差选项 (H={H_spectral}, 位置投影)", fontsize=14)
    if deviation_options:
        origin_x, origin_y = 0, 0
        num_to_plot = min(len(deviation_options), 10)
        colors_dev = plt.cm.viridis(np.linspace(0, 1, num_to_plot))

        plotted_labels = set()
        for i in range(num_to_plot):
            opt = deviation_options[i]
            label = f"选项 {i//2 + 1}{'+' if i%2==0 else '-'}: $\Delta$pos=({opt[0]:.0f}, {opt[1]:.0f})m"
            # 避免重复图例
            if label not in plotted_labels:
                ax_dev.arrow(origin_x, origin_y, opt[1], opt[0], 
                             head_width=max(0.02 * np.linalg.norm(opt[:2]), 0.5),
                             head_length=max(0.04 * np.linalg.norm(opt[:2]), 1.0),
                             fc=colors_dev[i], ec=colors_dev[i], label=label)
                plotted_labels.add(label)
            else: # 无标签，只画箭头
                 ax_dev.arrow(origin_x, origin_y, opt[1], opt[0], 
                             head_width=max(0.02 * np.linalg.norm(opt[:2]), 0.5),
                             head_length=max(0.04 * np.linalg.norm(opt[:2]), 1.0),
                             fc=colors_dev[i], ec=colors_dev[i])

        ax_dev.scatter(origin_x, origin_y, color='black', marker='o', s=50, label="相对零输入响应参考点")
    ax_dev.set_xlabel("沿迹向相对位置偏差 $\Delta y$ (m)")
    ax_dev.set_ylabel("径向相对位置偏差 $\Delta x$ (m)")
    ax_dev.grid(True)
    ax_dev.axhline(0, color='black', lw=0.5)
    ax_dev.axvline(0, color='black', lw=0.5)
    ax_dev.axis('equal')
    
    # 调整图例
    handles, labels = ax_dev.get_legend_handles_labels()
    if handles: # 仅当有图例项时才显示
        # 按标签排序图例项，使得+/-成对出现
        # sorted_legend_elements = sorted(zip(labels, handles), key=lambda x: x[0])
        # labels_sorted = [x[0] for x in sorted_legend_elements]
        # handles_sorted = [x[1] for x in sorted_legend_elements]
        # ax_dev.legend(handles_sorted, labels_sorted, loc='best', fontsize='small')
        ax_dev.legend(loc='best', fontsize='small')


    # 子图2: 特征向量（模态）对状态量的影响热力图
    ax_heatmap = axes[1]
    ax_heatmap.set_title("特征模态对状态量的影响 (特征向量分量绝对值)", fontsize=14)
    
    num_modes_for_heatmap = min(eigvecs.shape[1] if eigvecs.ndim > 1 and eigvecs.shape[1] > 0 else 0, 8) 
    if num_modes_for_heatmap > 0:
        heatmap_data = np.abs(eigvecs[:, :num_modes_for_heatmap]) 
        state_labels = ['$\Delta x$', '$\Delta y$', '$\Delta v_x$', '$\Delta v_y$']
        mode_labels = [f'模态 {i+1}\n($\lambda$={eigvals[i]:.1e})' for i in range(num_modes_for_heatmap)]
        
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                    xticklabels=mode_labels, yticklabels=state_labels, ax=ax_heatmap,
                    linewidths=.5, cbar_kws={'label': '影响程度 (分量绝对值)'})
        # Fix: Use set_xticklabels to control rotation and alignment
        ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), 
                                  rotation=30, 
                                  horizontalalignment='right')
        ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), 
                                  rotation=0)
    else:
        ax_heatmap.text(0.5, 0.5, "无有效特征模态可显示", 
                       horizontalalignment='center', 
                       verticalalignment='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    test_spectral_analyzer()