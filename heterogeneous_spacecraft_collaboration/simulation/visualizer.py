# heterogeneous_spacecraft_collaboration/simulation/visualizer.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Dict, List, Any, Optional, TYPE_CHECKING
import sys
import os
import copy
from pathlib import Path

# ---统一的导入路径管理---
_simulation_dir_viz = os.path.dirname(os.path.abspath(__file__))
_project_root_dir_viz = os.path.dirname(_simulation_dir_viz)
if _project_root_dir_viz not in sys.path:
    sys.path.insert(0, _project_root_dir_viz)
# ---路径管理结束---

try:
    from simulation.environment import SimulationEnvironment, SpacecraftAgentSim
    from strong_communication.ksc_coalition_formation import IDLE_TASK_ID
    from utils.font_utils import setup_matplotlib_fonts
except ImportError as e:
    print(f"ERROR [simulation/visualizer.py]: Failed to import modules: {e}")
    print("Attempting fallback imports for Visualizer...")
    try:
        from ..simulation.environment import SimulationEnvironment, SpacecraftAgentSim
        from ..strong_communication.ksc_coalition_formation import IDLE_TASK_ID
        from ..utils.font_utils import setup_matplotlib_fonts
    except ImportError as e2:
        print(f"  Fallback imports for Visualizer also failed: {e2}")
        def setup_matplotlib_fonts():
            print("CRITICAL WARNING: utils.font_utils.setup_matplotlib_fonts not found. Chinese characters may not display.")
            matplotlib.rcParams['axes.unicode_minus'] = False
            return False

if TYPE_CHECKING:
    pass

# ---全局样式和颜色(Nature风格)---
NATURE_COLORS = ['#0C5DA5', '#FF2C00', '#00B945', '#FF9500', '#845B97', '#474747', '#9e9e9e', '#F2C800']
EXTENDED_NATURE_COLORS = NATURE_COLORS + ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
LINESTYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'H', 'd']

def set_academic_style_and_fonts():
    font_found = setup_matplotlib_fonts() #
    plt.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
        'legend.frameon': False, 'legend.framealpha': 0.7, 'legend.facecolor': 'white',
        'figure.dpi': 150, 'savefig.dpi': 300,
        'lines.linewidth': 2.5, 'lines.markersize': 8,
        'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.6,
        'axes.formatter.useoffset': False, 'axes.formatter.limits': (-3, 4),
        'axes.edgecolor': 'black', 'axes.linewidth': 1.2,
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'xtick.minor.width': 0.8, 'ytick.minor.width': 0.8,
        'xtick.major.pad': 7, 'ytick.major.pad': 7,
        'axes.labelpad': 12, 'axes.titlepad': 18,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
        'axes.unicode_minus': False
    })

set_academic_style_and_fonts()

class Visualizer:
    def __init__(self, env: SimulationEnvironment):
        if not isinstance(env, SimulationEnvironment):
            raise TypeError("env 参数必须是 SimulationEnvironment 的实例。")
        self.env = env
        self.num_agents = env.num_agents
        self.agent_ids = env.agent_ids
        self.task_type_names = env.task_type_names if env.task_type_names else ["类型未知"]
        self.colors = EXTENDED_NATURE_COLORS
        self.linestyles = LINESTYLES
        self.markers = MARKERS
        
        viz_settings = self.env.scenario_config.get('visualization_settings', {})
        plot_traj_settings = viz_settings.get('plot_agent_trajectories_2d', {})
        self.show_all_tasks_in_trajectory_plot = plot_traj_settings.get('show_all_tasks_in_trajectory_plot', True)


    def _get_time_axis(self, num_points: Optional[int] = None, dt_override: Optional[float] = None) -> np.ndarray:
        steps = num_points if num_points is not None else self.env.time_step + 1
        dt_val = dt_override if dt_override is not None else self.env.dt
        return np.arange(steps) * dt_val

    def _get_ksc_decision_rounds_axis(self) -> np.ndarray:
        effective_num_rounds = self.env.ksc_decision_round_count
        return np.arange(1, effective_num_rounds + 1) if effective_num_rounds > 0 else np.array([])

    def _empty_plot_message(self, fig_or_figsize: Any, title: str, message: str, axs_list: Optional[List[plt.Axes]] = None) -> plt.Figure:
        # (Implementation from previous response)
        if isinstance(fig_or_figsize, plt.Figure):
            fig = fig_or_figsize
            if axs_list and axs_list[0]: ax = axs_list[0]
            else: ax = fig.gca() if fig.get_axes() else fig.add_subplot(111)
        else:
            default_figsize = (10, 4)
            if isinstance(fig_or_figsize, tuple) and len(fig_or_figsize) == 2:
                 current_figsize = (fig_or_figsize[0], fig_or_figsize[1] if fig_or_figsize[1] is not None else default_figsize[1])
            else:
                 current_figsize = default_figsize
            fig, ax = plt.subplots(figsize=current_figsize)
        
        ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticks([]); ax.set_yticks([])
        if isinstance(fig_or_figsize, tuple) or not axs_list : plt.tight_layout()
        return fig

    def plot_agent_trajectories_2d(self,
                                 ax: Optional[plt.Axes] = None,
                                 # show_tasks is now effectively controlled by self.show_all_tasks_in_trajectory_plot
                                 show_final_assignment_ksc: bool = True,
                                 show_communication_range_final_agent_id: Optional[str] = None,
                                 plot_every_n_steps: int = 1,
                                 show_aif_target_points: bool = True, 
                                 aif_target_plot_interval: int = 5
                                 ) -> Optional[plt.Figure]:
        if ax is None:
            fig, current_ax = plt.subplots(figsize=(14, 12))
            return_fig = fig
        else:
            current_ax = ax
            return_fig = current_ax.get_figure()

        current_ax.set_title(f"航天器轨迹与任务分布\n场景: {self.env.scenario_config.get('scenario_display_name', self.env.scenario_config.get('scenario_name_loaded_from','未知场景'))}")
        current_ax.set_xlabel("X 位置 (m)")
        current_ax.set_ylabel("Y 位置 (m)")
        current_ax.axis('equal')

        # 打印 self.show_all_tasks_in_trajectory_plot 的值进行调试
        # print(f"DEBUG [Visualizer Traj]: self.show_all_tasks_in_trajectory_plot = {self.show_all_tasks_in_trajectory_plot}")


        plotted_task_labels_legend = set()
        agent_legend_handles = []
        agent_legend_labels = []
        all_x_coords, all_y_coords = [], []

        # ... (绘制智能体轨迹的代码保持不变，确保它在任务绘制之前，以便任务图标在轨迹之上 zorder 控制) ...
        # (Agent plotting code from previous correct response)
        for i, agent_id in enumerate(self.agent_ids):
            agent_sim: Optional[SpacecraftAgentSim] = self.env.agents.get(agent_id)
            if not agent_sim or not agent_sim.history_states:
                continue

            agent_color = self.colors[i % len(self.colors)]
            agent_linestyle = self.linestyles[i % len(self.linestyles)]
            
            states_full = np.array(agent_sim.history_states)
            num_total_states = states_full.shape[0]
            indices_to_plot = list(range(0, num_total_states, plot_every_n_steps))
            if num_total_states > 0 and (num_total_states - 1) not in indices_to_plot :
                indices_to_plot.append(num_total_states - 1)
            
            states_for_path = states_full[indices_to_plot, :] if indices_to_plot else np.array([])

            if states_for_path.ndim == 2 and states_for_path.shape[0] > 0 and states_for_path.shape[1] >= 2:
                all_x_coords.extend(states_for_path[:, 0])
                all_y_coords.extend(states_for_path[:, 1])
                line, = current_ax.plot(states_for_path[:, 0], states_for_path[:, 1], 
                                 linestyle=agent_linestyle, color=agent_color,
                                 label=f"智能体 {agent_id}", 
                                 linewidth=matplotlib.rcParams['lines.linewidth'], zorder=3)
                
                start_scatter_handle = current_ax.scatter(states_for_path[0, 0], states_for_path[0, 1], marker='o', s=120,
                                        color=agent_color, edgecolor='black', zorder=5, label=f"{agent_id} 起点")
                
                if i == 0: 
                    agent_legend_handles.extend([line, start_scatter_handle])
                    agent_legend_labels.extend([f"智能体 {agent_id}", f"{agent_id} 起点"])
                else: 
                    agent_legend_handles.append(line)
                    agent_legend_labels.append(f"智能体 {agent_id}")

                current_ax.text(states_for_path[-1, 0] + 1.5, states_for_path[-1, 1] + 1.5, agent_id, fontsize=10, color=agent_color, zorder=6,
                                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.1))

                if show_aif_target_points and agent_sim.history_delta_s_planned:
                    aif_targets_x, aif_targets_y = [], []
                    num_delta_s = len(agent_sim.history_delta_s_planned)
                    
                    for k in range(num_delta_s):
                        if k % aif_target_plot_interval == 0: 
                            if k < num_total_states: 
                                current_state_for_delta_s = states_full[k, :2]
                                delta_s_val = agent_sim.history_delta_s_planned[k]
                                if delta_s_val is not None and len(delta_s_val) >= 2:
                                    aif_target = current_state_for_delta_s + delta_s_val[:2]
                                    aif_targets_x.append(aif_target[0])
                                    aif_targets_y.append(aif_target[1])
                    
                    if aif_targets_x:
                        aif_target_handle = current_ax.scatter(aif_targets_x, aif_targets_y, marker='+', s=60,
                                         color=agent_color, alpha=0.7, zorder=4, 
                                         label=f"{agent_id} AIF短期目标")
                        is_new_label_aif = True
                        for lbl_aif in agent_legend_labels:
                            if f"{agent_id} AIF短期目标" == lbl_aif:
                                is_new_label_aif = False; break
                        if is_new_label_aif:
                           agent_legend_handles.append(aif_target_handle)
                           agent_legend_labels.append(f"{agent_id} AIF短期目标")
        
        task_legend_handles, task_legend_labels = [], []
        # --- 修改开始：直接从 self.env.tasks 获取最新的任务列表进行迭代 ---
        current_env_tasks = self.env.tasks 
        
        if current_env_tasks: # 检查是否有任务
            # print(f"DEBUG [Visualizer Traj]: Tasks to iterate for plotting: {list(current_env_tasks.keys())}")
            for i_task, task_id in enumerate(current_env_tasks.keys()): # 直接迭代当前环境中的任务ID
                task_info = current_env_tasks.get(task_id) # task_info 必定存在，因为是从keys迭代的
                # ... (后续的任务位置提取、状态判断、绘制逻辑与之前版本一致)
                if task_info: # 理论上 task_info 不会是 None
                    task_pos_val = None
                    pos_keys_to_try = ["position", "current_state", "initial_state"]
                    for key in pos_keys_to_try:
                        if key in task_info and task_info[key] is not None:
                            try:
                                pos_candidate = np.array(task_info[key])
                                if pos_candidate.ndim == 1 and len(pos_candidate) >= 2: task_pos_val = pos_candidate[:2]; break 
                                elif pos_candidate.ndim == 2 and pos_candidate.shape[0] > 0 and pos_candidate.shape[1] >=2: task_pos_val = pos_candidate[-1, :2]; break
                            except Exception: pass
                    
                    if task_pos_val is not None and len(task_pos_val) == 2 :
                        all_x_coords.append(task_pos_val[0]); all_y_coords.append(task_pos_val[1])
                        task_status_completed = task_info.get("is_completed", False) or task_info.get("current_completion_percentage", 0.0) >= 100.0
                        completion_percentage = task_info.get("current_completion_percentage", 0.0)
                        
                        task_marker_style = '^'; task_alpha = 0.95
                        task_marker_color = self.colors[(self.num_agents + i_task) % len(self.colors)]
                        label_suffix_parts = []
                        task_type_name_display = task_info.get("true_type_name", "未知类型")
                        
                        if task_status_completed:
                            task_marker_style = 'X'; task_marker_color = 'dimgray'; task_alpha = 0.7
                            label_suffix_parts.append("已完成")
                        elif completion_percentage > 0:
                            task_marker_style = 'o'
                            label_suffix_parts.append(f"{completion_percentage:.0f}%")
                        
                        if task_type_name_display != "未知类型" and not task_status_completed:
                             label_suffix_parts.insert(0, task_type_name_display)
                        
                        label_suffix_display = f" ({', '.join(label_suffix_parts)})" if label_suffix_parts else ""
                        full_task_label_for_legend = f"任务 {task_id}{label_suffix_display}"
                        
                        should_plot_this_task = self.show_all_tasks_in_trajectory_plot or not task_status_completed

                        if should_plot_this_task:
                            scatter_handle = current_ax.scatter(task_pos_val[0], task_pos_val[1], 
                                            marker=task_marker_style, s=250, color=task_marker_color,
                                            label=full_task_label_for_legend if task_id not in plotted_task_labels_legend else "", 
                                            alpha=task_alpha, edgecolor='black', linewidth=1.0, zorder=10)
                            if task_id not in plotted_task_labels_legend:
                                task_legend_handles.append(scatter_handle); task_legend_labels.append(full_task_label_for_legend)
                                plotted_task_labels_legend.add(task_id)
                            
                            text_anno = f"{task_id}" + ("(C)" if task_status_completed and self.show_all_tasks_in_trajectory_plot else "")
                            current_ax.text(task_pos_val[0] + 2.0, task_pos_val[1] + 2.0, text_anno,
                                        fontsize=11, zorder=11, color=task_marker_color,
                                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1))
        else:
            print("[Visualizer INFO traj] self.env.tasks is empty or None. No tasks will be plotted.")
        # --- 修改结束 ---
        
        # ... (设置绘图范围、绘制KSC分配、通信范围、图例的代码保持不变) ...
        # (Bounding box, KSC assignment, communication range, and legend code from previous correct response)
        if all_x_coords and all_y_coords:
            min_x, max_x = min(all_x_coords), max(all_x_coords)
            min_y, max_y = min(all_y_coords), max(all_y_coords)
            padding_x = (max_x - min_x) * 0.1 + 10 
            padding_y = (max_y - min_y) * 0.1 + 10
            current_ax.set_xlim(min_x - padding_x, max_x + padding_x)
            current_ax.set_ylim(min_y - padding_y, max_y + padding_y)
        else: current_ax.set_xlim(-50, 50); current_ax.set_ylim(-50, 50)

        if show_final_assignment_ksc and self.env.ksc_algorithm and hasattr(self.env.ksc_algorithm, '_sim_best_assignment_final'):
            final_assignment = self.env.ksc_algorithm._sim_best_assignment_final
            if final_assignment:
                for agent_id_assigned, task_id_assigned in final_assignment.items():
                    # 使用 current_env_tasks 来检查任务是否存在
                    if task_id_assigned and task_id_assigned != IDLE_TASK_ID and task_id_assigned in current_env_tasks:
                        agent_sim_assigned = self.env.agents.get(agent_id_assigned)
                        task_def_assigned = current_env_tasks.get(task_id_assigned) # 从当前任务列表获取
                        task_pos_assigned = np.array(task_def_assigned["position"][:2]) if task_def_assigned and "position" in task_def_assigned else None
                        
                        if task_pos_assigned is not None and agent_sim_assigned and agent_sim_assigned.history_states:
                            if agent_sim_assigned.history_states[-1].ndim==1 and len(agent_sim_assigned.history_states[-1]) >=2:
                                agent_final_pos = agent_sim_assigned.history_states[-1][:2]
                                current_ax.plot([agent_final_pos[0], task_pos_assigned[0]], 
                                             [agent_final_pos[1], task_pos_assigned[1]],
                                             linestyle=':', color=self.colors[self.agent_ids.index(agent_id_assigned) % len(self.colors)],
                                             linewidth=1.5, alpha=0.7, zorder=4)
        
        if show_communication_range_final_agent_id and show_communication_range_final_agent_id in self.env.agents:
            agent_comm = self.env.agents[show_communication_range_final_agent_id]
            if agent_comm.history_states:
                agent_final_pos_comm = agent_comm.history_states[-1][:2]
                comm_range = self.env.sim_params.get("communication_range", 0)
                if isinstance(comm_range, (int, float)) and comm_range > 0:
                    circle = plt.Circle((agent_final_pos_comm[0], agent_final_pos_comm[1]), comm_range,
                                        color=self.colors[self.agent_ids.index(show_communication_range_final_agent_id) % len(self.colors)], 
                                        alpha=0.10, fill=True, zorder=1, 
                                        label=f"{show_communication_range_final_agent_id} 通信范围")
                    current_ax.add_artist(circle)
                    is_new_label_comm = True
                    for lbl_comm in agent_legend_labels:
                        if f"{show_communication_range_final_agent_id} 通信范围" == lbl_comm:
                            is_new_label_comm = False; break
                    if is_new_label_comm:
                        agent_legend_handles.append(circle)
                        agent_legend_labels.append(f"{show_communication_range_final_agent_id} 通信范围")

        all_handles = agent_legend_handles + task_legend_handles
        all_labels = agent_legend_labels + task_legend_labels
        unique_labels_handles_dict = {}
        for handle, label in zip(all_handles, all_labels):
            if label not in unique_labels_handles_dict:
                unique_labels_handles_dict[label] = handle
        
        num_legend_items = len(unique_labels_handles_dict)
        if num_legend_items > 0:
            current_ax.legend(unique_labels_handles_dict.values(), unique_labels_handles_dict.keys(), 
                        loc='upper left', bbox_to_anchor=(1.02, 1.0),
                        ncol=1 if num_legend_items <= 7 else 2)

        if ax is None:
            plt.tight_layout(rect=[0, 0, 0.80, 1]) 
            return return_fig
        return None

    def plot_all_agents_velocities(self, figsize=(12, 6)) -> Optional[plt.Figure]:
        fig, ax = plt.subplots(figsize=figsize)
        time_axis_states = self._get_time_axis(num_points=self.env.time_step + 1)
        plotted_anything = False
        for i, agent_id in enumerate(self.agent_ids):
            agent = self.env.agents.get(agent_id)
            if agent and agent.history_states:
                states_array = np.array(agent.history_states)
                if states_array.ndim == 2 and states_array.shape[0] == len(time_axis_states) and states_array.shape[1] >= 4:
                    velocities = np.linalg.norm(states_array[:, 2:4], axis=1)
                    ax.plot(time_axis_states, velocities, label=f"{agent_id} 速度",
                            color=self.colors[i % len(self.colors)],
                            linestyle=self.linestyles[i % len(self.linestyles)])
                    plotted_anything = True
        if not plotted_anything and len(self.agent_ids)>0:
            ax.text(0.5, 0.5, "无有效速度数据可绘制", ha='center', va='center', transform=ax.transAxes)
        elif not self.agent_ids:
            ax.text(0.5, 0.5, "无智能体", ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel("仿真时间 (s)"); ax.set_ylabel("速度大小 (m/s)")
        ax.set_title("所有智能体速度演化")
        if plotted_anything: ax.legend(loc='best')
        fig.tight_layout(); return fig

    def plot_all_agents_accelerations(self, figsize=(12, 6)) -> Optional[plt.Figure]:
        fig, ax = plt.subplots(figsize=figsize)
        num_control_points = self.env.time_step 
        if num_control_points == 0 :
            ax.text(0.5,0.5, "无控制数据可绘制加速度", ha='center',va='center',transform=ax.transAxes)
            ax.set_title("所有智能体控制输入（加速度）演化"); fig.tight_layout(); return fig
        time_axis_controls = np.arange(1, num_control_points + 1) * self.env.dt
        plotted_anything = False
        for i, agent_id in enumerate(self.agent_ids):
            agent = self.env.agents.get(agent_id)
            if agent and agent.history_controls:
                controls_array = np.array(agent.history_controls)
                if controls_array.ndim == 2 and controls_array.shape[0] == len(time_axis_controls) and controls_array.shape[1] >= 2:
                    accelerations = np.linalg.norm(controls_array, axis=1)
                    ax.plot(time_axis_controls, accelerations, label=f"{agent_id} 加速度",
                            color=self.colors[i % len(self.colors)],
                            linestyle=self.linestyles[i % len(self.linestyles)], 
                            drawstyle='steps-post')
                    plotted_anything = True
        if not plotted_anything and len(self.agent_ids)>0:
             ax.text(0.5, 0.5, "无有效加速度数据可绘制", ha='center', va='center', transform=ax.transAxes)
        elif not self.agent_ids:
            ax.text(0.5, 0.5, "无智能体", ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel("仿真时间 (s)"); ax.set_ylabel("加速度大小 ($m/s^2$)")
        ax.set_title("所有智能体控制输入（加速度）演化")
        if plotted_anything: ax.legend(loc='best')
        fig.tight_layout(); return fig

    def plot_ksc_global_utility(self, ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
        if not self.env.history_global_utility_ksc:
            return self._empty_plot_message(ax.get_figure() if ax else (10,6), "KSC系统总期望效用演化", "无KSC效用数据")
        if ax is None: fig, current_ax = plt.subplots(figsize=(10, 6)); return_fig = fig
        else: current_ax = ax; return_fig = current_ax.get_figure()
        ksc_rounds = self._get_ksc_decision_rounds_axis(); utility_data = np.array(self.env.history_global_utility_ksc)
        plot_len = min(len(ksc_rounds), len(utility_data))
        if plot_len == 0 : current_ax.text(0.5, 0.5, "无KSC效用数据", ha='center', va='center', transform=current_ax.transAxes)
        else:
            current_ax.plot(ksc_rounds[:plot_len], utility_data[:plot_len], marker=self.markers[0], linestyle=self.linestyles[0], color=self.colors[0])
            current_ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=min(plot_len, 10)))
        current_ax.set_xlabel("KSC决策轮次"); current_ax.set_ylabel("系统总期望效用"); current_ax.set_title("KSC系统总期望效用演化")
        if ax is None: plt.tight_layout(); return return_fig
        return None

    def plot_belief_convergence(self, tracked_agents_tasks: Dict[str, List[str]], figsize=(14, None)) -> Optional[plt.Figure]:
        if not self.env.history_beliefs_at_ksc_step or self.env.belief_manager_ksc is None or not self.task_type_names:
            return self._empty_plot_message(figsize, "KSC任务类型信念收敛图", "依赖数据缺失 (信念历史/管理器/任务类型)")
        valid_tracked_pairs = []
        for agent_id, tasks_for_agent in tracked_agents_tasks.items():
            if agent_id in self.agent_ids:
                for task_id_check in tasks_for_agent:
                    if task_id_check in self.env.tasks: valid_tracked_pairs.append((agent_id, task_id_check))
        num_plots = len(valid_tracked_pairs)
        if num_plots == 0: return self._empty_plot_message(figsize, "KSC任务类型信念收敛图", "未指定有效的追踪目标或无数据")
        fig_height = figsize[1] if figsize[1] is not None else max(5, num_plots * 4.0)
        fig, axs = plt.subplots(num_plots, 1, figsize=(figsize[0], fig_height), sharex=True, squeeze=False); axs_flat = axs.flatten()
        ksc_rounds_axis_all = self._get_ksc_decision_rounds_axis()
        if len(ksc_rounds_axis_all) == 0: return self._empty_plot_message(fig, "KSC任务类型信念收敛图", "无KSC决策轮次数据", axs_list=axs_flat)
        num_task_types = len(self.task_type_names)
        if num_task_types == 0: return self._empty_plot_message(fig, "KSC任务类型信念收敛图", "无任务类型定义", axs_list=axs_flat)
        plot_idx_counter = 0
        for agent_id_track, task_id_track in valid_tracked_pairs:
            if plot_idx_counter >= len(axs_flat): break 
            ax_current = axs_flat[plot_idx_counter]; belief_histories = [np.full(len(ksc_rounds_axis_all), np.nan) for _ in range(num_task_types)]; has_data = False
            for round_idx, ksc_round in enumerate(ksc_rounds_axis_all): 
                round_data = self.env.history_beliefs_at_ksc_step.get(int(ksc_round), {}); belief_dist = round_data.get(agent_id_track, {}).get(task_id_track)
                if belief_dist is not None and len(belief_dist) == num_task_types:
                    has_data = True
                    for type_idx in range(num_task_types): belief_histories[type_idx][round_idx] = belief_dist[type_idx]
            if not has_data: ax_current.text(0.5, 0.5, f"智能体 {agent_id_track} 对任务 {task_id_track}\n无信念数据", ha='center', va='center', transform=ax_current.transAxes)
            else:
                task_meta = self.env.tasks.get(task_id_track, {}); true_type_name = task_meta.get('true_type_name', '未知'); true_type_idx = task_meta.get("true_type_idx", -1)
                for type_idx in range(num_task_types):
                    is_true = (type_idx == true_type_idx)
                    ax_current.plot(ksc_rounds_axis_all, belief_histories[type_idx], marker=self.markers[type_idx % len(self.markers)], linestyle=self.linestyles[type_idx % len(self.linestyles)], color=self.colors[type_idx % len(self.colors)], linewidth=3.0 if is_true else 2.0, alpha=1.0 if is_true else 0.8, label=f"{self.task_type_names[type_idx]}{' (真实)' if is_true else ''}")
                ax_current.set_title(f"智能体 {agent_id_track} - 任务 {task_id_track} (真实: {true_type_name})"); ax_current.set_ylabel("信念概率 P(类型)"); ax_current.legend(loc='best'); ax_current.set_ylim(-0.05, 1.05)
            ax_current.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=min(len(ksc_rounds_axis_all) if ksc_rounds_axis_all.size >0 else 5,8)))
            plot_idx_counter +=1
        if plot_idx_counter > 0 and len(axs_flat) > plot_idx_counter -1 : axs_flat[plot_idx_counter-1].set_xlabel("KSC决策轮次")
        elif num_plots == 1 and len(axs_flat) > 0: axs_flat[0].set_xlabel("KSC决策轮次")
        fig.suptitle(f"KSC任务类型信念收敛图 (场景: {self.env.scenario_config.get('scenario_display_name', '未知')})"); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); return fig

    def plot_ksc_communication_metrics(self, ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
        if not (self.env.ksc_stats_messages_broadcasted or self.env.ksc_stats_active_links):
            return self._empty_plot_message(ax.get_figure() if ax else (10,6), "KSC通信开销指标", "无KSC通信数据")
        if ax is None: fig, current_ax = plt.subplots(figsize=(10, 6)); return_fig = fig
        else: current_ax = ax; return_fig = current_ax.get_figure()
        ksc_rounds = self._get_ksc_decision_rounds_axis(); min_len = len(ksc_rounds)
        lists_to_check = [self.env.ksc_stats_messages_broadcasted if isinstance(self.env.ksc_stats_messages_broadcasted, list) else [], self.env.ksc_stats_messages_processed if isinstance(self.env.ksc_stats_messages_processed, list) else [], self.env.ksc_stats_active_links if isinstance(self.env.ksc_stats_active_links, list) else []]
        for hist_list in lists_to_check:
            if hist_list: min_len = min(min_len, len(hist_list))
        if min_len == 0 and len(ksc_rounds) == 0: current_ax.text(0.5, 0.5, "无KSC通信数据点", ha='center', va='center', transform=current_ax.transAxes)
        elif min_len == 0 and len(ksc_rounds) > 0: min_len = len(ksc_rounds); current_ax.text(0.5, 0.5, "KSC通信统计数据缺失", ha='center', va='center', transform=current_ax.transAxes)
        if min_len > 0 :
            ksc_plot_rounds = ksc_rounds[:min_len]; lns = []
            if self.env.ksc_stats_messages_broadcasted and len(self.env.ksc_stats_messages_broadcasted) >= min_len: lns.extend(current_ax.plot(ksc_plot_rounds, self.env.ksc_stats_messages_broadcasted[:min_len], marker=self.markers[1], linestyle=self.linestyles[1], color=self.colors[1], label="广播消息数"))
            if self.env.ksc_stats_messages_processed and len(self.env.ksc_stats_messages_processed) >= min_len: lns.extend(current_ax.plot(ksc_plot_rounds, self.env.ksc_stats_messages_processed[:min_len], marker=self.markers[2], linestyle=self.linestyles[2], color=self.colors[2], label="处理消息数"))
            current_ax.set_ylabel("消息数量", color=self.colors[1]); current_ax.tick_params(axis='y', labelcolor=self.colors[1]); current_ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
            if self.env.ksc_stats_active_links and len(self.env.ksc_stats_active_links) >= min_len:
                ax2 = current_ax.twinx(); lns.extend(ax2.plot(ksc_plot_rounds, self.env.ksc_stats_active_links[:min_len], marker=self.markers[3], linestyle=self.linestyles[3], color=self.colors[3], label="活动链接数")); ax2.set_ylabel("活动链接数", color=self.colors[3]); ax2.tick_params(axis='y', labelcolor=self.colors[3]); ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
            if lns: current_ax.legend(lns, [l.get_label() for l in lns], loc='best')
            current_ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=min(len(ksc_plot_rounds) if ksc_plot_rounds.size > 0 else 5,10)))
        current_ax.set_xlabel("KSC决策轮次"); current_ax.set_title("KSC通信开销指标")
        if ax is None: plt.tight_layout(); return return_fig
        return None

    def plot_overall_performance(self, figsize=(16, 12)) -> Optional[plt.Figure]:
        fig, axs = plt.subplots(2, 2, figsize=figsize); fig.suptitle(f"仿真总体性能 ({self.env.scenario_config.get('scenario_display_name', '未知')})")
        self.plot_ksc_global_utility(ax=axs[0,0])
        time_ax_active = self._get_time_axis(num_points=len(self.env.history_num_active_tasks))
        if len(time_ax_active) == len(self.env.history_num_active_tasks) and self.env.history_num_active_tasks: axs[0,1].plot(time_ax_active, self.env.history_num_active_tasks, marker='.', color=self.colors[1 % len(self.colors)]); axs[0,1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        else: axs[0,1].text(0.5,0.5,"无活动任务数据",ha='center',va='center',transform=axs[0,1].transAxes)
        axs[0,1].set_xlabel("仿真时间 (s)"); axs[0,1].set_ylabel("活动任务数量"); axs[0,1].set_title("活动任务数量变化")
        self.plot_ksc_communication_metrics(ax=axs[1,0])
        time_ax_env_links = self._get_time_axis(num_points=len(self.env.env_level_active_comm_links))
        if len(time_ax_env_links) == len(self.env.env_level_active_comm_links) and self.env.env_level_active_comm_links: axs[1,1].plot(time_ax_env_links, self.env.env_level_active_comm_links, marker='.', color=self.colors[4 % len(self.colors)]); axs[1,1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        else: axs[1,1].text(0.5,0.5,"无环境链接数据",ha='center',va='center',transform=axs[1,1].transAxes)
        axs[1,1].set_xlabel("仿真时间 (s)"); axs[1,1].set_ylabel("环境通信链接数"); axs[1,1].set_title("环境通信链接数 (基于范围)")
        plt.tight_layout(rect=[0, 0, 1, 0.95]); return fig

    def plot_single_agent_details(self, agent_id_to_plot: str, figsize=(12, 15)) -> Optional[plt.Figure]:
        agent = self.env.agents.get(agent_id_to_plot);
        if not agent: return self._empty_plot_message(figsize, f"智能体 {agent_id_to_plot} 详细状态", f"智能体 {agent_id_to_plot} 未找到")
        fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True); fig.suptitle(f"智能体 {agent_id_to_plot} 详细状态演化")
        time_axis_states = self._get_time_axis(num_points=len(agent.history_states)); states_arr = np.array(agent.history_states)
        if states_arr.ndim == 2 and states_arr.shape[0] == len(time_axis_states) and states_arr.shape[0] > 0:
            axs[0].plot(time_axis_states, states_arr[:, 0], label="X 位置", color=self.colors[0], linestyle=self.linestyles[0]); axs[0].plot(time_axis_states, states_arr[:, 1], label="Y 位置", color=self.colors[1], linestyle=self.linestyles[1]); axs[0].set_ylabel("位置 (m)"); axs[0].legend(loc='upper right'); ax0_twin = axs[0].twinx()
            ax0_twin.plot(time_axis_states, states_arr[:, 2], label="$V_x$", color=self.colors[2], linestyle=self.linestyles[2]); ax0_twin.plot(time_axis_states, states_arr[:, 3], label="$V_y$", color=self.colors[3], linestyle=self.linestyles[3]); ax0_twin.set_ylabel("速度 (m/s)"); ax0_twin.legend(loc='lower right')
        else: axs[0].text(0.5,0.5, "无位置/速度数据", ha='center',va='center',transform=axs[0].transAxes)
        axs[0].set_title("位置与速度")
        if agent.history_controls:
            controls_arr = np.array(agent.history_controls); time_axis_ctrl = np.arange(1, len(controls_arr) + 1) * self.env.dt if len(controls_arr) > 0 else np.array([])
            if controls_arr.ndim == 2 and controls_arr.shape[0] == len(time_axis_ctrl) and controls_arr.shape[0] > 0: axs[1].plot(time_axis_ctrl, controls_arr[:, 0], label="$u_x$", color=self.colors[0]); axs[1].plot(time_axis_ctrl, controls_arr[:, 1], label="$u_y$", color=self.colors[1])
            else: axs[1].text(0.5,0.5, "无有效控制数据", ha='center',va='center',transform=axs[1].transAxes)
        else: axs[1].text(0.5,0.5, "无控制历史数据", ha='center',va='center',transform=axs[1].transAxes)
        axs[1].set_ylabel("控制输入 ($m/s^2$)"); axs[1].legend(loc='best'); axs[1].set_title("控制输入 (加速度)")
        if agent.history_decision_mode:
            time_axis_dec = self._get_time_axis(num_points=len(agent.history_decision_mode)); modes_num = [{"strong":2,"transition":1,"weak":0, "unknown":-1}.get(m,-1) for m in agent.history_decision_mode]
            if len(modes_num) == len(time_axis_dec) and len(modes_num)>0: axs[2].plot(time_axis_dec, modes_num, marker='.', drawstyle='steps-post', color=self.colors[2 % len(self.colors)], label="决策模式"); axs[2].set_yticks([0,1,2]); axs[2].set_yticklabels(["弱","过渡","强"])
            else: axs[2].text(0.5,0.5, "无有效决策模式数据", ha='center',va='center',transform=axs[2].transAxes)
        else: axs[2].text(0.5,0.5, "无决策模式历史", ha='center',va='center',transform=axs[2].transAxes)
        axs[2].set_ylabel("决策模式"); axs[2].legend(loc='best'); axs[2].set_title("决策模式演化")
        if agent.history_csi_smooth:
            time_axis_csi = self._get_time_axis(num_points=len(agent.history_csi_smooth))
            if len(agent.history_csi_smooth) == len(time_axis_csi) and len(agent.history_csi_smooth) > 0:
                axs[3].plot(time_axis_csi, agent.history_csi_smooth, marker='.', color=self.colors[3 % len(self.colors)], label="CSI (平滑)"); adaptive_cfg = self.env.scenario_config.get("adaptive_controller_params",{}); axs[3].axhline(adaptive_cfg.get("csi_threshold_low",0.3),c='gray',ls=':',lw=1.2,label="低阈值"); axs[3].axhline(adaptive_cfg.get("csi_threshold_high",0.7),c='gray',ls=':',lw=1.2,label="高阈值"); axs[3].set_ylim(0, 1.05)
            else: axs[3].text(0.5,0.5, "无有效CSI数据", ha='center',va='center',transform=axs[3].transAxes)
        else: axs[3].text(0.5,0.5, "无CSI历史数据", ha='center',va='center',transform=axs[3].transAxes)
        axs[3].set_xlabel("仿真时间 (s)"); axs[3].set_ylabel("CSI值"); axs[3].legend(loc='best'); axs[3].set_title("通信状态指数 (CSI) 演化")
        plt.tight_layout(rect=[0, 0, 1, 0.96]); return fig

    def plot_task_status_gantt(self, figsize=(16, None)) -> Optional[plt.Figure]:
        if not self.env.history_task_status or not self.env.tasks: return self._empty_plot_message(figsize, "任务状态甘特图", "无任务数据或历史")
        current_task_ids = list(self.env.tasks.keys());
        if not current_task_ids: return self._empty_plot_message(figsize, "任务状态甘特图", "无任务ID")
        num_tasks = len(current_task_ids); fig_height = figsize[1] if figsize[1] is not None else max(6, num_tasks * 0.9 + 4); fig, ax = plt.subplots(figsize=(figsize[0], fig_height)); y_pos = {tid: i for i, tid in enumerate(reversed(current_task_ids))}
        cmap_gantt = {"active": self.colors[5 % len(self.colors)], "assigned": self.colors[4 % len(self.colors)], "completed": self.colors[2 % len(self.colors)]}; max_steps = 0
        for tid in current_task_ids: max_steps = max(max_steps, len(self.env.history_task_status.get(tid,[])))
        if max_steps == 0: return self._empty_plot_message(fig, "任务状态甘特图", "无任务状态历史点")
        legend_handles_gantt = {}
        for tid in current_task_ids:
            statuses_raw = self.env.history_task_status.get(tid, []);
            if not statuses_raw: continue
            statuses = list(statuses_raw);
            if len(statuses) < max_steps: statuses.extend([statuses[-1] if statuses else "active"] * (max_steps - len(statuses)))
            current_stat = None; start_step = 0
            for step, stat_val_raw in enumerate(statuses):
                stat_val = "active" if not stat_val_raw or stat_val_raw == "unknown" else stat_val_raw
                if stat_val != current_stat:
                    if current_stat is not None: bar = ax.barh(y_pos[tid], (step - start_step) * self.env.dt, left=start_step * self.env.dt, height=0.7, color=cmap_gantt.get(current_stat,'lightgrey'), edgecolor='black', alpha=0.9);
                    if current_stat not in legend_handles_gantt and current_stat is not None: legend_handles_gantt[current_stat] = bar
                    current_stat = stat_val; start_step = step
            if current_stat is not None and max_steps > start_step : bar = ax.barh(y_pos[tid], (max_steps - start_step) * self.env.dt, left=start_step * self.env.dt, height=0.7, color=cmap_gantt.get(current_stat,'lightgrey'), edgecolor='black', alpha=0.9)
            if current_stat not in legend_handles_gantt and current_stat is not None: legend_handles_gantt[current_stat] = bar
            comp_hist = self.env.history_task_completion_percentage.get(tid,[]);
            if comp_hist:
                text_interval = max(1, max_steps // 8 if max_steps > 0 else 1); last_shown_comp_val = -10; comp_hist_padded = list(comp_hist)
                if len(comp_hist_padded) < max_steps: comp_hist_padded.extend([comp_hist_padded[-1] if comp_hist_padded else 0] * (max_steps - len(comp_hist_padded)))
                for i_c in range(0, max_steps, text_interval):
                    c_val = comp_hist_padded[i_c]
                    if c_val > 0 and (abs(c_val - last_shown_comp_val) > 5 or c_val >=100 or i_c==0 or i_c + text_interval >= max_steps): ax.text(i_c*self.env.dt + 0.1*self.env.dt, y_pos[tid], f"{c_val:.0f}%", ha='left',va='center', fontsize=10, bbox=dict(fc='white',alpha=0.6,pad=0.1,ec='none')); last_shown_comp_val = c_val
                if len(comp_hist_padded) > 0 and comp_hist_padded[-1] >= 100 and last_shown_comp_val < 100 and (max_steps-1) % text_interval != 0 : ax.text((max_steps-1)*self.env.dt + 0.1*self.env.dt, y_pos[tid], f"{comp_hist_padded[-1]:.0f}%", ha='left',va='center', fontsize=10, bbox=dict(fc='white',alpha=0.6,pad=0.1,ec='none'))
        ax.set_xlabel("仿真时间 (s)"); ax.set_ylabel("任务"); ax.set_yticks(list(y_pos.values()))
        yt_labels = [f"{tid} [{self.env.tasks.get(tid,{}).get('true_type_name','?')} ({self.env.tasks.get(tid,{}).get('current_completion_percentage',0):.0f}%)]" for tid in reversed(current_task_ids)]
        ax.set_yticklabels(yt_labels, fontsize=11); ax.set_title("任务状态与完成进度甘特图")
        if legend_handles_gantt: ax.legend(legend_handles_gantt.values(), [k.capitalize() for k in legend_handles_gantt.keys()], loc='upper right', bbox_to_anchor=(1.0,1.18), ncol=len(legend_handles_gantt))
        ax.grid(axis='x', linestyle=':', alpha=0.7); ax.set_xlim(-0.05 * self.env.current_time, self.env.current_time * 1.05 if self.env.current_time > 0 else 1.0)
        plt.tight_layout(rect=[0.05,0.05,0.95,0.90]); return fig

    def plot_ksc_assignments_evolution(self, figsize=(16, None)) -> Optional[plt.Figure]:
        if not self.env.history_assignments_ksc or not self.agent_ids: return self._empty_plot_message(figsize, "KSC任务分配演化", "无分配数据或智能体")
        ksc_rounds = self._get_ksc_decision_rounds_axis();
        if len(ksc_rounds) == 0: return self._empty_plot_message(figsize, "KSC任务分配演化", "无KSC决策轮次")
        hist_assign = self.env.history_assignments_ksc; plot_len = min(len(ksc_rounds), len(hist_assign));
        if plot_len == 0: return self._empty_plot_message(figsize, "KSC任务分配演化", "历史数据长度为0")
        ksc_rounds, hist_assign = ksc_rounds[:plot_len], hist_assign[:plot_len]
        all_task_ids_set = set([IDLE_TASK_ID]); [all_task_ids_set.update(ad.values()) for ad in hist_assign]; current_env_task_ids = self.env.tasks.keys(); [all_task_ids_set.add(tid_env) for tid_env in current_env_task_ids]
        sorted_task_ids = [IDLE_TASK_ID] + sorted([tid for tid in all_task_ids_set if tid != IDLE_TASK_ID and tid in current_env_task_ids]); task_map = {tid: i for i, tid in enumerate(sorted_task_ids)}; num_map_tasks = len(sorted_task_ids)
        assign_matrix = np.full((self.num_agents, plot_len), task_map.get(IDLE_TASK_ID,0), dtype=int)
        for r_idx, assign_d in enumerate(hist_assign):
            for a_idx, aid in enumerate(self.agent_ids): assign_matrix[a_idx, r_idx] = task_map.get(assign_d.get(aid, IDLE_TASK_ID), task_map.get(IDLE_TASK_ID,0))
        fig_height = figsize[1] if figsize[1] is not None else max(7, self.num_agents * 0.9 + 4); fig, ax = plt.subplots(figsize=(figsize[0], fig_height))
        plot_colors = ['#D3D3D3'] + [self.colors[i % len(self.colors)] for i in range(num_map_tasks-1)]; cmap = mcolors.ListedColormap(plot_colors[:num_map_tasks]) if num_map_tasks > 0 else mcolors.ListedColormap(['#D3D3D3'])
        im = ax.imshow(assign_matrix, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0, vmax=max(0,num_map_tasks-1))
        xtick_step = max(1, plot_len // 12 if plot_len >0 else 1); ax.set_xticks(np.arange(0,plot_len,xtick_step)); ax.set_xticklabels(ksc_rounds[::xtick_step].astype(int)); ax.set_yticks(np.arange(self.num_agents)); ax.set_yticklabels(self.agent_ids, fontsize=11); ax.set_xlabel("KSC决策轮次"); ax.set_ylabel("智能体ID"); ax.set_title("KSC任务分配演化")
        if num_map_tasks > 0:
            cbar = fig.colorbar(im, ax=ax, ticks=np.arange(num_map_tasks), aspect=max(15, 3*num_map_tasks), pad=0.02); cbar_labs = ["未分配" if tid==IDLE_TASK_ID else f"{tid} ({self.env.tasks.get(tid,{}).get('true_type_name','?')})" for tid in sorted_task_ids]; cbar.set_ticklabels(cbar_labs); cbar.set_label("分配的任务", rotation=270, labelpad=25); cbar.ax.tick_params(labelsize=10)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"); ax.grid(which="minor", color="grey", linestyle=':', linewidth=0.5, alpha=0.3); ax.set_xticks(np.arange(-.5, plot_len, 1), minor=True); ax.set_yticks(np.arange(-.5, self.num_agents, 1), minor=True); plt.tight_layout(rect=[0,0.05,0.95,0.95]); return fig

    def plot_all_agents_csi_and_decision_modes(self, figsize=(12, None)) -> Optional[plt.Figure]:
        if not self.agent_ids: return self._empty_plot_message(figsize, "所有智能体的CSI与决策模式演化", "无智能体数据")
        num_plot_agents = len(self.agent_ids); fig_height = figsize[1] if figsize[1] is not None else max(5, num_plot_agents * 3.5); fig, axs = plt.subplots(num_plot_agents, 1, figsize=(figsize[0],fig_height), sharex=True, squeeze=False); axs_flat = axs.flatten(); time_ax = self._get_time_axis(num_points=self.env.time_step + 1)
        any_agent_plotted = False
        for i, aid in enumerate(self.agent_ids):
            if i >= len(axs_flat): break
            agent_obj = self.env.agents.get(aid); ax_csi_current = axs_flat[i]; ax_mode_current = ax_csi_current.twinx()
            if not agent_obj: ax_csi_current.text(0.5,0.5, f"智能体 {aid} 数据缺失", ha='center',va='center',transform=ax_csi_current.transAxes); ax_csi_current.set_title(f"智能体 {aid}: CSI 与决策模式"); continue
            csi_hist = agent_obj.history_csi_smooth; mode_hist = agent_obj.history_decision_mode
            csi_plot_data = np.pad(csi_hist, (0, len(time_ax) - len(csi_hist)), 'edge') if csi_hist and len(csi_hist) < len(time_ax) else csi_hist
            mode_plot_data_raw = np.pad(mode_hist, (0, len(time_ax) - len(mode_hist)), 'edge') if mode_hist and len(mode_hist) < len(time_ax) else mode_hist
            mode_map_plot = {"strong":2, "transition":1, "weak":0, "unknown":-1}; modes_numeric_plot = [mode_map_plot.get(m, -1) for m in mode_plot_data_raw] if mode_plot_data_raw else []
            lns_csi, lns_mode, lns_thresh = [], [], []
            if csi_plot_data and len(csi_plot_data) == len(time_ax) and len(csi_plot_data)>0:
                lns_csi = ax_csi_current.plot(time_ax, csi_plot_data, color=self.colors[0], label="CSI (平滑)"); adaptive_params = self.env.scenario_config.get("adaptive_controller_params",{}); lns_thresh.append(ax_csi_current.axhline(adaptive_params.get("csi_threshold_low",0.3),c='darkgray',ls=':',lw=1.5,label="低阈值")); lns_thresh.append(ax_csi_current.axhline(adaptive_params.get("csi_threshold_high",0.7),c='dimgray',ls=':',lw=1.5,label="高阈值"))
            else: ax_csi_current.text(0.5,0.4,"无CSI数据",ha='center',va='center',transform=ax_csi_current.transAxes,color=self.colors[0])
            ax_csi_current.set_ylabel("CSI值", color=self.colors[0]); ax_csi_current.tick_params(axis='y',labelcolor=self.colors[0]); ax_csi_current.set_ylim(-0.05,1.05)
            if modes_numeric_plot and len(modes_numeric_plot) == len(time_ax) and len(modes_numeric_plot)>0: lns_mode = ax_mode_current.plot(time_ax, modes_numeric_plot, color=self.colors[1 % len(self.colors)], linestyle='--', drawstyle='steps-post', label="决策模式"); ax_mode_current.set_yticks([0,1,2]); ax_mode_current.set_yticklabels(["弱","过渡","强"])
            else: ax_mode_current.text(0.5,0.6,"无决策模式数据",ha='center',va='center',transform=ax_mode_current.transAxes,color=self.colors[1 % len(self.colors)])
            ax_mode_current.set_ylabel("决策模式", color=self.colors[1 % len(self.colors)]); ax_mode_current.tick_params(axis='y',labelcolor=self.colors[1 % len(self.colors)]); ax_mode_current.set_ylim(-0.5,2.5); ax_csi_current.set_title(f"智能体 {aid}");
            all_lns_current = lns_csi + lns_mode + lns_thresh
            if all_lns_current: ax_csi_current.legend(all_lns_current, [l.get_label() for l in all_lns_current], loc='center left', bbox_to_anchor=(1.05,0.5))
            any_agent_plotted = True
        if not any_agent_plotted and num_plot_agents > 0: axs_flat[0].text(0.5,0.5,"所有智能体数据均缺失",ha='center',va='center',transform=axs_flat[0].transAxes)
        if num_plot_agents > 0: axs_flat[num_plot_agents-1].set_xlabel("仿真时间 (s)")
        fig.suptitle(f"所有智能体的CSI与决策模式演化 ({self.env.scenario_config.get('scenario_display_name','未知')})"); plt.tight_layout(rect=[0,0.03,0.85,0.95]); return fig

    def plot_aif_efe_history(self, agent_id_to_plot: str, figsize=(10, 6)) -> Optional[plt.Figure]:
        agent = self.env.agents.get(agent_id_to_plot);
        if not agent: return self._empty_plot_message(figsize, f"智能体 {agent_id_to_plot} EFE演化", f"智能体 {agent_id_to_plot} 未找到")
        efe_history = None
        if hasattr(agent, 'adaptive_controller') and agent.adaptive_controller and hasattr(agent.adaptive_controller, 'active_inference_agent') and agent.adaptive_controller.active_inference_agent: efe_history = getattr(agent.adaptive_controller.active_inference_agent, 'history_selected_efe_values', None)
        if not efe_history or len(efe_history) == 0: return self._empty_plot_message(figsize, f"智能体 {agent_id_to_plot} EFE演化", f"智能体 {agent_id_to_plot} 无EFE历史数据")
        dt_aif_decision = self.env.sim_params.get("dt_decision_aif", self.env.dt); num_efe_points = len(efe_history); time_axis_efe = np.arange(num_efe_points) * dt_aif_decision
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(time_axis_efe, efe_history, marker='.', label="选择动作的EFE", color=self.colors[0]); ax.set_xlabel("仿真时间 (s) (AIF决策点)"); ax.set_ylabel("期望自由能 (EFE)"); ax.legend(loc='best'); ax.set_title(f"智能体 {agent_id_to_plot} AIF期望自由能演化"); fig.tight_layout(); return fig

    def plot_final_agent_positions_and_comm_links(self, comm_range_override: Optional[float] = None, ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
        if ax is None: fig, current_ax = plt.subplots(figsize=(10,10)); return_fig = fig
        else: current_ax = ax; return_fig = ax.get_figure()
        current_tasks_viz = self.env.tasks; task_handles, task_labels_legend = [], []
        if current_tasks_viz:
            for i, task_id in enumerate(current_tasks_viz.keys()):
                task_info = current_tasks_viz.get(task_id)
                if task_info and "position" in task_info:
                    pos = np.array(task_info["position"][:2]); status = task_info.get("status", "active"); completion = task_info.get("current_completion_percentage",0.0); is_done = status == 'completed' or completion >= 100.0; marker = 'P' if is_done else 'X'; color_idx = (self.num_agents + i) % len(self.colors); color = 'grey' if is_done else self.colors[color_idx]
                    label_text = f"任务 {task_id} ({'完成' if is_done else f'{completion:.0f}%'})"; scatter_h = current_ax.scatter(pos[0], pos[1], marker=marker, s=150, color=color, label=label_text, edgecolor='black', zorder=10)
                    if label_text not in task_labels_legend: task_handles.append(scatter_h); task_labels_legend.append(label_text)
                    current_ax.text(pos[0] + 1.0, pos[1] + 1.0, f"{task_id}", fontsize=10, zorder=11, color=color)
        final_agent_positions = {}; agent_handles, agent_labels_legend = [], []
        for i, agent_id in enumerate(self.agent_ids):
            agent_sim = self.env.agents.get(agent_id)
            if agent_sim and agent_sim.history_states and agent_sim.history_states[-1].ndim == 1 and len(agent_sim.history_states[-1]) >=2:
                final_pos = agent_sim.history_states[-1][:2]; final_agent_positions[agent_id] = final_pos; label_text_agent = f"智能体 {agent_id}"; scatter_h_agent = current_ax.scatter(final_pos[0], final_pos[1], marker=self.markers[i % len(self.markers)], s=120, color=self.colors[i % len(self.colors)], label=label_text_agent, edgecolor='black', zorder=5)
                if label_text_agent not in agent_labels_legend: agent_handles.append(scatter_h_agent); agent_labels_legend.append(label_text_agent)
                current_ax.text(final_pos[0] + 1.0, final_pos[1] - 1.0, agent_id, fontsize=10, zorder=6, color=self.colors[i%len(self.colors)])
        comm_range = comm_range_override if comm_range_override is not None else self.env.sim_params.get("communication_range")
        if isinstance(comm_range, (int,float)) and comm_range > 0:
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    id1, id2 = self.agent_ids[i], self.agent_ids[j]; pos1, pos2 = final_agent_positions.get(id1), final_agent_positions.get(id2)
                    if pos1 is not None and pos2 is not None and np.linalg.norm(pos1 - pos2) <= comm_range: current_ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], linestyle='--', color='dimgray', alpha=0.7, linewidth=1.5, zorder=1)
        elif comm_range is None: 
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    pos1, pos2 = final_agent_positions.get(self.agent_ids[i]), final_agent_positions.get(self.agent_ids[j])
                    if pos1 is not None and pos2 is not None: current_ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], linestyle=':', color='lightgray', alpha=0.6, linewidth=1.0, zorder=1)
        current_ax.set_xlabel("X 位置 (m)"); current_ax.set_ylabel("Y 位置 (m)"); current_ax.set_title(f"系统最终布局与通信图 (t={self.env.current_time:.1f}s)")
        all_legend_handles = agent_handles + task_handles; all_legend_labels = agent_labels_legend + task_labels_legend; unique_hl_dict = {lbl:h for h,lbl in reversed(list(zip(all_legend_handles,all_legend_labels)))}
        if unique_hl_dict: current_ax.legend(unique_hl_dict.values(), unique_hl_dict.keys(), loc='upper left', bbox_to_anchor=(1.02,1.0))
        current_ax.axis('equal');
        if ax is None: plt.tight_layout(rect=[0,0,0.8,1]); return return_fig
        return None

    def save_plot(self, fig: plt.Figure, filename_prefix: str, plot_name: str, results_dir_override: Optional[str]=None):
        results_path = Path(results_dir_override or getattr(self.env, 'results_dir', Path("results/plots")))
        results_path.mkdir(parents=True, exist_ok=True)
        base_fn = f"{filename_prefix}_{plot_name}"
        for fmt in ['png', 'pdf']:
            fn = results_path / f"{base_fn}.{fmt}"
            try:
                fig.savefig(fn)
                print(f"绘图日志: 已保存图像到 {fn}")
            except Exception as e: print(f"绘图错误: 保存 {fn} 失败: {e}")
        plt.close(fig)

    def show_plots(self): plt.show()


if __name__ == '__main__':
    print("Visualizer 模块自测试 (需要模拟环境或加载数据)。")
    
    class MockAgent:
        def __init__(self, id_str):
            self.id = id_str
            self.history_states = [np.array([0,0,0,0]), np.array([1,1,0,0]), np.array([2,0,0,0])]
            self.history_controls = [np.array([0.1,0]), np.array([0,-0.1])]
            self.history_decision_mode = ["strong", "strong", "transition"]
            self.history_csi_smooth = [0.8, 0.85, 0.6]
            self.history_delta_s_planned = [np.array([1,1,0,0]), np.array([1,-1,0,0]), np.array([-1,-1,0,0])]
            self.adaptive_controller = type('MockAC', (object,), {
                'active_inference_agent': type('MockAIA', (object,), {'history_selected_efe_values': [10.5, 9.2, 8.7]})
            })()

    class MockEnv(SimulationEnvironment):
        def __init__(self):
            self.scenario_config = {
                "scenario_name_loaded_from": "mock_scenario_viz",
                "scenario_display_name": "可视化测试场景",
                "tasks": {
                    "T01": {"position": np.array([15,10]), "true_type_name": "TypeA", "is_completed": False, "current_completion_percentage": 20.0, "status": "active", "initial_state": [15,10,0,0]},
                    "T02": {"position": np.array([-10,15]), "true_type_name": "TypeB", "is_completed": True, "current_completion_percentage": 100.0, "status": "completed", "initial_state":[-10,15,0,0]},
                    "T03": {"position": np.array([0, -20]), "true_type_name": "TypeA", "is_completed": False, "current_completion_percentage": 0.0, "status": "active", "initial_state": [0,-20,0,0]}
                },
                "task_type_names": ["TypeA", "TypeB"],
                "spacecrafts": [{"id": "A1", "initial_state": [0,0,0,0]}, {"id": "A2", "initial_state": [0,1,0,0]}],
                "simulation_params": {"dt_simulation": 0.1, "communication_range": 20.0, "dt_decision_aif": 0.1},
                "adaptive_controller_params": {"csi_threshold_low": 0.3, "csi_threshold_high": 0.7},
                "ksc_algorithm_config": {},
                "visualization_settings": { 
                    'plot_agent_trajectories_2d': { 'show_all_tasks_in_trajectory_plot': True }
                }
            }
            self.agent_ids = ["A1", "A2"]
            self.num_agents = 2
            self.tasks = copy.deepcopy(self.scenario_config["tasks"])
            self.task_type_names = self.scenario_config["task_type_names"]
            self.dt = self.scenario_config["simulation_params"]["dt_simulation"]
            self.time_step = 2 
            self.current_time = self.time_step * self.dt
            self.ksc_decision_round_count = 2
            
            self.agents = {"A1": MockAgent("A1"), "A2": MockAgent("A2")}
            self.agents["A2"].history_states = [np.array([0,1,0,0]), np.array([-1,2,0,0]), np.array([-2,1,0,0])]
            
            self.history_agent_states = {aid: ag.history_states for aid, ag in self.agents.items()}
            self.history_global_utility_ksc = [10.0, 12.5]
            self.history_beliefs_at_ksc_step = {
                1: {"A1": {"T01": np.array([0.8, 0.2]), "T02": np.array([0.1, 0.9])}},
                2: {"A1": {"T01": np.array([0.9, 0.1]), "T02": np.array([0.05, 0.95])}}
            }
            self.belief_manager_ksc = True 
            self.ksc_stats_messages_broadcasted = [5, 4]
            self.ksc_stats_messages_processed = [10, 8]
            self.ksc_stats_active_links = [1, 1]
            self.history_num_active_tasks = [3, 2, 2]
            self.env_level_active_comm_links = [1, 1, 1]
            self.history_task_status = {"T01": ["active", "active", "active"], "T02": ["active", "completed", "completed"], "T03": ["active", "active", "active"]}
            self.history_task_completion_percentage = {"T01": [0, 10, 20], "T02": [0, 100, 100], "T03": [0,0,0]}
            self.history_assignments_ksc = [{ "A1": "T01", "A2": "T02"}, { "A1": "T01", "A2": IDLE_TASK_ID}]
            self.ksc_algorithm = type('MockKSC', (object,), {'_sim_best_assignment_final': {"A1": "T01"}})()
            self.sim_params = self.scenario_config["simulation_params"]
            self.results_dir = Path("results/test_plots_output_final_v3_all_methods")


    print("创建模拟环境进行可视化测试...")
    mock_env = MockEnv()
    visualizer = Visualizer(mock_env)

    plot_functions_to_test = {
        "trajectories_with_aif_targets_all_tasks": lambda: visualizer.plot_agent_trajectories_2d(show_communication_range_final_agent_id="A1", show_aif_target_points=True, aif_target_plot_interval=1),
        "velocities": visualizer.plot_all_agents_velocities,
        "accelerations": visualizer.plot_all_agents_accelerations,
        "ksc_utility": visualizer.plot_ksc_global_utility,
        "belief_convergence": lambda: visualizer.plot_belief_convergence(tracked_agents_tasks={"A1": ["T01"]}),
        "task_gantt": visualizer.plot_task_status_gantt,
        "csi_modes": visualizer.plot_all_agents_csi_and_decision_modes,
        "aif_efe_A1": lambda: visualizer.plot_aif_efe_history(agent_id_to_plot="A1"),
        "final_layout": visualizer.plot_final_agent_positions_and_comm_links,
        "overall_performance": visualizer.plot_overall_performance,
        "ksc_assignments": visualizer.plot_ksc_assignments_evolution,
        "agent_A1_details": lambda: visualizer.plot_single_agent_details(agent_id_to_plot="A1")
    }

    for plot_name, func in plot_functions_to_test.items():
        print(f"测试 {plot_name} 图...")
        try:
            fig_test = func()
            if fig_test:
                visualizer.save_plot(fig_test, "test_run_all", plot_name)
            else:
                print(f"  {plot_name} 未生成图像对象 (可能因为数据不足或内部逻辑返回None)。")
        except Exception as e_plot_test:
            print(f"  !!! 测试 {plot_name} 图时发生错误: {e_plot_test}")
            import traceback
            traceback.print_exc()


    print(f"可视化测试完毕。请检查 '{mock_env.results_dir}' 目录下的图像。")
    # visualizer.show_plots()
