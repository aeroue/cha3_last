# simulation/run_simulation.py
import numpy as np
import copy
import pickle
import json
import os
import sys
import time
import random
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import platform

# Add at the beginning of your script
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # or 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

def setup_matplotlib_fonts():
    """Configure matplotlib to properly handle Chinese characters"""
    
    # Define a list of Chinese fonts to try, in order of preference
    if platform.system() == 'Windows':
        chinese_fonts = [
            'Microsoft YaHei',
            'SimHei',
            'SimSun',
            'DengXian',
            'KaiTi'
        ]
    else:  # Linux/Mac
        chinese_fonts = [
            'Noto Sans CJK SC',
            'WenQuanYi Micro Hei',
            'Hiragino Sans GB',
            'Arial Unicode MS'
        ]

    # Try to find a suitable font
    font_found = False
    for font_name in chinese_fonts:
        try:
            font_path = fm.findfont(FontProperties(family=font_name))
            if font_path:
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                print(f"Using font: {font_name}")
                font_found = True
                break
        except:
            continue

    if not font_found:
        print("Warning: No suitable Chinese font found. Text may not display correctly.")
        
    # Additional matplotlib configurations
    plt.rcParams['axes.unicode_minus'] = False  # Correct minus sign display
    plt.rcParams['font.size'] = 12  # Set default font size
    
    return font_found

# Call this function early in your script
setup_matplotlib_fonts()

# Modify your visualization code to use fallback text if needed
def fallback_text(text):
    """Convert Chinese text to ASCII if font rendering fails"""
    try:
        # Try to render the text normally
        return text
    except:
        # Fall back to ASCII representation
        return text.encode('ascii', 'replace').decode('ascii')

# In your plotting functions, wrap text with fallback_text():
def create_plot(title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.set_title(fallback_text(title))
    ax.set_xlabel(fallback_text(xlabel))
    ax.set_ylabel(fallback_text(ylabel))
    return fig, ax

# --- 导入路径管理 (与您要求一致) ---
try:
    # 尝试相对导入
    from .environment import SimulationEnvironment
    from .visualizer import Visualizer # <--- 取消注释 Visualizer
    from .scenarios import load_scenario # 用于可能的参数覆盖或检查
except ImportError:
    # 如果相对导入失败，则尝试绝对导入
    _current_file_directory_run = os.path.dirname(os.path.abspath(__file__))
    _project_root_directory_run = os.path.dirname(_current_file_directory_run)
    if _project_root_directory_run not in sys.path:
        sys.path.insert(0, _project_root_directory_run)

    from simulation.environment import SimulationEnvironment
    from simulation.visualizer import Visualizer # <--- 取消注释 Visualizer
    from simulation.scenarios import load_scenario

class NumpyJSONEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，用于处理包含NumPy数组的字典。
    将NumPy数组转换为列表，以便JSON序列化。
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)): # 处理 numpy.bool_ 和 Python bool
            return bool(obj)
        if isinstance(obj, (set, tuple)): # 将集合和元组也转为列表
            return list(obj)
        return json.JSONEncoder.default(self, obj)

def run_single_simulation(scenario_name: str,
                          max_steps_override: Optional[int] = None,
                          random_seed: Optional[int] = None,
                          results_dir_base: str = "results",
                          enable_visualization: bool = True,
                          show_plots: bool = False,
                          save_full_history_pickle: bool = True,
                          save_summary_json: bool = True,
                          custom_suffix: Optional[str] = None
                         ) -> Optional[Dict[str, Any]]:
    """
    运行单个仿真实验。

    参数:
        scenario_name (str): 要运行的场景名称。
        max_steps_override (Optional[int]): 如果提供，则覆盖场景中定义的最大仿真步数。
        random_seed (Optional[int]): 如果提供，则设置随机种子以保证可复现性。
        results_dir_base (str): 存放所有仿真结果的基础目录。
        enable_visualization (bool): 是否生成并保存可视化图表。
        show_plots (bool): 是否在生成图表后尝试显示它们 (plt.show())。
        save_full_history_pickle (bool): 是否将完整的仿真环境历史数据保存为pickle文件。
        save_summary_json (bool): 是否将仿真结果的摘要保存为JSON文件。
        custom_suffix (Optional[str]): 为本次运行的结果目录和文件名添加自定义后缀。

    返回:
        Optional[Dict[str, Any]]: 包含仿真结果和历史数据的字典，如果仿真成功运行。
                                    如果场景加载失败或发生其他严重错误，则返回None。
    """
    start_time_total_run = time.time()

    # 1. 设置随机种子
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        print(f"随机种子已设置为: {random_seed}")
    else:
        # 如果没有提供种子，但希望记录所用的种子，可以生成一个
        random_seed = random.randint(0, 2**32 - 1)
        np.random.seed(random_seed)
        random.seed(random_seed)
        print(f"使用随机生成的种子: {random_seed} (用于复现)")


    # 2. 创建结果目录
    # 使用场景名和时间戳（或自定义后缀）创建唯一的子目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_specific_suffix = f"_{custom_suffix}" if custom_suffix else f"_{timestamp}"
    scenario_results_dir = os.path.join(results_dir_base, f"{scenario_name}{run_specific_suffix}")

    # 检查基础结果目录是否存在，如果不存在则创建
    if not os.path.exists(results_dir_base):
        try:
            os.makedirs(results_dir_base)
            print(f"已创建基础结果目录: {results_dir_base}")
        except OSError as e:
            print(f"错误: 无法创建基础结果目录 '{results_dir_base}'. OS错误: {e}")
            return None # 无法保存结果，提前退出

    # 创建本次运行的特定子目录
    if not os.path.exists(scenario_results_dir):
        try:
            os.makedirs(scenario_results_dir)
            print(f"结果将保存在: {scenario_results_dir}")
        except OSError as e:
            print(f"错误: 无法创建场景结果目录 '{scenario_results_dir}'. OS错误: {e}")
            return None # 无法保存结果，提前退出
    else: # 如果目录已存在 (例如，使用固定的custom_suffix)，则打印提示
        print(f"注意: 结果目录 '{scenario_results_dir}' 已存在，结果将被覆盖或追加。")


    # 3. 初始化仿真环境
    env = None
    try:
        env = SimulationEnvironment(scenario_name)
    except ValueError as e_val:
        print(f"错误: 初始化仿真环境失败 - {e_val}")
        return None
    except Exception as e_init: # 捕获其他可能的初始化错误
        print(f"错误: 初始化仿真环境时发生未知错误: {e_init}")
        import traceback
        traceback.print_exc()
        return None

    # 覆盖最大仿真步数 (如果提供)
    if max_steps_override is not None:
        env.sim_params["max_time_steps"] = max_steps_override
        print(f"最大仿真步数已覆盖为: {max_steps_override}")

    # 4. 运行仿真循环
    print(f"\n--- 开始仿真: {scenario_name} (共 {env.sim_params.get('max_time_steps')} 步) ---")
    simulation_run_start_time = time.time()

    progress_print_interval = max(1, env.sim_params.get('max_time_steps', 100) // 20) # 大约打印20次进度

    # 从 ksc_coalition_formation 导入 IDLE_TASK_ID，如果之前没有导入的话
    try:
        from ..strong_communication.ksc_coalition_formation import IDLE_TASK_ID
    except ImportError:
        from strong_communication.ksc_coalition_formation import IDLE_TASK_ID


    for step_num in range(env.sim_params.get('max_time_steps', 100)):
        step_loop_start_time = time.time()
        should_continue = env.step()
        step_loop_end_time = time.time()

        if (step_num + 1) % progress_print_interval == 0 or not should_continue:
            ksc_util_latest = env.history_global_utility_ksc[-1] if env.history_global_utility_ksc else "N/A"
            num_active_tasks_latest = len(env.active_task_ids) if env.active_task_ids is not None else "N/A"

            # 安全地获取KSC分配以显示任务数量
            num_agents_assigned_ksc = 0
            if env.history_assignments_ksc and env.history_assignments_ksc[-1]:
                num_agents_assigned_ksc = sum(1 for task_id in env.history_assignments_ksc[-1].values() if task_id != IDLE_TASK_ID)

            print(f"  SimStep {env.time_step:4d}/{env.sim_params.get('max_time_steps')}, "
                  f"Time: {env.current_time:7.1f}s, "
                  f"KSC Util: {ksc_util_latest if isinstance(ksc_util_latest, str) else ksc_util_latest:.1f}, "
                  f"Active Tasks: {num_active_tasks_latest}, "
                  f"Assigned (KSC): {num_agents_assigned_ksc}, "
                  f"LoopTime: {(step_loop_end_time - step_loop_start_time)*1000:.1f}ms")

        if not should_continue:
            break

    simulation_run_end_time = time.time()
    print(f"--- 仿真结束 (共 {env.time_step} 物理步, {env.ksc_decision_round_count} KSC决策轮) ---")
    print(f"仿真运行耗时: {simulation_run_end_time - simulation_run_start_time:.2f} 秒")

    # 5. 准备并保存结果
    results_to_save = {
        "scenario_name": scenario_name,
        "scenario_config_used": copy.deepcopy(env.scenario_config), # 保存实际使用的场景配置
        "random_seed_used": random_seed,
        "simulation_params_final": copy.deepcopy(env.sim_params),
        "total_simulation_steps_run": env.time_step,
        "total_ksc_decision_rounds": env.ksc_decision_round_count,
        "final_simulation_time_s": env.current_time,
        "tasks_final_state": copy.deepcopy(env.tasks),
        "active_task_ids_final": copy.deepcopy(env.active_task_ids),
        # KSC历史
        "history_global_utility_ksc": copy.deepcopy(env.history_global_utility_ksc),
        "history_assignments_ksc": copy.deepcopy(env.history_assignments_ksc), # 列表，每个元素是该轮KSC后的分配字典
        "history_beliefs_at_ksc_step": copy.deepcopy(env.history_beliefs_at_ksc_step), # 嵌套字典
        "ksc_stats_messages_broadcasted": copy.deepcopy(env.ksc_stats_messages_broadcasted),
        "ksc_stats_messages_processed": copy.deepcopy(env.ksc_stats_messages_processed),
        "ksc_stats_active_links_ksc_graph": copy.deepcopy(env.ksc_stats_active_links),
        # 环境级历史
        "history_num_active_tasks_env": copy.deepcopy(env.history_num_active_tasks),
        "history_task_status_env": copy.deepcopy(env.history_task_status), # 字典，键是task_id, 值是状态列表
        "history_env_level_active_comm_links": copy.deepcopy(env.env_level_active_comm_links),
        # 各智能体的详细历史 (注意这会使文件非常大)
        "agents_history": {}
    }
    for agent_id, agent_obj in env.agents.items():
        results_to_save["agents_history"][agent_id] = {
            "id": agent_obj.id,
            "physical_params": copy.deepcopy(agent_obj.physical_params),
            "history_states": copy.deepcopy(agent_obj.history_states),
            "history_controls": copy.deepcopy(agent_obj.history_controls),
            "history_delta_s_planned": copy.deepcopy(agent_obj.history_delta_s_planned),
            "history_decision_mode": copy.deepcopy(agent_obj.history_decision_mode),
            "history_csi_smooth": copy.deepcopy(agent_obj.history_csi_smooth),
        }

    if save_full_history_pickle:
        pickle_filename = os.path.join(scenario_results_dir, f"full_results_{scenario_name}{run_specific_suffix}.pkl")
        try:
            with open(pickle_filename, "wb") as f_pickle:
                pickle.dump(results_to_save, f_pickle)
            print(f"完整仿真结果已保存到: {pickle_filename}")
        except Exception as e_pickle:
            print(f"错误: 保存pickle文件失败 - {e_pickle}")

    if save_summary_json:
        summary_results = {
            "scenario_name": scenario_name,
            "scenario_display_name": env.scenario_config.get("scenario_display_name", scenario_name),
            "random_seed_used": random_seed,
            "total_steps": env.time_step,
            "total_time_s": env.current_time,
            "final_ksc_utility": env.history_global_utility_ksc[-1] if env.history_global_utility_ksc else None,
            "num_tasks_completed": sum(1 for t_stat in env.tasks.values() if t_stat.get("status") == "completed"),
            "total_value_realized": sum(t_stat.get("value_realized", 0.0) for t_stat in env.tasks.values()),
            "average_ksc_messages_broadcasted_per_round": np.mean(env.ksc_stats_messages_broadcasted) if env.ksc_stats_messages_broadcasted else 0,
            "average_env_comm_links": np.mean(env.env_level_active_comm_links) if env.env_level_active_comm_links else 0,
            "final_ksc_assignment": env.history_assignments_ksc[-1] if env.history_assignments_ksc else None,
        }
        json_filename = os.path.join(scenario_results_dir, f"summary_results_{scenario_name}{run_specific_suffix}.json")
        try:
            with open(json_filename, "w", encoding='utf-8') as f_json:
                json.dump(summary_results, f_json, cls=NumpyJSONEncoder, indent=4, ensure_ascii=False)
            print(f"仿真摘要已保存到: {json_filename}")
        except Exception as e_json:
            print(f"错误: 保存JSON摘要文件失败 - {e_json}")
            import traceback
            traceback.print_exc()


    # 生成并保存可视化图表 (如果启用)
    if enable_visualization:
        print("\n--- 生成可视化图表 ---")
        
        # 确保已初始化中文字体
        setup_matplotlib_fonts()
        
        visualizer = Visualizer(env)

        # 1. 生成轨迹图
                # 修改后的代码：
        fig_traj = visualizer.plot_agent_trajectories_2d(
            show_final_assignment_ksc=env.scenario_config.get('visualization_settings', {}).get('plot_agent_trajectories_2d', {}).get('show_final_ksc_assignment_links', True),
            show_communication_range_final_agent_id=env.scenario_config.get('visualization_settings', {}).get('plot_agent_trajectories_2d', {}).get('show_communication_range_final_agent_id', None), # 根据visualizer方法签名调整
            plot_every_n_steps=env.scenario_config.get('visualization_settings', {}).get('plot_agent_trajectories_2d', {}).get('plot_every_n_steps', 1), # 根据visualizer方法签名调整
            show_aif_target_points=env.scenario_config.get('visualization_settings', {}).get('plot_agent_trajectories_2d', {}).get('show_aif_target_points', True), # 根据visualizer方法签名调整
            aif_target_plot_interval=env.scenario_config.get('visualization_settings', {}).get('plot_agent_trajectories_2d', {}).get('aif_target_plot_interval', 5) # 根据visualizer方法签名调整
        )
        if fig_traj:
            traj_filepath = os.path.join(scenario_results_dir, "trajectories.png")
            fig_traj.savefig(traj_filepath, dpi=300, bbox_inches='tight')
            print(f"轨迹图已保存至: {traj_filepath}")
            if show_plots:
                plt.figure(fig_traj.number)
                plt.show()
            else:
                plt.close(fig_traj)

        # 2. 生成信念收敛图
        tracked_beliefs_config = env.scenario_config.get("visualization_settings", {}).get("plot_belief_convergence_for", {})
        if tracked_beliefs_config:
            print("生成信念收敛图...")
            fig_belief = visualizer.plot_belief_convergence(tracked_beliefs_config)
            if fig_belief:
                belief_filepath = os.path.join(scenario_results_dir, "belief_convergence.png")
                fig_belief.savefig(belief_filepath, dpi=300, bbox_inches='tight')
                print(f"信念收敛图已保存至: {belief_filepath}")
                if show_plots:
                    plt.figure(fig_belief.number)
                    plt.show()
                else:
                    plt.close(fig_belief)

        # 3. 生成任务甘特图
        fig_gantt = visualizer.plot_task_status_gantt()
        if fig_gantt:
            gantt_filepath = os.path.join(scenario_results_dir, "task_gantt.png")
            fig_gantt.savefig(gantt_filepath, dpi=300, bbox_inches='tight')
            print(f"任务甘特图已保存至: {gantt_filepath}")
            if show_plots:
                plt.figure(fig_gantt.number)
                plt.show()
            else:
                plt.close(fig_gantt)

        # 4. 生成KSC任务分配演化图
        fig_assign = visualizer.plot_ksc_assignments_evolution()
        if fig_assign:
            assign_filepath = os.path.join(scenario_results_dir, "ksc_assignments_evolution.png")
            fig_assign.savefig(assign_filepath, dpi=300, bbox_inches='tight')
            print(f"KSC任务分配演化图已保存至: {assign_filepath}")
            if show_plots:
                plt.figure(fig_assign.number)
                plt.show()
            else:
                plt.close(fig_assign)
        
        # 5. 生成所有智能体速度图
        fig_vel = visualizer.plot_all_agents_velocities()
        if fig_vel:
            vel_filepath = os.path.join(scenario_results_dir, "velocities.png")
            fig_vel.savefig(vel_filepath, dpi=300, bbox_inches='tight')
            print(f"速度图已保存至: {vel_filepath}")
            if show_plots:
                plt.figure(fig_vel.number)
                plt.show()
            else:
                plt.close(fig_vel)
        
        # 6. 生成所有智能体加速度图
        fig_acc = visualizer.plot_all_agents_accelerations()
        if fig_acc:
            acc_filepath = os.path.join(scenario_results_dir, "accelerations.png")
            fig_acc.savefig(acc_filepath, dpi=300, bbox_inches='tight')
            print(f"加速度图已保存至: {acc_filepath}")
            if show_plots:
                plt.figure(fig_acc.number)
                plt.show()
            else:
                plt.close(fig_acc)
        
        # 7. 生成KSC全局效用图
        fig_utility = visualizer.plot_ksc_global_utility()
        if fig_utility:
            utility_filepath = os.path.join(scenario_results_dir, "ksc_global_utility.png")
            fig_utility.savefig(utility_filepath, dpi=300, bbox_inches='tight')
            print(f"KSC效用图已保存至: {utility_filepath}")
            if show_plots:
                plt.figure(fig_utility.number)
                plt.show()
            else:
                plt.close(fig_utility)
        
        # 8. 生成KSC通信指标图
        fig_comm = visualizer.plot_ksc_communication_metrics()
        if fig_comm:
            comm_filepath = os.path.join(scenario_results_dir, "ksc_communication_metrics.png")
            fig_comm.savefig(comm_filepath, dpi=300, bbox_inches='tight')
            print(f"KSC通信指标图已保存至: {comm_filepath}")
            if show_plots:
                plt.figure(fig_comm.number)
                plt.show()
            else:
                plt.close(fig_comm)
        
        # 9. 生成CSI和决策模式图
        if env.scenario_config.get("visualization_settings", {}).get("plot_all_agents_csi_modes", True):
            fig_csi = visualizer.plot_all_agents_csi_and_decision_modes()
            if fig_csi:
                csi_filepath = os.path.join(scenario_results_dir, "csi_decision_modes.png")
                fig_csi.savefig(csi_filepath, dpi=300, bbox_inches='tight')
                print(f"CSI和决策模式图已保存至: {csi_filepath}")
                if show_plots:
                    plt.figure(fig_csi.number)
                    plt.show()
                else:
                    plt.close(fig_csi)
        
        # 10. 生成总体性能图
        fig_perf = visualizer.plot_overall_performance()
        if fig_perf:
            perf_filepath = os.path.join(scenario_results_dir, "overall_performance.png")
            fig_perf.savefig(perf_filepath, dpi=300, bbox_inches='tight')
            print(f"总体性能图已保存至: {perf_filepath}")
            if show_plots:
                plt.figure(fig_perf.number)
                plt.show()
            else:
                plt.close(fig_perf)
                
        print("图表生成完成!")

    end_time_total_run = time.time()
    print(f"\n--- 单次仿真运行总结 ---")
    print(f"场景: {scenario_name}")
    print(f"总耗时: {end_time_total_run - start_time_total_run:.2f} 秒")
    print(f"结果目录: {scenario_results_dir}")

    return results_to_save

# --- 模块自测试 ---
if __name__ == '__main__':
    print("--- 测试 run_single_simulation 函数 ---")
    # 选择一个在scenarios.py中定义的场景进行测试
    # scenario_to_run = "strong_comms_basic_ksc"
    scenario_to_run = "dynamic_comms_adaptive_switching" # 测试更复杂的场景

    # 检查场景是否存在
    try:
        temp_scen_config = load_scenario(scenario_to_run)
        if temp_scen_config is None:
            print(f"错误: 场景 '{scenario_to_run}' 无法加载，无法进行测试。")
            exit()
    except Exception as e_load_test:
        print(f"错误: 加载场景 '{scenario_to_run}' 以进行测试时出错: {e_load_test}")
        exit()

    print(f"将运行测试场景: {scenario_to_run}")

    test_results_dir = os.path.join("results", "run_simulation_tests") # 单独的测试结果目录
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir, exist_ok=True)

    # 为了快速测试，可以覆盖最大步数
    test_max_steps = temp_scen_config.get("simulation_params",{}).get("max_time_steps", 50)
    # test_max_steps = 15 # 进一步减少测试步数

    results_data = run_single_simulation(
        scenario_name=scenario_to_run,
        max_steps_override=test_max_steps,
        random_seed=12345, # 固定种子以便调试
        results_dir_base=test_results_dir,
        enable_visualization=True, # 测试时启用可视化
        show_plots=False,          # 测试时通常不阻塞显示图像
        save_full_history_pickle=True,
        save_summary_json=True,
        custom_suffix="test_run_viz_enabled" # 修改后缀以区分
    )

    if results_data:
        print(f"\n--- run_single_simulation 测试成功完成 ---")
        print(f"  最终KSC效用: {results_data.get('history_global_utility_ksc', [None])[-1]}")
        completed_tasks_count = sum(1 for t_id, t_data in results_data.get('tasks_final_state', {}).items() if t_data.get('status') == 'completed')
        print(f"  完成的任务数量: {completed_tasks_count}")
        print(f"  结果保存在 '{os.path.join(test_results_dir, scenario_to_run + '_test_run_viz_enabled')}' 子目录中。")

        # 简单验证一下 agents_history 是否被填充
        if results_data.get("agents_history"):
            first_agent_id_hist = list(results_data["agents_history"].keys())[0]
            if results_data["agents_history"][first_agent_id_hist].get("history_states"):
                print(f"  智能体 {first_agent_id_hist} 的历史状态记录了 {len(results_data['agents_history'][first_agent_id_hist]['history_states'])} 个条目。")
            else:
                print(f"  警告: 智能体 {first_agent_id_hist} 的历史状态未被记录。")
        else:
            print("  警告: agents_history 未在结果中找到。")

    else:
        print(f"--- run_single_simulation 测试失败 ---")