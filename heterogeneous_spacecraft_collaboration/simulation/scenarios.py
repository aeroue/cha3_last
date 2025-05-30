# # simulation/scenarios.py - 修复版本
# import numpy as np
# import copy
# import random
# from typing import Dict, List, Any, Tuple, Optional
# import sys
# import os

# # --- 临时路径处理 ---
# _current_file_directory_scen = os.path.dirname(os.path.abspath(__file__))
# _project_root_directory_scen = os.path.dirname(_current_file_directory_scen)
# if _project_root_directory_scen not in sys.path:
#     sys.path.insert(0, _project_root_directory_scen)

# try:
#     from ..weak_communication.aif_functions_isobeliefs_convergent import generate_possible_goal_configs
# except ImportError:
#     from weak_communication.aif_functions_isobeliefs_convergent import generate_possible_goal_configs

# # --- 默认参数定义 ---
# DEFAULT_SC_PHYSICAL_PARAMS: Dict[str, Any] = {
#     "initial_mass_kg": 100.0,
#     "isp_s": 300.0,
#     "agent_radius": 0.5,
#     "capabilities": ["default_sensor"],
#     "type": 's',
#     "observation_accuracy_p_corr": 0.85  # 添加观测精度
# }

# DEFAULT_KSC_AGENT_PARAMS: Dict[str, Any] = {"k_value": 2}

# DEFAULT_AIF_GLOBAL_HYPERPARAMS: Dict[str, Any] = {
#     "reasoning_level": 1,  # 改为1以启用推理
#     "use_epistemic_planning": True,  # 启用认知规划
#     "alpha_epistemic": 1.0,
#     "beta_pragmatic": 1.0, 
#     "gamma_control_effort": 0.01, 
#     "policy_temperature": 0.1,
#     "belief_update_temperature": 1.0, 
#     "perspective_fusion_lambda": 0.5,
#     "obs_variance_pos": 0.5, 
#     "obs_variance_vel": 0.05,
#     "evidence_sensitivity_eta": 20.0, 
#     "vel_mag_thresh": 0.01,
#     "dist_thresh_evidence": 200.0, 
#     "evidence_norm_factor": 1.0, 
#     "EPS": 1e-9,
#     "C_target_dist_G": None, 
#     "utility_of_G_configs": None,
#     "angle_evidence_weight_factor": 100.0, 
#     "dist_evidence_weight": 1.0,
#     "safety_distance_sq": 25.0, 
#     "collision_penalty_h_factor": 100.0,
#     "initial_belief_alpha_ksc": 2.0  # 增加初始信念强度
# }

# DEFAULT_ADAPTIVE_PARAMS: Dict[str, Any] = {
#     "csi_threshold_low": 0.3, 
#     "csi_threshold_high": 0.7, 
#     "csi_ema_alpha": 0.2
# }

# DEFAULT_SPECTRAL_PARAMS: Dict[str, Any] = {
#     "horizon_H": 10, 
#     "alpha_scale": 25.0,
#     "u_max_for_norm": 0.1, 
#     "num_modes_to_consider": 4
# }

# DEFAULT_MPC_PARAMS: Dict[str, Any] = {
#     "prediction_horizon": 15,
#     "Q_terminal_diag": [100.0, 100.0, 10.0, 10.0],
#     "R_control_diag": [0.01, 0.01],
#     "u_max_abs": 0.2,
#     "prediction_horizon_dv_est": 10,
#     "Q_terminal_diag_dv_est": [50.0, 50.0, 1.0, 1.0],
#     "R_control_diag_dv_est": [0.001, 0.001],
#     "solver_print_level": 0  # 减少求解器输出
# }

# def create_task_definition(task_id_str: str,
#                           num_total_task_types: int,
#                           task_type_names_list: List[str],
#                           position_range: Tuple[float, float] = (-250.0, 250.0),
#                           reward_ranges_by_type: Optional[Dict[str, Tuple[float, float]]] = None,
#                           risk_ranges_by_type: Optional[Dict[str, Tuple[float, float]]] = None,
#                           override_true_type_name: Optional[str] = None,
#                           min_agents_needed_override: Optional[int] = None,
#                           max_agents_allowed_override: Optional[int] = None,
#                           required_capabilities_options: Optional[List[List[str]]] = None,
#                           total_workload_range: Tuple[float, float] = (80.0, 120.0),
#                           work_rate_per_agent_dt_range: Tuple[float, float] = (0.5, 1.5),
#                           existing_task_positions: List[np.ndarray] = None
#                           ) -> Dict[str, Any]:
    
#     if len(task_type_names_list) != num_total_task_types and num_total_task_types > 0:
#         task_type_names_list = [f"类型{i+1}" for i in range(num_total_task_types)]
#         print(f"警告: task_type_names_list 与 num_total_task_types 不匹配，已使用默认类型名。")

#     if override_true_type_name and override_true_type_name in task_type_names_list:
#         true_type_name = override_true_type_name
#         true_type_idx = task_type_names_list.index(true_type_name)
#     elif num_total_task_types > 0:
#         true_type_idx = random.randint(0, num_total_task_types - 1)
#         true_type_name = task_type_names_list[true_type_idx]
#     else:
#         true_type_idx = -1
#         true_type_name = "无特定类型"

#     # 设置任务奖励和风险
#     task_specific_type_rewards = {}
#     task_specific_type_risks = {}
#     default_reward_range = (800, 1200)
#     default_true_reward_range = (1500, 2500)
#     default_risk_range = (10, 30)
#     default_true_risk_range = (5, 15)

#     if num_total_task_types > 0:
#         for type_name_iter in task_type_names_list:
#             current_reward_range = (reward_ranges_by_type or {}).get(type_name_iter, default_reward_range)
#             current_risk_range = (risk_ranges_by_type or {}).get(type_name_iter, default_risk_range)
#             task_specific_type_rewards[type_name_iter] = random.uniform(current_reward_range[0], current_reward_range[1])
#             task_specific_type_risks[type_name_iter] = random.uniform(current_risk_range[0], current_risk_range[1])
        
#         if true_type_name != "无特定类型":
#             true_type_reward_range_final = (reward_ranges_by_type or {}).get(true_type_name, default_true_reward_range)
#             true_type_risk_range_final = (risk_ranges_by_type or {}).get(true_type_name, default_true_risk_range)
#             task_specific_type_rewards[true_type_name] = random.uniform(true_type_reward_range_final[0], true_type_reward_range_final[1])
#             task_specific_type_risks[true_type_name] = random.uniform(true_type_risk_range_final[0], true_type_risk_range_final[1])

#     # 设置任务能力需求
#     if required_capabilities_options and len(required_capabilities_options) > 0:
#         required_caps = random.choice(required_capabilities_options)
#     else:
#         required_caps = ["cap_sensor_dist"]

#     # 设置智能体需求
#     min_agents = min_agents_needed_override if min_agents_needed_override is not None else 1
#     max_agents = max_agents_allowed_override if max_agents_allowed_override is not None else min(2, num_total_task_types if num_total_task_types > 0 else 2)
#     max_agents = max(min_agents, max_agents)

#     # 生成任务位置，避免与现有任务太近
#     min_distance_between_tasks = 80.0  # 增大最小距离
#     position = np.random.uniform(position_range[0], position_range[1], 2)
    
#     if existing_task_positions:
#         max_attempts = 50  # 增加尝试次数
#         for _ in range(max_attempts):
#             too_close = False
#             for existing_pos in existing_task_positions:
#                 if np.linalg.norm(position - existing_pos) < min_distance_between_tasks:
#                     too_close = True
#                     break
            
#             if not too_close:
#                 break
                
#             position = np.random.uniform(position_range[0], position_range[1], 2)

#     # 创建任务定义
#     task_def = {
#         "id": task_id_str,
#         "position": position.tolist(),  # 确保是列表格式
#         "target_velocity": [0.0, 0.0],
#         "initial_state": position.tolist() + [0.0, 0.0],  # [x, y, vx, vy]
#         "current_state": position.tolist() + [0.0, 0.0],  # 确保current_state存在
#         "true_type_idx": true_type_idx,
#         "true_type_name": true_type_name,
#         "type_rewards": task_specific_type_rewards,
#         "type_risks": task_specific_type_risks,
#         "required_capabilities": required_caps,
#         "status": "active",
#         "assigned_agents": [],
#         "completion_time": -1.0,
#         "value_realized": 0.0,
#         "min_agents_needed": min_agents,
#         "max_agents_allowed": max_agents,
#         # 任务完成相关参数 - 调整为更慢完成
#         "total_workload": random.uniform(200.0, 300.0),  # 增加工作量
#         "current_completion_percentage": 0.0,
#         "work_rate_per_agent_per_dt": random.uniform(1.0, 2.0)  # 降低工作效率
#     }
    
#     return task_def

# def create_spacecraft_definition(sc_id_str: str,
#                                  agent_idx_for_placement: int,
#                                  total_num_agents_for_placement: int,
#                                  physical_base_config: Optional[Dict] = None, # 包含了 fuel_cost_coeff
#                                  ksc_base_config: Optional[Dict] = None,
#                                  aif_agent_type_options: Optional[List[str]] = None,
#                                  initial_pos_placement_radius: float = 200.0,
#                                  initial_velocity_range: Tuple[float, float] = (-0.1, 0.1)
#                                  ) -> Dict[str, Any]:
#     sc_physical_params = copy.deepcopy(DEFAULT_SC_PHYSICAL_PARAMS)
#     if physical_base_config:
#         sc_physical_params.update(physical_base_config) # physical_base_config 应包含 fuel_cost_coeff
    
#     # 确保 fuel_cost_coeff 有一个默认值，如果模板中没有提供
#     if 'fuel_cost_coeff' not in sc_physical_params:
#         sc_physical_params['fuel_cost_coeff'] = 1.0 # 或一个更合适的默认值
    
#     sc_ksc_params = copy.deepcopy(DEFAULT_KSC_AGENT_PARAMS)
#     if ksc_base_config:
#         sc_ksc_params.update(ksc_base_config)
    
#     if "type" not in sc_physical_params or not sc_physical_params["type"]:
#         if aif_agent_type_options and len(aif_agent_type_options) > 0:
#             sc_physical_params["type"] = random.choice(aif_agent_type_options)
#         else:
#             sc_physical_params["type"] = "s_default"
    
#     if "capabilities" not in sc_physical_params or not sc_physical_params["capabilities"]:
#         all_caps_options = ["cap_sensor_dist", "cap_sensor_angle", "cap_actuator_A", "cap_comm_relay"]
#         num_caps_to_select = random.randint(1, min(2, len(all_caps_options)))
#         sc_physical_params["capabilities"] = random.sample(all_caps_options, k=num_caps_to_select)
    
#     # 计算初始位置
#     angle = 2 * np.pi * agent_idx_for_placement / total_num_agents_for_placement if total_num_agents_for_placement > 0 else 0
#     radius = initial_pos_placement_radius * (0.6 + random.random() * 0.4)
#     initial_pos_x = radius * np.cos(angle)
#     initial_pos_y = radius * np.sin(angle)
#     initial_vel_x = random.uniform(initial_velocity_range[0], initial_velocity_range[1])
#     initial_vel_y = random.uniform(initial_velocity_range[0], initial_velocity_range[1])
#     initial_state = np.array([initial_pos_x, initial_pos_y, initial_vel_x, initial_vel_y])
    
#     sc_def = {
#         "id": sc_id_str,
#         "initial_state": initial_state.tolist(),
#         "physical_params": sc_physical_params,
#         "ksc_params": sc_ksc_params
#     }
    
#     return sc_def

# # --- 场景字典 ---
# SCENARIOS: Dict[str, Dict[str, Any]] = {}

# # 修复的调试场景
# SCENARIOS["debug_belief_display"] = {
#     "scenario_display_name": "信念显示调试场景 (5A-8T)",
#     "description": "5个航天器，8个任务，专门用于调试信念显示和任务显示问题。",
#     "num_agents": 5,
#     "num_tasks": 8,
#     "task_type_names": ["高价值科研", "普通巡查", "紧急维修"],
#     "aif_num_abstract_goals": 8,
#     "aif_allow_multiple_to_same_goal": True,

#     "simulation_params": {
#         "max_time_steps": 800,  # 增加仿真步数
#         "dt_simulation": 2.0,
#         "dt_decision_ksc": 6.0,  # 增加KSC决策间隔，让信念有时间积累
#         "dt_decision_aif": 4.0,
#         "communication_range": 300.0,  # 设置固定通信范围
#         "strong_mode_components_enabled": True,  # 确保启用强通信组件
#         "stop_if_all_tasks_done": False,  # 不要提前结束
#         "orbit_mean_motion_n": 0.0011,
#         "belief_update_interval_ksc_steps": 1,  # 每轮都更新信念
#         "task_completion_distance_threshold": 30.0,
#         "physical_state_update_interval_sim_steps": -1
#     },

#     "agent_definitions_template": {
#         "physical_params_base": {
#             "initial_mass_kg": 150.0,
#             "isp_s": 310.0,
#             "capabilities": ["cap_sensor_dist", "cap_actuator_A"],
#             "observation_accuracy_p_corr": 0.75  # 降低观测精度，让信念变化更明显
#         },
#         "ksc_params_base": {"k_value": 2},
#         "aif_agent_type_options": ['s', 'A', 'B'],
#         "initial_pos_placement_radius": 200.0,
#         "initial_velocity_range": (-0.03, 0.03)
#     },

#     "agent_specific_overrides": {
#         "SC1": {"physical_params": {"type": "s"}, "ksc_params": {"k_value": 3}},
#         "SC2": {"physical_params": {"type": "A"}, "ksc_params": {"k_value": 2}},
#         "SC3": {"physical_params": {"type": "B"}, "ksc_params": {"k_value": 2}},
#         "SC4": {"physical_params": {"type": "s"}, "ksc_params": {"k_value": 1}},
#         "SC5": {"physical_params": {"type": "A"}, "ksc_params": {"k_value": 2}}
#     },

#     "task_definitions_template": {
#         "position_range": (-300.0, 300.0),  # 稍微扩大范围
#         "reward_ranges_by_type": {
#             "高价值科研": (2500, 3500), 
#             "普通巡查": (800, 1200),
#             "紧急维修": (1500, 2000)
#         },
#         "risk_ranges_by_type": {
#             "高价值科研": (20, 40), 
#             "普通巡查": (5, 15),
#             "紧急维修": (10, 25)
#         },
#         "required_capabilities_options": [["cap_sensor_dist"], ["cap_actuator_A"], ["cap_sensor_dist", "cap_actuator_A"]],
#         "total_workload_range": (150.0, 250.0),  # 增加工作量让任务不会太快完成
#         "work_rate_per_agent_dt_range": (1.0, 2.0)  # 降低工作效率
#     },

#     # 手动指定任务类型，确保有多样性
#     "manual_task_true_types": {
#         "Task1": "高价值科研", 
#         "Task2": "普通巡查", 
#         "Task3": "高价值科研",
#         "Task4": "紧急维修", 
#         "Task5": "普通巡查", 
#         "Task6": "高价值科研",
#         "Task7": "紧急维修", 
#         "Task8": "普通巡查"
#     },

#     # 设置初始信念，确保有差异
#     "initial_belief_overrides": {
#         "SC1": {
#             "Task1": np.array([0.8, 0.1, 0.1]),  # 强烈认为是高价值科研
#             "Task2": np.array([0.1, 0.8, 0.1]),  # 强烈认为是普通巡查
#             "Task3": np.array([0.4, 0.4, 0.2])   # 不确定
#         },
#         "SC2": {
#             "Task1": np.array([0.2, 0.6, 0.2]),  # 与SC1的信念不同
#             "Task2": np.array([0.3, 0.3, 0.4]),  # 不确定
#             "Task4": np.array([0.1, 0.1, 0.8])   # 认为是紧急维修
#         },
#         "SC3": {
#             "Task1": np.array([0.5, 0.3, 0.2]),
#             "Task3": np.array([0.2, 0.2, 0.6]),
#             "Task5": np.array([0.1, 0.7, 0.2])
#         }
#     },

#     "adaptive_controller_params": {**copy.deepcopy(DEFAULT_ADAPTIVE_PARAMS)},
    
#     "aif_global_hyperparams": {
#         **copy.deepcopy(DEFAULT_AIF_GLOBAL_HYPERPARAMS),
#         "reasoning_level": 1,  # 启用一阶推理
#         "use_epistemic_planning": True,
#         "initial_belief_alpha_ksc": 2.0,
#         "belief_update_temperature": 0.5  # 让信念更新更敏感
#     },

#     "spectral_params": {**copy.deepcopy(DEFAULT_SPECTRAL_PARAMS), "alpha_scale": 20.0},
    
#     "mpc_params": {**copy.deepcopy(DEFAULT_MPC_PARAMS), 
#                    "R_control_diag": [0.008, 0.008], 
#                    "u_max_abs": 0.15,
#                    "solver_print_level": 0},

#     "ksc_algorithm_config": {
#         "max_dfs_branching_factor": 3,
#         "ksc_max_coalition_size_per_task": {
#             "高价值科研": 2, 
#             "普通巡查": 1, 
#             "紧急维修": 2
#         },
#         "ksc_min_agents_per_task": {
#             "高价值科研": 1, 
#             "紧急维修": 1
#         }
#     },

#     # 重要：确保可视化设置正确
#     "visualization_settings": {
#         "plot_overall_performance": True,
#         "plot_agent_trajectories": True,
#         "plot_single_agent_details_for_agent": "SC1",
#         "plot_task_gantt": True,
#         # 关键：指定要跟踪信念的智能体和任务
#         "plot_belief_convergence_for": {
#             "SC1": ["Task1", "Task2", "Task3"],
#             "SC2": ["Task1", "Task2", "Task4"], 
#             "SC3": ["Task1", "Task3", "Task5"]
#         },
#         "plot_ksc_assignments_evolution": True,
#         "plot_agent_velocities": True,
#         "plot_agent_accelerations": True,
#         "plot_ksc_communication_stats": True,
#         "plot_final_agent_locations_comm_graph": True,
#         "plot_all_agents_csi_modes": True,
#         "plot_task_completion_percentage": True
#     }
# }

# # 保持原有的强通信场景作为备选
# SCENARIOS: Dict[str, Dict[str, Any]] = {
#     "strong_comms_basic_ksc_completion_test": {
#         "scenario_display_name": "强通信KSC-完成度测试 (3A-6T)",
#         "description": "3个航天器，6个任务，测试KSC分配和任务完成。",
#         "num_agents": 3,
#         "num_tasks": 6, # 确保有足够任务供分配
#         "task_type_names": ["高价值科研", "普通巡查"],
#         "aif_num_abstract_goals": 6, 
#         "aif_allow_multiple_to_same_goal": True,

#         "simulation_params": {
#             "max_time_steps": 500, # 足够的仿真时间
#             "dt_simulation": 2.0,           
#             "dt_decision_ksc": 4.0, # KSC决策频率
#             "dt_decision_aif": 2.0, 
#             "communication_range": None, # 强通信
#             "strong_mode_components_enabled": True, 
#             "stop_if_all_tasks_done": True, 
#             "orbit_mean_motion_n": 0.0011,
#             "belief_update_interval_ksc_steps": 1,
#             "task_completion_check_interval_sim_steps": 1, 
#             "task_completion_distance_threshold": 50.0,   
#             "physical_state_update_interval_sim_steps": -1 # 无随机扰动，方便调试
#         },
#         "agent_definitions_template": { 
#             "physical_params_base": {
#                 "initial_mass_kg": 100.0, # 降低初始质量以减少燃料消耗
#                 "isp_s": 350.0,           # 提高比冲以减少燃料消耗
#                 "capabilities": ["cap_sensor_dist", "cap_actuator_A"],
#                 "observation_accuracy_p_corr": 0.95, # 提高观测准确率
#                 "work_efficiency_per_step": 1.0, 
#                 "fuel_cost_coeff": 5.0, # 降低燃料成本在效用中的权重 (重要!)
#             },
#             "ksc_params_base": {"k_value": 2}, # KSC的K值
#             "aif_agent_type_options": ['s', 'A', 'B'], 
#             "initial_pos_placement_radius": 100.0, # 智能体初始离原点近一些
#             "initial_velocity_range": (-0.01, 0.01) # 初始速度小一些
#         },
#         "task_definitions_template": { 
#             "position_range": (-150.0, 150.0), # 任务位置范围不要太大，减少Delta-V
#             "reward_ranges_by_type": { # 显著提高任务奖励
#                 "高价值科研": (5000, 7000), 
#                 "普通巡查": (2000, 3000)
#             },
#             "risk_ranges_by_type": { # 适当降低风险或保持不变
#                 "高价值科研": (10, 20), 
#                 "普通巡查": (2, 8)
#             },
#             "required_capabilities_options": [["cap_sensor_dist"], ["cap_actuator_A"]],
#             "total_workload_range": (100.0, 150.0), # 保持中等工作量，避免过快完成
#             "work_rate_per_agent_dt_range": (1.5, 2.5) 
#         },
#         "manual_task_true_types": { # 确保所有任务都有真实类型
#             "Task1": "高价值科研", "Task2": "普通巡查", "Task3": "高价值科研", 
#             "Task4": "普通巡查", "Task5": "高价值科研", "Task6": "普通巡查"
#         },
#         "initial_belief_overrides": { # 初始信念可以更乐观一些
#              "SC1": {"Task1": np.array([3.0, 0.5]), "Task2": np.array([0.5, 3.0])},
#              "SC2": {"Task1": np.array([2.5, 0.5]), "Task3": np.array([2.5, 0.5])},
#              "SC3": {"Task2": np.array([0.5, 2.5]), "Task4": np.array([0.5, 2.5])},
#         },
#         "adaptive_controller_params": {**copy.deepcopy(DEFAULT_ADAPTIVE_PARAMS)},
#         "aif_global_hyperparams": { 
#             **copy.deepcopy(DEFAULT_AIF_GLOBAL_HYPERPARAMS),
#             "reasoning_level": 0, "use_epistemic_planning": False,
#             "initial_belief_alpha_ksc": 1.0 # KSC信念管理器的初始alpha可以高一些，表示更强的初始均匀信念
#         },
#         "spectral_params": {**copy.deepcopy(DEFAULT_SPECTRAL_PARAMS), "alpha_scale": 25.0}, # 适中的谱分析尺度
#         "mpc_params": { # MPC参数调整以鼓励机动
#             **copy.deepcopy(DEFAULT_MPC_PARAMS), 
#             "R_control_diag": [0.001, 0.001], # 大幅降低控制代价权重
#             "Q_terminal_diag": [200.0, 200.0, 20.0, 20.0], # 提高终端状态权重
#             "u_max_abs": 0.25, # 允许稍大的加速度
#             "prediction_horizon_dv_est": 8, # DV估算时域可以短一些，如果R小
#             "R_control_diag_dv_est": [0.0005, 0.0005] # DV估算也用非常小的R
#         },
#         "ksc_algorithm_config": { 
#              "max_dfs_branching_factor": 2,
#              "ksc_max_coalition_size_per_task": {"高价值科研": 2, "普通巡查": 1},
#              "ksc_min_agents_per_task": {"高价值科研": 1} # 确保高价值任务至少有一个智能体
#         },
#         "visualization_settings": { # 确保追踪目标有效
#             "plot_overall_performance": True, "plot_agent_trajectories": True,
#             "plot_single_agent_details_for_agent": "SC1", "plot_task_gantt": True,
#             "plot_belief_convergence_for": {"SC1": ["Task1", "Task3"], "SC2": ["Task2", "Task4"]}, 
#             "plot_ksc_assignments_evolution": True,
#         }
#     }
# }

# def load_scenario(scenario_name: str) -> Optional[Dict[str, Any]]:
#     """
#     加载指定名称的场景配置
    
#     参数:
#         scenario_name (str): 场景名称
        
#     返回:
#         Optional[Dict[str, Any]]: 场景配置字典，如果场景不存在则返回None
#     """
#     if scenario_name not in SCENARIOS:
#         print(f"错误: 场景 '{scenario_name}' 未在 SCENARIOS 中定义。")
#         print(f"可用场景: {list(SCENARIOS.keys())}")
#         return None
    
#     config = copy.deepcopy(SCENARIOS[scenario_name])
#     config["scenario_name_loaded_from"] = scenario_name
    
#     # 获取基本配置信息
#     num_agents = config.get("num_agents", 0)
#     num_tasks = config.get("num_tasks", 0)
#     task_type_names = config.get("task_type_names", [])
#     num_task_types = len(task_type_names)
    
#     print(f"加载场景: {scenario_name}")
#     print(f"  智能体数量: {num_agents}")
#     print(f"  任务数量: {num_tasks}")
#     print(f"  任务类型: {task_type_names}")
    
#     # 创建航天器
#     config["spacecrafts"] = []
#     agent_template_config = config.get("agent_definitions_template", {})
#     template_physical_base = agent_template_config.get("physical_params_base")
#     template_ksc_base = agent_template_config.get("ksc_params_base")
#     aif_type_options = agent_template_config.get("aif_agent_type_options")
#     placement_radius = agent_template_config.get("initial_pos_placement_radius", 200.0)
#     velocity_range = agent_template_config.get("initial_velocity_range", (-0.1, 0.1))
    
#     for i in range(num_agents):
#         sc_id = f"SC{i+1}"
#         agent_overrides = config.get("agent_specific_overrides", {}).get(sc_id, {})
#         current_physical_params = copy.deepcopy(template_physical_base or {})
#         current_physical_params.update(agent_overrides.get("physical_params", {}))
#         current_ksc_params = copy.deepcopy(template_ksc_base or {})
#         current_ksc_params.update(agent_overrides.get("ksc_params", {}))
        
#         sc_def = create_spacecraft_definition(
#             sc_id_str=sc_id, agent_idx_for_placement=i, total_num_agents_for_placement=num_agents,
#             physical_base_config=current_physical_params, ksc_base_config=current_ksc_params,
#             aif_agent_type_options=aif_type_options, initial_pos_placement_radius=placement_radius,
#             initial_velocity_range=velocity_range
#         )
#         config["spacecrafts"].append(sc_def)
    
#     # 创建任务
#     tasks_dict_final = {}
    
#     if num_tasks > 0:
#         print(f"开始创建任务，数量: {num_tasks}")
        
#         task_template_config = config.get("task_definitions_template", {})
#         ksc_algo_specific_cfg = config.get("ksc_algorithm_config", {})
#         min_agents_constraints_per_task = ksc_algo_specific_cfg.get("ksc_min_agents_per_task", {})
#         max_agents_constraints_per_task = ksc_algo_specific_cfg.get("ksc_max_coalition_size_per_task", {})
        
#         existing_task_positions = []

#         for i in range(num_tasks):
#             task_id = f"Task{i+1}"
#             print(f"  创建任务: {task_id}")
            
#             task_true_type_override = config.get("manual_task_true_types", {}).get(task_id)
            
#             # 根据任务类型确定最小/最大智能体需求
#             if task_true_type_override:
#                 min_agents_override_val = min_agents_constraints_per_task.get(task_true_type_override)
#                 max_agents_override_val = max_agents_constraints_per_task.get(task_true_type_override)
#             else:
#                 min_agents_override_val = min_agents_constraints_per_task.get(task_id)
#                 max_agents_override_val = max_agents_constraints_per_task.get(task_id)
            
#             try:
#                 task_def = create_task_definition(
#                     task_id_str=task_id, 
#                     num_total_task_types=num_task_types, 
#                     task_type_names_list=task_type_names,
#                     position_range=task_template_config.get("position_range", (-300.0, 300.0)),
#                     reward_ranges_by_type=task_template_config.get("reward_ranges_by_type"),
#                     risk_ranges_by_type=task_template_config.get("risk_ranges_by_type"),
#                     override_true_type_name=task_true_type_override,
#                     min_agents_needed_override=min_agents_override_val,
#                     max_agents_allowed_override=max_agents_override_val,
#                     required_capabilities_options=task_template_config.get("required_capabilities_options"),
#                     total_workload_range=task_template_config.get("total_workload_range", (80.0,120.0)),
#                     work_rate_per_agent_dt_range=task_template_config.get("work_rate_per_agent_dt_range", (0.5,1.5)),
#                     existing_task_positions=existing_task_positions
#                 )
                
#                 tasks_dict_final[task_id] = task_def
                
#                 # 记录任务位置用于后续任务的位置生成
#                 if "position" in task_def:
#                     existing_task_positions.append(np.array(task_def["position"]))
#                     print(f"    任务 {task_id} 位置: {task_def['position']}, 类型: {task_def['true_type_name']}")
                    
#             except Exception as e:
#                 print(f"创建任务 {task_id} 时出错: {e}")
#                 continue
    
#     config["tasks"] = tasks_dict_final
#     print(f"任务创建完成，成功创建了 {len(tasks_dict_final)} 个任务")
    
#     # 配置AIF目标位置
#     num_aif_goals_from_config = config.get("aif_num_abstract_goals", 0)
#     if "aif_goal_positions" in config and isinstance(config["aif_goal_positions"], list):
#         config["aif_num_abstract_goals"] = len(config["aif_goal_positions"])
#     elif num_aif_goals_from_config > 0:
#         aif_goals_pos_list = []
#         ksc_task_positions = [t_def["position"] for t_def in config["tasks"].values() if "position" in t_def]
#         if len(ksc_task_positions) >= num_aif_goals_from_config:
#             aif_goals_pos_list = ksc_task_positions[:num_aif_goals_from_config]
#         else:
#             aif_goals_pos_list.extend(ksc_task_positions)
#             num_to_generate_randomly = num_aif_goals_from_config - len(aif_goals_pos_list)
#             for _ in range(num_to_generate_randomly):
#                 aif_goals_pos_list.append(np.random.uniform(-150.0, 150.0, 2))
#         config["aif_goal_positions"] = [np.array(p) for p in aif_goals_pos_list]
#     else:
#         config["aif_goal_positions"] = []
#         config["aif_num_abstract_goals"] = 0
    
#     # 设置AIF参数
#     final_aif_hyperparams = copy.deepcopy(DEFAULT_AIF_GLOBAL_HYPERPARAMS)
#     final_aif_hyperparams.update(config.get("aif_global_hyperparams", {}))
    
#     num_actual_aif_goals_for_g_config = len(config["aif_goal_positions"])
#     allow_multiple_to_aif_g = config.get("aif_allow_multiple_to_same_goal", False)
#     possible_G_configs_array = generate_possible_goal_configs(
#         num_agents, num_actual_aif_goals_for_g_config, allow_multiple_to_aif_g
#     )
#     num_total_g_configs = possible_G_configs_array.shape[0] if possible_G_configs_array.size > 0 else 0
    
#     if num_total_g_configs > 0:
#         c_target_from_scenario = final_aif_hyperparams.get("C_target_dist_G")
#         if not (c_target_from_scenario is not None and isinstance(c_target_from_scenario, np.ndarray) and len(c_target_from_scenario) == num_total_g_configs):
#             final_aif_hyperparams["C_target_dist_G"] = np.ones(num_total_g_configs) / num_total_g_configs
#         utility_g_from_scenario = final_aif_hyperparams.get("utility_of_G_configs")
#         if not (utility_g_from_scenario is not None and isinstance(utility_g_from_scenario, np.ndarray) and len(utility_g_from_scenario) == num_total_g_configs):
#             final_aif_hyperparams["utility_of_G_configs"] = np.zeros(num_total_g_configs)
#     else:
#         final_aif_hyperparams["C_target_dist_G"] = np.array([])
#         final_aif_hyperparams["utility_of_G_configs"] = np.array([])
    
#     config["aif_global_hyperparams"] = final_aif_hyperparams
    
#     # 设置其他参数
#     config["adaptive_controller_params"] = {**copy.deepcopy(DEFAULT_ADAPTIVE_PARAMS), **config.get("adaptive_controller_params", {})}
#     config["spectral_params"] = {**copy.deepcopy(DEFAULT_SPECTRAL_PARAMS), **config.get("spectral_params", {})}
#     config["mpc_params"] = {**copy.deepcopy(DEFAULT_MPC_PARAMS), **config.get("mpc_params", {})}
    
#     # 清理临时数据
#     config.pop("agent_definitions_template", None)
#     config.pop("task_definitions_template", None)
#     config.pop("agent_specific_overrides", None)
#     config.pop("manual_task_true_types", None)
#     config.pop("initial_belief_overrides", None)
    
#     print(f"场景 '{scenario_name}' 加载完成")
    
#     return config

# if __name__ == '__main__':
#     print("\n--- 测试修复后的场景加载 ---")
    
#     # 测试调试场景
#     scenario_name = "debug_belief_display"
#     print(f"尝试加载调试场景: {scenario_name}")
#     test_config = load_scenario(scenario_name)
    
#     if test_config:
#         print(f"✓ 场景加载成功")
#         print(f"  航天器数量: {len(test_config.get('spacecrafts', []))}")
#         print(f"  任务数量: {len(test_config.get('tasks', []))}")
#         print(f"  任务ID列表: {list(test_config.get('tasks', {}).keys())}")
#         print(f"  AIF目标数量: {len(test_config.get('aif_goal_positions', []))}")
        
#         # 检查可视化设置
#         viz_settings = test_config.get('visualization_settings', {})
#         belief_tracking = viz_settings.get('plot_belief_convergence_for', {})
#         print(f"  信念跟踪配置: {belief_tracking}")
        
#         # 检查任务位置
#         tasks = test_config.get('tasks', {})
#         for task_id, task_data in tasks.items():
#             pos = task_data.get('position', 'N/A')
#             task_type = task_data.get('true_type_name', 'N/A')
#             print(f"    {task_id}: 位置={pos}, 类型={task_type}")
#     else:
#         print(f"✗ 场景加载失败")
# simulation/scenarios.py
import numpy as np
import copy
import random
from typing import Dict, List, Any, Tuple, Optional
import sys
import os

# --- 统一的导入路径管理 ---
_scenarios_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir_scen = os.path.dirname(_scenarios_dir)
if _project_root_dir_scen not in sys.path:
    sys.path.insert(0, _project_root_dir_scen)
# --- 路径管理结束 ---

try:
    from weak_communication.aif_functions_isobeliefs_convergent import generate_possible_goal_configs
except ImportError as e:
    print(f"ERROR [simulation/scenarios.py]: Failed to import generate_possible_goal_configs: {e}")
    def generate_possible_goal_configs(num_agents, num_goals, allow_multiple):
        print("WARNING: Using dummy generate_possible_goal_configs.")
        return np.array([])

# --- 默认参数大幅调整以鼓励任务执行 ---
DEFAULT_SC_PHYSICAL_PARAMS: Dict[str, Any] = {
    "initial_mass_kg": 80.0,        # 更轻的智能体
    "isp_s": 400.0,                 # 更高效的引擎
    "agent_radius": 0.5,
    "capabilities": ["default_sensor"],
    "type": 's',
    "fuel_cost_coeff": 0.5,         # 大幅降低燃料成本在效用中的负面影响
    "observation_accuracy_p_corr": 0.95,
    "work_efficiency_per_step": 1.5,
}
DEFAULT_KSC_AGENT_PARAMS: Dict[str, Any] = {"k_value": 2}
DEFAULT_AIF_GLOBAL_HYPERPARAMS: Dict[str, Any] = {
    "reasoning_level": 0, "use_epistemic_planning": False, "alpha_epistemic": 1.0,
    "beta_pragmatic": 1.0, "gamma_control_effort": 0.005, "policy_temperature": 0.05, # 降低控制努力惩罚和策略温度
    "belief_update_temperature": 1.0, "perspective_fusion_lambda": 0.5,
    "obs_variance_pos": 0.1, "obs_variance_vel": 0.01, # 更精确的观测
    "evidence_sensitivity_eta": 35.0, "vel_mag_thresh": 0.005,
    "dist_thresh_evidence": 250.0, "evidence_norm_factor": 0.8, "EPS": 1e-9,
    "C_target_dist_G": None, "utility_of_G_configs": None,
    "angle_evidence_weight_factor": 100.0, "dist_evidence_weight": 1.0,
    "safety_distance_sq": 16.0, "collision_penalty_h_factor": 150.0,
    "initial_belief_alpha_ksc": 2.0 # 更强的初始均匀信念
}
DEFAULT_ADAPTIVE_PARAMS: Dict[str, Any] = {
    "csi_threshold_low": 0.3, "csi_threshold_high": 0.7, "csi_ema_alpha": 0.2
}
DEFAULT_SPECTRAL_PARAMS: Dict[str, Any] = {
    "horizon_H": 8, "alpha_scale": 15.0, # 减小尺度以产生更可达的偏差
    "u_max_for_norm": 0.2, "num_modes_to_consider": 4 # 与MPC u_max匹配
}
DEFAULT_MPC_PARAMS: Dict[str, Any] = {
    "prediction_horizon": 10, # 缩短预测时域，可能更快计算
    "Q_terminal_diag": [200.0, 200.0, 20.0, 20.0], # 强终端状态权重
    "R_control_diag": [0.0001, 0.0001],      # 极低控制代价权重，鼓励使用控制
    "u_max_abs": 0.3,                       # 允许更大的加速度
    "prediction_horizon_dv_est": 6,
    "Q_terminal_diag_dv_est": [100.0, 100.0, 10.0, 10.0],
    "R_control_diag_dv_est": [0.00005, 0.00005], # DeltaV估算也用极小的R
    "solver_print_level": 0
}

def create_task_definition(task_id_str: str,
                           num_total_task_types: int,
                           task_type_names_list: List[str],
                           position_range: Tuple[float, float] = (-100.0, 100.0), # 任务更集中
                           reward_ranges_by_type: Optional[Dict[str, Tuple[float, float]]] = None,
                           risk_ranges_by_type: Optional[Dict[str, Tuple[float, float]]] = None,
                           override_true_type_name: Optional[str] = None,
                           min_agents_needed_override: Optional[int] = None,
                           max_agents_allowed_override: Optional[int] = None,
                           required_capabilities_options: Optional[List[List[str]]] = None,
                           total_workload_range: Tuple[float, float] = (80.0, 120.0),
                           work_rate_per_agent_dt_range: Tuple[float, float] = (2.0, 3.0), # 稍快完成
                           existing_task_positions: Optional[List[np.ndarray]] = None,
                           min_distance_between_tasks: float = 40.0
                           ) -> Dict[str, Any]:
    # (函数体与上一版本提供的修改后版本一致，这里不再重复以节省空间，确保使用该版本)
    # ... (代码与上一个回答中提供的 create_task_definition 一致) ...
    if not task_type_names_list and num_total_task_types > 0 :
        task_type_names_list = [f"类型{i+1}" for i in range(num_total_task_types)]
    elif len(task_type_names_list) != num_total_task_types and num_total_task_types > 0:
        task_type_names_list = [f"类型{i+1}" for i in range(num_total_task_types)]

    if override_true_type_name and override_true_type_name in task_type_names_list:
        true_type_name = override_true_type_name
        true_type_idx = task_type_names_list.index(true_type_name)
    elif num_total_task_types > 0:
        true_type_idx = random.randint(0, num_total_task_types - 1)
        true_type_name = task_type_names_list[true_type_idx]
    else:
        true_type_idx = -1; true_type_name = "无特定类型"

    task_specific_type_rewards = {}
    task_specific_type_risks = {}
    # 显著提高默认奖励，降低默认风险
    default_reward_range = (8000, 12000); default_true_reward_range = (15000, 20000) # 大幅增加奖励
    default_risk_range = (2, 8); default_true_risk_range = (1, 5) # 大幅减少风险

    if num_total_task_types > 0:
        for type_name_iter in task_type_names_list:
            current_reward_range = (reward_ranges_by_type or {}).get(type_name_iter, default_reward_range)
            current_risk_range = (risk_ranges_by_type or {}).get(type_name_iter, default_risk_range)
            task_specific_type_rewards[type_name_iter] = random.uniform(current_reward_range[0], current_reward_range[1])
            task_specific_type_risks[type_name_iter] = random.uniform(current_risk_range[0], current_risk_range[1])
        if true_type_name != "无特定类型" and true_type_name in task_type_names_list: 
            true_type_reward_range_final = (reward_ranges_by_type or {}).get(true_type_name, default_true_reward_range)
            true_type_risk_range_final = (risk_ranges_by_type or {}).get(true_type_name, default_true_risk_range)
            task_specific_type_rewards[true_type_name] = random.uniform(true_type_reward_range_final[0], true_type_reward_range_final[1])
            task_specific_type_risks[true_type_name] = random.uniform(true_type_risk_range_final[0], true_type_risk_range_final[1])

    required_caps = random.choice(required_capabilities_options) if required_capabilities_options else [f"cap_task_{random.randint(1,2)}"]
    min_agents = min_agents_needed_override if min_agents_needed_override is not None else 1 # 通常任务只需1个智能体
    max_agents = max_agents_allowed_override if max_agents_allowed_override is not None else random.choices([min_agents, min_agents + 1], weights=[0.8, 0.2], k=1)[0]
    max_agents = max(min_agents, max_agents)

    position = np.array([0.0,0.0]) 
    max_attempts_pos = 30; attempt_count_pos = 0 # 增加尝试次数
    while attempt_count_pos < max_attempts_pos:
        position = np.random.uniform(position_range[0], position_range[1], 2)
        if not existing_task_positions: break
        too_close = any(np.linalg.norm(position - existing_pos) < min_distance_between_tasks for existing_pos in existing_task_positions)
        if not too_close: break
        attempt_count_pos += 1
    if attempt_count_pos == max_attempts_pos:
        print(f"警告: 未能为任务 {task_id_str} 在 {max_attempts_pos} 次尝试内找到足够远的位置（最小距离 {min_distance_between_tasks}m）。可能导致任务重叠。")


    task_def = {
        "id": task_id_str, "position": position.tolist(), "target_velocity": np.array([0.0, 0.0]).tolist(),
        "true_type_idx": true_type_idx, "true_type_name": true_type_name,
        "type_rewards": task_specific_type_rewards, "type_risks": task_specific_type_risks,
        "required_capabilities": required_caps, "status": "active", "assigned_agents": [],
        "completion_time": -1.0, "value_realized": 0.0,
        "min_agents_needed": min_agents, "max_agents_allowed": max_agents,
        "total_workload": random.uniform(total_workload_range[0], total_workload_range[1]),
        "current_completion_percentage": 0.0,
        "work_rate_per_agent_per_dt": random.uniform(work_rate_per_agent_dt_range[0], work_rate_per_agent_dt_range[1])
    }
    return task_def


def create_spacecraft_definition(sc_id_str: str,
                                 agent_idx_for_placement: int, total_num_agents_for_placement: int,
                                 physical_base_config: Optional[Dict] = None, ksc_base_config: Optional[Dict] = None,
                                 aif_agent_type_options: Optional[List[str]] = None,
                                 initial_pos_placement_radius: float = 80.0, # 智能体更集中
                                 initial_velocity_range: Tuple[float, float] = (-0.005, 0.005) # 初始速度更小
                                 ) -> Dict[str, Any]:
    # (函数体与上一版本提供的修改后版本一致，确保 physical_base_config 包含调整后的 fuel_cost_coeff)
    # ... (代码与上一个回答中提供的 create_spacecraft_definition 一致) ...
    sc_physical_params = copy.deepcopy(DEFAULT_SC_PHYSICAL_PARAMS) 
    if physical_base_config: sc_physical_params.update(physical_base_config)
    if 'fuel_cost_coeff' not in sc_physical_params: sc_physical_params['fuel_cost_coeff'] = DEFAULT_SC_PHYSICAL_PARAMS['fuel_cost_coeff']

    sc_ksc_params = copy.deepcopy(DEFAULT_KSC_AGENT_PARAMS)
    if ksc_base_config: sc_ksc_params.update(ksc_base_config)

    if "type" not in sc_physical_params or not sc_physical_params["type"]:
        sc_physical_params["type"] = random.choice(aif_agent_type_options) if aif_agent_type_options else "s_default"
    if "capabilities" not in sc_physical_params or not sc_physical_params["capabilities"]:
        all_caps = ["cap_sensor_dist", "cap_sensor_angle", "cap_actuator_A"]
        sc_physical_params["capabilities"] = random.sample(all_caps, k=random.randint(1,min(2,len(all_caps))))

    angle = 2 * np.pi * agent_idx_for_placement / total_num_agents_for_placement if total_num_agents_for_placement > 0 else 0
    radius = initial_pos_placement_radius * (0.8 + random.random() * 0.2) # Tighter initial spread
    initial_state = np.array([
        radius * np.cos(angle), radius * np.sin(angle),
        random.uniform(initial_velocity_range[0], initial_velocity_range[1]),
        random.uniform(initial_velocity_range[0], initial_velocity_range[1])
    ])
    return {"id": sc_id_str, "initial_state": initial_state.tolist(),
            "physical_params": sc_physical_params, "ksc_params": sc_ksc_params}


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "strong_comms_basic_ksc_completion_test": {
        "scenario_display_name": "强通信KSC-效用调优 (3A-6T)", # 名称更新
        "description": "3智能体6任务，调整参数以鼓励KSC进行任务分配，测试任务完成。",
        "num_agents": 3, "num_tasks": 6,
        "task_type_names": ["甲类目标", "乙类目标"], # 更通用的类型名
        "aif_num_abstract_goals": 6, "aif_allow_multiple_to_same_goal": True,
        "simulation_params": {
            "max_time_steps": 500, "dt_simulation": 2.0, "dt_decision_ksc": 4.0, "dt_decision_aif": 2.0,
            "communication_range": None, "strong_mode_components_enabled": True, "stop_if_all_tasks_done": True,
            "orbit_mean_motion_n": 0.0011, "belief_update_interval_ksc_steps": 2, # 降低信念更新频率，减少干扰
            "task_completion_check_interval_sim_steps": 1,
            "task_completion_distance_threshold": 60.0, # 稍放宽距离阈值
            "physical_state_update_interval_sim_steps": -1
        },
        "agent_definitions_template": {
            "physical_params_base": { # 将使用上面调整后的 DEFAULT_SC_PHYSICAL_PARAMS
                "capabilities": ["cap_type1", "cap_type2"], # 确保能力与任务需求匹配
            },
            "ksc_params_base": {"k_value": 2},
            "aif_agent_type_options": ['s', 'A'], # 确保这些类型在AIF中有定义
            "initial_pos_placement_radius": 80.0, # 智能体初始更集中
            "initial_velocity_range": (-0.005, 0.005)
        },
        "task_definitions_template": {
            "position_range": (-120.0, 120.0), # 任务更集中
            "reward_ranges_by_type": {"甲类目标": (10000, 15000), "乙类目标": (6000, 9000)}, # 大幅提高奖励
            "risk_ranges_by_type": {"甲类目标": (1, 5), "乙类目标": (0, 2)}, # 大幅降低风险
            "required_capabilities_options": [["cap_type1"], ["cap_type2"]], # 任务需要的能力
            "total_workload_range": (100.0, 160.0),
            "work_rate_per_agent_dt_range": (2.5, 4.0) # 稍快的完成速率
        },
        # 手动指定所有6个任务的类型，确保与 reward_ranges_by_type 匹配
        "manual_task_true_types": {
            "Task1": "甲类目标", "Task2": "乙类目标", "Task3": "甲类目标",
            "Task4": "乙类目标", "Task5": "甲类目标", "Task6": "乙类目标"
        },
        "initial_belief_overrides": { # 初始信念可以更乐观或更集中
             "SC1": {"Task1": np.array([5.0, 0.1]), "Task2": np.array([0.1, 5.0])},
             "SC2": {"Task3": np.array([4.0, 0.2]), "Task4": np.array([0.2, 4.0])},
             "SC3": {"Task5": np.array([4.5, 0.3]), "Task6": np.array([0.3, 4.5])},
        },
        "adaptive_controller_params": {**copy.deepcopy(DEFAULT_ADAPTIVE_PARAMS)},
        "aif_global_hyperparams": {**copy.deepcopy(DEFAULT_AIF_GLOBAL_HYPERPARAMS)},
        "spectral_params": {**copy.deepcopy(DEFAULT_SPECTRAL_PARAMS)},
        "mpc_params": {**copy.deepcopy(DEFAULT_MPC_PARAMS)},
        "ksc_algorithm_config": {
             "max_dfs_branching_factor": 2,
             # 确保这里的类型名与 task_type_names 一致
             "ksc_max_coalition_size_per_task": {"甲类目标": 2, "乙类目标": 1},
             "ksc_min_agents_per_task": {"甲类目标": 1, "乙类目标": 1 } # 确保任务至少被分配
        },
        "visualization_settings": {
            "plot_overall_performance": True, "plot_agent_trajectories": True,
            "plot_single_agent_details_for_agent": "SC1", "plot_task_gantt": True,
            # 确保追踪的任务ID和智能体ID存在
            "plot_belief_convergence_for": {"SC1": ["Task1", "Task3"], "SC2": ["Task2", "Task4"]},
            "plot_ksc_assignments_evolution": True,
        }
    },
    # (dynamic_comms_adaptive_switching_benchmark_base 场景定义也应检查并应用类似的参数调整原则)
    "dynamic_comms_adaptive_switching_benchmark_base": { # 确保此场景也调优
        "scenario_display_name": "动态通信与自适应切换基准-调优 (3A-4T)",
        "description": "3智能体4任务，动态通信，调整参数以促使决策。",
        "num_agents": 3, "num_tasks": 4,
        "task_type_names": ["紧急救援", "常规扫描"],
        "aif_num_abstract_goals": 4, "aif_allow_multiple_to_same_goal": False,
        "simulation_params": {
            "max_time_steps": 600, "dt_simulation": 2.0, "dt_decision_ksc": 4.0, "dt_decision_aif": 2.0,
            "communication_range": 200.0, "strong_mode_components_enabled": True,
            "stop_if_all_tasks_done": True, "orbit_mean_motion_n": 0.0011,
            "belief_update_interval_ksc_steps": 1, "task_completion_check_interval_sim_steps": 1,
            "task_completion_distance_threshold": 40.0, "physical_state_update_interval_sim_steps": 25
        },
        "agent_definitions_template": { # 使用上面调整后的 DEFAULT_SC_PHYSICAL_PARAMS
            "physical_params_base": {"capabilities": ["cap_sensor_multi", "cap_actuator_B"]},
            "ksc_params_base": {"k_value": 2},
            "aif_agent_type_options": ['s', 'A'],
            "initial_pos_placement_radius": 120.0, "initial_velocity_range": (-0.01, 0.01)
        },
         "task_definitions_template": {
            "position_range": (-180.0, 180.0),
            "reward_ranges_by_type": {"紧急救援": (9000, 12000), "常规扫描": (2000, 3000)},
            "risk_ranges_by_type": {"紧急救援": (20, 40), "常规扫描": (3, 10)},
            "required_capabilities_options": [["cap_sensor_multi"]],
            "total_workload_range": (150.0, 220.0),
            "work_rate_per_agent_dt_range": (1.2, 1.8)
        },
        "manual_task_true_types": {"Task1": "紧急救援", "Task2": "常规扫描", "Task3": "紧急救援", "Task4": "常规扫描"},
        "adaptive_controller_params": {**copy.deepcopy(DEFAULT_ADAPTIVE_PARAMS)},
        "aif_global_hyperparams": {**copy.deepcopy(DEFAULT_AIF_GLOBAL_HYPERPARAMS), "reasoning_level": 1, "use_epistemic_planning": True},
        "spectral_params": {**copy.deepcopy(DEFAULT_SPECTRAL_PARAMS)},
        "mpc_params": {**copy.deepcopy(DEFAULT_MPC_PARAMS)},
        "ksc_algorithm_config": {
             "max_dfs_branching_factor": 2,
             "ksc_max_coalition_size_per_task": {"紧急救援": 1, "常规扫描": 2},
             "ksc_min_agents_per_task": {"紧急救援": 1}
        },
        "simulation_callbacks": {
            "on_step_begin": [{"condition": "env.time_step > 0 and env.time_step % 50 == 0",
                               "action": "update_env_param", "param_path": "sim_params.communication_range",
                               "value": "100.0 + 100.0 * (1 + np.sin(env.current_time / 100.0))",
                               "message": "通信范围发生周期性变化"}]
        },
        "visualization_settings": {"plot_all_agents_csi_modes": True, "plot_belief_convergence_for": {"SC1": ["Task1"]}}
    },
}

# ... (load_scenario 函数与上一版本修改后一致，确保它能正确处理新的默认参数和任务/智能体创建) ...
def load_scenario(scenario_name: str) -> Optional[Dict[str, Any]]:
    if scenario_name not in SCENARIOS:
        print(f"错误: 场景 '{scenario_name}' 未在 SCENARIOS 中定义。")
        return None
    
    config = copy.deepcopy(SCENARIOS[scenario_name])
    config["scenario_name_loaded_from"] = scenario_name # Store the original key
    
    num_agents = config.get("num_agents", 0)
    num_tasks = config.get("num_tasks", 0)
    task_type_names = config.get("task_type_names", [])
    num_task_types = len(task_type_names) if task_type_names else 0
    
    # Create spacecrafts, ensuring physical_params_base from template is used if agent_specific_overrides doesn't provide one
    config["spacecrafts"] = []
    agent_template_config = config.get("agent_definitions_template", {})
    # Use DEFAULT_SC_PHYSICAL_PARAMS as a fallback for the template's base
    template_physical_base_from_scen = agent_template_config.get("physical_params_base", {})
    
    for i in range(num_agents):
        sc_id = f"SC{i+1}"
        agent_overrides = config.get("agent_specific_overrides", {}).get(sc_id, {})
        
        # Start with global default, then template base, then specific override
        current_physical_params = copy.deepcopy(DEFAULT_SC_PHYSICAL_PARAMS)
        current_physical_params.update(template_physical_base_from_scen) # Apply template base
        current_physical_params.update(agent_overrides.get("physical_params", {})) # Apply specific override

        current_ksc_params = copy.deepcopy(DEFAULT_KSC_AGENT_PARAMS)
        current_ksc_params.update(agent_template_config.get("ksc_params_base", {}))
        current_ksc_params.update(agent_overrides.get("ksc_params", {}))
        
        sc_def = create_spacecraft_definition(
            sc_id_str=sc_id, agent_idx_for_placement=i, total_num_agents_for_placement=num_agents,
            physical_base_config=current_physical_params, ksc_base_config=current_ksc_params,
            aif_agent_type_options=agent_template_config.get("aif_agent_type_options"), 
            initial_pos_placement_radius=agent_template_config.get("initial_pos_placement_radius", 150.0),
            initial_velocity_range=agent_template_config.get("initial_velocity_range", (-0.01, 0.01))
        )
        config["spacecrafts"].append(sc_def)
    
    tasks_dict_final_scen: Dict[str, Dict[str, Any]] = {}
    if num_tasks > 0:
        task_template_config_scen = config.get("task_definitions_template", {})
        ksc_algo_specific_cfg_scen = config.get("ksc_algorithm_config", {})
        min_agents_constraints_scen = ksc_algo_specific_cfg_scen.get("ksc_min_agents_per_task", {})
        max_agents_constraints_scen = ksc_algo_specific_cfg_scen.get("ksc_max_coalition_size_per_task", {})
        existing_task_positions_scen: List[np.ndarray] = []

        for i in range(num_tasks):
            task_id_scen = f"Task{i+1}"
            task_true_type_override_scen = config.get("manual_task_true_types", {}).get(task_id_scen)
            
            min_agents_override_val_scen = None
            if task_true_type_override_scen and isinstance(min_agents_constraints_scen, dict):
                 min_agents_override_val_scen = min_agents_constraints_scen.get(task_true_type_override_scen)

            max_agents_override_val_scen = None
            if task_true_type_override_scen and isinstance(max_agents_constraints_scen, dict):
                 max_agents_override_val_scen = max_agents_constraints_scen.get(task_true_type_override_scen)
            
            task_def_scen = create_task_definition(
                task_id_str=task_id_scen, num_total_task_types=num_task_types, task_type_names_list=task_type_names,
                position_range=task_template_config_scen.get("position_range", (-100.0, 100.0)),
                reward_ranges_by_type=task_template_config_scen.get("reward_ranges_by_type"),
                risk_ranges_by_type=task_template_config_scen.get("risk_ranges_by_type"),
                override_true_type_name=task_true_type_override_scen,
                min_agents_needed_override=min_agents_override_val_scen,
                max_agents_allowed_override=max_agents_override_val_scen,
                required_capabilities_options=task_template_config_scen.get("required_capabilities_options"),
                total_workload_range=task_template_config_scen.get("total_workload_range", (100.0,150.0)),
                work_rate_per_agent_dt_range=task_template_config_scen.get("work_rate_per_agent_dt_range", (1.5,2.5)),
                existing_task_positions=existing_task_positions_scen,
                min_distance_between_tasks=task_template_config_scen.get("min_distance_between_tasks", 40.0)
            )
            tasks_dict_final_scen[task_id_scen] = task_def_scen
            if "position" in task_def_scen and isinstance(task_def_scen["position"], list) and len(task_def_scen["position"])==2 :
                existing_task_positions_scen.append(np.array(task_def_scen["position"]))
    config["tasks"] = tasks_dict_final_scen
    
    # AIF Goal Positions and Hyperparameters (ensure this logic remains robust)
    num_aif_goals_from_config_scen = config.get("aif_num_abstract_goals", 0)
    if not ("aif_goal_positions" in config and isinstance(config["aif_goal_positions"], list) and config["aif_goal_positions"]):
        if num_aif_goals_from_config_scen > 0:
            aif_goals_pos_list_scen = []
            ksc_task_positions_scen = [t_def["position"] for t_def in config["tasks"].values() if "position" in t_def and t_def["position"]]
            if len(ksc_task_positions_scen) >= num_aif_goals_from_config_scen:
                aif_goals_pos_list_scen = [np.array(p) for p in ksc_task_positions_scen[:num_aif_goals_from_config_scen]]
            else:
                aif_goals_pos_list_scen.extend([np.array(p) for p in ksc_task_positions_scen])
                num_to_generate_randomly_scen = num_aif_goals_from_config_scen - len(aif_goals_pos_list_scen)
                for _ in range(num_to_generate_randomly_scen): aif_goals_pos_list_scen.append(np.random.uniform(-100.0, 100.0, 2)) # Smaller range for AIF goals too
            config["aif_goal_positions"] = aif_goals_pos_list_scen
        else: config["aif_goal_positions"] = []
    config["aif_num_abstract_goals"] = len(config["aif_goal_positions"])

    final_aif_hyperparams_scen = copy.deepcopy(DEFAULT_AIF_GLOBAL_HYPERPARAMS)
    final_aif_hyperparams_scen.update(config.get("aif_global_hyperparams", {}))
    num_actual_aif_goals_for_g_config_scen = config["aif_num_abstract_goals"]
    allow_multiple_to_aif_g_scen = config.get("aif_allow_multiple_to_same_goal", False)

    if num_agents > 0 and num_actual_aif_goals_for_g_config_scen > 0: # Only generate if agents and goals exist
        possible_G_configs_array_scen = generate_possible_goal_configs(num_agents, num_actual_aif_goals_for_g_config_scen, allow_multiple_to_aif_g_scen)
        num_total_g_configs_scen = possible_G_configs_array_scen.shape[0] if possible_G_configs_array_scen.size > 0 else 0
        if num_total_g_configs_scen > 0:
            c_target_from_scenario_scen = final_aif_hyperparams_scen.get("C_target_dist_G")
            if not (c_target_from_scenario_scen is not None and isinstance(c_target_from_scenario_scen, np.ndarray) and len(c_target_from_scenario_scen) == num_total_g_configs_scen):
                final_aif_hyperparams_scen["C_target_dist_G"] = np.ones(num_total_g_configs_scen) / num_total_g_configs_scen
            utility_g_from_scenario_scen = final_aif_hyperparams_scen.get("utility_of_G_configs")
            if not (utility_g_from_scenario_scen is not None and isinstance(utility_g_from_scenario_scen, np.ndarray) and len(utility_g_from_scenario_scen) == num_total_g_configs_scen):
                final_aif_hyperparams_scen["utility_of_G_configs"] = np.zeros(num_total_g_configs_scen)
        else: final_aif_hyperparams_scen["C_target_dist_G"] = np.array([]); final_aif_hyperparams_scen["utility_of_G_configs"] = np.array([])
    else: # No agents or no AIF goals, so no G configs
        final_aif_hyperparams_scen["C_target_dist_G"] = np.array([])
        final_aif_hyperparams_scen["utility_of_G_configs"] = np.array([])
        
    config["aif_global_hyperparams"] = final_aif_hyperparams_scen
    
    config["adaptive_controller_params"] = {**copy.deepcopy(DEFAULT_ADAPTIVE_PARAMS), **config.get("adaptive_controller_params", {})}
    config["spectral_params"] = {**copy.deepcopy(DEFAULT_SPECTRAL_PARAMS), **config.get("spectral_params", {})}
    config["mpc_params"] = {**copy.deepcopy(DEFAULT_MPC_PARAMS), **config.get("mpc_params", {})}
    
    # Clean up template keys if they are not meant to be part of the final loaded config
    config.pop("agent_definitions_template", None); config.pop("task_definitions_template", None)
    config.pop("agent_specific_overrides", None); config.pop("manual_task_true_types", None) # Keep initial_belief_overrides for env to use
    
    return config

# ... (模块自测试 __main__ 保持不变) ...