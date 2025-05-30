# experiments/strong_comm_belief_learning/experiment_config.py
"""
强通信信念自学习实验配置模块
用于生成不同实验场景的配置参数
"""

import numpy as np
import copy
import random
from typing import Dict, List, Tuple, Any, Optional
import itertools

class BeliefLearningExperimentConfig:
    """强通信信念自学习实验配置生成器"""
    
    def __init__(self, base_output_dir: str = "results/strong_comm_belief_experiments"):
        self.base_output_dir = base_output_dir
        
        # 基础实验参数
        self.base_params = {
            "simulation_params": {
                "max_time_steps": 800,
                "dt_simulation": 2.0,
                "dt_decision_ksc": 4.0,
                "dt_decision_aif": 4.0,
                "communication_range": None,  # 强通信，无限制
                "strong_mode_components_enabled": True,
                "stop_if_all_tasks_done": True,
                "orbit_mean_motion_n": 0.0011,
                "belief_update_interval_ksc_steps": 1,
                "task_completion_distance_threshold": 80.0,  # 增大完成距离阈值
                "physical_state_update_interval_sim_steps": -1,
                "perturbation_pos_drift_max_per_dt": 0.02,  # 小幅随机扰动
                "perturbation_vel_drift_max_per_dt": 0.001
            },
            
            "agent_params_template": {
                "initial_mass_kg": 120.0,
                "isp_s": 350.0,
                "agent_radius": 0.8,
                "fuel_cost_coeff": 0.8,  # 降低燃料成本权重，鼓励机动
                "observation_accuracy_p_corr": 0.90,  # 高观测精度
                "capabilities": ["sensor_multi", "actuator_main"],
                "type": "strong_comm_agent"
            },
            
            "task_params_template": {
                "position_range": (-2000.0, 2000.0),  # 5km x 5km 范围
                "min_distance_between_tasks": 200.0,  # 任务间最小距离200m
                "total_workload_range": (180.0, 280.0),
                "work_rate_per_agent_dt_range": (2.0, 3.5),
                "reward_ranges_by_type": {
                    "高价值科研": (8000, 12000),
                    "普通巡查": (3000, 5000), 
                    "紧急维修": (5000, 8000)
                },
                "risk_ranges_by_type": {
                    "高价值科研": (15, 30),
                    "普通巡查": (2, 8),
                    "紧急维修": (8, 20)
                }
            },
            
            "ksc_params": {
                "max_dfs_branching_factor": 3,
                "max_dfs_depth_for_others": 2,
                "ksc_max_coalition_size_per_task": {
                    "高价值科研": 3, 
                    "普通巡查": 2, 
                    "紧急维修": 2
                },
                "ksc_min_agents_per_task": {
                    "高价值科研": 1,
                    "紧急维修": 1
                }
            },
            
            "belief_params": {
                "initial_belief_alpha_base": 1.5,  # 初始信念强度
                "num_task_types": 3,
                "task_type_names": ["高价值科研", "普通巡查", "紧急维修"]
            },
            
            "mpc_params": {
                "prediction_horizon": 12,
                "Q_terminal_diag": [150.0, 150.0, 15.0, 15.0],
                "R_control_diag": [0.001, 0.001],  # 极低控制成本
                "u_max_abs": 0.4,
                "solver_print_level": 0
            }
        }
        
    def generate_scalability_configs(self) -> List[Dict[str, Any]]:
        """生成扩展性实验配置"""
        configs = []
        
        # 实验1: 固定任务数量，变化航天器数量
        fixed_num_tasks = 8
        for num_agents in [3, 4, 5, 6, 7, 8]:
            config = self._create_base_config(
                num_agents=num_agents,
                num_tasks=fixed_num_tasks,
                experiment_name=f"scalability_agents_{num_agents}a_{fixed_num_tasks}t"
            )
            configs.append(config)
            
        # 实验2: 固定航天器数量，变化任务数量  
        fixed_num_agents = 5
        for num_tasks in [4, 6, 8, 10, 12]:
            config = self._create_base_config(
                num_agents=fixed_num_agents,
                num_tasks=num_tasks,
                experiment_name=f"scalability_tasks_{fixed_num_agents}a_{num_tasks}t"
            )
            configs.append(config)
            
        return configs
    
    def generate_comparison_configs(self) -> List[Dict[str, Any]]:
        """生成对比实验配置"""
        configs = []
        base_agents, base_tasks = 5, 8
        
        # 实验3: 不同K值对比
        for k_value in [1, 2, 3, 4, 5]:
            config = self._create_base_config(
                num_agents=base_agents,
                num_tasks=base_tasks,
                experiment_name=f"k_value_comparison_k{k_value}",
                custom_k_value=k_value
            )
            configs.append(config)
            
        # 实验4: 不同初始信念强度对比
        for alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
            config = self._create_base_config(
                num_agents=base_agents,
                num_tasks=base_tasks,
                experiment_name=f"belief_alpha_comparison_a{alpha}",
                custom_belief_alpha=alpha
            )
            configs.append(config)
            
        # 实验5: 不同观测精度对比
        for accuracy in [0.70, 0.80, 0.90, 0.95, 0.99]:
            config = self._create_base_config(
                num_agents=base_agents,
                num_tasks=base_tasks,
                experiment_name=f"observation_accuracy_comparison_acc{int(accuracy*100)}",
                custom_observation_accuracy=accuracy
            )
            configs.append(config)
            
        # 实验6: 不同信念更新频率对比
        for update_interval in [1, 2, 3, 5, 8]:
            config = self._create_base_config(
                num_agents=base_agents,
                num_tasks=base_tasks,
                experiment_name=f"belief_update_freq_comparison_int{update_interval}",
                custom_belief_update_interval=update_interval
            )
            configs.append(config)
            
        return configs
    
    def generate_robustness_configs(self) -> List[Dict[str, Any]]:
        """生成鲁棒性测试配置"""
        configs = []
        base_agents, base_tasks = 6, 10
        
        # 实验7: 不同任务分布密度
        for min_distance in [100.0, 150.0, 200.0, 300.0, 500.0]:
            config = self._create_base_config(
                num_agents=base_agents,
                num_tasks=base_tasks,
                experiment_name=f"task_density_comparison_dist{int(min_distance)}",
                custom_task_min_distance=min_distance
            )
            configs.append(config)
            
        # 实验8: 不同任务完成难度
        for completion_threshold in [50.0, 80.0, 120.0, 150.0, 200.0]:
            config = self._create_base_config(
                num_agents=base_agents,
                num_tasks=base_tasks,
                experiment_name=f"completion_difficulty_thresh{int(completion_threshold)}",
                custom_completion_threshold=completion_threshold
            )
            configs.append(config)
            
        return configs
        
    def _create_base_config(self, 
                           num_agents: int, 
                           num_tasks: int,
                           experiment_name: str,
                           custom_k_value: Optional[int] = None,
                           custom_belief_alpha: Optional[float] = None,
                           custom_observation_accuracy: Optional[float] = None,
                           custom_belief_update_interval: Optional[int] = None,
                           custom_task_min_distance: Optional[float] = None,
                           custom_completion_threshold: Optional[float] = None) -> Dict[str, Any]:
        """创建基础实验配置"""
        
        config = copy.deepcopy(self.base_params)
        
        # 基础信息
        config["scenario_name"] = experiment_name
        config["scenario_display_name"] = f"强通信信念自学习实验: {experiment_name}"
        config["description"] = f"{num_agents}个航天器, {num_tasks}个任务的信念自学习实验"
        config["num_agents"] = num_agents
        config["num_tasks"] = num_tasks
        
        # 生成航天器配置
        config["spacecrafts"] = self._generate_spacecraft_configs(num_agents, custom_k_value, custom_observation_accuracy)
        
        # 生成任务配置
        config["tasks"] = self._generate_task_configs(num_tasks, custom_task_min_distance, custom_completion_threshold)
        
        # 设置AIF相关参数（虽然主要用强通信，但框架需要）
        config["aif_goal_positions"] = [np.array(task["position"]) for task in config["tasks"].values()]
        config["aif_num_abstract_goals"] = num_tasks
        config["aif_allow_multiple_to_same_goal"] = False
        
        # 自定义参数
        if custom_belief_alpha is not None:
            config["belief_params"]["initial_belief_alpha_base"] = custom_belief_alpha
            
        if custom_belief_update_interval is not None:
            config["simulation_params"]["belief_update_interval_ksc_steps"] = custom_belief_update_interval
            
        # 初始信念设置
        config["initial_belief_overrides"] = self._generate_initial_beliefs(num_agents, num_tasks, custom_belief_alpha)
        
        # 可视化设置
        config["visualization_settings"] = {
            "plot_overall_performance": True,
            "plot_agent_trajectories": True,
            "plot_task_gantt": True,
            "plot_belief_convergence_for": self._get_belief_tracking_config(num_agents, num_tasks),
            "plot_ksc_assignments_evolution": True,
            "plot_ksc_communication_stats": True,
            "plot_agent_velocities": True,
            "plot_agent_accelerations": True,
            "plot_final_agent_locations_comm_graph": True,
            "plot_task_completion_percentage": True
        }
        
        return config
    
    def _generate_spacecraft_configs(self, 
                                   num_agents: int, 
                                   custom_k_value: Optional[int] = None,
                                   custom_observation_accuracy: Optional[float] = None) -> List[Dict[str, Any]]:
        """生成航天器配置"""
        spacecrafts = []
        
        # 在5km x 5km范围内均匀分布初始位置
        initial_radius = 800.0  # 初始分布半径
        
        for i in range(num_agents):
            agent_id = f"SC{i+1:02d}"
            
            # 均匀分布初始位置
            angle = 2 * np.pi * i / num_agents
            radius_variation = initial_radius * (0.7 + 0.3 * random.random())
            initial_x = radius_variation * np.cos(angle)
            initial_y = radius_variation * np.sin(angle)
            initial_vx = random.uniform(-0.02, 0.02)
            initial_vy = random.uniform(-0.02, 0.02)
            
            physical_params = copy.deepcopy(self.base_params["agent_params_template"])
            if custom_observation_accuracy is not None:
                physical_params["observation_accuracy_p_corr"] = custom_observation_accuracy
                
            ksc_params = {
                "k_value": custom_k_value if custom_k_value is not None else (2 + i % 3)  # K值在2-4之间变化
            }
            
            spacecraft = {
                "id": agent_id,
                "initial_state": [initial_x, initial_y, initial_vx, initial_vy],
                "physical_params": physical_params,
                "ksc_params": ksc_params
            }
            
            spacecrafts.append(spacecraft)
            
        return spacecrafts
    
    def _generate_task_configs(self, 
                             num_tasks: int, 
                             custom_min_distance: Optional[float] = None,
                             custom_completion_threshold: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """生成任务配置"""
        tasks = {}
        task_positions = []
        
        min_distance = custom_min_distance if custom_min_distance is not None else self.base_params["task_params_template"]["min_distance_between_tasks"]
        completion_threshold = custom_completion_threshold if custom_completion_threshold is not None else self.base_params["simulation_params"]["task_completion_distance_threshold"]
        
        # 任务类型循环分配
        task_types = self.base_params["belief_params"]["task_type_names"]
        
        for i in range(num_tasks):
            task_id = f"Task{i+1:02d}"
            task_type = task_types[i % len(task_types)]
            
            # 生成任务位置，确保距离足够远
            max_attempts = 100
            for attempt in range(max_attempts):
                # 在更大范围内生成位置，确保分散
                x = random.uniform(-2200, 2200)
                y = random.uniform(-2200, 2200)
                position = np.array([x, y])
                
                # 检查与已有任务的距离
                if not task_positions:
                    task_positions.append(position)
                    break
                    
                min_dist_to_existing = min(np.linalg.norm(position - existing_pos) 
                                         for existing_pos in task_positions)
                
                if min_dist_to_existing >= min_distance:
                    task_positions.append(position)
                    break
            else:
                # 如果无法找到足够远的位置，使用最后一个位置
                task_positions.append(position)
            
            # 任务参数
            reward_range = self.base_params["task_params_template"]["reward_ranges_by_type"][task_type]
            risk_range = self.base_params["task_params_template"]["risk_ranges_by_type"][task_type]
            workload_range = self.base_params["task_params_template"]["total_workload_range"]
            work_rate_range = self.base_params["task_params_template"]["work_rate_per_agent_dt_range"]
            
            task_config = {
                "id": task_id,
                "position": task_positions[-1].tolist(),
                "target_velocity": [0.0, 0.0],
                "initial_state": task_positions[-1].tolist() + [0.0, 0.0],
                "current_state": task_positions[-1].tolist() + [0.0, 0.0],
                "true_type_idx": task_types.index(task_type),
                "true_type_name": task_type,
                "type_rewards": {task_type: random.uniform(reward_range[0], reward_range[1])},
                "type_risks": {task_type: random.uniform(risk_range[0], risk_range[1])},
                "required_capabilities": ["sensor_multi"],
                "status": "active",
                "assigned_agents": [],
                "completion_time": -1.0,
                "value_realized": 0.0,
                "min_agents_needed": 1,
                "max_agents_allowed": self.base_params["ksc_params"]["ksc_max_coalition_size_per_task"].get(task_type, 2),
                "total_workload": random.uniform(workload_range[0], workload_range[1]),
                "current_completion_percentage": 0.0,
                "work_rate_per_agent_per_dt": random.uniform(work_rate_range[0], work_rate_range[1])
            }
            
            tasks[task_id] = task_config
            
        return tasks
    
    def _generate_initial_beliefs(self, 
                                num_agents: int, 
                                num_tasks: int, 
                                custom_alpha: Optional[float] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """生成初始信念配置"""
        initial_beliefs = {}
        base_alpha = custom_alpha if custom_alpha is not None else self.base_params["belief_params"]["initial_belief_alpha_base"]
        num_types = self.base_params["belief_params"]["num_task_types"]
        
        for i in range(num_agents):
            agent_id = f"SC{i+1:02d}"
            initial_beliefs[agent_id] = {}
            
            # 为每个任务生成初始信念
            for j in range(min(num_tasks, 8)):  # 限制初始信念设置的任务数量
                task_id = f"Task{j+1:02d}"
                
                # 生成略有偏向的初始信念
                alphas = np.full(num_types, base_alpha)
                
                # 给某个类型增加一些偏向
                biased_type = (i + j) % num_types
                alphas[biased_type] += random.uniform(0.5, 2.0)
                
                initial_beliefs[agent_id][task_id] = alphas
                
        return initial_beliefs
    
    def _get_belief_tracking_config(self, num_agents: int, num_tasks: int) -> Dict[str, List[str]]:
        """获取信念跟踪配置"""
        tracking_config = {}
        
        # 为前几个智能体设置信念跟踪
        for i in range(min(num_agents, 4)):
            agent_id = f"SC{i+1:02d}"
            # 跟踪前几个任务的信念
            tracked_tasks = [f"Task{j+1:02d}" for j in range(min(num_tasks, 6))]
            tracking_config[agent_id] = tracked_tasks
            
        return tracking_config

    def get_all_experiment_configs(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取所有实验配置"""
        all_configs = {
            "scalability": self.generate_scalability_configs(),
            "comparison": self.generate_comparison_configs(), 
            "robustness": self.generate_robustness_configs()
        }
        return all_configs