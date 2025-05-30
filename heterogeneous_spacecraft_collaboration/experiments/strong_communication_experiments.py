# experiments/strong_communication_experiments.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import os
import sys
import copy
import time
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from simulation.environment import SimulationEnvironment
from simulation.visualizer import Visualizer
from simulation.scenarios import load_scenario, SCENARIOS
from utils.font_utils import setup_matplotlib_fonts

# 设置学术论文风格的matplotlib参数
def setup_academic_style():
    """设置符合学术论文要求的matplotlib样式"""
    # 设置中文字体支持
    setup_matplotlib_fonts()
    
    # Nature期刊风格配色方案
    nature_colors = [
        '#0C5DA5',  # 深蓝色
        '#FF2C00',  # 红色  
        '#00B945',  # 绿色
        '#FF9500',  # 橙色
        '#845B97',  # 紫色
        '#474747',  # 深灰色
        '#9E9E9E',  # 浅灰色
        '#F2C800',  # 黄色
        '#E64B35',  # 深红色
        '#4DBBD5',  # 青色
        '#00A087',  # 青绿色
        '#3C5488'   # 深蓝紫色
    ]
    
    # 设置matplotlib参数
    plt.rcParams.update({
        'font.size': 16,           # 基础字体大小
        'axes.labelsize': 18,      # 坐标轴标签字体大小
        'axes.titlesize': 20,      # 标题字体大小
        'xtick.labelsize': 14,     # x轴刻度标签字体大小
        'ytick.labelsize': 14,     # y轴刻度标签字体大小
        'legend.fontsize': 14,     # 图例字体大小
        'figure.titlesize': 22,    # 图片标题字体大小
        
        # 线条样式
        'lines.linewidth': 3.0,    # 线条粗细
        'lines.markersize': 8,     # 标记大小
        'lines.markeredgewidth': 2, # 标记边缘宽度
        
        # 图表样式
        'axes.linewidth': 1.5,     # 坐标轴线宽
        'axes.grid': True,         # 显示网格
        'grid.alpha': 0.3,         # 网格透明度
        'grid.linewidth': 1.0,     # 网格线宽
        
        # 图片质量
        'figure.dpi': 300,         # 显示分辨率
        'savefig.dpi': 600,        # 保存分辨率
        'savefig.bbox': 'tight',   # 紧凑布局
        'savefig.pad_inches': 0.1, # 边距
        
        # 颜色设置
        'axes.prop_cycle': plt.cycler('color', nature_colors),
        
        # 其他设置
        'axes.unicode_minus': False,  # 正确显示负号
        'font.family': ['sans-serif'],
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    return nature_colors

class StrongCommunicationExperiments:
    """强通信信念自学习实验类"""
    
    def __init__(self, results_dir: str = "results/strong_communication_experiments"):
        """
        初始化实验管理器
        
        Args:
            results_dir: 结果保存目录
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置学术风格
        self.colors = setup_academic_style()
        
        # 线型和标记样式
        self.line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        self.markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P']
        
        # 实验配置
        self.base_scenario_config = self._create_base_scenario()
        
        # 结果存储
        self.experiment_results = {}
        
        print(f"强通信实验管理器初始化完成")
        print(f"结果保存路径: {self.results_dir}")
    
    def _create_base_scenario(self) -> Dict[str, Any]:
        """创建基础实验场景配置"""
        return {
            "scenario_display_name": "强通信信念自学习基础场景",
            "description": "用于强通信KSC算法性能测试的基础场景",
            "num_agents": 5,
            "num_tasks": 8,
            "task_type_names": ["高价值科研", "普通巡查", "紧急维修"],
            "aif_num_abstract_goals": 8,
            "aif_allow_multiple_to_same_goal": True,

            "simulation_params": {
                "max_time_steps": 500,
                "dt_simulation": 2.0,
                "dt_decision_ksc": 6.0,
                "dt_decision_aif": 4.0,
                "communication_range": None,  # 强通信，无限制
                "strong_mode_components_enabled": True,
                "stop_if_all_tasks_done": False,
                "orbit_mean_motion_n": 0.0011,
                "belief_update_interval_ksc_steps": 1,
                "task_completion_distance_threshold": 30.0,
                "physical_state_update_interval_sim_steps": -1
            },

            "agent_definitions_template": {
                "physical_params_base": {
                    "initial_mass_kg": 120.0,
                    "isp_s": 320.0,
                    "capabilities": ["cap_sensor_dist", "cap_actuator_A"],
                    "observation_accuracy_p_corr": 0.85,
                    "fuel_cost_coeff": 1.0
                },
                "ksc_params_base": {"k_value": 3},
                "aif_agent_type_options": ['s', 'A', 'B'],
                "initial_pos_placement_radius": 800.0,  # 在5km×5km空间中分布
                "initial_velocity_range": (-0.02, 0.02)
            },

            "task_definitions_template": {
                "position_range": (-2500.0, 2500.0),  # 5km×5km空间
                "reward_ranges_by_type": {
                    "高价值科研": (8000, 12000),
                    "普通巡查": (3000, 5000),
                    "紧急维修": (5000, 8000)
                },
                "risk_ranges_by_type": {
                    "高价值科研": (15, 25),
                    "普通巡查": (5, 10),
                    "紧急维修": (10, 20)
                },
                "required_capabilities_options": [
                    ["cap_sensor_dist"], 
                    ["cap_actuator_A"], 
                    ["cap_sensor_dist", "cap_actuator_A"]
                ],
                "total_workload_range": (200.0, 350.0),
                "work_rate_per_agent_dt_range": (2.0, 4.0),
                "min_distance_between_tasks": 500.0  # 任务间最小距离500m
            },

            "ksc_algorithm_config": {
                "max_dfs_branching_factor": 3,
                "ksc_max_coalition_size_per_task": {
                    "高价值科研": 3,
                    "普通巡查": 1,
                    "紧急维修": 2
                },
                "ksc_min_agents_per_task": {
                    "高价值科研": 1,
                    "紧急维修": 1
                }
            },

            "aif_global_hyperparams": {
                "reasoning_level": 1,
                "use_epistemic_planning": True,
                "alpha_epistemic": 1.0,
                "beta_pragmatic": 1.0,
                "gamma_control_effort": 0.01,
                "policy_temperature": 0.1,
                "belief_update_temperature": 0.8,
                "initial_belief_alpha_ksc": 2.0
            },

            "adaptive_controller_params": {
                "csi_threshold_low": 0.3,
                "csi_threshold_high": 0.7,
                "csi_ema_alpha": 0.2
            },

            "spectral_params": {
                "horizon_H": 8,
                "alpha_scale": 20.0,
                "u_max_for_norm": 0.15,
                "num_modes_to_consider": 4
            },

            "mpc_params": {
                "prediction_horizon": 12,
                "Q_terminal_diag": [150.0, 150.0, 15.0, 15.0],
                "R_control_diag": [0.005, 0.005],
                "u_max_abs": 0.2,
                "solver_print_level": 0
            }
        }
    
    def _create_task_positions_well_distributed(self, num_tasks: int, 
                                               space_range: Tuple[float, float] = (-2500.0, 2500.0),
                                               min_distance: float = 500.0) -> List[Tuple[float, float]]:
        """
        创建分布良好的任务位置，确保任务间距离不小于min_distance
        
        Args:
            num_tasks: 任务数量
            space_range: 空间范围
            min_distance: 最小间距
            
        Returns:
            任务位置列表
        """
        positions = []
        max_attempts = 1000
        
        for i in range(num_tasks):
            attempts = 0
            while attempts < max_attempts:
                # 生成随机位置
                x = np.random.uniform(space_range[0], space_range[1])
                y = np.random.uniform(space_range[0], space_range[1])
                
                # 检查与已有位置的距离
                valid = True
                for existing_pos in positions:
                    distance = np.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                    if distance < min_distance:
                        valid = False
                        break
                
                if valid:
                    positions.append((x, y))
                    break
                    
                attempts += 1
            
            # 如果无法找到合适位置，使用网格布局
            if attempts >= max_attempts:
                print(f"警告: 第{i+1}个任务无法找到合适位置，使用网格布局")
                grid_size = int(np.ceil(np.sqrt(num_tasks)))
                grid_x = (i % grid_size) * (space_range[1] - space_range[0]) / grid_size + space_range[0]
                grid_y = (i // grid_size) * (space_range[1] - space_range[0]) / grid_size + space_range[0]
                positions.append((grid_x, grid_y))
        
        return positions
    
    def run_single_experiment(self, scenario_config: Dict[str, Any], 
                             experiment_name: str, 
                             num_runs: int = 5,
                             max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        运行单个实验
        
        Args:
            scenario_config: 场景配置
            experiment_name: 实验名称
            num_runs: 运行次数
            max_steps: 最大步数
            
        Returns:
            实验结果
        """
        print(f"\n开始实验: {experiment_name}")
        print(f"运行次数: {num_runs}")
        
        results = {
            'experiment_name': experiment_name,
            'num_runs': num_runs,
            'runs_data': [],
            'statistics': {}
        }
        
        for run_id in range(num_runs):
            print(f"  运行 {run_id + 1}/{num_runs}")
            
            try:
                # 创建临时场景文件
                temp_scenario_name = f"temp_{experiment_name}_{run_id}"
                
                # 修改场景配置
                config = copy.deepcopy(scenario_config)
                if max_steps:
                    config["simulation_params"]["max_time_steps"] = max_steps
                
                # 创建任务位置
                task_positions = self._create_task_positions_well_distributed(
                    config["num_tasks"],
                    config["task_definitions_template"]["position_range"],
                    config["task_definitions_template"]["min_distance_between_tasks"]
                )
                
                # 手动设置任务位置和类型
                config["manual_task_positions"] = task_positions
                config["manual_task_true_types"] = {}
                for i in range(config["num_tasks"]):
                    task_id = f"Task{i+1}"
                    type_idx = i % len(config["task_type_names"])
                    config["manual_task_true_types"][task_id] = config["task_type_names"][type_idx]
                
                # 临时保存场景到SCENARIOS字典
                from simulation.scenarios import SCENARIOS
                SCENARIOS[temp_scenario_name] = config
                
                # 运行仿真
                env = SimulationEnvironment(temp_scenario_name)
                
                # 记录开始时间
                start_time = time.time()
                
                # 运行仿真循环
                step_count = 0
                while step_count < config["simulation_params"]["max_time_steps"]:
                    if not env.step():
                        break
                    step_count += 1
                
                # 记录结束时间
                end_time = time.time()
                
                # 收集结果
                run_result = {
                    'run_id': run_id,
                    'execution_time': end_time - start_time,
                    'total_steps': env.time_step,
                    'total_time': env.current_time,
                    'ksc_rounds': env.ksc_decision_round_count,
                    'final_utility': env.history_global_utility_ksc[-1] if env.history_global_utility_ksc else 0,
                    'tasks_completed': len([t for t in env.tasks.values() if t.get('status') == 'completed']),
                    'total_tasks': len(env.tasks),
                    'completion_rate': len([t for t in env.tasks.values() if t.get('status') == 'completed']) / len(env.tasks),
                    'total_value': sum(t.get('value_realized', 0) for t in env.tasks.values()),
                    'avg_messages_per_round': np.mean(env.ksc_stats_messages_broadcasted) if env.ksc_stats_messages_broadcasted else 0,
                    'avg_active_links': np.mean(env.ksc_stats_active_links) if env.ksc_stats_active_links else 0,
                    
                    # 详细历史数据
                    'utility_history': copy.deepcopy(env.history_global_utility_ksc),
                    'assignment_history': copy.deepcopy(env.history_assignments_ksc),
                    'belief_history': copy.deepcopy(env.history_beliefs_at_ksc_step),
                    'task_completion_history': copy.deepcopy(env.history_task_completion_percentage),
                    'message_stats': {
                        'broadcasted': copy.deepcopy(env.ksc_stats_messages_broadcasted),
                        'processed': copy.deepcopy(env.ksc_stats_messages_processed),
                        'active_links': copy.deepcopy(env.ksc_stats_active_links)
                    },
                    
                    # 保存环境实例用于可视化
                    'environment': env
                }
                
                results['runs_data'].append(run_result)
                
                # 清理临时场景
                if temp_scenario_name in SCENARIOS:
                    del SCENARIOS[temp_scenario_name]
                    
            except Exception as e:
                print(f"    运行 {run_id + 1} 失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # 计算统计数据
        if results['runs_data']:
            results['statistics'] = self._calculate_statistics(results['runs_data'])
        
        # 保存结果
        results_file = self.results_dir / f"{experiment_name}_results.pkl"
        with open(results_file, 'wb') as f:
            # 移除environment对象以避免pickle问题
            clean_results = copy.deepcopy(results)
            for run_data in clean_results['runs_data']:
                if 'environment' in run_data:
                    del run_data['environment']
            pickle.dump(clean_results, f)
        
        print(f"实验 {experiment_name} 完成")
        print(f"成功运行: {len(results['runs_data'])}/{num_runs}")
        
        return results
    
    def _calculate_statistics(self, runs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算统计数据"""
        if not runs_data:
            return {}
        
        # 提取各项指标
        metrics = {
            'execution_time': [r['execution_time'] for r in runs_data],
            'final_utility': [r['final_utility'] for r in runs_data],
            'completion_rate': [r['completion_rate'] for r in runs_data],
            'total_value': [r['total_value'] for r in runs_data],
            'avg_messages_per_round': [r['avg_messages_per_round'] for r in runs_data],
            'avg_active_links': [r['avg_active_links'] for r in runs_data],
            'total_steps': [r['total_steps'] for r in runs_data],
            'ksc_rounds': [r['ksc_rounds'] for r in runs_data]
        }
        
        # 计算统计量
        statistics = {}
        for metric_name, values in metrics.items():
            if values:
                statistics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
        
        return statistics