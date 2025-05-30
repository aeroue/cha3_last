# experiments/monte_carlo_experiments.py
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# 导入项目路径
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
_project_root_directory = os.path.dirname(_current_file_directory)
if _project_root_directory not in sys.path:
    sys.path.insert(0, _project_root_directory)

from .experiment_manager import ExperimentManager
from .scenario_generators import ScenarioGenerator

class MonteCarloExperiments:
    """
    蒙特卡洛实验类：评估算法在随机初始条件下的稳健性
    """
    def __init__(self,
                 results_dir: str = "results/experiments/monte_carlo",
                 save_plots: bool = True,
                 show_plots: bool = True):
        """
        初始化蒙特卡洛实验
        
        参数:
            results_dir (str): 结果保存目录
            save_plots (bool): 是否保存图表
            show_plots (bool): 是否显示图表
        """
        self.results_dir = results_dir
        self.save_plots = save_plots
        self.show_plots = show_plots
        
        # 创建实验管理器
        self.experiment_manager = ExperimentManager(
            results_base_dir=results_dir,
            save_visualization=save_plots,
            show_plots=show_plots
        )
        
        # 创建场景生成器
        self.scenario_generator = ScenarioGenerator()
    
    def run_basic_monte_carlo(self,
                             base_scenario: Optional[str] = None,
                             num_runs: int = 10,
                             max_steps: Optional[int] = None) -> None:
        """
        在单一场景上运行基本蒙特卡洛实验
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            num_runs (int): 蒙特卡洛运行次数
            max_steps (int, optional): 最大仿真步数
        """
        # 如果未提供基础场景，则选择一个适合的场景
        if base_scenario is None:
            candidates = [s for s in self.scenario_generator.get_available_scenarios() 
                         if "dynamic_comms" in s or "adaptive" in s]
            if candidates:
                base_scenario = candidates[0]
            else:
                print("错误: 未找到合适的场景作为基础")
                return
        
        print(f"开始基本蒙特卡洛实验，基础场景: {base_scenario}，运行次数: {num_runs}")
        
        # 运行蒙特卡洛实验
        self.experiment_manager.run_monte_carlo_experiment(
            scenario_name=base_scenario,
            num_runs=num_runs,
            experiment_name="基本蒙特卡洛实验",
            max_steps=max_steps
        )
    
    def run_random_task_position_monte_carlo(self,
                                           base_scenario: Optional[str] = None,
                                           num_runs: int = 10,
                                           max_steps: Optional[int] = None) -> None:
        """
        随机任务位置蒙特卡洛实验
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            num_runs (int): 蒙特卡洛运行次数
            max_steps (int, optional): 最大仿真步数
        """
        # 如果未提供基础场景，则选择一个适合的场景
        if base_scenario is None:
            candidates = [s for s in self.scenario_generator.get_available_scenarios() 
                         if "strong_comms" in s or "ksc" in s]
            if candidates:
                base_scenario = candidates[0]
            else:
                print("错误: 未找到合适的场景作为基础")
                return
        
        print(f"开始随机任务位置蒙特卡洛实验，基础场景: {base_scenario}，运行次数: {num_runs}")
        
        # 创建随机任务位置场景
        scenario_list = []
        for i in range(num_runs):
            # 加载基础场景
            base_config = self.scenario_generator.base_scenarios.get(base_scenario)
            if not base_config:
                print(f"错误: 基础场景 '{base_scenario}' 不存在")
                return
                
            # 创建新的随机任务位置场景
            new_config = copy.deepcopy(base_config)
            
            # 修改任务位置为随机值
            position_range = (-250.0, 250.0)  # 默认位置范围
            if "task_definitions_template" in new_config and "position_range" in new_config["task_definitions_template"]:
                position_range = new_config["task_definitions_template"]["position_range"]
                
            for task_id, task_def in new_config.get("tasks", {}).items():
                # 随机生成新位置
                new_pos = np.random.uniform(position_range[0], position_range[1], 2)
                task_def["position"] = new_pos.tolist()
                if "current_state" in task_def:
                    task_def["current_state"][:2] = new_pos
                
            # 保存修改后的场景
            new_scenario_name = f"{base_scenario}_random_task_pos_{i+1}"
            new_config["scenario_display_name"] = f"{new_config.get('scenario_display_name', base_scenario)} (随机任务位置 #{i+1})"
            
            # 注册到全局场景
            from simulation.scenarios import SCENARIOS
            SCENARIOS[new_scenario_name] = new_config
            scenario_list.append(new_scenario_name)
        
        # 运行比较实验
        self.experiment_manager.compare_scenarios(
            scenario_list=scenario_list,
            comparison_name="随机任务位置蒙特卡洛实验",
            max_steps=max_steps,
            random_seed=42,  # 使用固定种子以便稳定比较
            num_runs=1  # 每个场景只运行一次，因为场景本身已经随机化
        )
        
        # 清理临时场景
        from simulation.scenarios import SCENARIOS
        for scenario_name in scenario_list:
            if scenario_name in SCENARIOS:
                del SCENARIOS[scenario_name]
    
    def run_noisy_observation_monte_carlo(self,
                                        base_scenario: Optional[str] = None,
                                        num_runs: int = 10,
                                        noise_levels: Optional[List[float]] = None,
                                        max_steps: Optional[int] = None) -> None:
        """
        观测噪声影响蒙特卡洛实验
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            num_runs (int): 每个噪声级别的运行次数
            noise_levels (List[float], optional): 要测试的噪声级别列表
            max_steps (int, optional): 最大仿真步数
        """
        # 如果未提供基础场景，则选择一个适合的场景
        if base_scenario is None:
            candidates = [s for s in self.scenario_generator.get_available_scenarios() 
                         if "weak_comms" in s or "aif" in s]
            if candidates:
                base_scenario = candidates[0]
            else:
                print("错误: 未找到合适的场景作为基础")
                return
        
        # 如果未提供噪声级别，则使用默认值
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
            
        print(f"开始观测噪声影响实验，基础场景: {base_scenario}，噪声级别: {noise_levels}")
        
        # 为每个噪声级别创建场景
        for noise in noise_levels:
            # 创建噪声场景
            noise_scenario = self._create_noise_level_scenario(base_scenario, noise)
            
            # 运行蒙特卡洛实验
            self.experiment_manager.run_monte_carlo_experiment(
                scenario_name=noise_scenario,
                num_runs=num_runs,
                experiment_name=f"噪声级别{noise}蒙特卡洛实验",
                max_steps=max_steps
            )
    
    def _create_noise_level_scenario(self, 
                                   base_scenario_name: str, 
                                   noise_level: float) -> str:
        """
        创建特定观测噪声级别的场景
        
        参数:
            base_scenario_name (str): 基础场景名称
            noise_level (float): 观测噪声级别
            
        返回:
            str: 新场景名称
        """
        # 加载基础场景
        base_config = self.scenario_generator.base_scenarios.get(base_scenario_name)
        if not base_config:
            raise ValueError(f"基础场景 '{base_scenario_name}' 不存在")
            
        # 创建新配置
        new_config = copy.deepcopy(base_config)
        
        # 修改噪声参数
        if "aif_global_hyperparams" in new_config:
            new_config["aif_global_hyperparams"]["obs_variance_pos"] = noise_level
            new_config["aif_global_hyperparams"]["obs_variance_vel"] = noise_level / 10
        
        # 生成场景名称
        new_scenario_name = f"{base_scenario_name}_noise_{int(noise_level*100)}"
        new_config["scenario_display_name"] = f"{new_config.get('scenario_display_name', base_scenario_name)} (噪声={noise_level:.2f})"
        
        # 注册到全局场景
        from simulation.scenarios import SCENARIOS
        SCENARIOS[new_scenario_name] = new_config
        
        return new_scenario_name
    
    def run_all_monte_carlo_experiments(self,
                                      max_steps: Optional[int] = None,
                                      num_runs: int = 10) -> None:
        """
        运行所有蒙特卡洛实验
        
        参数:
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 蒙特卡洛运行次数
        """
        print("开始运行所有蒙特卡洛实验...")
        
        # 运行基本蒙特卡洛实验
        self.run_basic_monte_carlo(max_steps=max_steps, num_runs=num_runs)
        
        # 运行随机任务位置蒙特卡洛实验
        self.run_random_task_position_monte_carlo(max_steps=max_steps, num_runs=5)  # 减少运行次数
        
        # 运行观测噪声影响实验
        self.run_noisy_observation_monte_carlo(max_steps=max_steps, num_runs=5)  # 减少运行次数
        
        print("所有蒙特卡洛实验完成！")

# 导入必要的库
import copy  # 确保导入copy模块

# 如果直接运行此脚本，执行蒙特卡洛实验
if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="运行异构航天器协同任务分配蒙特卡洛实验")
    
    parser.add_argument("--experiment", type=str, default="all", 
                       choices=["all", "basic", "task_position", "observation_noise"],
                       help="要运行的实验类型")
    
    parser.add_argument("--max_steps", type=int, default=None,
                       help="最大仿真步数")
    
    parser.add_argument("--num_runs", type=int, default=10,
                       help="蒙特卡洛运行次数")
    
    parser.add_argument("--no_show", action="store_true",
                       help="不显示图表")
    
    parser.add_argument("--no_save", action="store_true",
                       help="不保存图表")
    
    args = parser.parse_args()
    
    # 创建实验对象
    monte_carlo = MonteCarloExperiments(
        save_plots=not args.no_save,
        show_plots=not args.no_show
    )
    
    # 根据参数运行指定实验
    if args.experiment == "all":
        monte_carlo.run_all_monte_carlo_experiments(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "basic":
        monte_carlo.run_basic_monte_carlo(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "task_position":
        monte_carlo.run_random_task_position_monte_carlo(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "observation_noise":
        monte_carlo.run_noisy_observation_monte_carlo(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    else:
        print(f"未知的实验类型: {args.experiment}")