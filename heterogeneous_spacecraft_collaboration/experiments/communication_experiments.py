# experiments/communication_experiments.py
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple

# 导入项目路径
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
_project_root_directory = os.path.dirname(_current_file_directory)
if _project_root_directory not in sys.path:
    sys.path.insert(0, _project_root_directory)

from .experiment_manager import ExperimentManager
from .scenario_generators import ScenarioGenerator

class CommunicationExperiments:
    """
    通信实验类：研究不同通信条件对系统性能的影响
    """
    def __init__(self,
                 results_dir: str = "results/experiments/communication",
                 save_plots: bool = True,
                 show_plots: bool = True):
        """
        初始化通信实验
        
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
    
    def run_communication_degradation_experiment(self,
                                               base_scenario: Optional[str] = None,
                                               max_steps: Optional[int] = None,
                                               num_runs: int = 3) -> None:
        """
        通信质量下降实验
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个配置运行的次数
        """
        # 如果未提供基础场景，则选择一个适合的场景
        if base_scenario is None:
            candidates = [s for s in self.scenario_generator.get_available_scenarios() 
                         if "dynamic_comms" in s or "adaptive" in s]
            if candidates:
                base_scenario = candidates[0]
            else:
                print("错误: 未找到合适的自适应场景作为基础")
                return
        
        print(f"开始通信质量下降实验，基础场景: {base_scenario}")
        
        # 创建带有通信中断的场景
        scenarios = []
        
        # 1. 创建通信范围动态下降场景
        degradation_scenario = self._create_degradation_scenario(base_scenario)
        scenarios.append(degradation_scenario)
        
        # 2. 创建通信突然中断场景
        blackout_scenario = self._create_blackout_scenario(base_scenario)
        scenarios.append(blackout_scenario)
        
        # 3. 创建通信范围波动场景
        fluctuation_scenario = self._create_fluctuation_scenario(base_scenario)
        scenarios.append(fluctuation_scenario)
        
        # 4. 创建基础场景，作为比较基准
        scenarios.append(base_scenario)
        
        # 运行比较实验
        self.experiment_manager.compare_scenarios(
            scenario_list=scenarios,
            comparison_name="通信质量变化实验",
            max_steps=max_steps,
            num_runs=num_runs
        )
    
    def _create_degradation_scenario(self, base_scenario_name: str) -> str:
        """
        创建通信质量逐渐下降的场景
        
        参数:
            base_scenario_name (str): 基础场景名称
            
        返回:
            str: 新场景名称
        """
        # 加载基础场景
        base_config = self.scenario_generator.base_scenarios.get(base_scenario_name)
        if not base_config:
            raise ValueError(f"基础场景 '{base_scenario_name}' 不存在")
            
        # 创建新配置
        new_config = copy.deepcopy(base_config)
        new_scenario_name = f"{base_scenario_name}_degradation"
        
        # 添加通信范围随时间变化的回调
        if "simulation_callbacks" not in new_config:
            new_config["simulation_callbacks"] = {"on_step_begin": []}
        
        # 初始通信范围
        initial_range = 300.0
        if "simulation_params" in new_config and "communication_range" in new_config["simulation_params"]:
            initial_range = new_config["simulation_params"]["communication_range"]
        
        # 添加通信范围逐渐下降的回调
        degradation_callback = {
            "condition": "env.time_step > 0 and env.time_step % 20 == 0 and env.time_step <= 200",
            "action": "update_env_param",
            "param_path": "sim_params.communication_range",
            "value": "max(10, env.sim_params.get('communication_range', 300.0) - 30)",
            "message": "通信范围下降"
        }
        
        # 确保初始范围被设置
        if "simulation_params" not in new_config:
            new_config["simulation_params"] = {}
        new_config["simulation_params"]["communication_range"] = initial_range
        
        new_config["simulation_callbacks"]["on_step_begin"].append(degradation_callback)
        new_config["scenario_display_name"] = f"{new_config.get('scenario_display_name', base_scenario_name)} (通信逐渐下降)"
        
        # 注册到全局场景
        from simulation.scenarios import SCENARIOS
        SCENARIOS[new_scenario_name] = new_config
        
        return new_scenario_name
    
    def _create_blackout_scenario(self, base_scenario_name: str) -> str:
        """
        创建通信突然中断的场景
        
        参数:
            base_scenario_name (str): 基础场景名称
            
        返回:
            str: 新场景名称
        """
        # 加载基础场景
        base_config = self.scenario_generator.base_scenarios.get(base_scenario_name)
        if not base_config:
            raise ValueError(f"基础场景 '{base_scenario_name}' 不存在")
            
        # 创建新配置
        new_config = copy.deepcopy(base_config)
        new_scenario_name = f"{base_scenario_name}_blackout"
        
        # 添加通信中断和恢复的回调
        if "simulation_callbacks" not in new_config:
            new_config["simulation_callbacks"] = {"on_step_begin": []}
        
        # 初始通信范围
        initial_range = 300.0
        if "simulation_params" in new_config and "communication_range" in new_config["simulation_params"]:
            initial_range = new_config["simulation_params"]["communication_range"]
        
        # 添加通信中断的回调
        blackout_callbacks = [
            {
                "condition": "env.time_step == 100",
                "action": "update_env_param",
                "param_path": "sim_params.communication_range",
                "value": 0,
                "message": "通信完全中断"
            },
            {
                "condition": "env.time_step == 200",
                "action": "update_env_param",
                "param_path": "sim_params.communication_range",
                "value": initial_range,
                "message": "通信恢复"
            }
        ]
        
        # 确保初始范围被设置
        if "simulation_params" not in new_config:
            new_config["simulation_params"] = {}
        new_config["simulation_params"]["communication_range"] = initial_range
        
        for callback in blackout_callbacks:
            new_config["simulation_callbacks"]["on_step_begin"].append(callback)
        
        new_config["scenario_display_name"] = f"{new_config.get('scenario_display_name', base_scenario_name)} (通信突然中断)"
        
        # 注册到全局场景
        from simulation.scenarios import SCENARIOS
        SCENARIOS[new_scenario_name] = new_config
        
        return new_scenario_name
    
    def _create_fluctuation_scenario(self, base_scenario_name: str) -> str:
        """
        创建通信范围波动的场景
        
        参数:
            base_scenario_name (str): 基础场景名称
            
        返回:
            str: 新场景名称
        """
        # 加载基础场景
        base_config = self.scenario_generator.base_scenarios.get(base_scenario_name)
        if not base_config:
            raise ValueError(f"基础场景 '{base_scenario_name}' 不存在")
            
        # 创建新配置
        new_config = copy.deepcopy(base_config)
        new_scenario_name = f"{base_scenario_name}_fluctuation"
        
        # 添加通信范围波动的回调
        if "simulation_callbacks" not in new_config:
            new_config["simulation_callbacks"] = {"on_step_begin": []}
        
        # 初始通信范围
        initial_range = 200.0
        if "simulation_params" in new_config and "communication_range" in new_config["simulation_params"]:
            initial_range = new_config["simulation_params"]["communication_range"]
        
        # 添加通信范围波动的回调
        fluctuation_callback = {
            "condition": "env.time_step > 0 and env.time_step % 20 == 0",
            "action": "update_env_param",
            "param_path": "sim_params.communication_range",
            "value": f"max(50, min(350, {initial_range} + 100 * np.sin(env.time_step * 0.05)))",
            "message": "通信范围波动"
        }
        
        # 确保初始范围被设置
        if "simulation_params" not in new_config:
            new_config["simulation_params"] = {}
        new_config["simulation_params"]["communication_range"] = initial_range
        
        new_config["simulation_callbacks"]["on_step_begin"].append(fluctuation_callback)
        new_config["scenario_display_name"] = f"{new_config.get('scenario_display_name', base_scenario_name)} (通信范围波动)"
        
        # 注册到全局场景
        from simulation.scenarios import SCENARIOS
        SCENARIOS[new_scenario_name] = new_config
        
        return new_scenario_name
    
    def run_csi_sensitivity_experiment(self,
                                     base_scenario: Optional[str] = None,
                                     max_steps: Optional[int] = None,
                                     num_runs: int = 3) -> None:
        """
        CSI灵敏度实验
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个配置运行的次数
        """
        # 如果未提供基础场景，则选择一个适合的场景
        if base_scenario is None:
            candidates = [s for s in self.scenario_generator.get_available_scenarios() 
                         if "dynamic_comms" in s or "adaptive" in s]
            if candidates:
                base_scenario = candidates[0]
            else:
                print("错误: 未找到合适的自适应场景作为基础")
                return
        
        print(f"开始CSI灵敏度实验，基础场景: {base_scenario}")
        
        # 创建不同CSI灵敏度的场景
        alpha_values = [0.05, 0.1, 0.2, 0.5, 0.8]
        scenarios = []
        
        for alpha in alpha_values:
            # 创建场景
            scenario_name = self._create_csi_alpha_scenario(base_scenario, alpha)
            scenarios.append(scenario_name)
        
        # 运行比较实验
        self.experiment_manager.compare_scenarios(
            scenario_list=scenarios,
            comparison_name="CSI灵敏度实验",
            max_steps=max_steps,
            num_runs=num_runs
        )
    
    def _create_csi_alpha_scenario(self, 
                                 base_scenario_name: str, 
                                 csi_alpha: float) -> str:
        """
        创建特定CSI平滑系数的场景
        
        参数:
            base_scenario_name (str): 基础场景名称
            csi_alpha (float): CSI平滑系数 (0-1)
            
        返回:
            str: 新场景名称
        """
        # 加载基础场景
        base_config = self.scenario_generator.base_scenarios.get(base_scenario_name)
        if not base_config:
            raise ValueError(f"基础场景 '{base_scenario_name}' 不存在")
            
        # 创建新配置
        new_config = copy.deepcopy(base_config)
        
        # 修改CSI平滑系数
        if "adaptive_controller_params" in new_config:
            new_config["adaptive_controller_params"]["csi_ema_alpha"] = csi_alpha
        
        # 生成场景名称
        new_scenario_name = f"{base_scenario_name}_csi_alpha_{int(csi_alpha*100)}"
        new_config["scenario_display_name"] = f"{new_config.get('scenario_display_name', base_scenario_name)} (CSI平滑α={csi_alpha:.2f})"
        
        # 注册到全局场景
        from simulation.scenarios import SCENARIOS
        SCENARIOS[new_scenario_name] = new_config
        
        return new_scenario_name
    
    def run_varying_delay_experiment(self,
                                   base_scenario: Optional[str] = None,
                                   max_steps: Optional[int] = None,
                                   num_runs: int = 3) -> None:
        """
        通信延迟实验
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个配置运行的次数
        """
        # 如果未提供基础场景，则选择一个适合的场景
        if base_scenario is None:
            candidates = [s for s in self.scenario_generator.get_available_scenarios() 
                         if "strong_comms" in s or "ksc" in s]
            if candidates:
                base_scenario = candidates[0]
            else:
                print("错误: 未找到合适的KSC场景作为基础")
                return
        
        print(f"开始通信延迟实验，基础场景: {base_scenario}")
        
        # 创建不同延迟的场景
        delay_values = [1.0, 2.0, 4.0, 8.0]  # 决策延迟 (单位: s)
        scenarios = []
        
        # 基准场景
        scenarios.append(base_scenario)
        
        for delay in delay_values:
            # 创建场景
            scenario_name = self._create_decision_delay_scenario(base_scenario, delay)
            scenarios.append(scenario_name)
        
        # 运行比较实验
        self.experiment_manager.compare_scenarios(
            scenario_list=scenarios,
            comparison_name="通信延迟实验",
            max_steps=max_steps,
            num_runs=num_runs
        )
    
    def _create_decision_delay_scenario(self, 
                                      base_scenario_name: str, 
                                      delay: float) -> str:
        """
        创建特定决策延迟的场景
        
        参数:
            base_scenario_name (str): 基础场景名称
            delay (float): 决策延迟 (s)
            
        返回:
            str: 新场景名称
        """
        # 加载基础场景
        base_config = self.scenario_generator.base_scenarios.get(base_scenario_name)
        if not base_config:
            raise ValueError(f"基础场景 '{base_scenario_name}' 不存在")
            
        # 创建新配置
        new_config = copy.deepcopy(base_config)
        
        # 修改决策间隔
        if "simulation_params" in new_config:
            new_config["simulation_params"]["dt_decision_ksc"] = delay
        
        # 生成场景名称
        new_scenario_name = f"{base_scenario_name}_delay_{int(delay)}"
        new_config["scenario_display_name"] = f"{new_config.get('scenario_display_name', base_scenario_name)} (决策延迟={delay:.1f}s)"
        
        # 注册到全局场景
        from simulation.scenarios import SCENARIOS
        SCENARIOS[new_scenario_name] = new_config
        
        return new_scenario_name
    
    def run_all_communication_experiments(self,
                                        max_steps: Optional[int] = None,
                                        num_runs: int = 3) -> None:
        """
        运行所有通信实验
        
        参数:
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个配置运行的次数
        """
        print("开始运行所有通信实验...")
        
        # 运行通信质量下降实验
        self.run_communication_degradation_experiment(max_steps=max_steps, num_runs=num_runs)
        
        # 运行CSI灵敏度实验
        self.run_csi_sensitivity_experiment(max_steps=max_steps, num_runs=num_runs)
        
        # 运行通信延迟实验
        self.run_varying_delay_experiment(max_steps=max_steps, num_runs=num_runs)
        
        print("所有通信实验完成！")

# 导入必要的库
import copy  # 确保导入copy模块

# 如果直接运行此脚本，执行通信实验
if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="运行异构航天器协同任务分配通信实验")
    
    parser.add_argument("--experiment", type=str, default="all", 
                       choices=["all", "degradation", "csi_sensitivity", "delay"],
                       help="要运行的实验类型")
    
    parser.add_argument("--max_steps", type=int, default=None,
                       help="最大仿真步数")
    
    parser.add_argument("--num_runs", type=int, default=3,
                       help="每个配置运行的次数")
    
    parser.add_argument("--no_show", action="store_true",
                       help="不显示图表")
    
    parser.add_argument("--no_save", action="store_true",
                       help="不保存图表")
    
    args = parser.parse_args()
    
    # 创建实验对象
    comm_experiments = CommunicationExperiments(
        save_plots=not args.no_save,
        show_plots=not args.no_show
    )
    
    # 根据参数运行指定实验
    if args.experiment == "all":
        comm_experiments.run_all_communication_experiments(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "degradation":
        comm_experiments.run_communication_degradation_experiment(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "csi_sensitivity":
        comm_experiments.run_csi_sensitivity_experiment(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "delay":
        comm_experiments.run_varying_delay_experiment(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    else:
        print(f"未知的实验类型: {args.experiment}")