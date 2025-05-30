# experiments/scalability_experiments.py
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
import copy

class ScalabilityExperiments:
    """
    扩展性实验类：测试算法在不同规模下的性能
    """
    def __init__(self,
                 results_dir: str = "results/experiments/scalability",
                 save_plots: bool = True,
                 show_plots: bool = True):
        """
        初始化扩展性实验
        
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
    
    def run_agent_scaling_experiment(self,
                                   base_scenario: Optional[str] = None,
                                   agent_counts: Optional[List[int]] = None,
                                   max_steps: Optional[int] = None,
                                   num_runs: int = 3) -> None:
        """
        智能体规模扩展性实验
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            agent_counts (List[int], optional): 要测试的智能体数量列表
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
        
        # 如果未提供智能体数量列表，则使用默认值
        if agent_counts is None:
            agent_counts = [3, 5, 8, 10]
            
        print(f"开始智能体规模扩展性实验，基础场景: {base_scenario}，智能体数量: {agent_counts}")
        
        # 创建不同智能体数量的场景
        scenarios = []
        for count in agent_counts:
            scenario_name = self._create_agent_count_scenario(base_scenario, count)
            scenarios.append(scenario_name)
        
        # 运行比较实验
        self.experiment_manager.compare_scenarios(
            scenario_list=scenarios,
            comparison_name="智能体规模扩展性实验",
            max_steps=max_steps,
            num_runs=num_runs
        )
    
    def _create_agent_count_scenario(self, 
                                   base_scenario_name: str, 
                                   agent_count: int) -> str:
        """
        创建特定智能体数量的场景
        
        参数:
            base_scenario_name (str): 基础场景名称
            agent_count (int): 智能体数量
            
        返回:
            str: 新场景名称
        """
        # 加载基础场景
        base_config = self.scenario_generator.base_scenarios.get(base_scenario_name)
        if not base_config:
            raise ValueError(f"基础场景 '{base_scenario_name}' 不存在")
            
        # 创建新配置
        new_config = copy.deepcopy(base_config)
        new_config["num_agents"] = agent_count
        
        # 生成场景名称
        new_scenario_name = f"{base_scenario_name}_agents{agent_count}"
        new_config["scenario_display_name"] = f"{new_config.get('scenario_display_name', base_scenario_name)} ({agent_count}智能体)"
        
        # 重新生成航天器
        sc_template = new_config.get("agent_definitions_template", {})
        new_config["spacecrafts"] = []
        
        for i in range(agent_count):
            sc_id = f"SC{i+1}"
            
            # 获取模板参数
            physical_params_base = sc_template.get("physical_params_base", {})
            ksc_params_base = sc_template.get("ksc_params_base", {})
            aif_agent_type_options = sc_template.get("aif_agent_type_options", ['s', 'A'])
            initial_radius = sc_template.get("initial_pos_placement_radius", 200.0)
            velocity_range = sc_template.get("initial_velocity_range", (-0.1, 0.1))
            
            # 计算初始位置
            angle = 2 * np.pi * i / agent_count
            radius = initial_radius * (0.8 + 0.2 * np.random.random())
            pos_x = radius * np.cos(angle)
            pos_y = radius * np.sin(angle)
            
            # 随机初始速度
            vel_x = np.random.uniform(velocity_range[0], velocity_range[1])
            vel_y = np.random.uniform(velocity_range[0], velocity_range[1])
            
            # 智能体参数
            sc_def = {
                "id": sc_id,
                "initial_state": [pos_x, pos_y, vel_x, vel_y],
                "physical_params": copy.deepcopy(physical_params_base),
                "ksc_params": copy.deepcopy(ksc_params_base)
            }
            
            # 随机分配智能体类型
            if aif_agent_type_options:
                sc_def["physical_params"]["type"] = np.random.choice(aif_agent_type_options)
            
            new_config["spacecrafts"].append(sc_def)
        
        # 注册到全局场景
        from simulation.scenarios import SCENARIOS
        SCENARIOS[new_scenario_name] = new_config
        
        return new_scenario_name
    
    def run_task_scaling_experiment(self,
                                  base_scenario: Optional[str] = None,
                                  task_counts: Optional[List[int]] = None,
                                  max_steps: Optional[int] = None,
                                  num_runs: int = 3) -> None:
        """
        任务规模扩展性实验
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            task_counts (List[int], optional): 要测试的任务数量列表
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
        
        # 如果未提供任务数量列表，则使用默认值
        if task_counts is None:
            task_counts = [2, 4, 6, 8]
            
        print(f"开始任务规模扩展性实验，基础场景: {base_scenario}，任务数量: {task_counts}")
        
        # 创建不同任务数量的场景
        scenarios = []
        for count in task_counts:
            scenario_name = self._create_task_count_scenario(base_scenario, count)
            scenarios.append(scenario_name)
        
        # 运行比较实验
        self.experiment_manager.compare_scenarios(
            scenario_list=scenarios,
            comparison_name="任务规模扩展性实验",
            max_steps=max_steps,
            num_runs=num_runs
        )
    
    def _create_task_count_scenario(self, 
                                  base_scenario_name: str, 
                                  task_count: int) -> str:
        """
        创建特定任务数量的场景
        
        参数:
            base_scenario_name (str): 基础场景名称
            task_count (int): 任务数量
            
        返回:
            str: 新场景名称
        """
        # 加载基础场景
        base_config = self.scenario_generator.base_scenarios.get(base_scenario_name)
        if not base_config:
            raise ValueError(f"基础场景 '{base_scenario_name}' 不存在")
            
        # 创建新配置
        new_config = copy.deepcopy(base_config)
        new_config["num_tasks"] = task_count
        
        # 生成场景名称
        new_scenario_name = f"{base_scenario_name}_tasks{task_count}"
        new_config["scenario_display_name"] = f"{new_config.get('scenario_display_name', base_scenario_name)} ({task_count}任务)"
        
        # 获取任务模板参数
        task_template = new_config.get("task_definitions_template", {})
        position_range = task_template.get("position_range", (-300.0, 300.0))
        reward_ranges = task_template.get("reward_ranges_by_type", {})
        risk_ranges = task_template.get("risk_ranges_by_type", {})
        required_caps = task_template.get("required_capabilities_options", [["cap_sensor_dist"]])
        task_type_names = new_config.get("task_type_names", ["默认类型"])
        
        # 重新生成任务
        from simulation.scenarios import create_task_definition
        
        new_tasks = {}
        for i in range(task_count):
            task_id = f"Task{i+1}"
            task_def = create_task_definition(
                task_id_str=task_id,
                num_total_task_types=len(task_type_names),
                task_type_names_list=task_type_names,
                position_range=position_range,
                reward_ranges_by_type=reward_ranges,
                risk_ranges_by_type=risk_ranges,
                required_capabilities_options=required_caps,
                total_workload_range=task_template.get("total_workload_range", (80.0, 120.0)),
                work_rate_per_agent_dt_range=task_template.get("work_rate_per_agent_dt_range", (0.5, 1.5))
            )
            new_tasks[task_id] = task_def
        
        new_config["tasks"] = new_tasks
        
        # 注册到全局场景
        from simulation.scenarios import SCENARIOS
        SCENARIOS[new_scenario_name] = new_config
        
        return new_scenario_name
    
    def run_computation_time_experiment(self,
                                      base_scenario: Optional[str] = None,
                                      parameter_variations: Optional[Dict[str, List[Any]]] = None,
                                      max_steps: Optional[int] = None,
                                      num_runs: int = 3) -> None:
        """
        计算时间扩展性实验
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            parameter_variations (Dict[str, List[Any]], optional): 要测试的参数变化
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
        
        # 如果未提供参数变化，则使用默认值
        if parameter_variations is None:
            parameter_variations = {
                "k_value": [1, 2, 3, 4, 5],
                "agents": [3, 5, 8],
                "tasks": [2, 4, 6]
            }
            
        print(f"开始计算时间扩展性实验，基础场景: {base_scenario}")
        
        # 运行不同K值的计算时间实验
        if "k_value" in parameter_variations:
            k_values = parameter_variations["k_value"]
            print(f"测试K值计算时间: {k_values}")
            self.experiment_manager.analyze_k_impact(
                base_scenario=base_scenario,
                k_values=k_values,
                experiment_name="K值计算时间分析",
                max_steps=max_steps,
                num_runs=num_runs
            )
        
        # 运行不同智能体数量的计算时间实验
        if "agents" in parameter_variations:
            agent_counts = parameter_variations["agents"]
            scenarios = []
            for count in agent_counts:
                scenario_name = self._create_agent_count_scenario(base_scenario, count)
                scenarios.append(scenario_name)
                
            self.experiment_manager.compare_scenarios(
                scenario_list=scenarios,
                comparison_name="智能体数量计算时间分析",
                max_steps=max_steps,
                num_runs=num_runs
            )
        
        # 运行不同任务数量的计算时间实验
        if "tasks" in parameter_variations:
            task_counts = parameter_variations["tasks"]
            scenarios = []
            for count in task_counts:
                scenario_name = self._create_task_count_scenario(base_scenario, count)
                scenarios.append(scenario_name)
                
            self.experiment_manager.compare_scenarios(
                scenario_list=scenarios,
                comparison_name="任务数量计算时间分析",
                max_steps=max_steps,
                num_runs=num_runs
            )
    
    def run_all_scalability_experiments(self,
                                      max_steps: Optional[int] = None,
                                      num_runs: int = 3) -> None:
        """
        运行所有扩展性实验
        
        参数:
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个配置运行的次数
        """
        print("开始运行所有扩展性实验...")
        
        # 运行智能体规模扩展性实验
        self.run_agent_scaling_experiment(max_steps=max_steps, num_runs=num_runs)
        
        # 运行任务规模扩展性实验
        self.run_task_scaling_experiment(max_steps=max_steps, num_runs=num_runs)
        
        # 运行计算时间扩展性实验
        self.run_computation_time_experiment(max_steps=max_steps, num_runs=num_runs)
        
        print("所有扩展性实验完成！")

# 如果直接运行此脚本，执行扩展性实验
if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="运行异构航天器协同任务分配扩展性实验")
    
    parser.add_argument("--experiment", type=str, default="all", 
                       choices=["all", "agent_scaling", "task_scaling", "computation_time"],
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
    scalability = ScalabilityExperiments(
        save_plots=not args.no_save,
        show_plots=not args.no_show
    )
    
    # 根据参数运行指定实验
    if args.experiment == "all":
        scalability.run_all_scalability_experiments(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "agent_scaling":
        scalability.run_agent_scaling_experiment(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "task_scaling":
        scalability.run_task_scaling_experiment(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "computation_time":
        scalability.run_computation_time_experiment(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    else:
        print(f"未知的实验类型: {args.experiment}")