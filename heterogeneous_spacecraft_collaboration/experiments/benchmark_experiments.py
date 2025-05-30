# experiments/benchmark_experiments.py
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import argparse

# 导入项目路径
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
_project_root_directory = os.path.dirname(_current_file_directory)
if _project_root_directory not in sys.path:
    sys.path.insert(0, _project_root_directory)

from .experiment_manager import ExperimentManager
from .scenario_generators import ScenarioGenerator

class BenchmarkExperiments:
    """
    基准比较实验：比较不同通信模式下的系统性能
    """
    def __init__(self, 
                 results_dir: str = "results/experiments/benchmark",
                 save_plots: bool = True,
                 show_plots: bool = True):
        """
        初始化基准比较实验
        
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
        
    def run_communication_mode_comparison(self, 
                                         base_scenario: Optional[str] = None,
                                         max_steps: Optional[int] = None,
                                         num_runs: int = 3) -> None:
        """
        运行通信模式比较实验
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个场景运行的次数
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
        
        print(f"开始通信模式比较实验，基础场景: {base_scenario}")
        
        # 创建比较场景
        strong_scenario = self.scenario_generator.create_communication_mode_scenario(
            base_scenario, "强通信", enable_aif=False, 
            enable_adaptive=False, enable_ksc=True
        )
        weak_scenario = self.scenario_generator.create_communication_mode_scenario(
            base_scenario, "弱通信", enable_aif=True, 
            enable_adaptive=False, enable_ksc=False
        )
        adaptive_scenario = self.scenario_generator.create_communication_mode_scenario(
            base_scenario, "自适应", enable_aif=True, 
            enable_adaptive=True, enable_ksc=True
        )
        
        # 运行比较实验
        comparison_scenarios = [strong_scenario, weak_scenario, adaptive_scenario]
        self.experiment_manager.compare_scenarios(
            scenario_list=comparison_scenarios,
            comparison_name="通信模式比较",
            max_steps=max_steps,
            num_runs=num_runs
        )
        
    def run_adaptive_vs_fixed_comparison(self, 
                                       base_scenario: Optional[str] = None,
                                       max_steps: Optional[int] = None,
                                       num_runs: int = 3) -> None:
        """
        运行自适应vs固定通信模式比较实验
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个场景运行的次数
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
        
        print(f"开始自适应vs固定通信模式比较实验，基础场景: {base_scenario}")
        
        # 创建自适应阈值变化场景
        adaptive_default = self.scenario_generator.create_adaptive_threshold_scenario(
            base_scenario, 0.3, 0.7, f"{base_scenario}_adaptive_default"
        )
        
        adaptive_sensitive = self.scenario_generator.create_adaptive_threshold_scenario(
            base_scenario, 0.4, 0.6, f"{base_scenario}_adaptive_sensitive"
        )
        
        adaptive_conservative = self.scenario_generator.create_adaptive_threshold_scenario(
            base_scenario, 0.2, 0.8, f"{base_scenario}_adaptive_conservative"
        )
        
        # 运行比较实验
        comparison_scenarios = [adaptive_default, adaptive_sensitive, adaptive_conservative]
        self.experiment_manager.compare_scenarios(
            scenario_list=comparison_scenarios,
            comparison_name="自适应阈值比较",
            max_steps=max_steps,
            num_runs=num_runs
        )
        
    def run_dynamic_communication_experiment(self, 
                                           base_scenario: Optional[str] = None,
                                           max_steps: Optional[int] = None,
                                           num_runs: int = 3) -> None:
        """
        运行动态通信条件下的性能实验
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个场景运行的次数
        """
        # 如果未提供基础场景，则选择一个适合的场景
        if base_scenario is None:
            candidates = [s for s in self.scenario_generator.get_available_scenarios() 
                         if "dynamic_comms" in s]
            if candidates:
                base_scenario = candidates[0]
            else:
                print("错误: 未找到合适的动态通信场景作为基础")
                return
        
        print(f"开始动态通信条件实验，基础场景: {base_scenario}")
        
        # 创建不同通信范围的场景
        range_high = self.scenario_generator.create_comm_range_scenario(
            base_scenario, 300.0, f"{base_scenario}_range_high"
        )
        
        range_medium = self.scenario_generator.create_comm_range_scenario(
            base_scenario, 150.0, f"{base_scenario}_range_medium"
        )
        
        range_low = self.scenario_generator.create_comm_range_scenario(
            base_scenario, 75.0, f"{base_scenario}_range_low"
        )
        
        # 运行比较实验
        comparison_scenarios = [range_high, range_medium, range_low]
        self.experiment_manager.compare_scenarios(
            scenario_list=comparison_scenarios,
            comparison_name="动态通信条件比较",
            max_steps=max_steps,
            num_runs=num_runs
        )
        
    def run_all_benchmark_experiments(self, 
                                     max_steps: Optional[int] = None,
                                     num_runs: int = 3) -> None:
        """
        运行所有基准比较实验
        
        参数:
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个场景运行的次数
        """
        print("开始运行所有基准比较实验...")
        
        # 运行通信模式比较实验
        self.run_communication_mode_comparison(max_steps=max_steps, num_runs=num_runs)
        
        # 运行自适应阈值比较实验
        self.run_adaptive_vs_fixed_comparison(max_steps=max_steps, num_runs=num_runs)
        
        # 运行动态通信条件实验
        self.run_dynamic_communication_experiment(max_steps=max_steps, num_runs=num_runs)
        
        print("所有基准比较实验完成！")

# 如果直接运行此脚本，执行基准比较实验
if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="运行异构航天器协同任务分配基准比较实验")
    
    parser.add_argument("--experiment", type=str, default="all", 
                       choices=["all", "modes", "adaptive", "dynamic"],
                       help="要运行的实验类型")
    
    parser.add_argument("--max_steps", type=int, default=None,
                       help="最大仿真步数")
    
    parser.add_argument("--num_runs", type=int, default=3,
                       help="每个场景运行的次数")
    
    parser.add_argument("--no_show", action="store_true",
                       help="不显示图表")
    
    parser.add_argument("--no_save", action="store_true",
                       help="不保存图表")
    
    args = parser.parse_args()
    
    # 创建实验对象
    benchmark = BenchmarkExperiments(
        save_plots=not args.no_save,
        show_plots=not args.no_show
    )
    
    # 根据参数运行指定实验
    if args.experiment == "all":
        benchmark.run_all_benchmark_experiments(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "modes":
        benchmark.run_communication_mode_comparison(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "adaptive":
        benchmark.run_adaptive_vs_fixed_comparison(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "dynamic":
        benchmark.run_dynamic_communication_experiment(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    else:
        print(f"未知的实验类型: {args.experiment}")