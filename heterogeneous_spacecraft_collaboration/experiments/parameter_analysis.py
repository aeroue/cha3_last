# experiments/parameter_analysis.py
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

class ParameterAnalysisExperiments:
    """
    参数敏感性分析实验：研究各参数对系统性能的影响
    """
    def __init__(self,
                 results_dir: str = "results/experiments/parameter_analysis",
                 save_plots: bool = True,
                 show_plots: bool = True):
        """
        初始化参数敏感性分析实验
        
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
    
    def run_k_value_analysis(self,
                            base_scenario: Optional[str] = None,
                            k_values: Optional[List[int]] = None,
                            max_steps: Optional[int] = None,
                            num_runs: int = 3) -> None:
        """
        运行K值敏感性分析
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            k_values (List[int], optional): 要测试的K值列表，如果未提供则使用默认值
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个K值运行的次数
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
        
        # 如果未提供K值列表，则使用默认值
        if k_values is None:
            k_values = [1, 2, 3, 4, 5]
            
        print(f"开始K值敏感性分析，基础场景: {base_scenario}, K值: {k_values}")
        
        # 运行K值分析
        self.experiment_manager.analyze_k_impact(
            base_scenario=base_scenario,
            k_values=k_values,
            experiment_name="K值敏感性分析",
            max_steps=max_steps,
            num_runs=num_runs
        )
    
    def run_communication_range_analysis(self,
                                       base_scenario: Optional[str] = None,
                                       comm_ranges: Optional[List[float]] = None,
                                       max_steps: Optional[int] = None,
                                       num_runs: int = 3) -> None:
        """
        运行通信范围敏感性分析
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            comm_ranges (List[float], optional): 要测试的通信范围列表，如果未提供则使用默认值
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个通信范围运行的次数
        """
        # 如果未提供基础场景，则选择一个适合的场景
        if base_scenario is None:
            candidates = [s for s in self.scenario_generator.get_available_scenarios() 
                         if "dynamic_comms" in s or "adaptive" in s]
            if candidates:
                base_scenario = candidates[0]
            else:
                print("错误: 未找到合适的动态通信场景作为基础")
                return
        
        # 如果未提供通信范围列表，则使用默认值
        if comm_ranges is None:
            comm_ranges = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
            
        print(f"开始通信范围敏感性分析，基础场景: {base_scenario}, 通信范围: {comm_ranges}")
        
        # 运行通信范围分析
        self.experiment_manager.analyze_communication_impact(
            base_scenario=base_scenario,
            comm_range_values=comm_ranges,
            experiment_name="通信范围敏感性分析",
            max_steps=max_steps,
            num_runs=num_runs
        )
    
    def run_reasoning_level_analysis(self,
                                   base_scenario: Optional[str] = None,
                                   max_steps: Optional[int] = None,
                                   num_runs: int = 3) -> None:
        """
        运行推理级别敏感性分析
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个推理级别配置运行的次数
        """
        # 如果未提供基础场景，则选择一个适合的场景
        if base_scenario is None:
            candidates = [s for s in self.scenario_generator.get_available_scenarios() 
                         if "weak_comms" in s or "aif" in s]
            if candidates:
                base_scenario = candidates[0]
            else:
                print("错误: 未找到合适的AIF场景作为基础")
                return
        
        print(f"开始推理级别敏感性分析，基础场景: {base_scenario}")
        
        # 运行推理级别分析
        self.experiment_manager.analyze_reasoning_level_impact(
            base_scenario=base_scenario,
            reasoning_levels=[0, 1, 2],  # 测试0阶、1阶、2阶推理
            experiment_name="推理级别敏感性分析",
            max_steps=max_steps,
            num_runs=num_runs
        )
    
    def run_csi_threshold_analysis(self,
                                 base_scenario: Optional[str] = None,
                                 max_steps: Optional[int] = None,
                                 num_runs: int = 3) -> None:
        """
        运行CSI阈值敏感性分析
        
        参数:
            base_scenario (str, optional): 基础场景名称，如果未提供则自动选择
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个CSI阈值配置运行的次数
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
        
        print(f"开始CSI阈值敏感性分析，基础场景: {base_scenario}")
        
        # 创建不同CSI阈值的场景
        threshold_configs = [
            (0.2, 0.8, "保守切换"),
            (0.3, 0.7, "默认切换"),
            (0.4, 0.6, "敏感切换"),
            (0.5, 0.5, "无迟滞切换")
        ]
        
        scenarios = []
        for low, high, name in threshold_configs:
            scenario = self.scenario_generator.create_adaptive_threshold_scenario(
                base_scenario, low, high, f"{base_scenario}_thresh_{int(low*10)}_{int(high*10)}"
            )
            scenarios.append(scenario)
        
        # 运行比较实验
        self.experiment_manager.compare_scenarios(
            scenario_list=scenarios,
            comparison_name="CSI阈值敏感性分析",
            max_steps=max_steps,
            num_runs=num_runs
        )
    
    def run_all_parameter_analyses(self,
                                 max_steps: Optional[int] = None,
                                 num_runs: int = 3) -> None:
        """
        运行所有参数敏感性分析实验
        
        参数:
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个配置运行的次数
        """
        print("开始运行所有参数敏感性分析实验...")
        
        # 运行K值敏感性分析
        self.run_k_value_analysis(max_steps=max_steps, num_runs=num_runs)
        
        # 运行通信范围敏感性分析
        self.run_communication_range_analysis(max_steps=max_steps, num_runs=num_runs)
        
        # 运行推理级别敏感性分析
        self.run_reasoning_level_analysis(max_steps=max_steps, num_runs=num_runs)
        
        # 运行CSI阈值敏感性分析
        self.run_csi_threshold_analysis(max_steps=max_steps, num_runs=num_runs)
        
        print("所有参数敏感性分析实验完成！")

# 如果直接运行此脚本，执行参数敏感性分析实验
if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="运行异构航天器协同任务分配参数敏感性分析实验")
    
    parser.add_argument("--experiment", type=str, default="all", 
                       choices=["all", "k_value", "comm_range", "reasoning", "csi_threshold"],
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
    parameter_analysis = ParameterAnalysisExperiments(
        save_plots=not args.no_save,
        show_plots=not args.no_show
    )
    
    # 根据参数运行指定实验
    if args.experiment == "all":
        parameter_analysis.run_all_parameter_analyses(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "k_value":
        parameter_analysis.run_k_value_analysis(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "comm_range":
        parameter_analysis.run_communication_range_analysis(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "reasoning":
        parameter_analysis.run_reasoning_level_analysis(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.experiment == "csi_threshold":
        parameter_analysis.run_csi_threshold_analysis(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    else:
        print(f"未知的实验类型: {args.experiment}")