# run_all_experiments.py
import os
import sys
import argparse
import time
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from experiments.fix_chinese import setup_chinese_fonts
setup_chinese_fonts()


# 在imports后添加
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STSong', 'DengXian', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print("已启用中文字体支持")
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

# 创建一个辅助函数
# 在run_all_experiments.py中完善中文字体支持函数
def setup_chinese_fonts():
    """设置中文字体支持"""
    import matplotlib.font_manager as fm
    
    # 尝试找到系统中存在的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STSong', 'DengXian', 
                     'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans CN']
    
    # 检查是否有自定义字体目录
    custom_font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
    if os.path.exists(custom_font_dir):
        for font_file in os.listdir(custom_font_dir):
            if font_file.endswith(('.ttf', '.otf')):
                fm.fontManager.addfont(os.path.join(custom_font_dir, font_file))
                print(f"已添加自定义字体: {font_file}")
    
    # 尝试设置中文字体
    for font in chinese_fonts:
        try:
            if fm.findfont(font, fallback_to_default=False):
                matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
                matplotlib.rcParams['axes.unicode_minus'] = False
                print(f"成功设置中文字体: {font}")
                return True
        except:
            continue
    
    print("警告: 未找到合适的中文字体，尝试使用备用方案...")
    matplotlib.rcParams['font.sans-serif'] = chinese_fonts + ['Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return False

# 在main函数开始处调用
setup_chinese_fonts()  # 设置中文字体

# 当前目录加入导入路径
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入实验模块
from experiments.experiment_manager import ExperimentManager
from experiments.benchmark_experiments import BenchmarkExperiments
from experiments.parameter_analysis import ParameterAnalysisExperiments
from experiments.communication_experiments import CommunicationExperiments
from experiments.monte_carlo_experiments import MonteCarloExperiments
from experiments.scalability_experiments import ScalabilityExperiments
from experiments.analysis_utilities import ResultsAnalyzer
from experiments.fix_chinese import setup_chinese_fonts
setup_chinese_fonts()  # 设置中文字体
# 设置字体以支持中文显示
def setup_parser() -> argparse.ArgumentParser:
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(description="运行异构航天器协同任务分配算法实验")
    
    parser.add_argument("--experiment_type", type=str, default="all",
                       choices=["all", "benchmark", "parameter", "communication", "monte_carlo", "scalability"],
                       help="要运行的实验类型")
    
    parser.add_argument("--specific_experiment", type=str, default="",
                       help="要运行的特定实验名称 (如 'k_value', 'comm_range' 等)")
    
    parser.add_argument("--max_steps", type=int, default=None,
                       help="最大仿真步数")
    
    parser.add_argument("--num_runs", type=int, default=3,
                       help="每个配置运行的次数")
    
    parser.add_argument("--results_dir", type=str, default="results/experiments",
                       help="结果保存目录")
    
    parser.add_argument("--no_show", action="store_true",
                       help="不显示图表")
    
    parser.add_argument("--no_save", action="store_true",
                       help="不保存图表")
    
    parser.add_argument("--generate_reports", action="store_true",
                       help="生成实验摘要报告")
    
    return parser

def run_benchmark_experiments(args: argparse.Namespace) -> None:
    """运行基准比较实验"""
    print("\n====== 运行基准比较实验 ======\n")
    
    benchmark = BenchmarkExperiments(
        results_dir=os.path.join(args.results_dir, "benchmark"),
        save_plots=not args.no_save,
        show_plots=not args.no_show
    )
    
    if args.specific_experiment == "":
        benchmark.run_all_benchmark_experiments(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "modes":
        benchmark.run_communication_mode_comparison(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "adaptive":
        benchmark.run_adaptive_vs_fixed_comparison(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "dynamic":
        benchmark.run_dynamic_communication_experiment(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    else:
        print(f"未知的基准比较实验: {args.specific_experiment}")

def run_parameter_analysis(args: argparse.Namespace) -> None:
    """运行参数敏感性分析实验"""
    print("\n====== 运行参数敏感性分析实验 ======\n")
    
    parameter_analysis = ParameterAnalysisExperiments(
        results_dir=os.path.join(args.results_dir, "parameter_analysis"),
        save_plots=not args.no_save,
        show_plots=not args.no_show
    )
    
    if args.specific_experiment == "":
        parameter_analysis.run_all_parameter_analyses(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "k_value":
        parameter_analysis.run_k_value_analysis(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "comm_range":
        parameter_analysis.run_communication_range_analysis(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "reasoning":
        parameter_analysis.run_reasoning_level_analysis(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "csi_threshold":
        parameter_analysis.run_csi_threshold_analysis(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    else:
        print(f"未知的参数分析实验: {args.specific_experiment}")

def run_communication_experiments(args: argparse.Namespace) -> None:
    """运行通信实验"""
    print("\n====== 运行通信实验 ======\n")
    
    comm_experiments = CommunicationExperiments(
        results_dir=os.path.join(args.results_dir, "communication"),
        save_plots=not args.no_save,
        show_plots=not args.no_show
    )
    
    if args.specific_experiment == "":
        comm_experiments.run_all_communication_experiments(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "degradation":
        comm_experiments.run_communication_degradation_experiment(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "csi_sensitivity":
        comm_experiments.run_csi_sensitivity_experiment(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "delay":
        comm_experiments.run_varying_delay_experiment(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    else:
        print(f"未知的通信实验: {args.specific_experiment}")

def run_monte_carlo_experiments(args: argparse.Namespace) -> None:
    """运行蒙特卡洛实验"""
    print("\n====== 运行蒙特卡洛实验 ======\n")
    
    monte_carlo = MonteCarloExperiments(
        results_dir=os.path.join(args.results_dir, "monte_carlo"),
        save_plots=not args.no_save,
        show_plots=not args.no_show
    )
    
    if args.specific_experiment == "":
        monte_carlo.run_all_monte_carlo_experiments(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "basic":
        monte_carlo.run_basic_monte_carlo(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "task_position":
        monte_carlo.run_random_task_position_monte_carlo(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "observation_noise":
        monte_carlo.run_noisy_observation_monte_carlo(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    else:
        print(f"未知的蒙特卡洛实验: {args.specific_experiment}")

def run_scalability_experiments(args: argparse.Namespace) -> None:
    """运行扩展性实验"""
    print("\n====== 运行扩展性实验 ======\n")
    
    scalability = ScalabilityExperiments(
        results_dir=os.path.join(args.results_dir, "scalability"),
        save_plots=not args.no_save,
        show_plots=not args.no_show
    )
    
    if args.specific_experiment == "":
        scalability.run_all_scalability_experiments(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "agent_scaling":
        scalability.run_agent_scaling_experiment(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "task_scaling":
        scalability.run_task_scaling_experiment(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    elif args.specific_experiment == "computation_time":
        scalability.run_computation_time_experiment(
            max_steps=args.max_steps,
            num_runs=args.num_runs
        )
    else:
        print(f"未知的扩展性实验: {args.specific_experiment}")

def generate_final_reports(args: argparse.Namespace) -> None:
    """生成所有实验的最终报告"""
    print("\n====== 生成实验摘要报告 ======\n")
    
    analyzer = ResultsAnalyzer(results_base_dir=args.results_dir)
    
    # 查找所有实验目录
    import glob
    experiment_dirs = glob.glob(os.path.join(args.results_dir, "*"))
    
    for dir_path in experiment_dirs:
        if os.path.isdir(dir_path):
            dir_name = os.path.basename(dir_path)
            print(f"正在生成 '{dir_name}' 的摘要报告...")
            analyzer.generate_summary_report(dir_name)

def main() -> None:
    """主函数，运行所有实验"""
    # 解析命令行参数
    parser = setup_parser()
    args = parser.parse_args()
    
    # 显示实验配置
    print("\n====== 异构航天器协同任务分配算法实验 ======")
    print(f"实验类型: {args.experiment_type}")
    if args.specific_experiment:
        print(f"特定实验: {args.specific_experiment}")
    print(f"最大仿真步数: {args.max_steps if args.max_steps else '使用场景默认值'}")
    print(f"每个配置运行次数: {args.num_runs}")
    print(f"结果目录: {args.results_dir}")
    print(f"显示图表: {not args.no_show}")
    print(f"保存图表: {not args.no_save}")
    print(f"生成报告: {args.generate_reports}")
    
    # 确保结果目录存在
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 根据指定类型运行实验
    if args.experiment_type == "all" or args.experiment_type == "benchmark":
        run_benchmark_experiments(args)
    
    if args.experiment_type == "all" or args.experiment_type == "parameter":
        run_parameter_analysis(args)
    
    if args.experiment_type == "all" or args.experiment_type == "communication":
        run_communication_experiments(args)
    
    if args.experiment_type == "all" or args.experiment_type == "monte_carlo":
        run_monte_carlo_experiments(args)
    
    if args.experiment_type == "all" or args.experiment_type == "scalability":
        run_scalability_experiments(args)
    
    # 如果需要，生成报告
    if args.generate_reports:
        generate_final_reports(args)
    
    # 显示总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n====== 所有实验完成！======")
    print(f"总耗时: {hours}小时 {minutes}分钟 {seconds}秒")

if __name__ == "__main__":
    
    main()