# experiments/experiment_manager.py
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from experiments.fix_chinese import setup_chinese_fonts
setup_chinese_fonts()

# 导入项目路径
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
_project_root_directory = os.path.dirname(_current_file_directory)
if _project_root_directory not in sys.path:
    sys.path.insert(0, _project_root_directory)

from simulation.run_simulation import run_single_simulation
from simulation.scenarios import load_scenario, SCENARIOS
from simulation.visualizer import Visualizer

class ExperimentManager:
    """
    实验管理器类：负责设置、运行和分析各种实验
    """
    def __init__(self, 
                 results_base_dir: str = "results/experiments",
                 save_visualization: bool = True,
                 show_plots: bool = True,
                 save_data: bool = True):
        """
        初始化实验管理器
        
        参数:
            results_base_dir (str): 结果保存的基础路径
            save_visualization (bool): 是否保存可视化图表
            show_plots (bool): 是否显示图表
            save_data (bool): 是否保存实验数据
        """
        self.results_base_dir = results_base_dir
        self.save_visualization = save_visualization
        self.show_plots = show_plots
        self.save_data = save_data
        
        # 确保结果目录存在
        if not os.path.exists(results_base_dir):
            os.makedirs(results_base_dir, exist_ok=True)
            
        # 设置matplotlib参数以支持中文显示
        self._setup_matplotlib_style()
        
    def _setup_matplotlib_style(self):
        """设置matplotlib风格为Nature学术风格，并支持中文显示"""
        # 尝试寻找系统中存在的中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STSong', 'DengXian']
        chinese_font_found = False
        
        for font in chinese_fonts:
            try:
                font_path = matplotlib.font_manager.findfont(font, fallback_to_default=False)
                if font_path:
                    matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
                    chinese_font_found = True
                    break
            except:
                continue
                
        if not chinese_font_found:
            print("警告: 未找到合适的中文字体，图表中的中文可能无法正确显示")
            
        # 解决负号显示问题
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        # Nature学术风格设置
        plt.style.use('default')  # 先重置为默认风格
        
        # 颜色方案
        nature_colors = [
            '#0C5DA5', '#FF2C00', '#00B945', '#FF9500',
            '#845B97', '#474747', '#9e9e9e', '#F2C800'
        ]
        sns.set_palette(nature_colors)
        
        # 线条风格
        matplotlib.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11, 
            'axes.titlesize': 12,
            'xtick.labelsize': 9, 
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'legend.frameon': False, 
            'figure.dpi': 100,
            'savefig.dpi': 300, 
            'savefig.bbox': 'tight',
            'lines.linewidth': 1.8,  # 较粗线条
            'lines.markersize': 6,
            'axes.grid': True, 
            'grid.linestyle': '--', 
            'grid.alpha': 0.7,
            'axes.formatter.useoffset': False, 
            'axes.formatter.limits': (-3, 4)
        })
    
    def create_experiment_dir(self, experiment_name: str) -> str:
        """
        为特定实验创建结果目录
        
        参数:
            experiment_name (str): 实验名称
            
        返回:
            str: 实验结果目录路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(self.results_base_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir
    
    def run_single_scenario(self, 
                           scenario_name: str,
                           max_steps_override: Optional[int] = None,
                           random_seed: Optional[int] = None,
                           generate_visualizations: bool = True,
                           custom_suffix: Optional[str] = None) -> Dict[str, Any]:
        """
        运行单个场景并返回结果
        
        参数:
            scenario_name (str): 场景名称，必须存在于scenarios.py中
            max_steps_override (int, optional): 覆盖场景中定义的最大步数
            random_seed (int, optional): 随机种子
            generate_visualizations (bool): 是否生成可视化
            custom_suffix (str, optional): 自定义后缀
            
        返回:
            Dict[str, Any]: 仿真结果
        """
        start_time = time.time()
        print(f"开始运行场景: {scenario_name}")
        
        # 如果未提供suffix，使用时间戳
        if custom_suffix is None:
            custom_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # 运行仿真
        results = run_single_simulation(
            scenario_name=scenario_name,
            max_steps_override=max_steps_override,
            random_seed=random_seed,
            results_dir_base=self.results_base_dir,
            enable_visualization=generate_visualizations and self.save_visualization,
            show_plots=generate_visualizations and self.show_plots,
            save_full_history_pickle=self.save_data,
            save_summary_json=self.save_data,
            custom_suffix=custom_suffix
        )
        
        end_time = time.time()
        print(f"场景 {scenario_name} 运行完成，耗时: {end_time - start_time:.2f} 秒")
        
        return results
        
    def compare_scenarios(self, 
                          scenario_list: List[str],
                          comparison_name: str,
                          max_steps: Optional[int] = None,
                          random_seed: Optional[int] = None,
                          num_runs: int = 1) -> None:
        """
        比较多个场景的性能，并生成对比图表
        
        参数:
            scenario_list (List[str]): 要比较的场景列表
            comparison_name (str): 比较的名称，用于结果保存
            max_steps (int, optional): 覆盖所有场景的最大步数
            random_seed (int, optional): 基础随机种子
            num_runs (int): 每个场景运行的次数
        """
        experiment_dir = self.create_experiment_dir(comparison_name)
        print(f"比较实验 '{comparison_name}' 开始，结果将保存至: {experiment_dir}")
        
        # 收集所有场景的结果
        all_results = {}
        metrics_data = {
            "scenario": [], 
            "run": [], 
            "utility": [], 
            "tasks_completed": [],
            "total_time": [], 
            "messages_broadcasted": []
        }
        
        for scenario_name in scenario_list:
            all_results[scenario_name] = []
            
            for run in range(num_runs):
                # 为每个运行设置不同的随机种子
                if random_seed is not None:
                    curr_seed = random_seed + run
                else:
                    curr_seed = None
                
                results = self.run_single_scenario(
                    scenario_name=scenario_name,
                    max_steps_override=max_steps,
                    random_seed=curr_seed,
                    generate_visualizations=(run == 0),  # 只为第一次运行生成可视化
                    custom_suffix=f"{comparison_name}_run{run+1}"
                )
                
                all_results[scenario_name].append(results)
                
                # 收集关键指标
                if results:
                    metrics_data["scenario"].append(scenario_name)
                    metrics_data["run"].append(run + 1)
                    
                    # 提取关键性能指标
                    utility = results.get("history_global_utility_ksc", [0])[-1] if "history_global_utility_ksc" in results else 0
                    tasks_completed = sum(1 for t in results.get("tasks_final_state", {}).values() 
                                        if t.get("status") == "completed")
                    total_time = results.get("final_simulation_time_s", 0)
                    msgs_broadcasted = np.mean(results.get("ksc_stats_messages_broadcasted", [0])) if "ksc_stats_messages_broadcasted" in results else 0
                    
                    metrics_data["utility"].append(utility)
                    metrics_data["tasks_completed"].append(tasks_completed)
                    metrics_data["total_time"].append(total_time)
                    metrics_data["messages_broadcasted"].append(msgs_broadcasted)
        
        # 将指标数据转换为DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        
        # 保存原始指标数据到CSV
        if self.save_data:
            metrics_csv_path = os.path.join(experiment_dir, "comparison_metrics.csv")
            metrics_df.to_csv(metrics_csv_path, index=False)
            print(f"比较指标已保存到: {metrics_csv_path}")
        
        # 生成比较图表
        if self.save_visualization or self.show_plots:
            self._generate_comparison_plots(metrics_df, experiment_dir, comparison_name)
        
        print(f"比较实验 '{comparison_name}' 完成!")
        
    def _generate_comparison_plots(self, 
                                  metrics_df: pd.DataFrame, 
                                  save_dir: str, 
                                  comparison_name: str) -> None:
        """
        生成场景比较的图表
        
        参数:
            metrics_df (pd.DataFrame): 包含比较指标的DataFrame
            save_dir (str): 图表保存目录
            comparison_name (str): 比较的名称
        """
        if metrics_df.empty:
            print("警告: 没有数据可用于生成比较图表")
            return
            
        # 为每个指标创建对比图
        metrics_to_plot = ["utility", "tasks_completed", "total_time", "messages_broadcasted"]
        titles = {
            "utility": "系统总体效用",
            "tasks_completed": "完成任务数量", 
            "total_time": "完成时间 (秒)",
            "messages_broadcasted": "广播消息数量"
        }
        ylabels = {
            "utility": "系统效用值",
            "tasks_completed": "任务数量", 
            "total_time": "时间 (秒)",
            "messages_broadcasted": "平均消息数"
        }
        
        for metric in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 计算每个场景的均值和标准差
            summary = metrics_df.groupby('scenario')[metric].agg(['mean', 'std']).reset_index()
            
            # 生成条形图
            bars = ax.bar(
                summary['scenario'], 
                summary['mean'],
                yerr=summary['std'],
                capsize=5,
                error_kw={'elinewidth': 1.5, 'capthick': 1.5},
                width=0.6
            )
            
            # 添加数值标签
            for bar, mean, std in zip(bars, summary['mean'], summary['std']):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + std + 0.1,
                    f'{mean:.2f}±{std:.2f}', 
                    ha='center', 
                    va='bottom', 
                    fontsize=9
                )
            
            # 设置图表标题和标签
            ax.set_title(f"{titles.get(metric, metric)}", fontsize=12)
            ax.set_xlabel("场景", fontsize=11)
            ax.set_ylabel(ylabels.get(metric, metric), fontsize=11)
            ax.set_xticks(range(len(summary['scenario'])))
            ax.set_xticklabels(summary['scenario'], rotation=45, ha='right')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if self.save_visualization:
                fig_path = os.path.join(save_dir, f"{comparison_name}_{metric}_comparison.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"比较图表已保存: {fig_path}")
            
            if self.show_plots:
                plt.show()
            else:
                plt.close(fig)
                
        # 创建一个综合视图 (雷达图)
        self._create_radar_comparison(metrics_df, save_dir, comparison_name)
    
    def _create_radar_comparison(self, 
                               metrics_df: pd.DataFrame, 
                               save_dir: str, 
                               comparison_name: str) -> None:
        """
        创建雷达图进行场景综合比较
        
        参数:
            metrics_df (pd.DataFrame): 包含比较指标的DataFrame
            save_dir (str): 图表保存目录
            comparison_name (str): 比较的名称
        """
        # 计算每个指标的平均值
        radar_metrics = ["utility", "tasks_completed", "messages_broadcasted"]
        
        # 获取不同场景
        scenarios = metrics_df['scenario'].unique()
        
        if len(scenarios) < 2:
            print("警告: 需要至少2个场景来创建雷达图对比")
            return
            
        # 准备数据
        means = metrics_df.groupby('scenario')[radar_metrics].mean()
        
        # 标准化数据 (0-1)，对于消息数，认为越少越好
        normalized = means.copy()
        for col in radar_metrics:
            if col == "messages_broadcasted":
                # 对于消息数，值越小越好，所以取倒数再归一化
                max_val = means[col].max()
                normalized[col] = max_val / means[col]
                normalized[col] = normalized[col] / normalized[col].max()
            else:
                # 对于其他指标，值越大越好
                normalized[col] = (means[col] - means[col].min()) / (means[col].max() - means[col].min() + 1e-10)
        
        # 创建雷达图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        metric_labels = {
            "utility": "系统效用",
            "tasks_completed": "任务完成数", 
            "messages_broadcasted": "通信效率" # 已反转，所以高值表示更好的通信效率（更少的消息）
        }
        
        ax.set_theta_offset(np.pi / 2)  # 从顶部开始
        ax.set_theta_direction(-1)  # 顺时针
        
        # 设置标签
        plt_labels = [metric_labels.get(m, m) for m in radar_metrics]
        plt_labels += plt_labels[:1]  # 闭合
        ax.set_xticks(angles)
        ax.set_xticklabels(plt_labels, fontsize=12)
        
        # 绘制每个场景的雷达图
        for i, scenario in enumerate(scenarios):
            values = normalized.loc[scenario].values.tolist()
            values += values[:1]  # 闭合
            ax.plot(angles, values, linewidth=2, linestyle='-', 
                   label=scenario, marker='o', markersize=6)
            ax.fill(angles, values, alpha=0.1)
        
        # 设置Y轴
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["20%", "40%", "60%", "80%"], fontsize=9)
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title(f"{comparison_name} - 场景性能综合对比", fontsize=14, y=1.08)
        
        # 保存图表
        if self.save_visualization:
            fig_path = os.path.join(save_dir, f"{comparison_name}_radar_comparison.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"雷达对比图已保存: {fig_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def analyze_k_impact(self, 
                        base_scenario: str,
                        k_values: List[int],
                        experiment_name: str = "k_impact_analysis",
                        max_steps: Optional[int] = None,
                        num_runs: int = 3) -> None:
        """
        分析K值对KSC算法性能的影响
        
        参数:
            base_scenario (str): 基础场景名称
            k_values (List[int]): 要测试的K值列表
            experiment_name (str): 实验名称
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个K值运行的次数
        """
        experiment_dir = self.create_experiment_dir(experiment_name)
        print(f"开始分析K值对性能的影响，实验: '{experiment_name}'")
        
        # 收集结果数据
        metrics_data = {
            "k_value": [], 
            "run": [], 
            "utility": [], 
            "tasks_completed": [],
            "total_time": [], 
            "convergence_rounds": [], # KSC算法收敛轮次
            "messages_broadcasted": []
        }
        
        # 为每个K值生成并运行场景
        for k in k_values:
            print(f"\n--- 测试 K = {k} ---")
            
            for run in range(num_runs):
                # 加载基础场景并修改K值
                scenario_config = load_scenario(base_scenario)
                if not scenario_config:
                    print(f"错误: 无法加载基础场景 '{base_scenario}'")
                    continue
                
                # 修改场景中所有智能体的K值
                for sc_def in scenario_config["spacecrafts"]:
                    if "ksc_params" in sc_def:
                        sc_def["ksc_params"]["k_value"] = k
                
                # 保存修改后的场景配置
                modified_scenario_name = f"{base_scenario}_k{k}"
                SCENARIOS[modified_scenario_name] = scenario_config
                
                # 运行修改后的场景
                curr_seed = None if run == 0 else run * 100 + k
                results = self.run_single_scenario(
                    scenario_name=modified_scenario_name,
                    max_steps_override=max_steps,
                    random_seed=curr_seed,
                    generate_visualizations=(run == 0),  # 只为第一次运行生成可视化
                    custom_suffix=f"{experiment_name}_k{k}_run{run+1}"
                )
                
                # 收集指标
                if results:
                    metrics_data["k_value"].append(k)
                    metrics_data["run"].append(run + 1)
                    
                    utility = results.get("history_global_utility_ksc", [0])[-1] if "history_global_utility_ksc" in results else 0
                    tasks_completed = sum(1 for t in results.get("tasks_final_state", {}).values() 
                                        if t.get("status") == "completed")
                    total_time = results.get("final_simulation_time_s", 0)
                    # 计算收敛轮次 (效用不再显著变化的轮次)
                    utils = results.get("history_global_utility_ksc", [])
                    convergence_round = self._calculate_convergence_round(utils)
                    msgs_broadcasted = np.sum(results.get("ksc_stats_messages_broadcasted", [0])) if "ksc_stats_messages_broadcasted" in results else 0
                    
                    metrics_data["utility"].append(utility)
                    metrics_data["tasks_completed"].append(tasks_completed)
                    metrics_data["total_time"].append(total_time)
                    metrics_data["convergence_rounds"].append(convergence_round)
                    metrics_data["messages_broadcasted"].append(msgs_broadcasted)
                
                # 删除临时场景
                if modified_scenario_name in SCENARIOS:
                    del SCENARIOS[modified_scenario_name]
        
        # 将指标数据转换为DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        
        # 保存原始指标数据到CSV
        if self.save_data:
            metrics_csv_path = os.path.join(experiment_dir, "k_impact_metrics.csv")
            metrics_df.to_csv(metrics_csv_path, index=False)
            print(f"K值分析指标已保存到: {metrics_csv_path}")
        
        # 生成分析图表
        if self.save_visualization or self.show_plots:
            self._generate_k_impact_plots(metrics_df, experiment_dir, experiment_name)
        
        print(f"K值影响分析实验 '{experiment_name}' 完成!")
    
    def _calculate_convergence_round(self, utility_history: List[float], threshold: float = 0.01) -> int:
        """
        计算KSC算法收敛的轮次
        
        参数:
            utility_history (List[float]): 效用历史
            threshold (float): 收敛阈值
            
        返回:
            int: 收敛轮次
        """
        if not utility_history or len(utility_history) < 2:
            return 0
            
        for i in range(1, len(utility_history)):
            # 计算相对变化
            if abs(utility_history[i] - utility_history[i-1]) < threshold * abs(utility_history[i-1] + 1e-10):
                return i
                
        return len(utility_history)  # 未收敛，返回总轮次
    
    def _generate_k_impact_plots(self, 
                               metrics_df: pd.DataFrame, 
                               save_dir: str, 
                               experiment_name: str) -> None:
        """
        生成K值影响分析的图表
        
        参数:
            metrics_df (pd.DataFrame): 包含分析指标的DataFrame
            save_dir (str): 图表保存目录
            experiment_name (str): 实验名称
        """
        if metrics_df.empty:
            print("警告: 没有数据可用于生成K值影响分析图表")
            return
            
        # 计算每个K值的均值和标准差
        summary = metrics_df.groupby('k_value').agg({
            'utility': ['mean', 'std'],
            'tasks_completed': ['mean', 'std'],
            'total_time': ['mean', 'std'],
            'convergence_rounds': ['mean', 'std'],
            'messages_broadcasted': ['mean', 'std']
        })
        
        # 设置K值顺序和颜色映射
        k_values = sorted(metrics_df['k_value'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
        
        # 分析指标与K值的关系
        plot_metrics = [
            ('utility', '系统总体效用'), 
            ('convergence_rounds', 'KSC算法收敛轮次'),
            ('tasks_completed', '完成任务数量'), 
            ('messages_broadcasted', '广播消息总数')
        ]
        
        for metric, title in plot_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            mean_values = [summary.loc[k][(metric, 'mean')] for k in k_values]
            std_values = [summary.loc[k][(metric, 'std')] for k in k_values]
            
            # 绘制折线图（带误差区域）
            ax.plot(k_values, mean_values, 'o-', linewidth=2, markersize=8)
            ax.fill_between(k_values, 
                           [m-s for m,s in zip(mean_values, std_values)],
                           [m+s for m,s in zip(mean_values, std_values)],
                           alpha=0.2)
            
            # 添加数据点标签
            for i, (k, mean, std) in enumerate(zip(k_values, mean_values, std_values)):
                ax.annotate(f'{mean:.2f}±{std:.2f}', 
                           (k, mean), 
                           xytext=(0, 10), 
                           textcoords='offset points',
                           ha='center', va='bottom', fontsize=9)
            
            # 设置图表标题和标签
            ax.set_title(f"K值对{title}的影响", fontsize=12)
            ax.set_xlabel("K值", fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_xticks(k_values)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 添加K值的含义说明
            ax.text(0.02, 0.02, 
                   "注: K表示KS-COAL算法中允许的最大转换链长度", 
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图表
            if self.save_visualization:
                fig_path = os.path.join(save_dir, f"{experiment_name}_{metric}_vs_k.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"K值影响分析图表已保存: {fig_path}")
            
            if self.show_plots:
                plt.show()
            else:
                plt.close(fig)
        
        # 创建K值与效用-消息数权衡图
        self._create_k_tradeoff_plot(summary, k_values, save_dir, experiment_name)
    
    def _create_k_tradeoff_plot(self, 
                              summary: pd.DataFrame, 
                              k_values: List[int], 
                              save_dir: str, 
                              experiment_name: str) -> None:
        """
        创建K值与效用-消息数权衡的散点图
        
        参数:
            summary (pd.DataFrame): 汇总指标
            k_values (List[int]): K值列表 
            save_dir (str): 图表保存目录
            experiment_name (str): 实验名称
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 提取效用和消息数据
        utility_means = [summary.loc[k][('utility', 'mean')] for k in k_values]
        message_means = [summary.loc[k][('messages_broadcasted', 'mean')] for k in k_values]
        
        # 创建散点图，点的大小表示K值
        sizes = [k*50 for k in k_values]  # 点大小与K值成正比
        scatter = ax.scatter(message_means, utility_means, s=sizes, c=k_values, 
                           cmap='viridis', alpha=0.7, edgecolor='k')
        
        # 添加K值标签
        for i, k in enumerate(k_values):
            ax.annotate(f'K={k}', 
                       (message_means[i], utility_means[i]),
                       xytext=(5, 5), 
                       textcoords='offset points')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label('K值', fontsize=11)
        
        # 设置图表标题和标签
        ax.set_title("K值对系统效用和通信开销的权衡分析", fontsize=12)
        ax.set_xlabel("广播消息总数", fontsize=11)
        ax.set_ylabel("系统总体效用", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加最佳区域指示 (右上角)
        ax.text(0.98, 0.98, "理想区域\n(高效用低通信开销)", 
               ha='right', va='top', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green", alpha=0.8))
        
        # 添加最佳K值分析
        # 计算归一化后的指标
        norm_utility = [(u - min(utility_means))/(max(utility_means)-min(utility_means)+1e-10) for u in utility_means]
        norm_messages = [1 - (m - min(message_means))/(max(message_means)-min(message_means)+1e-10) for m in message_means]  # 反向，因为消息数量越少越好
        
        # 计算综合得分
        scores = [(u + m)/2 for u, m in zip(norm_utility, norm_messages)]
        best_k_idx = np.argmax(scores)
        best_k = k_values[best_k_idx]
        
        ax.text(0.02, 0.98, 
               f"最佳K值分析: K={best_k}\n(综合考虑效用和通信开销)", 
               ha='left', va='top', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        if self.save_visualization:
            fig_path = os.path.join(save_dir, f"{experiment_name}_k_tradeoff.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"K值权衡分析图表已保存: {fig_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def analyze_communication_impact(self, 
                                    base_scenario: str,
                                    comm_range_values: List[float],
                                    experiment_name: str = "comm_impact_analysis",
                                    max_steps: Optional[int] = None,
                                    num_runs: int = 3) -> None:
        """
        分析通信范围对系统性能的影响
        
        参数:
            base_scenario (str): 基础场景名称
            comm_range_values (List[float]): 要测试的通信范围列表
            experiment_name (str): 实验名称
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个通信范围运行的次数
        """
        experiment_dir = self.create_experiment_dir(experiment_name)
        print(f"开始分析通信范围对性能的影响，实验: '{experiment_name}'")
        
        # 收集结果数据
        metrics_data = {
            "comm_range": [], 
            "run": [], 
            "utility": [], 
            "tasks_completed": [],
            "total_time": [], 
            "active_comm_links": [],
            "csi_average": [],
            "messages_broadcasted": []
        }
        
        # 为每个通信范围生成并运行场景
        for comm_range in comm_range_values:
            print(f"\n--- 测试通信范围 = {comm_range} ---")
            
            for run in range(num_runs):
                # 加载基础场景并修改通信范围
                scenario_config = load_scenario(base_scenario)
                if not scenario_config:
                    print(f"错误: 无法加载基础场景 '{base_scenario}'")
                    continue
                
                # 修改场景中的通信范围
                if "simulation_params" in scenario_config:
                    scenario_config["simulation_params"]["communication_range"] = comm_range
                
                # 保存修改后的场景配置
                modified_scenario_name = f"{base_scenario}_comm{int(comm_range)}"
                SCENARIOS[modified_scenario_name] = scenario_config
                
                # 运行修改后的场景
                curr_seed = None if run == 0 else run * 100 + int(comm_range)
                results = self.run_single_scenario(
                    scenario_name=modified_scenario_name,
                    max_steps_override=max_steps,
                    random_seed=curr_seed,
                    generate_visualizations=(run == 0),  # 只为第一次运行生成可视化
                    custom_suffix=f"{experiment_name}_comm{int(comm_range)}_run{run+1}"
                )
                
                # 收集指标
                if results:
                    metrics_data["comm_range"].append(comm_range)
                    metrics_data["run"].append(run + 1)
                    
                    utility = results.get("history_global_utility_ksc", [0])[-1] if "history_global_utility_ksc" in results else 0
                    tasks_completed = sum(1 for t in results.get("tasks_final_state", {}).values() 
                                        if t.get("status") == "completed")
                    total_time = results.get("final_simulation_time_s", 0)
                    
                    # 计算平均活动通信链接数
                    active_links = np.mean(results.get("stats_active_links_ksc_graph", [0])) if "stats_active_links_ksc_graph" in results else 0
                    
                    # 计算平均CSI值 (从智能体历史中提取)
                    csi_values = []
                    for agent_id, agent_data in results.get("agents_history", {}).items():
                        if "history_csi_smooth" in agent_data:
                            csi_values.extend(agent_data["history_csi_smooth"])
                    csi_avg = np.mean(csi_values) if csi_values else 0
                    
                    msgs_broadcasted = np.mean(results.get("ksc_stats_messages_broadcasted", [0])) if "ksc_stats_messages_broadcasted" in results else 0
                    
                    metrics_data["utility"].append(utility)
                    metrics_data["tasks_completed"].append(tasks_completed)
                    metrics_data["total_time"].append(total_time)
                    metrics_data["active_comm_links"].append(active_links)
                    metrics_data["csi_average"].append(csi_avg)
                    metrics_data["messages_broadcasted"].append(msgs_broadcasted)
                
                # 删除临时场景
                if modified_scenario_name in SCENARIOS:
                    del SCENARIOS[modified_scenario_name]
        
        # 将指标数据转换为DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        
        # 保存原始指标数据到CSV
        if self.save_data:
            metrics_csv_path = os.path.join(experiment_dir, "comm_impact_metrics.csv")
            metrics_df.to_csv(metrics_csv_path, index=False)
            print(f"通信范围分析指标已保存到: {metrics_csv_path}")
        
        # 生成分析图表
        if self.save_visualization or self.show_plots:
            self._generate_comm_impact_plots(metrics_df, experiment_dir, experiment_name)
        
        print(f"通信范围影响分析实验 '{experiment_name}' 完成!")
    
    def _generate_comm_impact_plots(self, 
                                  metrics_df: pd.DataFrame, 
                                  save_dir: str, 
                                  experiment_name: str) -> None:
        """
        生成通信范围影响分析的图表
        
        参数:
            metrics_df (pd.DataFrame): 包含分析指标的DataFrame
            save_dir (str): 图表保存目录
            experiment_name (str): 实验名称
        """
        if metrics_df.empty:
            print("警告: 没有数据可用于生成通信范围影响分析图表")
            return
            
        # 计算每个通信范围的均值和标准差
        summary = metrics_df.groupby('comm_range').agg({
            'utility': ['mean', 'std'],
            'tasks_completed': ['mean', 'std'],
            'total_time': ['mean', 'std'],
            'active_comm_links': ['mean', 'std'],
            'csi_average': ['mean', 'std'],
            'messages_broadcasted': ['mean', 'std']
        })
        
        # 设置通信范围顺序和颜色映射
        comm_ranges = sorted(metrics_df['comm_range'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(comm_ranges)))
        
        # 创建通信范围与各指标的关系图
        plt_metrics = [
            ('active_comm_links', '平均活动通信链接数'),
            ('csi_average', '平均通信状态指数 (CSI)'),
            ('utility', '系统总体效用'),
            ('tasks_completed', '完成任务数量'),
            ('messages_broadcasted', '平均广播消息数')
        ]
        
        for metric, title in plt_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            mean_values = [summary.loc[r][(metric, 'mean')] for r in comm_ranges]
            std_values = [summary.loc[r][(metric, 'std')] for r in comm_ranges]
            
            # 绘制折线图（带误差区域）
            ax.plot(comm_ranges, mean_values, 'o-', linewidth=2, markersize=8)
            ax.fill_between(comm_ranges, 
                           [m-s for m,s in zip(mean_values, std_values)],
                           [m+s for m,s in zip(mean_values, std_values)],
                           alpha=0.2)
            
            # 添加数据点标签
            for i, (r, mean, std) in enumerate(zip(comm_ranges, mean_values, std_values)):
                ax.annotate(f'{mean:.2f}±{std:.2f}', 
                           (r, mean), 
                           xytext=(0, 10), 
                           textcoords='offset points',
                           ha='center', va='bottom', fontsize=9)
            
            # 设置图表标题和标签
            ax.set_title(f"通信范围对{title}的影响", fontsize=12)
            ax.set_xlabel("通信范围 (m)", fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 设置X轴刻度和标签
            plt.xticks(comm_ranges)
            
            # 特别情况：为CSI添加阈值线
            if metric == 'csi_average':
                # 假设CSI阈值约为0.3(低)和0.7(高)
                ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.7, label='CSI低阈值')
                ax.axhline(y=0.7, color='g', linestyle='--', alpha=0.7, label='CSI高阈值')
                ax.fill_between(ax.get_xlim(), 0, 0.3, color='r', alpha=0.1)
                ax.fill_between(ax.get_xlim(), 0.3, 0.7, color='y', alpha=0.1)
                ax.fill_between(ax.get_xlim(), 0.7, 1.0, color='g', alpha=0.1)
                ax.text(ax.get_xlim()[0], 0.15, "弱通信区域", fontsize=9, alpha=0.7)
                ax.text(ax.get_xlim()[0], 0.5, "过渡区域", fontsize=9, alpha=0.7)
                ax.text(ax.get_xlim()[0], 0.85, "强通信区域", fontsize=9, alpha=0.7)
                ax.legend()
            
            plt.tight_layout()
            
            # 保存图表
            if self.save_visualization:
                fig_path = os.path.join(save_dir, f"{experiment_name}_{metric}_vs_comm_range.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"通信范围影响分析图表已保存: {fig_path}")
            
            if self.show_plots:
                plt.show()
            else:
                plt.close(fig)
        
        # 创建通信范围与决策模式关系图
        self._create_comm_decision_mode_plot(metrics_df, save_dir, experiment_name)
    
    def _create_comm_decision_mode_plot(self, 
                                      metrics_df: pd.DataFrame, 
                                      save_dir: str, 
                                      experiment_name: str) -> None:
        """
        创建通信范围与决策模式的关系图
        
        参数:
            metrics_df (pd.DataFrame): 原始指标数据
            save_dir (str): 图表保存目录
            experiment_name (str): 实验名称
        """
        # 假设我们有CSI平均值作为参考
        if 'csi_average' not in metrics_df.columns:
            print("警告: 缺少CSI数据，无法创建决策模式分析图")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 按通信范围分组
        comm_ranges = sorted(metrics_df['comm_range'].unique())
        avg_csi = [metrics_df[metrics_df['comm_range']==r]['csi_average'].mean() for r in comm_ranges]
        
        # 根据CSI值估计各决策模式的比例
        strong_mode = [max(0, min(1, (csi - 0.7)/0.3)) for csi in avg_csi]
        weak_mode = [max(0, min(1, (0.3 - csi)/0.3)) for csi in avg_csi]
        transition_mode = [1 - s - w for s, w in zip(strong_mode, weak_mode)]
        
        # 创建堆叠柱状图
        bar_width = 0.6
        bar_positions = np.arange(len(comm_ranges))
        
        ax.bar(bar_positions, weak_mode, bar_width, label='弱通信模式', 
              color='#FF9500', edgecolor='k', linewidth=0.5, alpha=0.8)
        ax.bar(bar_positions, transition_mode, bar_width, bottom=weak_mode, 
              label='过渡模式', color='#474747', edgecolor='k', linewidth=0.5, alpha=0.8)
        ax.bar(bar_positions, strong_mode, bar_width, 
              bottom=[w+t for w, t in zip(weak_mode, transition_mode)],
              label='强通信模式', color='#0C5DA5', edgecolor='k', linewidth=0.5, alpha=0.8)
        
        # 设置图表标题和标签
        ax.set_title("通信范围对决策模式分布的影响", fontsize=12)
        ax.set_xlabel("通信范围 (m)", fontsize=11)
        ax.set_ylabel("各模式比例", fontsize=11)
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(comm_ranges)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        
        # 添加图例
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        # 添加通信范围与决策模式的描述
        ax.text(0.02, 0.02, 
               "通信范围影响系统CSI值，进而导致不同决策模式的触发比例变化", 
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
        
        # 在每个柱子上标注比例
        for i, (w, t, s) in enumerate(zip(weak_mode, transition_mode, strong_mode)):
            # 标注弱通信模式比例
            if w > 0.1:  # 只标注具有一定比例的模式
                ax.text(i, w/2, f'{w*100:.0f}%', ha='center', va='center', color='black', fontsize=8)
            
            # 标注过渡模式比例
            if t > 0.1:
                ax.text(i, w + t/2, f'{t*100:.0f}%', ha='center', va='center', color='white', fontsize=8)
            
            # 标注强通信模式比例
            if s > 0.1:
                ax.text(i, w + t + s/2, f'{s*100:.0f}%', ha='center', va='center', color='white', fontsize=8)
        
        plt.tight_layout()
        
        # 保存图表
        if self.save_visualization:
            fig_path = os.path.join(save_dir, f"{experiment_name}_decision_modes.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"决策模式分析图表已保存: {fig_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def run_monte_carlo_experiment(self, 
                                  scenario_name: str,
                                  num_runs: int = 10,
                                  experiment_name: str = "monte_carlo",
                                  max_steps: Optional[int] = None) -> None:
        """
        运行蒙特卡洛实验
        
        参数:
            scenario_name (str): 场景名称
            num_runs (int): 运行次数
            experiment_name (str): 实验名称
            max_steps (int, optional): 最大仿真步数
        """
        experiment_dir = self.create_experiment_dir(experiment_name)
        print(f"开始蒙特卡洛实验 '{experiment_name}'，场景: {scenario_name}，运行 {num_runs} 次")
        
        # 收集结果数据
        metrics_data = {
            "run": [], 
            "seed": [],
            "utility": [], 
            "tasks_completed": [],
            "total_time": [], 
            "convergence_rounds": [],
            "messages_broadcasted": []
        }
        
        # 收集更详细的数据
        utility_histories = []
        
        # 运行多次，每次使用不同的随机种子
        for run in range(num_runs):
            seed = run * 100 + 42  # 一种设置不同种子的方式
            print(f"\n--- 蒙特卡洛运行 {run+1}/{num_runs} (种子: {seed}) ---")
            
            results = self.run_single_scenario(
                scenario_name=scenario_name,
                max_steps_override=max_steps,
                random_seed=seed,
                generate_visualizations=(run == 0),  # 只为第一次运行生成可视化
                custom_suffix=f"{experiment_name}_run{run+1}"
            )
            
            # 收集指标
            if results:
                utility_history = results.get("history_global_utility_ksc", [])
                if utility_history:
                    utility_histories.append(utility_history)
                    
                    metrics_data["run"].append(run + 1)
                    metrics_data["seed"].append(seed)
                    
                    utility = utility_history[-1] if utility_history else 0
                    tasks_completed = sum(1 for t in results.get("tasks_final_state", {}).values() 
                                        if t.get("status") == "completed")
                    total_time = results.get("final_simulation_time_s", 0)
                    convergence_round = self._calculate_convergence_round(utility_history)
                    msgs_broadcasted = np.mean(results.get("ksc_stats_messages_broadcasted", [0])) if "ksc_stats_messages_broadcasted" in results else 0
                    
                    metrics_data["utility"].append(utility)
                    metrics_data["tasks_completed"].append(tasks_completed)
                    metrics_data["total_time"].append(total_time)
                    metrics_data["convergence_rounds"].append(convergence_round)
                    metrics_data["messages_broadcasted"].append(msgs_broadcasted)
        
        # 将指标数据转换为DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        
        # 保存原始指标数据到CSV
        if self.save_data:
            metrics_csv_path = os.path.join(experiment_dir, "monte_carlo_metrics.csv")
            metrics_df.to_csv(metrics_csv_path, index=False)
            print(f"蒙特卡洛实验指标已保存到: {metrics_csv_path}")
        
        # 生成分析图表
        if self.save_visualization or self.show_plots:
            self._generate_monte_carlo_plots(metrics_df, utility_histories, experiment_dir, experiment_name)
        
        print(f"蒙特卡洛实验 '{experiment_name}' 完成!")
    
    def _generate_monte_carlo_plots(self, 
                                  metrics_df: pd.DataFrame, 
                                  utility_histories: List[List[float]],
                                  save_dir: str, 
                                  experiment_name: str) -> None:
        """
        生成蒙特卡洛实验的统计图表
        
        参数:
            metrics_df (pd.DataFrame): 包含各次运行指标的DataFrame
            utility_histories (List[List[float]]): 每次运行的效用历史
            save_dir (str): 图表保存目录
            experiment_name (str): 实验名称
        """
        if metrics_df.empty:
            print("警告: 没有数据可用于生成蒙特卡洛分析图表")
            return
            
        # 1. 创建分布统计图
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        metrics_to_plot = [
            ('utility', '系统总体效用', axs[0, 0]),
            ('tasks_completed', '完成任务数量', axs[0, 1]),
            ('convergence_rounds', 'KSC算法收敛轮次', axs[1, 0]),
            ('messages_broadcasted', '平均广播消息数', axs[1, 1])
        ]
        
        for metric, title, ax in metrics_to_plot:
            # 使用直方图可视化分布
            sns.histplot(metrics_df[metric], kde=True, ax=ax, bins=min(10, len(metrics_df)))
            
            # 添加均值和标准差
            mean_val = metrics_df[metric].mean()
            std_val = metrics_df[metric].std()
            
            ax.axvline(mean_val, color='r', linestyle='--', alpha=0.8, label=f'均值: {mean_val:.2f}')
            ax.axvline(mean_val + std_val, color='g', linestyle=':', alpha=0.5, label=f'+1σ: {mean_val+std_val:.2f}')
            ax.axvline(mean_val - std_val, color='g', linestyle=':', alpha=0.5, label=f'-1σ: {mean_val-std_val:.2f}')
            
            ax.set_title(f"{title}分布", fontsize=12)
            ax.set_xlabel(title, fontsize=11)
            ax.set_ylabel("频率", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        fig.suptitle(f"蒙特卡洛实验 - 性能指标统计分布 (N={len(metrics_df)})", fontsize=14)
        
        # 保存图表
        if self.save_visualization:
            fig_path = os.path.join(save_dir, f"{experiment_name}_distributions.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"蒙特卡洛分布图表已保存: {fig_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        # 2. 创建效用历史收敛对比图
        if utility_histories:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # 找出最大长度，用于对齐
            max_length = max(len(hist) for hist in utility_histories)
            x_values = np.arange(1, max_length + 1)  # 从1开始的轮次
            
            # 对齐所有效用历史的长度 (填充较短的历史)
            aligned_histories = []
            for hist in utility_histories:
                if len(hist) < max_length:
                    padded = hist + [hist[-1]] * (max_length - len(hist))
                else:
                    padded = hist
                aligned_histories.append(padded)
            
            # 转换为NumPy数组以便计算
            utility_array = np.array(aligned_histories)
            
            # 计算每轮的均值和标准差
            mean_utility = np.mean(utility_array, axis=0)
            std_utility = np.std(utility_array, axis=0)
            
            # 绘制每次运行的薄线
            for i, hist in enumerate(utility_histories):
                if i == 0:  # 第一条线添加标签
                    ax2.plot(range(1, len(hist) + 1), hist, linewidth=0.8, alpha=0.3, color='gray', label='单次运行')
                else:
                    ax2.plot(range(1, len(hist) + 1), hist, linewidth=0.8, alpha=0.3, color='gray')
            
            # 绘制平均线和误差带
            ax2.plot(x_values, mean_utility, 'b-', linewidth=2.5, label='平均效用')
            ax2.fill_between(x_values, 
                            mean_utility - std_utility,
                            mean_utility + std_utility,
                            alpha=0.2, color='b', label='±1σ区间')
            
            # 设置图表标题和标签
            ax2.set_title(f"蒙特卡洛实验 - KSC效用收敛过程 (N={len(utility_histories)})", fontsize=12)
            ax2.set_xlabel("KSC决策轮次", fontsize=11)
            ax2.set_ylabel("系统总体效用", fontsize=11)
            ax2.legend(fontsize=10)
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 设置X轴为整数
            ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            # 添加收敛分析
            # 计算50%、75%和95%收敛时的平均轮次
            conv_50 = self._calculate_convergence_percentage(mean_utility, 0.5)
            conv_75 = self._calculate_convergence_percentage(mean_utility, 0.75)
            conv_95 = self._calculate_convergence_percentage(mean_utility, 0.95)
            
            if conv_50 > 0:
                ax2.axvline(x=conv_50, color='r', linestyle='--', alpha=0.5, 
                           label=f'50%收敛: {conv_50}轮')
            if conv_75 > 0:
                ax2.axvline(x=conv_75, color='g', linestyle='--', alpha=0.5, 
                           label=f'75%收敛: {conv_75}轮')
            if conv_95 > 0:
                ax2.axvline(x=conv_95, color='b', linestyle='--', alpha=0.5, 
                           label=f'95%收敛: {conv_95}轮')
            
            # 更新图例
            ax2.legend(fontsize=9)
            
            plt.tight_layout()
            
            # 保存图表
            if self.save_visualization:
                fig_path = os.path.join(save_dir, f"{experiment_name}_utility_convergence.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"蒙特卡洛效用收敛图表已保存: {fig_path}")
            
            if self.show_plots:
                plt.show()
            else:
                plt.close(fig2)
        else:
            print("警告: 没有效用历史数据可用于创建收敛图")
            
    def _calculate_convergence_percentage(self, utility_history, percentage=0.95):
        """计算达到最终值一定百分比所需的轮次"""
        if not utility_history.size:
            return 0
            
        final_value = utility_history[-1]
        if abs(final_value) < 1e-10:
            return 0
            
        target_value = final_value * percentage
        
        for i, val in enumerate(utility_history):
            if val >= target_value:
                return i + 1
                
        return len(utility_history)

    def analyze_reasoning_level_impact(self, 
                                      base_scenario: str,
                                      reasoning_levels: List[int],
                                      experiment_name: str = "reasoning_level_analysis",
                                      max_steps: Optional[int] = None,
                                      num_runs: int = 3) -> None:
        """
        分析AIF推理级别对系统性能的影响
        
        参数:
            base_scenario (str): 基础场景名称
            reasoning_levels (List[int]): 要测试的推理级别列表 (通常为0,1,2)
            experiment_name (str): 实验名称
            max_steps (int, optional): 最大仿真步数
            num_runs (int): 每个推理级别运行的次数
        """
        experiment_dir = self.create_experiment_dir(experiment_name)
        print(f"开始分析推理级别对性能的影响，实验: '{experiment_name}'")
        
        # 收集结果数据
        metrics_data = {
            "reasoning_level": [], 
            "use_ep": [],
            "run": [], 
            "utility": [], 
            "tasks_completed": [],
            "total_time": [], 
            "messages_broadcasted": []
        }
        
        ep_values = [False, True]  # 是否使用认知规划(epistemic planning)
        
        # 为每个推理级别和EP设置生成并运行场景
        for level in reasoning_levels:
            for use_ep in ep_values:
                # 如果level=0，通常不使用EP，跳过
                if level == 0 and use_ep:
                    continue
                    
                print(f"\n--- 测试推理级别 = {level}, 使用EP = {use_ep} ---")
                
                for run in range(num_runs):
                    # 加载基础场景并修改推理级别
                    scenario_config = load_scenario(base_scenario)
                    if not scenario_config:
                        print(f"错误: 无法加载基础场景 '{base_scenario}'")
                        continue
                    
                    # 修改场景中的推理级别
                    if "aif_global_hyperparams" in scenario_config:
                        scenario_config["aif_global_hyperparams"]["reasoning_level"] = level
                        scenario_config["aif_global_hyperparams"]["use_epistemic_planning"] = use_ep
                    
                    # 保存修改后的场景配置
                    ep_suffix = "ep" if use_ep else "noep"
                    modified_scenario_name = f"{base_scenario}_r{level}_{ep_suffix}"
                    SCENARIOS[modified_scenario_name] = scenario_config
                    
                    # 运行修改后的场景
                    curr_seed = None if run == 0 else run * 100 + level * 10 + (1 if use_ep else 0)
                    results = self.run_single_scenario(
                        scenario_name=modified_scenario_name,
                        max_steps_override=max_steps,
                        random_seed=curr_seed,
                        generate_visualizations=(run == 0),  # 只为第一次运行生成可视化
                        custom_suffix=f"{experiment_name}_r{level}_{ep_suffix}_run{run+1}"
                    )
                    
                    # 收集指标
                    if results:
                        metrics_data["reasoning_level"].append(level)
                        metrics_data["use_ep"].append(use_ep)
                        metrics_data["run"].append(run + 1)
                        
                        utility = results.get("history_global_utility_ksc", [0])[-1] if "history_global_utility_ksc" in results else 0
                        tasks_completed = sum(1 for t in results.get("tasks_final_state", {}).values() 
                                            if t.get("status") == "completed")
                        total_time = results.get("final_simulation_time_s", 0)
                        msgs_broadcasted = np.mean(results.get("ksc_stats_messages_broadcasted", [0])) if "ksc_stats_messages_broadcasted" in results else 0
                        
                        metrics_data["utility"].append(utility)
                        metrics_data["tasks_completed"].append(tasks_completed)
                        metrics_data["total_time"].append(total_time)
                        metrics_data["messages_broadcasted"].append(msgs_broadcasted)
                    
                    # 删除临时场景
                    if modified_scenario_name in SCENARIOS:
                        del SCENARIOS[modified_scenario_name]
        
        # 将指标数据转换为DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        
        # 保存原始指标数据到CSV
        if self.save_data:
            metrics_csv_path = os.path.join(experiment_dir, "reasoning_level_metrics.csv")
            metrics_df.to_csv(metrics_csv_path, index=False)
            print(f"推理级别分析指标已保存到: {metrics_csv_path}")
        
        # 生成分析图表
        if self.save_visualization or self.show_plots:
            self._generate_reasoning_level_plots(metrics_df, experiment_dir, experiment_name)
        
        print(f"推理级别影响分析实验 '{experiment_name}' 完成!")
    
    def _generate_reasoning_level_plots(self, 
                                      metrics_df: pd.DataFrame, 
                                      save_dir: str, 
                                      experiment_name: str) -> None:
        """
        生成推理级别影响分析的图表
        
        参数:
            metrics_df (pd.DataFrame): 包含分析指标的DataFrame
            save_dir (str): 图表保存目录
            experiment_name (str): 实验名称
        """
        if metrics_df.empty:
            print("警告: 没有数据可用于生成推理级别影响分析图表")
            return
            
        # 为每个组合创建标签
        metrics_df['config'] = metrics_df.apply(
            lambda row: f"L{row['reasoning_level']}" + ("+EP" if row['use_ep'] else ""), 
            axis=1
        )
        
        # 计算每个配置的均值和标准差
        summary = metrics_df.groupby('config').agg({
            'utility': ['mean', 'std'],
            'tasks_completed': ['mean', 'std'],
            'total_time': ['mean', 'std'],
            'messages_broadcasted': ['mean', 'std']
        })
        
        # 提取所有配置及其顺序
        configs = sorted(metrics_df['config'].unique())
        
        # 为每个指标创建柱状图
        metrics_to_plot = [
            ('utility', '系统总体效用'),
            ('tasks_completed', '完成任务数量'),
            ('total_time', '完成时间 (秒)'),
            ('messages_broadcasted', '平均广播消息数')
        ]
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        for i, (metric, title) in enumerate(metrics_to_plot):
            ax = axs[i//2, i%2]
            
            mean_values = [summary.loc[c][(metric, 'mean')] for c in configs]
            std_values = [summary.loc[c][(metric, 'std')] for c in configs]
            
            # 创建不同配置的颜色映射
            colors = []
            for config in configs:
                if "L0" in config:
                    colors.append('#FF9500')  # 零阶推理
                elif "L1" in config:
                    if "+EP" in config:
                        colors.append('#00B945')  # 一阶+EP
                    else:
                        colors.append('#474747')  # 一阶无EP
                elif "L2" in config:
                    if "+EP" in config:
                        colors.append('#0C5DA5')  # 二阶+EP
                    else:
                        colors.append('#845B97')  # 二阶无EP
            
            # 绘制柱状图
            bars = ax.bar(
                range(len(configs)), 
                mean_values,
                yerr=std_values,
                capsize=5,
                error_kw={'elinewidth': 1.5, 'capthick': 1.5},
                width=0.7,
                color=colors
            )
            
            # 添加数值标签
            for bar, mean, std in zip(bars, mean_values, std_values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + std + 0.1 * max(mean_values),
                    f'{mean:.2f}±{std:.2f}', 
                    ha='center', 
                    va='bottom', 
                    fontsize=9,
                    rotation=0
                )
            
            # 设置图表标题和标签
            ax.set_title(title, fontsize=12)
            ax.set_ylabel(title, fontsize=11)
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        fig.suptitle("推理级别与认知规划(EP)对性能的影响", fontsize=14)
        
        # 添加解释图例
        description = "推理级别说明:\n" + \
                      "L0: 零阶推理 (仅局部信息)\n" + \
                      "L1: 一阶推理 (考虑他人观测)\n" + \
                      "L1+EP: 一阶推理+认知规划\n" + \
                      "L2: 二阶推理 (考虑他人推理)\n" + \
                      "L2+EP: 二阶推理+认知规划"
                      
        fig.text(0.02, 0.02, description, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
        
        # 保存图表
        if self.save_visualization:
            fig_path = os.path.join(save_dir, f"{experiment_name}_reasoning_impact.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"推理级别影响分析图表已保存: {fig_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        # 创建推理级别对信念准确性的影响图（如果有数据）
        self._create_belief_accuracy_plot(metrics_df, save_dir, experiment_name)
    
    def _create_belief_accuracy_plot(self, 
                                    metrics_df: pd.DataFrame, 
                                    save_dir: str, 
                                    experiment_name: str) -> None:
        """
        创建推理级别对信念准确性的影响图
        
        参数:
            metrics_df (pd.DataFrame): 原始指标数据
            save_dir (str): 图表保存目录
            experiment_name (str): 实验名称
        """
        # 假设性数据，显示不同推理级别的信念准确性
        # 实际应用中，需要从仿真结果中提取相关数据
        belief_data = {
            'config': ['L0', 'L1', 'L1+EP', 'L2', 'L2+EP'],
            'accuracy': [0.55, 0.68, 0.75, 0.82, 0.92],
            'std': [0.1, 0.08, 0.07, 0.06, 0.05]
        }
        
        belief_df = pd.DataFrame(belief_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制信念准确性柱状图
        bars = ax.bar(
            belief_df['config'], 
            belief_df['accuracy'],
            yerr=belief_df['std'],
            capsize=5,
            error_kw={'elinewidth': 1.5, 'capthick': 1.5},
            width=0.6,
            color=['#FF9500', '#474747', '#00B945', '#845B97', '#0C5DA5']
        )
        
        # 添加数值标签
        for bar, acc, std in zip(bars, belief_df['accuracy'], belief_df['std']):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + std + 0.02,
                f'{acc:.2f}±{std:.2f}', 
                ha='center', 
                va='bottom', 
                fontsize=9
            )
        
        # 设置图表标题和标签
        ax.set_title("推理级别对任务类型信念准确性的影响", fontsize=12)
        ax.set_xlabel("推理级别配置", fontsize=11)
        ax.set_ylabel("信念准确性 (0-1)", fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例和注释
        ax.text(0.02, 0.05, 
               "信念准确性: 智能体对任务真实类型的信念概率\n推理级别越高，考虑越多他人视角，准确性越高", 
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        if self.save_visualization:
            fig_path = os.path.join(save_dir, f"{experiment_name}_belief_accuracy.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"信念准确性分析图表已保存: {fig_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)