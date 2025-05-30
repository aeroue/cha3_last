# experiments/visualization_tools.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator

# 导入项目路径
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
_project_root_directory = os.path.dirname(_current_file_directory)
if _project_root_directory not in sys.path:
    sys.path.insert(0, _project_root_directory)

from simulation.visualizer import Visualizer
from simulation.environment import SimulationEnvironment

class ExtendedVisualizer:
    """
    扩展可视化工具类：提供更丰富的可视化功能
    """
    def __init__(self, base_visualizer: Optional[Visualizer] = None):
        """
        初始化扩展可视化工具
        
        参数:
            base_visualizer (Visualizer, optional): 基础可视化器实例
        """
        self.base_visualizer = base_visualizer
        
        # Nature风格颜色方案
        self.nature_colors = [
            '#0C5DA5', '#FF2C00', '#00B945', '#FF9500',
            '#845B97', '#474747', '#9e9e9e', '#F2C800'
        ]
        
        # 不同的线型和标记
        self.linestyles = ['-', '--', '-.', ':']
        self.markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
        
        # 设置学术风格
        self._setup_academic_style()
    
    def _setup_academic_style(self):
        """设置matplotlib风格为Nature学术风格，并支持中文显示"""
        # 尝试寻找系统中存在的中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STSong', 'DengXian']
        chinese_font_found = False
        
        for font in chinese_fonts:
            try:
                font_path = fm.findfont(font, fallback_to_default=False)
                if font_path:
                    plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                    chinese_font_found = True
                    break
            except:
                continue
        
        if not chinese_font_found:
            print("警告: 未找到合适的中文字体，图表中的中文可能无法正确显示")
        
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        
        # Nature学术风格设置
        plt.style.use('default')  # 先重置为默认风格
        
        plt.rcParams.update({
            # 字体相关
            'font.size': 10,
            'axes.labelsize': 11, 
            'axes.titlesize': 12,
            'xtick.labelsize': 9, 
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            
            # 图形和网格
            'legend.frameon': False, 
            'figure.dpi': 100,
            'savefig.dpi': 300, 
            'savefig.bbox': 'tight',
            'lines.linewidth': 1.8,
            'lines.markersize': 6,
            'axes.grid': True, 
            'grid.linestyle': '--', 
            'grid.alpha': 0.7,
            
            # 其他设置
            'axes.formatter.useoffset': False, 
            'axes.formatter.limits': (-3, 4)
        })
    
    def plot_k_value_vs_performance(self, 
                                  results_data: Dict[int, List[Dict[str, Any]]],
                                  metric_name: str,
                                  title: Optional[str] = None,
                                  figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        绘制K值与性能指标的关系图
        
        参数:
            results_data (Dict[int, List[Dict[str, Any]]]): K值 -> 多次运行的结果列表
            metric_name (str): 要绘制的指标名称
            title (str, optional): 图表标题
            figsize (Tuple[int, int]): 图表大小
            
        返回:
            plt.Figure: 生成的图表对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        k_values = sorted(results_data.keys())
        means = []
        stds = []
        
        for k in k_values:
            # 提取指标值
            metrics = [run.get(metric_name, 0) for run in results_data[k] if metric_name in run]
            if metrics:
                means.append(np.mean(metrics))
                stds.append(np.std(metrics))
            else:
                means.append(0)
                stds.append(0)
        
        # 绘制带误差带的折线图
        ax.errorbar(k_values, means, yerr=stds, fmt='-o', 
                   ecolor='lightgray', capsize=5, capthick=2, 
                   linewidth=2, markersize=8, color=self.nature_colors[0])
        
        # 添加每个点的数值标签
        for i, (k, mean, std) in enumerate(zip(k_values, means, stds)):
            ax.annotate(f'{mean:.2f}±{std:.2f}', 
                       (k, mean), 
                       xytext=(0, 10), 
                       textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)
        
        # 设置图表标题和标签
        if title:
            ax.set_title(title, fontsize=12)
        else:
            ax.set_title(f"K值对{metric_name}的影响", fontsize=12)
            
        ax.set_xlabel("K值 (最大转换链长度)", fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        
        # 设置X轴为整数刻度
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加K值含义解释
        ax.text(0.02, 0.02, 
               "K表示KS-COAL算法中允许的最大转换链长度\n较大的K值提供更大的优化空间，但增加计算复杂度", 
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_comparison_radar(self, 
                            comparison_data: Dict[str, Dict[str, Union[float, List[float]]]],
                            metrics: List[str],
                            title: str = "算法性能比较",
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        绘制算法性能比较的雷达图
        
        参数:
            comparison_data (Dict[str, Dict[str, Union[float, List[float]]]]): 算法名称 -> 指标数据
            metrics (List[str]): 要比较的指标列表
            title (str): 图表标题
            figsize (Tuple[int, int]): 图表大小
            
        返回:
            plt.Figure: 生成的图表对象
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)
        
        # 设置雷达图角度和标签
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        # 设置标签
        ax.set_theta_offset(np.pi / 2)  # 从顶部开始
        ax.set_theta_direction(-1)      # 顺时针
        
        labels = metrics + [metrics[0]]  # 闭合标签
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=10)
        
        # 创建归一化的数据
        # 先提取每个指标的最大值和最小值
        max_values = {}
        min_values = {}
        for metric in metrics:
            all_values = []
            for algo_name, algo_data in comparison_data.items():
                if metric in algo_data:
                    if isinstance(algo_data[metric], list):
                        all_values.extend(algo_data[metric])
                    else:
                        all_values.append(algo_data[metric])
            
            if all_values:
                max_values[metric] = max(all_values)
                min_values[metric] = min(all_values)
            else:
                max_values[metric] = 1.0
                min_values[metric] = 0.0
        
        # 对每个算法绘制雷达图
        for i, (algo_name, algo_data) in enumerate(comparison_data.items()):
            values = []
            for metric in metrics:
                if metric in algo_data:
                    value = algo_data[metric]
                    if isinstance(value, list):
                        value = np.mean(value)
                        
                    # 归一化
                    if max_values[metric] > min_values[metric]:
                        normalized_value = (value - min_values[metric]) / (max_values[metric] - min_values[metric])
                    else:
                        normalized_value = 0.5
                        
                    values.append(normalized_value)
                else:
                    values.append(0.0)
            
            # 闭合雷达图
            values += values[:1]
            
            # 选择颜色和线型
            color = self.nature_colors[i % len(self.nature_colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            
            # 绘制雷达图
            ax.plot(angles, values, linewidth=2, linestyle=linestyle, 
                   label=algo_name, color=color, marker=self.markers[i % len(self.markers)],
                   markersize=6)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # 设置Y轴刻度范围和标签
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["20%", "40%", "60%", "80%"], fontsize=8)
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 添加标题
        plt.title(title, fontsize=14, y=1.08)
        
        return fig
    
    def plot_belief_convergence_animation(self, 
                                        history_beliefs: Dict[int, Dict[str, Dict[str, np.ndarray]]],
                                        tracked_agent_task: Tuple[str, str],
                                        task_true_type_idx: int,
                                        task_type_names: List[str],
                                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        绘制信念收敛过程的动画帧
        
        参数:
            history_beliefs (Dict[int, Dict[str, Dict[str, np.ndarray]]]): 历史信念数据
            tracked_agent_task (Tuple[str, str]): 要跟踪的(智能体ID, 任务ID)
            task_true_type_idx (int): 任务的真实类型索引
            task_type_names (List[str]): 任务类型名称列表
            figsize (Tuple[int, int]): 图表大小
            
        返回:
            plt.Figure: 生成的图表对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        agent_id, task_id = tracked_agent_task
        rounds = sorted(history_beliefs.keys())
        
        if not rounds:
            ax.text(0.5, 0.5, "没有信念历史数据", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return fig
        
        # 提取每轮中的信念数据
        beliefs_over_time = []
        for round_idx in rounds:
            if agent_id in history_beliefs[round_idx] and task_id in history_beliefs[round_idx][agent_id]:
                beliefs_over_time.append(history_beliefs[round_idx][agent_id][task_id])
            else:
                # 如果某轮缺失数据，使用之前的数据填充
                if beliefs_over_time:
                    beliefs_over_time.append(beliefs_over_time[-1])
                else:
                    # 如果是第一个就缺失，创建均匀分布
                    num_types = len(task_type_names)
                    beliefs_over_time.append(np.ones(num_types) / num_types)
        
        # 将数据转换为 (轮次 x 类型) 的数组
        belief_matrix = np.array(beliefs_over_time)
        
        # 绘制每种类型的信念演化
        for type_idx in range(belief_matrix.shape[1]):
            color = self.nature_colors[type_idx % len(self.nature_colors)]
            linestyle = self.linestyles[type_idx % len(self.linestyles)]
            marker = self.markers[type_idx % len(self.markers)]
            
            # 如果是真实类型，使用粗线
            linewidth = 2.5 if type_idx == task_true_type_idx else 1.8
            
            # 绘制信念演化曲线
            ax.plot(rounds, belief_matrix[:, type_idx], 
                   label=task_type_names[type_idx],
                   color=color, linestyle=linestyle, marker=marker,
                   linewidth=linewidth, markersize=6)
        
        # 标出真实类型
        ax.text(0.02, 0.98, f"真实类型: {task_type_names[task_true_type_idx]}", 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", fc=self.nature_colors[task_true_type_idx], 
                        ec="black", alpha=0.2))
        
        # 设置图表标题和标签
        ax.set_title(f"智能体 {agent_id} 对任务 {task_id} 的信念收敛过程", fontsize=12)
        ax.set_xlabel("KSC决策轮次", fontsize=11)
        ax.set_ylabel("信念概率 P(type)", fontsize=11)
        
        # Y轴范围
        ax.set_ylim(-0.05, 1.05)
        
        # X轴使用整数刻度
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 添加图例
        ax.legend(loc='center right')
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_adaptive_switching_animation(self,
                                        csi_history: List[float],
                                        mode_history: List[str],
                                        time_axis: List[float],
                                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制自适应切换过程的动画帧
        
        参数:
            csi_history (List[float]): CSI值历史
            mode_history (List[str]): 决策模式历史 ('weak', 'transition', 'strong')
            time_axis (List[float]): 时间轴
            figsize (Tuple[int, int]): 图表大小
            
        返回:
            plt.Figure: 生成的图表对象
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                      gridspec_kw={'height_ratios': [2, 1]})
        
        # 设置CSI阈值线
        low_threshold = 0.3
        high_threshold = 0.7
        
        # 绘制CSI历史
        ax1.plot(time_axis, csi_history, '-', color=self.nature_colors[0], 
               linewidth=2, label="CSI值")
        
        # 添加阈值线
        ax1.axhline(y=low_threshold, color='r', linestyle='--', 
                   label="低阈值 (弱通信<->过渡)", alpha=0.7)
        ax1.axhline(y=high_threshold, color='g', linestyle='--', 
                   label="高阈值 (过渡<->强通信)", alpha=0.7)
        
        # 填充不同通信区域
        ax1.fill_between(time_axis, 0, low_threshold, color='r', alpha=0.1)
        ax1.fill_between(time_axis, low_threshold, high_threshold, color='y', alpha=0.1)
        ax1.fill_between(time_axis, high_threshold, 1, color='g', alpha=0.1)
        
        # 设置区域标签
        ax1.text(time_axis[0], 0.15, "弱通信区域", fontsize=9, color='r', alpha=0.8)
        ax1.text(time_axis[0], 0.5, "过渡区域", fontsize=9, color='black', alpha=0.8)
        ax1.text(time_axis[0], 0.85, "强通信区域", fontsize=9, color='g', alpha=0.8)
        
        # 设置图表标题和标签
        ax1.set_title("自适应控制器模式切换", fontsize=12)
        ax1.set_ylabel("CSI值", fontsize=11)
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 转换决策模式为数值
        mode_mapping = {"weak": 0, "transition": 1, "strong": 2, "unknown": -1}
        mode_numeric = [mode_mapping.get(m, -1) for m in mode_history]
        
        # 绘制决策模式历史
        ax2.plot(time_axis, mode_numeric, drawstyle='steps-post', 
               color=self.nature_colors[1], linewidth=2, marker='o')
        
        # 设置Y轴刻度和标签
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(["弱通信", "过渡", "强通信"])
        ax2.set_ylabel("控制模式", fontsize=11)
        ax2.set_xlabel("仿真时间 (s)", fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 添加注释说明决策逻辑
        ax2.text(0.02, 0.1, 
               "决策逻辑: CSI < 0.3 → 弱通信模式 (AIF)\n"
               "0.3 ≤ CSI < 0.7 → 过渡模式\n"
               "CSI ≥ 0.7 → 强通信模式 (KSC)",
               transform=ax2.transAxes, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_task_completion_time_comparison(self,
                                           completion_times: Dict[str, List[float]],
                                           title: str = "任务完成时间比较",
                                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        绘制不同算法/场景的任务完成时间比较
        
        参数:
            completion_times (Dict[str, List[float]]): 算法/场景名称 -> 任务完成时间列表
            title (str): 图表标题
            figsize (Tuple[int, int]): 图表大小
            
        返回:
            plt.Figure: 生成的图表对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        algorithms = list(completion_times.keys())
        
        # 计算每个算法的平均完成时间和标准差
        means = []
        stds = []
        for algo in algorithms:
            times = completion_times[algo]
            means.append(np.mean(times))
            stds.append(np.std(times))
        
        # 绘制柱状图
        bars = ax.bar(algorithms, means, yerr=stds, capsize=5, 
                     color=[self.nature_colors[i % len(self.nature_colors)] for i in range(len(algorithms))],
                     alpha=0.8, edgecolor='black', linewidth=0.8)
        
        # 添加完成时间标签
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                   f'{mean:.1f}±{std:.1f}s', ha='center', va='bottom', fontsize=9)
        
        # 设置图表标题和标签
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("完成时间 (s)", fontsize=11)
        ax.set_xlabel("算法/场景", fontsize=11)
        
        # 设置Y轴起点为0
        ax.set_ylim(bottom=0)
        
        # 添加网格
        ax.grid(True, linestyle='--', axis='y', alpha=0.7)
        
        # 添加显著性标记
        # 假设完成时间越短越好，找出最佳算法
        best_idx = np.argmin(means)
        ax.text(best_idx, 5, "✓", fontsize=16, color='green', 
               ha='center', va='center', 
               bbox=dict(boxstyle="circle", fc='white', ec='green', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_monte_carlo_results(self,
                               monte_carlo_data: Dict[str, List[Dict[str, Any]]],
                               metric_name: str,
                               title: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制蒙特卡洛实验结果
        
        参数:
            monte_carlo_data (Dict[str, List[Dict[str, Any]]]): 场景名称 -> 多次运行的结果列表
            metric_name (str): 要绘制的指标名称
            title (str, optional): 图表标题
            figsize (Tuple[int, int]): 图表大小
            
        返回:
            plt.Figure: 生成的图表对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 所有场景的所有运行结果合并为一个列表
        all_results = []
        for scenario, results in monte_carlo_data.items():
            for result in results:
                if metric_name in result:
                    all_results.append({
                        "scenario": scenario,
                        "value": result[metric_name]
                    })
        
        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        if df.empty:
            ax1.text(0.5, 0.5, "没有蒙特卡洛实验数据", 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax2.text(0.5, 0.5, "没有蒙特卡洛实验数据", 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            return fig
        
        # 绘制小提琴图
        sns.violinplot(x="scenario", y="value", data=df, ax=ax1, 
                      palette=self.nature_colors, alpha=0.8)
        
        # 添加数据点
        sns.stripplot(x="scenario", y="value", data=df, ax=ax1, 
                     color='black', alpha=0.5, jitter=True, size=4)
        
        # 设置图表标题和标签
        if title:
            ax1.set_title(f"{title} - 分布", fontsize=12)
        else:
            ax1.set_title(f"{metric_name} - 分布", fontsize=12)
            
        ax1.set_xlabel("场景", fontsize=11)
        ax1.set_ylabel(metric_name, fontsize=11)
        
        # 旋转X轴标签
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        
        # 添加网格
        ax1.grid(True, linestyle='--', axis='y', alpha=0.7)
        
        # 绘制箱线图
        sns.boxplot(x="scenario", y="value", data=df, ax=ax2, 
                   palette=self.nature_colors, width=0.6, fliersize=4)
        
        # 设置图表标题和标签
        if title:
            ax2.set_title(f"{title} - 箱线图", fontsize=12)
        else:
            ax2.set_title(f"{metric_name} - 箱线图", fontsize=12)
            
        ax2.set_xlabel("场景", fontsize=11)
        ax2.set_ylabel(metric_name, fontsize=11)
        
        # 旋转X轴标签
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        
        # 添加网格
        ax2.grid(True, linestyle='--', axis='y', alpha=0.7)
        
        # 添加统计信息
        stats_text = "统计摘要:\n"
        for scenario in df["scenario"].unique():
            values = df[df["scenario"] == scenario]["value"]
            stats_text += f"{scenario}:\n"
            stats_text += f"  均值: {values.mean():.2f}\n"
            stats_text += f"  标准差: {values.std():.2f}\n"
            stats_text += f"  中位数: {values.median():.2f}\n"
            stats_text += f"  最小值: {values.min():.2f}\n"
            stats_text += f"  最大值: {values.max():.2f}\n"
        
        # 在图表底部添加统计信息
        fig.text(0.02, 0.02, stats_text, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # 为底部的统计文本留出空间
        return fig
    
    def plot_communication_metrics(self,
                                 comm_data: Dict[str, Dict[str, Any]],
                                 figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        绘制综合通信指标图表
        
        参数:
            comm_data (Dict[str, Dict[str, Any]]): 场景名称 -> 通信指标数据
            figsize (Tuple[int, int]): 图表大小
            
        返回:
            plt.Figure: 生成的图表对象
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # 上左：消息数量
        ax1 = fig.add_subplot(gs[0, 0])
        # 上右：链接数
        ax2 = fig.add_subplot(gs[0, 1])
        # 下：CSI与决策模式
        ax3 = fig.add_subplot(gs[1, :])
        
        # 场景名称和颜色映射
        scenarios = list(comm_data.keys())
        colors = [self.nature_colors[i % len(self.nature_colors)] for i in range(len(scenarios))]
        
        # 1. 绘制消息数量柱状图
        msgs_data = []
        for scenario in scenarios:
            if 'messages_broadcast' in comm_data[scenario]:
                msgs_data.append({
                    'scenario': scenario,
                    'value': comm_data[scenario]['messages_broadcast'],
                    'type': '广播'
                })
            if 'messages_processed' in comm_data[scenario]:
                msgs_data.append({
                    'scenario': scenario,
                    'value': comm_data[scenario]['messages_processed'],
                    'type': '处理'
                })
        
        if msgs_data:
            df_msgs = pd.DataFrame(msgs_data)
            sns.barplot(x='scenario', y='value', hue='type', data=df_msgs, ax=ax1, 
                       palette=[self.nature_colors[0], self.nature_colors[1]])
            
            ax1.set_title("通信消息数量", fontsize=12)
            ax1.set_xlabel("场景", fontsize=11)
            ax1.set_ylabel("消息数", fontsize=11)
            
            # 旋转X轴标签
            plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
            
            # 添加网格
            ax1.grid(True, linestyle='--', axis='y', alpha=0.7)
        else:
            ax1.text(0.5, 0.5, "没有消息数据", 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        
        # 2. 绘制链接数柱状图
        links_data = []
        for scenario in scenarios:
            if 'active_links' in comm_data[scenario]:
                links_data.append({
                    'scenario': scenario,
                    'value': comm_data[scenario]['active_links']
                })
        
        if links_data:
            df_links = pd.DataFrame(links_data)
            sns.barplot(x='scenario', y='value', data=df_links, ax=ax2, 
                       palette=colors[:len(df_links)])
            
            ax2.set_title("平均活动通信链接数", fontsize=12)
            ax2.set_xlabel("场景", fontsize=11)
            ax2.set_ylabel("链接数", fontsize=11)
            
            # 旋转X轴标签
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
            
            # 添加网格
            ax2.grid(True, linestyle='--', axis='y', alpha=0.7)
        else:
            ax2.text(0.5, 0.5, "没有链接数据", 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        # 3. 绘制CSI变化曲线
        any_csi_data = False
        for i, scenario in enumerate(scenarios):
            if 'csi_history' in comm_data[scenario] and 'time_axis' in comm_data[scenario]:
                csi_history = comm_data[scenario]['csi_history']
                time_axis = comm_data[scenario]['time_axis']
                
                if len(csi_history) > 0 and len(csi_history) == len(time_axis):
                    ax3.plot(time_axis, csi_history, label=scenario, 
                            color=colors[i], linewidth=2)
                    any_csi_data = True
        
        if any_csi_data:
            # 添加CSI阈值线
            ax3.axhline(y=0.3, color='r', linestyle='--', 
                       label="低阈值 (弱通信<->过渡)", alpha=0.7)
            ax3.axhline(y=0.7, color='g', linestyle='--', 
                       label="高阈值 (过渡<->强通信)", alpha=0.7)
            
            # 填充不同通信区域
            xmin, xmax = ax3.get_xlim()
            ax3.fill_between([xmin, xmax], 0, 0.3, color='r', alpha=0.1)
            ax3.fill_between([xmin, xmax], 0.3, 0.7, color='y', alpha=0.1)
            ax3.fill_between([xmin, xmax], 0.7, 1, color='g', alpha=0.1)
            
            ax3.set_title("CSI值随时间的变化", fontsize=12)
            ax3.set_xlabel("仿真时间 (s)", fontsize=11)
            ax3.set_ylabel("CSI值", fontsize=11)
            ax3.set_ylim(-0.05, 1.05)
            ax3.legend(loc='upper right')
            ax3.grid(True, linestyle='--', alpha=0.7)
        else:
            ax3.text(0.5, 0.5, "没有CSI数据", 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """
        保存图表为高分辨率图像
        
        参数:
            fig (plt.Figure): 图表对象
            filename (str): 文件名
            dpi (int): 分辨率
        """
        # 确保文件名的父目录存在
        dirname = os.path.dirname(filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
            
        try:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"图表已保存: {filename}")
        except Exception as e:
            print(f"保存图表失败: {e}")

# 如果直接运行此脚本，执行一些测试
if __name__ == "__main__":
    visualizer = ExtendedVisualizer()
    
    # 测试雷达图
    comparison_data = {
        "强通信": {"效用": 85.2, "任务完成数": 4, "通信开销": 120},
        "弱通信": {"效用": 65.8, "任务完成数": 3, "通信开销": 20},
        "自适应": {"效用": 80.5, "任务完成数": 4, "通信开销": 75}
    }
    
    radar_fig = visualizer.plot_comparison_radar(
        comparison_data,
        metrics=["效用", "任务完成数", "通信开销"],
        title="通信模式性能比较"
    )
    plt.figure(radar_fig.number)
    plt.show()
    
    # 测试蒙特卡洛结果图
    monte_carlo_data = {
        "K=2": [
            {"utility": 85.2, "time": 250.1, "tasks": 4},
            {"utility": 82.5, "time": 260.3, "tasks": 4},
            {"utility": 78.9, "time": 270.2, "tasks": 3}
        ],
        "K=3": [
            {"utility": 90.1, "time": 240.5, "tasks": 4},
            {"utility": 88.7, "time": 245.2, "tasks": 4},
            {"utility": 85.3, "time": 255.0, "tasks": 4}
        ]
    }
    
    mc_fig = visualizer.plot_monte_carlo_results(
        monte_carlo_data,
        metric_name="utility",
        title="系统效用蒙特卡洛分析"
    )
    plt.figure(mc_fig.number)
    plt.show()