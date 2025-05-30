# experiments/strong_comm_belief_learning/visualizer.py
"""
强通信信念自学习实验可视化器
生成高质量的学术论文图表
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyBboxPatch, Ellipse
from matplotlib.collections import LineCollection
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from scipy import stats
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.font_manager as fm
import platform
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class BeliefLearningVisualizer:
    """强通信信念自学习实验可视化器"""
    
    def __init__(self, experiment_dir: str, analysis_dir: Optional[str] = None):
        """
        初始化可视化器
        
        Args:
            experiment_dir: 实验结果目录
            analysis_dir: 分析结果目录
        """
        self.experiment_dir = Path(experiment_dir)
        self.analysis_dir = Path(analysis_dir) if analysis_dir else self.experiment_dir / "analysis"
        self.output_dir = self.experiment_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置学术风格和中文字体
        self.setup_academic_style()
        
        # Nature风格配色方案 - 扩展版本
        self.colors = [
            '#0C5DA5', '#FF2C00', '#00B945', '#FF9500', '#845B97', 
            '#474747', '#9e9e9e', '#F2C800', '#E64B35', '#4DBBD5',
            '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2',
            '#DC0000', '#7E6148', '#B09C85', '#1B9E77', '#D95F02',
            '#7570B3', '#E7298A', '#66A61E', '#E6AB02', '#A6761D'
        ]
        
        # 线型样式 - 更多样化
        self.line_styles = [
            '-', '--', '-.', ':', 
            (0, (3, 1, 1, 1)), (0, (5, 1)), 
            (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1))
        ]
        
        # 标记样式 - 更丰富
        self.markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', 'd', 
                       '>', '<', 'p', 'H', '8', '+', 'x', '|', '_']
        
        # 加载数据
        self.load_data()
        
        print(f"可视化器初始化完成，输出目录: {self.output_dir}")
        
    def setup_academic_style(self):
        """设置学术风格和中文字体"""
        # 中文字体设置 - 更全面的支持
        chinese_fonts = []
        if platform.system() == 'Windows':
            chinese_fonts = [
                'Microsoft YaHei', 'SimHei', 'SimSun', 'DengXian', 
                'KaiTi', 'FangSong', 'STSong', 'STKaiti'
            ]
        elif platform.system() == 'Darwin':  # macOS
            chinese_fonts = [
                'PingFang SC', 'Hiragino Sans GB', 'STSong', 
                'Arial Unicode MS', 'Songti SC'
            ]
        else:  # Linux
            chinese_fonts = [
                'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 
                'WenQuanYi Zen Hei', 'Source Han Sans CN',
                'Droid Sans Fallback'
            ]
        
        # 尝试设置中文字体
        font_found = False
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font_name in chinese_fonts:
            if font_name in available_fonts:
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                font_found = True
                print(f"使用中文字体: {font_name}")
                break
        
        if not font_found:
            print("警告: 未找到合适的中文字体，尝试设置备用方案")
            plt.rcParams['font.sans-serif'] = chinese_fonts + ['Arial', 'DejaVu Sans']
        
        # 重置matplotlib样式
        plt.style.use('default')
        
        # 详细的学术风格参数设置
        academic_params = {
            # 字体设置 - 更大更清晰
            'font.size': 18,                    # 基础字体大小
            'axes.labelsize': 22,               # 坐标轴标签字体大小
            'axes.titlesize': 24,               # 子图标题字体大小
            'xtick.labelsize': 16,              # x轴刻度标签大小
            'ytick.labelsize': 16,              # y轴刻度标签大小
            'legend.fontsize': 16,              # 图例字体大小
            'figure.titlesize': 28,             # 图形标题字体大小
            
            # 线条和标记设置 - 更粗更明显
            'lines.linewidth': 4.0,             # 线条宽度
            'lines.markersize': 12,             # 标记大小
            'lines.markeredgewidth': 2.0,       # 标记边缘宽度
            'patch.linewidth': 2.0,             # 图形边框宽度
            
            # 坐标轴设置 - 更明显的边框
            'axes.linewidth': 2.5,              # 坐标轴线宽度
            'axes.spines.top': False,           # 不显示上边框
            'axes.spines.right': False,         # 不显示右边框
            'axes.edgecolor': 'black',          # 坐标轴颜色
            'axes.labelpad': 18,                # 标签与坐标轴的距离
            'axes.titlepad': 25,                # 标题与图形的距离
            'axes.formatter.useoffset': False,   # 不使用偏移量
            'axes.xmargin': 0.02,               # x轴边距
            'axes.ymargin': 0.02,               # y轴边距
            
            # 网格设置 - 更清晰的网格
            'axes.grid': True,                  # 显示网格
            'grid.linewidth': 1.5,              # 网格线宽度
            'grid.alpha': 0.4,                  # 网格透明度
            'grid.linestyle': '--',             # 网格线样式
            'grid.color': 'gray',               # 网格颜色
            
            # 刻度设置 - 更大的刻度
            'xtick.major.size': 10,             # x轴主刻度大小
            'ytick.major.size': 10,             # y轴主刻度大小
            'xtick.minor.size': 6,              # x轴次刻度大小
            'ytick.minor.size': 6,              # y轴次刻度大小
            'xtick.major.width': 2.0,           # x轴主刻度宽度
            'ytick.major.width': 2.0,           # y轴主刻度宽度
            'xtick.minor.width': 1.5,           # x轴次刻度宽度
            'ytick.minor.width': 1.5,           # y轴次刻度宽度
            'xtick.major.pad': 10,              # x轴刻度标签距离
            'ytick.major.pad': 10,              # y轴刻度标签距离
            
            # 图例设置 - 更美观的图例
            'legend.frameon': True,             # 显示图例边框
            'legend.fancybox': True,            # 圆角图例框
            'legend.shadow': True,              # 图例阴影
            'legend.numpoints': 1,              # 图例中的点数
            'legend.scatterpoints': 1,          # 散点图图例点数
            'legend.handlelength': 2.5,         # 图例线长度
            'legend.handletextpad': 1.0,        # 图例文本间距
            'legend.columnspacing': 2.5,        # 图例列间距
            'legend.borderpad': 1.0,            # 图例边距
            'legend.framealpha': 0.9,           # 图例背景透明度
            
            # 图形设置 - 高质量输出
            'figure.facecolor': 'white',        # 图形背景色
            'axes.facecolor': 'white',          # 坐标轴背景色
            'figure.dpi': 150,                  # 图形DPI
            'savefig.dpi': 300,                 # 保存DPI
            'savefig.bbox': 'tight',            # 紧密裁剪
            'savefig.pad_inches': 0.3,          # 保存边距
            'savefig.facecolor': 'white',       # 保存背景色
            'savefig.edgecolor': 'none',        # 保存边框色
            'savefig.format': 'png',            # 默认保存格式
            
            # 文本设置
            'axes.unicode_minus': False,        # 正确显示负号
            'text.usetex': False,               # 不使用LaTeX
            'mathtext.default': 'regular',      # 数学文本样式
            
            # 颜色循环设置
            'axes.prop_cycle': plt.cycler('color', self.colors[:10])
        }
        
        # 应用参数设置
        plt.rcParams.update(academic_params)
        
        # 设置seaborn样式
        sns.set_palette(self.colors)
        sns.set_context("paper", font_scale=1.5, rc={
            "lines.linewidth": 4.0,
            "patch.linewidth": 2.0,
            "grid.linewidth": 1.5
        })
        
        print("学术风格设置完成")
    
    def load_data(self):
        """加载实验和分析数据"""
        print("加载可视化数据...")
        
        # 加载分析结果
        self.analysis_data = {}
        analysis_files = {
            'summary_stats': 'summary_statistics.json',
            'scalability': 'scalability_analysis.json',
            'k_value': 'k_value_analysis.json',
            'convergence': 'convergence_analysis.json',
            'statistical': 'statistical_tests.json'
        }
        
        for data_type, filename in analysis_files.items():
            file_path = self.analysis_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.analysis_data[data_type] = json.load(f)
                    print(f"成功加载 {filename}")
                except Exception as e:
                    print(f"加载{filename}失败: {e}")
                    self.analysis_data[data_type] = {}
            else:
                print(f"未找到文件: {filename}")
                self.analysis_data[data_type] = {}
        
        # 加载原始实验数据（用于详细可视化）
        self.experiment_data = self._load_experiment_details()
        
        # 统计加载的数据
        total_configs = sum(len(configs) for configs in self.experiment_data.values())
        print(f"数据加载完成，共加载 {total_configs} 个实验配置")
    
    def _load_experiment_details(self) -> Dict[str, Any]:
        """加载详细的实验数据"""
        experiment_data = {'scalability': {}, 'comparison': {}, 'robustness': {}}
        
        for exp_type in experiment_data.keys():
            exp_type_dir = self.experiment_dir / exp_type
            if not exp_type_dir.exists():
                continue
                
            for config_dir in exp_type_dir.iterdir():
                if not config_dir.is_dir():
                    continue
                    
                config_name = config_dir.name
                config_data = []
                
                for run_dir in config_dir.iterdir():
                    if not run_dir.is_dir() or not run_dir.name.startswith('run_'):
                        continue
                    
                    # 加载仿真环境状态
                    env_state_file = run_dir / "env_state.pkl"
                    result_file = run_dir / "results.pkl"
                    
                    if env_state_file.exists():
                        try:
                            with open(env_state_file, 'rb') as f:
                                env_data = pickle.load(f)
                            
                            # 如果有结果文件，也加载进来
                            if result_file.exists():
                                with open(result_file, 'rb') as f:
                                    result_data = pickle.load(f)
                                env_data['result_summary'] = result_data
                                
                            config_data.append(env_data)
                        except Exception as e:
                            print(f"加载数据失败: {run_dir}, 错误: {e}")
                            continue
                
                if config_data:
                    experiment_data[exp_type][config_name] = config_data
        
        return experiment_data
    
    def _select_representative_config(self) -> Optional[Dict[str, Any]]:
        """选择一个有代表性的配置用于详细可视化"""
        # 优先选择扩展性实验中的中等规模配置
        for exp_type in ['scalability', 'comparison', 'robustness']:
            type_data = self.experiment_data.get(exp_type, {})
            if not type_data:
                continue
            
            # 寻找合适的配置
            for config_name, runs in type_data.items():
                if runs and len(runs) > 0:
                    env_data = runs[0]  # 取第一次运行的数据
                    
                    # 检查数据完整性
                    agents_data = env_data.get('agents', {})
                    tasks = env_data.get('tasks', {})
                    
                    if len(agents_data) >= 3 and len(tasks) >= 4:
                        return {
                            'config_name': config_name,
                            'exp_type': exp_type,
                            'env_data': env_data
                        }
        
        print("警告: 未找到合适的代表性配置")
        return None
    
    def save_figure(self, fig: plt.Figure, filename: str, formats: List[str] = None):
        """保存图形到文件"""
        if formats is None:
            formats = ['png', 'pdf']
        
        for fmt in formats:
            filepath = self.output_dir / f"{filename}.{fmt}"
            try:
                fig.savefig(filepath, format=fmt, dpi=300 if fmt == 'png' else None,
                           bbox_inches='tight', pad_inches=0.3, 
                           facecolor='white', edgecolor='none')
                print(f"图表已保存: {filepath}")
            except Exception as e:
                print(f"保存图表失败 ({fmt}): {e}")
        
        # 自动关闭图形
        plt.close(fig)
    
    def create_all_visualizations(self):
        """创建所有可视化图表"""
        print("开始生成所有可视化图表...")
        
        visualization_methods = [
            ('01_initial_positions_communication', self.plot_initial_positions_communication),
            ('02_agent_trajectories', self.plot_agent_trajectories),
            ('03_key_moments_trajectories', self.plot_key_moments_trajectories),
            ('04_coalition_evolution', self.plot_coalition_evolution),
            ('05_task_gantt_chart', self.plot_task_gantt_chart),
            ('06_agent_execution_timeline', self.plot_agent_execution_timeline),
            ('07_belief_convergence', self.plot_belief_convergence),
            ('08_acceleration_curves', self.plot_acceleration_curves),
            ('09_velocity_curves', self.plot_velocity_curves),
            ('10_utility_convergence', self.plot_utility_convergence),
            ('11_k_value_comparison', self.plot_k_value_comparison),
            ('12_communication_statistics', self.plot_communication_statistics),
            ('13_scalability_analysis', self.plot_scalability_analysis),
            ('14_performance_comparison', self.plot_performance_comparison),
            ('15_statistical_significance', self.plot_statistical_significance)
        ]
        
        successful_plots = 0
        failed_plots = []
        
        for plot_name, method in visualization_methods:
            try:
                print(f"正在生成图表: {plot_name}")
                method()
                successful_plots += 1
                print(f"✓ {plot_name} 生成成功")
            except Exception as e:
                print(f"✗ 生成{plot_name}失败: {e}")
                failed_plots.append(plot_name)
                import traceback
                traceback.print_exc()
            finally:
                # 确保图形被关闭
                plt.close('all')
        
        print(f"\n图表生成完成！")
        print(f"成功生成: {successful_plots}/{len(visualization_methods)} 个图表")
        if failed_plots:
            print(f"失败的图表: {', '.join(failed_plots)}")
        print(f"图表保存位置: {self.output_dir}")
    
    def plot_initial_positions_communication(self):
        """绘制初始位置和通信连接图"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('强通信信念自学习系统初始配置分析', fontsize=28, fontweight='bold', y=0.95)
        
        # 选择四个不同的配置进行可视化
        configs_to_plot = []
        
        # 从不同实验类型中选择配置
        for exp_type in ['scalability', 'comparison', 'robustness']:
            type_data = self.experiment_data.get(exp_type, {})
            for config_name, runs in type_data.items():
                if runs and len(configs_to_plot) < 4:
                    configs_to_plot.append((exp_type, config_name, runs[0]))
        
        # 如果配置不足，重复使用
        while len(configs_to_plot) < 4 and configs_to_plot:
            configs_to_plot.extend(configs_to_plot[:4-len(configs_to_plot)])
        
        if not configs_to_plot:
            print("未找到配置数据进行初始位置可视化")
            plt.close(fig)
            return
        
        for idx, (exp_type, config_name, env_data) in enumerate(configs_to_plot[:4]):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            # 提取智能体初始位置
            agents_data = env_data.get('agents', {})
            agent_positions = {}
            
            for agent_id, agent_info in agents_data.items():
                history_states = agent_info.get('history_states', [])
                if history_states:
                    initial_pos = np.array(history_states[0][:2])
                    agent_positions[agent_id] = initial_pos
            
            # 提取任务位置
            tasks = env_data.get('tasks', {})
            task_positions = {}
            task_types = {}
            
            for task_id, task_info in tasks.items():
                if 'position' in task_info:
                    pos = np.array(task_info['position'][:2])
                    task_positions[task_id] = pos
                    task_types[task_id] = task_info.get('true_type_name', '未知')
            
            # 任务类型颜色映射
            type_colors = {
                '高价值科研': self.colors[0], 
                '普通巡查': self.colors[1], 
                '紧急维修': self.colors[2],
                '未知': self.colors[3]
            }
            
            # 绘制任务位置 - 更大更醒目
            plotted_types = set()
            for task_id, pos in task_positions.items():
                task_type = task_types.get(task_id, '未知')
                color = type_colors.get(task_type, self.colors[3])
                
                # 任务标记
                ax.scatter(pos[0], pos[1], c=color, s=500, marker='^', 
                          edgecolors='black', linewidth=3, alpha=0.9, zorder=10,
                          label=f'{task_type}任务' if task_type not in plotted_types else "")
                
                # 任务标签 - 更清晰的标注
                ax.annotate(task_id, (pos[0], pos[1]), xytext=(8, 8), 
                           textcoords='offset points', fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                   edgecolor='black', alpha=0.8))
                plotted_types.add(task_type)
            
            # 绘制智能体位置 - 更大更明显
            for i, (agent_id, pos) in enumerate(agent_positions.items()):
                color = self.colors[i + 4] if i + 4 < len(self.colors) else self.colors[i % len(self.colors)]
                
                # 智能体标记
                ax.scatter(pos[0], pos[1], c=color, s=400, marker='o', 
                          edgecolors='white', linewidth=3, alpha=1.0, zorder=8,
                          label=f'智能体' if i == 0 else "")
                
                # 智能体标签
                ax.annotate(agent_id, (pos[0], pos[1]), xytext=(-8, -20), 
                           textcoords='offset points', fontsize=12, fontweight='bold',
                           color=color, bbox=dict(boxstyle="round,pad=0.2", 
                                                 facecolor='white', alpha=0.7))
            
            # 绘制通信连接（强通信，全连接）- 更明显的连接线
            agent_pos_list = list(agent_positions.values())
            for i in range(len(agent_pos_list)):
                for j in range(i + 1, len(agent_pos_list)):
                    pos1, pos2 = agent_pos_list[i], agent_pos_list[j]
                    
                    # 计算距离并用不同颜色表示连接强度
                    distance = np.linalg.norm(pos1 - pos2)
                    alpha = max(0.2, 1.0 - distance / 3000)  # 距离越近，连接越明显
                    
                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                           'k-', alpha=alpha, linewidth=2.5, zorder=1)
            
            # 设置坐标轴范围 - 确保5km x 5km范围
            ax.set_xlim(-2800, 2800)
            ax.set_ylim(-2800, 2800)
            ax.set_aspect('equal')
            
            # 添加网格和边框
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)
            ax.set_axisbelow(True)
            
            # 坐标轴标签
            ax.set_xlabel('X坐标 (m)', fontsize=18, fontweight='bold')
            ax.set_ylabel('Y坐标 (m)', fontsize=18, fontweight='bold')
            
            # 子图标题
            num_agents = len(agent_positions)
            num_tasks = len(task_positions)
            exp_type_chinese = {'scalability': '扩展性', 'comparison': '对比', 'robustness': '鲁棒性'}
            
            ax.set_title(f'{num_agents}智能体-{num_tasks}任务配置\n({exp_type_chinese.get(exp_type, exp_type)}实验)', 
                        fontsize=16, fontweight='bold', pad=15)
            
            # 添加图例（只在第一个子图）
            if idx == 0:
                ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=12,
                         frameon=True, fancybox=True, shadow=True)
            
            # 添加配置信息文本框
            info_text = f'配置: {config_name.replace("_", " ")}\n智能体: {num_agents}\n任务: {num_tasks}'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', 
                           alpha=0.8, edgecolor='gray'))
        
        plt.tight_layout()
        self.save_figure(fig, '01_initial_positions_communication')
    
    def plot_agent_trajectories(self):
        """绘制智能体轨迹图"""
        selected_config = self._select_representative_config()
        if not selected_config:
            print("未找到合适的配置数据进行轨迹绘制")
            return
        
        env_data = selected_config['env_data']
        agents_data = env_data.get('agents', {})
        tasks = env_data.get('tasks', {})
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 14))
        
        # 绘制任务位置 - 更精美的任务标记
        task_type_colors = {
            '高价值科研': self.colors[0], 
            '普通巡查': self.colors[1], 
            '紧急维修': self.colors[2]
        }
        task_legends = set()
        
        for task_id, task_info in tasks.items():
            if 'position' in task_info:
                pos = np.array(task_info['position'][:2])
                task_type = task_info.get('true_type_name', '未知')
                color = task_type_colors.get(task_type, self.colors[3])
                completion = task_info.get('current_completion_percentage', 0)
                
                # 根据完成情况选择不同的视觉效果
                if completion >= 100:
                    marker = 'X'
                    alpha = 0.7
                    size = 600
                    edge_style = 'solid'
                    status_text = "已完成"
                else:
                    marker = '^'
                    alpha = 1.0
                    size = 500
                    edge_style = 'solid'
                    status_text = f"{completion:.0f}%"
                
                # 任务标记
                scatter = ax.scatter(pos[0], pos[1], c=color, s=size, marker=marker, 
                                   edgecolors='black', linewidth=3, alpha=alpha, 
                                   zorder=10, linestyle=edge_style)
                
                # 为已完成任务添加光晕效果
                if completion >= 100:
                    ax.scatter(pos[0], pos[1], c=color, s=size*1.5, marker=marker, 
                             alpha=0.3, zorder=9)
                
                # 任务标签 - 更美观的标注
                label_text = f'{task_id}\n({status_text})'
                text = ax.annotate(label_text, (pos[0], pos[1]), 
                                  xytext=(15, 15), textcoords='offset points', 
                                  fontsize=13, fontweight='bold', ha='left',
                                  bbox=dict(boxstyle="round,pad=0.4", 
                                          facecolor='white', edgecolor=color,
                                          alpha=0.9, linewidth=2))
                
                # 添加文字阴影效果
                text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
                
                # 图例
                legend_label = f'{task_type}任务'
                if legend_label not in task_legends:
                    task_legends.add(legend_label)
        
        # 绘制智能体轨迹 - 更丰富的轨迹可视化
        for i, (agent_id, agent_info) in enumerate(agents_data.items()):
            history_states = agent_info.get('history_states', [])
            if not history_states:
                continue
            
            states = np.array(history_states)
            if states.ndim != 2 or states.shape[1] < 2:
                continue
            
            positions = states[:, :2]
            color = self.colors[i + 4] if i + 4 < len(self.colors) else self.colors[i % len(self.colors)]
            
            # 创建轨迹的渐变效果
            segments = []
            colors_gradient = []
            
            for j in range(len(positions) - 1):
                segment = [positions[j], positions[j + 1]]
                segments.append(segment)
                # 轨迹后半部分颜色更深
                alpha = 0.4 + 0.6 * (j / (len(positions) - 1))
                colors_gradient.append((*plt.colors.hex2color(color), alpha))
            
            # 绘制渐变轨迹
            if segments:
                lc = LineCollection(segments, colors=colors_gradient, linewidths=5, zorder=5)
                ax.add_collection(lc)
            
            # 起点标记 - 更大更明显
            ax.scatter(positions[0, 0], positions[0, 1], 
                      c=color, s=300, marker='o', 
                      edgecolors='white', linewidth=4, 
                      alpha=1.0, zorder=8, 
                      label=f'智能体 {agent_id}')
            
            # 终点标记 - 正方形表示终点
            ax.scatter(positions[-1, 0], positions[-1, 1], 
                      c=color, s=300, marker='s', 
                      edgecolors='white', linewidth=4, 
                      alpha=1.0, zorder=8)
            
            # 智能体标签
            ax.annotate(f'{agent_id}\n起点', (positions[0, 0], positions[0, 1]), 
                       xytext=(-10, -25), textcoords='offset points', 
                       fontsize=12, fontweight='bold', ha='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, 
                               alpha=0.7, edgecolor='white'))
            
            ax.annotate(f'{agent_id}\n终点', (positions[-1, 0], positions[-1, 1]), 
                       xytext=(10, 10), textcoords='offset points', 
                       fontsize=12, fontweight='bold', ha='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, 
                               alpha=0.7, edgecolor='white'))
            
            # 方向箭头 - 在轨迹的几个关键点添加箭头
            arrow_positions = [len(positions) // 4, len(positions) // 2, 3 * len(positions) // 4]
            for pos_idx in arrow_positions:
                if pos_idx < len(positions) - 1:
                    start_pos = positions[pos_idx]
                    end_pos = positions[pos_idx + 1]
                    dx = end_pos[0] - start_pos[0]
                    dy = end_pos[1] - start_pos[1]
                    
                    if np.sqrt(dx**2 + dy**2) > 10:  # 只在有明显移动时添加箭头
                        ax.arrow(start_pos[0], start_pos[1], dx * 0.5, dy * 0.5, 
                                head_width=80, head_length=120, 
                                fc=color, ec=color, alpha=0.8, zorder=7,
                                linewidth=2)
        
        # 设置坐标轴和样式
        ax.set_xlim(-3000, 3000)
        ax.set_ylim(-3000, 3000)
        ax.set_aspect('equal')
        
        # 网格设置
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)
        ax.set_axisbelow(True)
        
        # 坐标轴标签
        ax.set_xlabel('X坐标 (m)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Y坐标 (m)', fontsize=20, fontweight='bold')
        
        # 标题
        config_name = selected_config.get('config_name', '示例配置')
        ax.set_title(f'智能体轨迹与任务执行可视化\n配置: {config_name.replace("_", " ")}', 
                    fontsize=22, fontweight='bold', pad=25)
        
        # 图例 - 分成两列显示
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1), 
                 fontsize=14, frameon=True, fancybox=True, shadow=True,
                 ncol=1 if len(handles) <= 8 else 2)
        
        # 添加比例尺
        scale_length = 1000  # 1km
        scale_x = ax.get_xlim()[1] - 500
        scale_y = ax.get_ylim()[0] + 300
        
        ax.plot([scale_x - scale_length/2, scale_x + scale_length/2], 
               [scale_y, scale_y], 'k-', linewidth=5, alpha=0.8)
        ax.text(scale_x, scale_y + 150, '1 km', ha='center', va='bottom', 
               fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        self.save_figure(fig, '02_agent_trajectories')
    
    def plot_key_moments_trajectories(self):
        """绘制关键时刻的轨迹图"""
        selected_config = self._select_representative_config()
        if not selected_config:
            return
        
        env_data = selected_config['env_data']
        agents_data = env_data.get('agents', {})
        tasks = env_data.get('tasks', {})
        final_time = env_data.get('final_time', 1000)
        
        # 定义关键时刻
        key_moments = [0.0, final_time * 0.2, final_time * 0.5, final_time * 0.8, final_time]
        moment_names = ['初始时刻', '20%进度', '50%进度', '80%进度', '最终时刻']
        
        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('关键时刻智能体位置与任务分配演化分析', fontsize=28, fontweight='bold', y=0.95)
        
        for moment_idx, (time_point, moment_name) in enumerate(zip(key_moments, moment_names)):
            if moment_idx >= 5:
                break
            
            # 创建子图
            if moment_idx < 3:
                ax = fig.add_subplot(gs[0, moment_idx])
            else:
                ax = fig.add_subplot(gs[1, moment_idx - 3])
            
            # 绘制任务 - 根据时刻显示不同状态
            task_type_colors = {
                '高价值科研': self.colors[0], 
                '普通巡查': self.colors[1], 
                '紧急维修': self.colors[2]
            }
            
            for task_id, task_info in tasks.items():
                if 'position' in task_info:
                    pos = np.array(task_info['position'][:2])
                    task_type = task_info.get('true_type_name', '未知')
                    color = task_type_colors.get(task_type, self.colors[3])
                    completion = task_info.get('current_completion_percentage', 0)
                    
                    # 估算该时刻的完成度（简化模型）
                    estimated_completion = min(100, completion * (time_point / final_time))
                    
                    # 根据估算完成度调整显示
                    if estimated_completion >= 100:
                        marker = 'X'
                        alpha = 0.6
                        size = 400
                    elif estimated_completion > 50:
                        marker = 'D'
                        alpha = 0.8
                        size = 350
                    else:
                        marker = '^'
                        alpha = 1.0
                        size = 300
                    
                    ax.scatter(pos[0], pos[1], c=color, s=size, marker=marker, 
                              edgecolors='black', linewidth=2, alpha=alpha, zorder=10)
                    
                    # 任务标签
                    ax.annotate(f'{task_id}\n{estimated_completion:.0f}%', 
                               (pos[0], pos[1]), xytext=(8, 8), 
                               textcoords='offset points', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            
            # 绘制智能体位置和轨迹
            for i, (agent_id, agent_info) in enumerate(agents_data.items()):
                history_states = agent_info.get('history_states', [])
                if not history_states:
                    continue
                
                states = np.array(history_states)
                if states.ndim != 2 or states.shape[1] < 4:
                    continue
                
                # 计算对应时间点的状态索引
                dt = 2.0  # 假设时间步长
                time_idx = min(int(time_point / dt), len(states) - 1)
                
                pos = states[time_idx, :2]
                vel = states[time_idx, 2:4] if states.shape[1] >= 4 else np.array([0, 0])
                color = self.colors[i + 4] if i + 4 < len(self.colors) else self.colors[i % len(self.colors)]
                
                # 智能体当前位置
                ax.scatter(pos[0], pos[1], c=color, s=250, marker='o', 
                          edgecolors='black', linewidth=2, alpha=0.9, zorder=8)
                
                # 智能体标签
                ax.annotate(agent_id, (pos[0], pos[1]), xytext=(-8, -20), 
                           textcoords='offset points', fontsize=11, fontweight='bold',
                           color=color)
                
                # 速度矢量 - 显示当前运动方向
                speed = np.linalg.norm(vel)
                if speed > 0.005:  # 只显示有意义的速度
                    scale = 200  # 速度缩放因子
                    ax.arrow(pos[0], pos[1], vel[0] * scale, vel[1] * scale,
                            head_width=100, head_length=150, fc=color, ec=color, 
                            alpha=0.7, zorder=7, linewidth=2)
                
                # 绘制到当前时刻的轨迹（渐变透明度）
                if time_idx > 0:
                    trajectory = states[:time_idx+1, :2]
                    
                    # 创建渐变轨迹
                    for j in range(len(trajectory) - 1):
                        alpha = 0.1 + 0.4 * (j / len(trajectory))
                        ax.plot([trajectory[j, 0], trajectory[j + 1, 0]], 
                               [trajectory[j, 1], trajectory[j + 1, 1]], 
                               color=color, linewidth=2, alpha=alpha, zorder=5)
            
            # 设置子图属性
            ax.set_xlim(-2800, 2800)
            ax.set_ylim(-2800, 2800)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 标题和标签
            ax.set_title(f'{moment_name}\n(t = {time_point:.0f}s)', 
                        fontsize=16, fontweight='bold', pad=15)
            
            if moment_idx >= 3:  # 底部行
                ax.set_xlabel('X坐标 (m)', fontsize=14, fontweight='bold')
            if moment_idx % 3 == 0:  # 左侧列
                ax.set_ylabel('Y坐标 (m)', fontsize=14, fontweight='bold')
            
            # 添加时刻信息框
            progress = time_point / final_time * 100
            info_text = f'仿真进度: {progress:.0f}%'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=12, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        self.save_figure(fig, '03_key_moments_trajectories')
    
    def plot_coalition_evolution(self):
        """绘制联盟演化图"""
        selected_config = self._select_representative_config()
        if not selected_config:
            return
        
        env_data = selected_config['env_data']
        assignment_history = env_data.get('history_assignments_ksc', [])
        
        if not assignment_history:
            print("未找到KSC分配历史数据")
            return
        
        # 准备数据
        agents = list(assignment_history[0].keys()) if assignment_history else []
        all_tasks = set()
        for assignment in assignment_history:
            all_tasks.update(assignment.values())
        
        # 移除IDLE任务标识
        IDLE_TASK = "IDLE_TASK_PLACEHOLDER"
        if IDLE_TASK in all_tasks:
            all_tasks.remove(IDLE_TASK)
        
        tasks = sorted(list(all_tasks))
        
        if not agents or not tasks:
            print("未找到足够的智能体或任务数据")
            return
        
        # 创建分配矩阵
        assignment_matrix = np.zeros((len(agents), len(assignment_history)))
        task_to_idx = {task: idx + 1 for idx, task in enumerate(tasks)}
        task_to_idx[IDLE_TASK] = 0
        
        for round_idx, assignment in enumerate(assignment_history):
            for agent_idx, agent_id in enumerate(agents):
                task_assigned = assignment.get(agent_id, IDLE_TASK)
                assignment_matrix[agent_idx, round_idx] = task_to_idx.get(task_assigned, 0)
        
        # 创建可视化
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[2, 2, 1], hspace=0.4, wspace=0.3)
        
        fig.suptitle('KSC联盟演化与任务分配动态分析', fontsize=24, fontweight='bold', y=0.95)
        
        # 子图1：分配矩阵热图
        ax1 = fig.add_subplot(gs[0, :])
        
        # 创建自定义颜色映射
        colors_for_heatmap = ['lightgray'] + self.colors[:len(tasks)]
        cmap = ListedColormap(colors_for_heatmap)
        
        im = ax1.imshow(assignment_matrix, cmap=cmap, aspect='auto', alpha=0.9, 
                       vmin=0, vmax=len(tasks))
        
        # 设置刻度
        ax1.set_xticks(range(0, len(assignment_history), max(1, len(assignment_history) // 15)))
        ax1.set_xticklabels([f'{i+1}' for i in range(0, len(assignment_history), 
                                                    max(1, len(assignment_history) // 15))])
        ax1.set_yticks(range(len(agents)))
        ax1.set_yticklabels(agents, fontsize=14)
        
        ax1.set_xlabel('KSC决策轮次', fontsize=16, fontweight='bold')
        ax1.set_ylabel('智能体', fontsize=16, fontweight='bold')
        ax1.set_title('任务分配矩阵演化热图', fontsize=18, fontweight='bold', pad=20)
        
        # 添加细网格
        ax1.set_xticks(np.arange(-0.5, len(assignment_history), 1), minor=True)
        ax1.set_yticks(np.arange(-0.5, len(agents), 1), minor=True)
        ax1.grid(which="minor", color="white", linestyle='-', linewidth=1.5)
        ax1.tick_params(which="minor", size=0)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax1, pad=0.02, aspect=40, shrink=0.8)
        cbar_labels = ['空闲'] + [f'任务{task}' for task in tasks]
        cbar.set_ticks(range(len(cbar_labels)))
        cbar.set_ticklabels(cbar_labels, fontsize=12)
        cbar.set_label('分配任务', fontsize=14, fontweight='bold')
        
        # 子图2：任务分配智能体数量统计
        ax2 = fig.add_subplot(gs[1, 0])
        
        task_assignments = {task: [] for task in tasks}
        
        for assignment in assignment_history:
            task_counts = {task: 0 for task in tasks}
            for agent_id, task_assigned in assignment.items():
                if task_assigned in task_counts:
                    task_counts[task_assigned] += 1
            
            for task in tasks:
                task_assignments[task].append(task_counts[task])
        
        rounds = list(range(1, len(assignment_history) + 1))
        
        for i, task in enumerate(tasks):
            ax2.plot(rounds, task_assignments[task], 
                    color=self.colors[i], linewidth=4, 
                    marker=self.markers[i % len(self.markers)], 
                    markersize=10, label=f'任务 {task}',
                    linestyle=self.line_styles[i % len(self.line_styles)],
                    alpha=0.8)
        
        ax2.set_xlabel('KSC决策轮次', fontsize=16, fontweight='bold')
        ax2.set_ylabel('分配的智能体数量', fontsize=16, fontweight='bold')
        ax2.set_title('各任务分配智能体数量变化', fontsize=18, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True)
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 子图3：分配变化频率分析
        ax3 = fig.add_subplot(gs[1, 1])
        
        # 计算每个智能体的任务变更次数
        change_counts = {agent: 0 for agent in agents}
        for agent_idx, agent_id in enumerate(agents):
            prev_task = None
            for round_idx in range(len(assignment_history)):
                current_task = assignment_history[round_idx].get(agent_id, IDLE_TASK)
                if prev_task is not None and current_task != prev_task:
                    change_counts[agent_id] += 1
                prev_task = current_task
        
        agents_sorted = sorted(agents)
        changes = [change_counts[agent] for agent in agents_sorted]
        colors_bar = [self.colors[i % len(self.colors)] for i in range(len(agents_sorted))]
        
        bars = ax3.bar(range(len(agents_sorted)), changes, 
                      color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
        
        ax3.set_xticks(range(len(agents_sorted)))
        ax3.set_xticklabels(agents_sorted, fontsize=14)
        ax3.set_xlabel('智能体', fontsize=16, fontweight='bold')
        ax3.set_ylabel('任务变更次数', fontsize=16, fontweight='bold')
        ax3.set_title('智能体任务分配变更频率', fontsize=18, fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.4, axis='y', linestyle='--')
        ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 在柱状图上添加数值标签
        for bar, count in zip(bars, changes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 子图4：联盟稳定性分析
        ax4 = fig.add_subplot(gs[2, :])
        
        # 计算每轮的联盟稳定性（与前一轮相同分配的比例）
        stability_scores = []
        for round_idx in range(1, len(assignment_history)):
            prev_assignment = assignment_history[round_idx - 1]
            current_assignment = assignment_history[round_idx]
            
            same_assignments = sum(1 for agent in agents 
                                 if prev_assignment.get(agent) == current_assignment.get(agent))
            stability = same_assignments / len(agents) if agents else 0
            stability_scores.append(stability)
        
        if stability_scores:
            stability_rounds = list(range(2, len(assignment_history) + 1))
            ax4.plot(stability_rounds, stability_scores, 
                    color=self.colors[0], linewidth=4, marker='o', markersize=8,
                    alpha=0.8, label='联盟稳定性')
            
            # 添加移动平均线
            if len(stability_scores) > 5:
                window_size = min(5, len(stability_scores) // 3)
                moving_avg = pd.Series(stability_scores).rolling(window=window_size).mean()
                ax4.plot(stability_rounds, moving_avg, 
                        color=self.colors[1], linewidth=3, linestyle='--',
                        alpha=0.7, label=f'{window_size}轮移动平均')
            
            ax4.set_xlabel('KSC决策轮次', fontsize=16, fontweight='bold')
            ax4.set_ylabel('稳定性得分', fontsize=16, fontweight='bold')
            ax4.set_title('联盟分配稳定性演化', fontsize=18, fontweight='bold', pad=15)
            ax4.set_ylim(0, 1.05)
            ax4.grid(True, alpha=0.4, linestyle='--')
            ax4.legend(loc='upper right', fontsize=12)
            
            # 添加稳定性阈值线
            ax4.axhline(y=0.8, color='red', linestyle=':', linewidth=2, 
                       alpha=0.7, label='高稳定性阈值')
            ax4.axhline(y=0.5, color='orange', linestyle=':', linewidth=2, 
                       alpha=0.7, label='中等稳定性阈值')
        
        plt.tight_layout()
        self.save_figure(fig, '04_coalition_evolution')
    
    def plot_task_gantt_chart(self):
        """绘制任务甘特图"""
        selected_config = self._select_representative_config()
        if not selected_config:
            return
        
        env_data = selected_config['env_data']
        tasks = env_data.get('tasks', {})
        assignment_history = env_data.get('history_assignments_ksc', [])
        
        if not tasks:
            print("未找到任务数据")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14))
        fig.suptitle('任务执行甘特图与进度分析', fontsize=24, fontweight='bold', y=0.95)
        
        # 准备数据
        task_ids = sorted(tasks.keys())
        y_positions = {task_id: i for i, task_id in enumerate(task_ids)}
        
        # 任务类型颜色
        type_colors = {
            '高价值科研': self.colors[0], 
            '普通巡查': self.colors[1], 
            '紧急维修': self.colors[2]
        }
        
        max_time = env_data.get('final_time', 1000)
        
        # 子图1：甘特图主体
        for task_id, task_info in tasks.items():
            y_pos = y_positions[task_id]
            task_type = task_info.get('true_type_name', '未知')
            color = type_colors.get(task_type, self.colors[3])
            
            # 任务状态信息
            completion_percentage = task_info.get('current_completion_percentage', 0)
            status = task_info.get('status', 'active')
            completion_time = task_info.get('completion_time', -1)
            
            # 计算任务时间范围
            start_time = 0
            if completion_time > 0:
                end_time = completion_time
            else:
                end_time = max_time
            
            # 绘制任务条 - 分为已完成和未完成部分
            total_width = end_time - start_time
            completed_width = (completion_percentage / 100.0) * total_width
            
            # 已完成部分
            if completed_width > 0:
                ax1.barh(y_pos, completed_width, left=start_time, height=0.7, 
                        color=color, alpha=0.9, edgecolor='black', linewidth=2,
                        label=f'{task_type}任务' if task_type not in [l.get_text().split('任务')[0] + '任务' for l in ax1.get_legend_handles_labels()[1]] else "")
            
            # 未完成部分
            remaining_width = total_width - completed_width
            if remaining_width > 0:
                ax1.barh(y_pos, remaining_width, left=start_time + completed_width, 
                        height=0.7, color=color, alpha=0.4, 
                        edgecolor='black', linewidth=2, hatch='///')
            
            # 添加完成百分比标签
            label_x = start_time + total_width / 2
            ax1.text(label_x, y_pos, f'{completion_percentage:.0f}%', 
                    ha='center', va='center', fontweight='bold', fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                            edgecolor=color, alpha=0.9, linewidth=2))
            
            # 任务信息标签（左侧）
            task_label = f'{task_id}\n({task_type})'
            ax1.text(-max_time * 0.08, y_pos, task_label, 
                    ha='right', va='center', fontweight='bold', fontsize=13,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, 
                            alpha=0.7, edgecolor='black'))
        
        # 设置坐标轴
        ax1.set_xlim(-max_time * 0.15, max_time * 1.05)
        ax1.set_ylim(-0.5, len(task_ids) - 0.5)
        ax1.set_xlabel('仿真时间 (s)', fontsize=18, fontweight='bold')
        ax1.set_ylabel('任务', fontsize=18, fontweight='bold')
        ax1.set_title('任务执行甘特图', fontsize=20, fontweight='bold', pad=20)
        
        # 移除y轴刻度标签（已经在左侧显示）
        ax1.set_yticks(range(len(task_ids)))
        ax1.set_yticklabels([])
        
        # 网格和样式
        ax1.grid(True, alpha=0.4, axis='x', linestyle='--')
        ax1.set_axisbelow(True)
        
        # 添加时间节点标记
        time_markers = [0, max_time * 0.25, max_time * 0.5, max_time * 0.75, max_time]
        for tm in time_markers:
            ax1.axvline(x=tm, color='gray', linestyle=':', alpha=0.6, linewidth=2)
        
        # 图例
        legend_elements = [
            mpatches.Patch(color=self.colors[0], alpha=0.9, label='高价值科研'),
            mpatches.Patch(color=self.colors[1], alpha=0.9, label='普通巡查'),
            mpatches.Patch(color=self.colors[2], alpha=0.9, label='紧急维修'),
            mpatches.Patch(color='gray', alpha=0.9, label='已完成'),
            mpatches.Patch(color='gray', alpha=0.4, hatch='///', label='未完成')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=14,
                  frameon=True, fancybox=True, shadow=True)
        
        # 子图2：任务完成度时间序列
        if assignment_history:
            dt_ksc = 4.0  # KSC决策间隔
            ksc_times = [i * dt_ksc for i in range(len(assignment_history))]
            
            for task_id, task_info in tasks.items():
                task_type = task_info.get('true_type_name', '未知')
                color = type_colors.get(task_type, self.colors[3])
                
                # 模拟任务完成度随时间的变化（简化模型）
                final_completion = task_info.get('current_completion_percentage', 0)
                completion_curve = []
                
                for t in ksc_times:
                    # 假设任务完成度按S曲线增长
                    if final_completion > 0:
                        progress = min(100, final_completion * (1 - np.exp(-3 * t / max_time)))
                    else:
                        progress = 0
                    completion_curve.append(progress)
                
                ax2.plot(ksc_times, completion_curve, 
                        color=color, linewidth=4, 
                        marker=self.markers[y_positions[task_id] % len(self.markers)], 
                        markersize=8, label=f'任务 {task_id}',
                        alpha=0.8, linestyle=self.line_styles[y_positions[task_id] % len(self.line_styles)])
            
            ax2.set_xlabel('仿真时间 (s)', fontsize=18, fontweight='bold')
            ax2.set_ylabel('完成度 (%)', fontsize=18, fontweight='bold')
            ax2.set_title('任务完成度时间演化', fontsize=20, fontweight='bold', pad=20)
            ax2.set_ylim(0, 105)
            ax2.grid(True, alpha=0.4, linestyle='--')
            ax2.legend(loc='center right', bbox_to_anchor=(1.15, 0.5), 
                      fontsize=12, frameon=True, fancybox=True)
            
            # 添加完成度阈值线
            for threshold in [25, 50, 75, 100]:
                ax2.axhline(y=threshold, color='gray', linestyle=':', 
                           alpha=0.5, linewidth=1)
                ax2.text(max_time * 1.02, threshold, f'{threshold}%', 
                        va='center', fontsize=11, alpha=0.7)
        
        plt.tight_layout()
        self.save_figure(fig, '05_task_gantt_chart')
    
    def plot_agent_execution_timeline(self):
        """绘制智能体执行时序图"""
        selected_config = self._select_representative_config()
        if not selected_config:
            return
        
        env_data = selected_config['env_data']
        agents_data = env_data.get('agents', {})
        assignment_history = env_data.get('history_assignments_ksc', [])
        
        if not agents_data or not assignment_history:
            print("未找到智能体数据或分配历史")
            return
        
        fig, axes = plt.subplots(len(agents_data), 1, figsize=(20, 4 * len(agents_data)))
        if len(agents_data) == 1:
            axes = [axes]
        
        fig.suptitle('智能体执行时序图与状态分析', fontsize=24, fontweight='bold', y=0.95)
        
        # 任务颜色映射
        all_tasks = set()
        for assignment in assignment_history:
            all_tasks.update(assignment.values())
        
        IDLE_TASK = "IDLE_TASK_PLACEHOLDER"
        all_tasks.discard(IDLE_TASK)
        
        # 扩展颜色映射
        task_colors = {IDLE_TASK: 'lightgray'}
        task_types = {}
        
        # 从任务信息中获取类型
        tasks_info = env_data.get('tasks', {})
        type_color_map = {
            '高价值科研': self.colors[0],
            '普通巡查': self.colors[1], 
            '紧急维修': self.colors[2]
        }
        
        for i, task in enumerate(sorted(all_tasks)):
            if task in tasks_info:
                task_type = tasks_info[task].get('true_type_name', '未知')
                task_colors[task] = type_color_map.get(task_type, self.colors[(i + 3) % len(self.colors)])
                task_types[task] = task_type
            else:
                task_colors[task] = self.colors[i % len(self.colors)]
                task_types[task] = '未知'
        
        # 计算时间轴
        dt_ksc = 4.0  # KSC决策间隔
        times = [i * dt_ksc for i in range(len(assignment_history))]
        
        for agent_idx, (agent_id, agent_info) in enumerate(agents_data.items()):
            ax = axes[agent_idx]
            
            # 提取该智能体的分配历史
            agent_assignments = [assignment.get(agent_id, IDLE_TASK) for assignment in assignment_history]
            
            # 获取决策模式历史（如果有）
            decision_modes = agent_info.get('history_decision_mode', [])
            
            # 绘制任务分配时序
            prev_task = None
            start_time = 0
            
            for i, current_task in enumerate(agent_assignments):
                current_time = times[i]
                
                if current_task != prev_task:
                    # 结束前一个任务段
                    if prev_task is not None:
                        color = task_colors.get(prev_task, 'gray')
                        task_type = task_types.get(prev_task, '未知')
                        label = f'{prev_task} ({task_type})' if prev_task != IDLE_TASK else '空闲'
                        
                        # 任务执行条
                        rect = ax.barh(0, current_time - start_time, left=start_time, height=0.6,
                                     color=color, alpha=0.8, edgecolor='black', linewidth=2)
                        
                        # 添加标签（在足够宽的段上）
                        segment_width = current_time - start_time
                        if segment_width > max(times) * 0.05:  # 只在足够宽的段上添加标签
                            label_text = prev_task if prev_task != IDLE_TASK else '空闲'
                            ax.text(start_time + segment_width / 2, 0, 
                                   label_text, ha='center', va='center', 
                                   fontweight='bold', fontsize=12,
                                   color='white' if prev_task != IDLE_TASK else 'black')
                    
                    start_time = current_time
                    prev_task = current_task
            
            # 处理最后一个任务段
            if prev_task is not None and times:
                final_time = times[-1] + dt_ksc
                color = task_colors.get(prev_task, 'gray')
                task_type = task_types.get(prev_task, '未知')
                
                rect = ax.barh(0, final_time - start_time, left=start_time, height=0.6,
                             color=color, alpha=0.8, edgecolor='black', linewidth=2)
                
                segment_width = final_time - start_time
                if segment_width > max(times) * 0.05:
                    label_text = prev_task if prev_task != IDLE_TASK else '空闲'
                    ax.text(start_time + segment_width / 2, 0, 
                           label_text, ha='center', va='center', 
                           fontweight='bold', fontsize=12,
                           color='white' if prev_task != IDLE_TASK else 'black')
            
            # 绘制决策模式变化（作为背景）
            if decision_modes:
                mode_dt = 2.0  # 假设AIF决策间隔
                mode_times = [i * mode_dt for i in range(len(decision_modes))]
                mode_colors = {'strong': 'blue', 'weak': 'red', 'transition': 'orange'}
                
                # 在时间轴上方绘制决策模式指示器
                for i, mode in enumerate(decision_modes):
                    if i < len(mode_times):
                        color = mode_colors.get(mode, 'gray')
                        ax.axvline(x=mode_times[i], ymin=0.8, ymax=1.0, 
                                  color=color, alpha=0.6, linewidth=3)
            
            # 添加任务切换点标记
            task_switch_times = []
            for i in range(1, len(agent_assignments)):
                if agent_assignments[i] != agent_assignments[i-1]:
                    task_switch_times.append(times[i])
            
            for switch_time in task_switch_times:
                ax.axvline(x=switch_time, color='red', linestyle='--', 
                          alpha=0.7, linewidth=2, zorder=10)
                ax.text(switch_time, 0.4, '切换', rotation=90, 
                       ha='right', va='bottom', fontsize=10, 
                       fontweight='bold', color='red')
            
            # 设置子图属性
            ax.set_xlim(0, times[-1] + dt_ksc if times else 100)
            ax.set_ylim(-0.4, 0.8)
            
            # 智能体标签
            ax.set_ylabel(f'{agent_id}', fontsize=16, fontweight='bold', 
                         rotation=0, ha='right', va='center')
            ax.set_yticks([])
            
            # 网格
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            ax.set_axisbelow(True)
            
            # 只在最后一个子图添加x轴标签
            if agent_idx == len(agents_data) - 1:
                ax.set_xlabel('仿真时间 (s)', fontsize=18, fontweight='bold')
            
            # 添加智能体状态信息
            history_states = agent_info.get('history_states', [])
            if history_states:
                total_distance = 0
                states = np.array(history_states)
                if states.ndim == 2 and states.shape[0] > 1:
                    distances = np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)
                    total_distance = np.sum(distances)
                
                info_text = f'总移动距离: {total_distance:.0f}m\n任务切换: {len(task_switch_times)}次'
                ax.text(0.98, 0.98, info_text, transform=ax.transAxes, 
                       fontsize=11, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', 
                               alpha=0.8, edgecolor='gray'))
        
        # 添加全局图例
        legend_elements = []
        for task, color in task_colors.items():
            if task == IDLE_TASK:
                label = '空闲状态'
            else:
                task_type = task_types.get(task, '未知')
                label = f'{task} ({task_type})'
            legend_elements.append(mpatches.Patch(color=color, alpha=0.8, label=label))
        
        # 添加决策模式图例
        if any(agent_info.get('history_decision_mode') for agent_info in agents_data.values()):
            legend_elements.extend([
                Line2D([0], [0], color='blue', lw=3, alpha=0.6, label='强通信模式'),
                Line2D([0], [0], color='orange', lw=3, alpha=0.6, label='过渡模式'),
                Line2D([0], [0], color='red', lw=3, alpha=0.6, label='弱通信模式')
            ])
        
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95), 
                  fontsize=12, ncol=min(len(legend_elements), 4),
                  frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        self.save_figure(fig, '06_agent_execution_timeline')
    
    def plot_belief_convergence(self):
        """绘制信念收敛图"""
        selected_config = self._select_representative_config()
        if not selected_config:
            return
        
        env_data = selected_config['env_data']
        belief_history = env_data.get('history_beliefs_at_ksc_step', {})
        tasks = env_data.get('tasks', {})
        
        if not belief_history or not tasks:
            print("未找到信念历史或任务数据")
            return
        
        # 选择前4个任务和前3个智能体进行可视化
        task_ids = sorted(list(tasks.keys()))[:4]
        
        # 获取所有智能体ID
        all_agents = set()
        for round_data in belief_history.values():
            all_agents.update(round_data.keys())
        agent_ids = sorted(list(all_agents))[:4]  # 增加到4个智能体
        
        if not task_ids or not agent_ids:
            print("没有足够的任务或智能体数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('信念收敛过程分析与置信度评估', fontsize=24, fontweight='bold', y=0.95)
        
        rounds = sorted(belief_history.keys())
        
        for task_idx, task_id in enumerate(task_ids):
            row, col = task_idx // 2, task_idx % 2
            ax = axes[row, col]
            
            task_info = tasks[task_id]
            true_type_idx = task_info.get('true_type_idx', -1)
            true_type_name = task_info.get('true_type_name', '未知')
            
            # 为每个智能体绘制信念演化
            for agent_idx, agent_id in enumerate(agent_ids):
                belief_values = []
                confidence_intervals = []
                valid_rounds = []
                
                for round_num in rounds:
                    round_data = belief_history.get(round_num, {})
                    agent_data = round_data.get(agent_id, {})
                    task_belief = agent_data.get(task_id)
                    
                    if (task_belief is not None and 
                        len(task_belief) > true_type_idx and 
                        true_type_idx >= 0):
                        belief_values.append(task_belief[true_type_idx])
                        valid_rounds.append(round_num)
                        
                        # 计算置信度（基于信念分布的熵）
                        entropy = -np.sum(task_belief * np.log(task_belief + 1e-10))
                        confidence = 1.0 - entropy / np.log(len(task_belief))
                        confidence_intervals.append(confidence * 0.1)  # 缩放置信区间
                
                if belief_values:
                    color = self.colors[agent_idx]
                    marker = self.markers[agent_idx]
                    linestyle = self.line_styles[agent_idx]
                    
                    # 主线条 - 信念值
                    ax.plot(valid_rounds, belief_values, 
                           color=color, linewidth=4, marker=marker, 
                           markersize=10, linestyle=linestyle,
                           label=f'智能体 {agent_id}', alpha=0.9)
                    
                    # 置信区间
                    if len(belief_values) > 2:
                        belief_array = np.array(belief_values)
                        confidence_array = np.array(confidence_intervals)
                        
                        upper_bound = np.minimum(belief_array + confidence_array, 1.0)
                        lower_bound = np.maximum(belief_array - confidence_array, 0.0)
                        
                        ax.fill_between(valid_rounds, lower_bound, upper_bound,
                                       color=color, alpha=0.25, zorder=1)
                    
                    # 添加收敛指标文本
                    if len(belief_values) > 5:
                        # 计算收敛速度（最后几个点的方差）
                        recent_values = belief_values[-5:]
                        convergence_score = 1.0 - np.std(recent_values)
                        final_accuracy = belief_values[-1]
                        
                        # 在图上添加收敛信息
                        ax.text(0.02, 0.98 - agent_idx * 0.1, 
                               f'{agent_id}: 收敛度={convergence_score:.2f}, 准确度={final_accuracy:.2f}', 
                               transform=ax.transAxes, fontsize=10,
                               verticalalignment='top', horizontalalignment='left',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, 
                                       alpha=0.3, edgecolor=color))
            
            # 添加真实类型的参考线和区域
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=3, 
                      alpha=0.8, label='理想收敛值 (100%)')
            
            # 添加收敛阈值区域
            ax.axhspan(0.8, 1.0, color='green', alpha=0.1, zorder=0, 
                      label='高置信度区域' if task_idx == 0 else "")
            ax.axhspan(0.6, 0.8, color='yellow', alpha=0.1, zorder=0,
                      label='中等置信度区域' if task_idx == 0 else "")
            
            # 设置子图属性
            ax.set_xlim(rounds[0] - 0.5, rounds[-1] + 0.5)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.4, linestyle='--')
            ax.set_xlabel('KSC决策轮次', fontsize=16, fontweight='bold')
            ax.set_ylabel('对真实类型的信念概率', fontsize=16, fontweight='bold')
            ax.set_title(f'任务 {task_id} 信念收敛分析\n(真实类型: {true_type_name})', 
                        fontsize=16, fontweight='bold', pad=15)
            
            # 图例（只在第一个子图显示）
            if task_idx == 0:
                ax.legend(loc='lower right', fontsize=12, frameon=True, 
                         fancybox=True, shadow=True)
            
            # 添加统计信息
            if belief_history:
                total_rounds = len(rounds)
                ax.text(0.98, 0.02, f'总轮次: {total_rounds}', 
                       transform=ax.transAxes, fontsize=11,
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', 
                               alpha=0.8))
        
        plt.tight_layout()
        self.save_figure(fig, '07_belief_convergence')
    
    def plot_acceleration_curves(self):
        """绘制加速度曲线图"""
        selected_config = self._select_representative_config()
        if not selected_config:
            return
        
        env_data = selected_config['env_data']
        agents_data = env_data.get('agents', {})
        
        if not agents_data:
            print("未找到智能体数据")
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[2, 2, 1], hspace=0.4, wspace=0.3)
        fig.suptitle('智能体加速度分析与控制性能评估', fontsize=24, fontweight='bold', y=0.95)
        
        # 计算加速度统计
        agent_accelerations = {}
        agent_controls = {}
        max_time_points = 0
        
        for agent_idx, (agent_id, agent_info) in enumerate(agents_data.items()):
            history_controls = agent_info.get('history_controls', [])
            if not history_controls:
                continue
            
            controls = np.array(history_controls)
            if controls.ndim != 2:
                continue
            
            # 计算加速度大小
            accelerations = np.linalg.norm(controls, axis=1)
            agent_accelerations[agent_id] = accelerations
            agent_controls[agent_id] = controls
            max_time_points = max(max_time_points, len(accelerations))
        
        if not agent_accelerations:
            print("未找到有效的控制数据")
            return
        
        dt = 2.0  # 控制时间步长
        
        # 子图1：加速度时间序列（上左）
        ax1 = fig.add_subplot(gs[0, 0])
        
        for agent_idx, (agent_id, accelerations) in enumerate(agent_accelerations.items()):
            time_points = np.arange(len(accelerations)) * dt
            color = self.colors[agent_idx]
            
            # 主曲线
            ax1.plot(time_points, accelerations, 
                    color=color, linewidth=4, alpha=0.8,
                    linestyle=self.line_styles[agent_idx % len(self.line_styles)],
                    label=f'智能体 {agent_id}')
            
            # 移动平均线
            if len(accelerations) > 10:
                window_size = min(15, len(accelerations) // 5)
                moving_avg = pd.Series(accelerations).rolling(window=window_size, center=True).mean()
                ax1.plot(time_points, moving_avg, 
                        color=color, linewidth=2, linestyle='-', alpha=0.6)
            
            # 标记峰值点
            if len(accelerations) > 5:
                peak_indices = []
                for i in range(1, len(accelerations) - 1):
                    if (accelerations[i] > accelerations[i-1] and 
                        accelerations[i] > accelerations[i+1] and 
                        accelerations[i] > np.mean(accelerations) + np.std(accelerations)):
                        peak_indices.append(i)
                
                if peak_indices:
                    peak_times = [time_points[i] for i in peak_indices]
                    peak_values = [accelerations[i] for i in peak_indices]
                    ax1.scatter(peak_times, peak_values, color=color, s=100, 
                               marker='*', edgecolors='black', linewidth=1, zorder=10)
        
        ax1.set_xlabel('仿真时间 (s)', fontsize=16, fontweight='bold')
        ax1.set_ylabel('加速度大小 (m/s²)', fontsize=16, fontweight='bold')
        ax1.set_title('智能体加速度时间序列', fontsize=18, fontweight='bold')
        ax1.grid(True, alpha=0.4, linestyle='--')
        ax1.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True)
        
        # 子图2：加速度方向分析（上右）
        ax2 = fig.add_subplot(gs[0, 1])
        
        for agent_idx, (agent_id, controls) in enumerate(agent_controls.items()):
            if len(controls) == 0:
                continue
                
            color = self.colors[agent_idx]
            
            # 绘制加速度矢量的方向分布
            ax_components = controls[:, 0]
            ay_components = controls[:, 1]
            
            # 创建方向角序列
            angles = np.arctan2(ay_components, ax_components) * 180 / np.pi
            time_points = np.arange(len(angles)) * dt
            
            # 绘制方向角变化
            ax2.plot(time_points, angles, 
                    color=color, linewidth=3, alpha=0.8,
                    linestyle=self.line_styles[agent_idx % len(self.line_styles)],
                    label=f'智能体 {agent_id}')
        
        ax2.set_xlabel('仿真时间 (s)', fontsize=16, fontweight='bold')
        ax2.set_ylabel('加速度方向角 (度)', fontsize=16, fontweight='bold')
        ax2.set_title('智能体加速度方向变化', fontsize=18, fontweight='bold')
        ax2.set_ylim(-180, 180)
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.legend(loc='upper right', fontsize=12)
        
        # 添加方向参考线
        for angle in [-90, 0, 90]:
            ax2.axhline(y=angle, color='gray', linestyle=':', alpha=0.5)
            direction_label = {-90: '向下', 0: '向右', 90: '向上'}
            ax2.text(ax2.get_xlim()[1] * 0.98, angle, direction_label.get(angle, ''), 
                    ha='right', va='center', fontsize=10, alpha=0.7)
        
        # 子图3：加速度分布箱线图（中左）
        ax3 = fig.add_subplot(gs[1, 0])
        
        acceleration_data = []
        agent_labels = []
        colors_for_box = []
        
        for agent_idx, (agent_id, accelerations) in enumerate(agent_accelerations.items()):
            acceleration_data.append(accelerations)
            agent_labels.append(agent_id)
            colors_for_box.append(self.colors[agent_idx])
        
        # 箱线图
        box_plot = ax3.boxplot(acceleration_data, labels=agent_labels, 
                              patch_artist=True, notch=True, 
                              showmeans=True, meanline=True,
                              flierprops=dict(marker='o', markerfacecolor='red', 
                                           markersize=5, alpha=0.5))
        
        # 设置颜色
        for patch, color in zip(box_plot['boxes'], colors_for_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
        
        # 添加散点图显示数据分布
        for i, accelerations in enumerate(acceleration_data):
            # 随机抖动x坐标以避免重叠
            x_jitter = np.random.normal(i + 1, 0.04, len(accelerations))
            ax3.scatter(x_jitter, accelerations, 
                       color=colors_for_box[i], alpha=0.4, s=30, zorder=10)
        
        ax3.set_ylabel('加速度大小 (m/s²)', fontsize=16, fontweight='bold')
        ax3.set_xlabel('智能体', fontsize=16, fontweight='bold')
        ax3.set_title('智能体加速度分布统计', fontsize=18, fontweight='bold')
        ax3.grid(True, alpha=0.4, axis='y', linestyle='--')
        
        # 子图4：控制效率分析（中右）
        ax4 = fig.add_subplot(gs[1, 1])
        
        # 计算控制效率指标
        efficiency_metrics = {}
        for agent_id, controls in agent_controls.items():
            if len(controls) == 0:
                continue
            
            # 计算总控制努力
            total_effort = np.sum(np.linalg.norm(controls, axis=1))
            
            # 计算控制平滑度（相邻控制输入的变化）
            if len(controls) > 1:
                control_changes = np.linalg.norm(np.diff(controls, axis=0), axis=1)
                smoothness = 1.0 / (1.0 + np.mean(control_changes))
            else:
                smoothness = 1.0
            
            # 计算控制频率（非零控制的比例）
            non_zero_controls = np.sum(np.linalg.norm(controls, axis=1) > 0.001)
            frequency = non_zero_controls / len(controls) if len(controls) > 0 else 0
            
            efficiency_metrics[agent_id] = {
                'total_effort': total_effort,
                'smoothness': smoothness,
                'frequency': frequency
            }
        
        # 绘制效率指标雷达图样式的对比
        agents_list = list(efficiency_metrics.keys())
        metrics = ['total_effort', 'smoothness', 'frequency']
        metric_labels = ['总控制努力', '控制平滑度', '控制频率']
        
        x_pos = np.arange(len(metrics))
        width = 0.8 / len(agents_list)
        
        for i, agent_id in enumerate(agents_list):
            values = [efficiency_metrics[agent_id][metric] for metric in metrics]
            # 标准化值以便比较
            normalized_values = []
            for j, val in enumerate(values):
                if metrics[j] == 'total_effort':
                    # 对于总努力，值越小越好，需要反转
                    max_val = max(efficiency_metrics[aid]['total_effort'] for aid in agents_list)
                    normalized_values.append(1.0 - val / max_val if max_val > 0 else 1.0)
                else:
                    normalized_values.append(val)
            
            color = self.colors[i]
            ax4.bar(x_pos + i * width, normalized_values, width, 
                   label=f'智能体 {agent_id}', color=color, alpha=0.8,
                   edgecolor='black', linewidth=1)
        
        ax4.set_xlabel('控制性能指标', fontsize=16, fontweight='bold')
        ax4.set_ylabel('标准化得分', fontsize=16, fontweight='bold')
        ax4.set_title('控制性能对比分析', fontsize=18, fontweight='bold')
        ax4.set_xticks(x_pos + width * (len(agents_list) - 1) / 2)
        ax4.set_xticklabels(metric_labels, fontsize=12)
        ax4.set_ylim(0, 1.05)
        ax4.grid(True, alpha=0.4, axis='y', linestyle='--')
        ax4.legend(loc='upper right', fontsize=12)
        
        # 子图5：统计汇总表（底部）
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # 创建统计表格
        stats_data = []
        for agent_id, accelerations in agent_accelerations.items():
            mean_acc = np.mean(accelerations)
            max_acc = np.max(accelerations)
            std_acc = np.std(accelerations)
            percentile_95 = np.percentile(accelerations, 95)
            
            stats_data.append([
                agent_id,
                f'{mean_acc:.4f}',
                f'{std_acc:.4f}',
                f'{max_acc:.4f}',
                f'{percentile_95:.4f}',
                f'{efficiency_metrics.get(agent_id, {}).get("smoothness", 0):.3f}'
            ])
        
        headers = ['智能体', '平均加速度', '标准差', '最大加速度', '95%分位数', '控制平滑度']
        
        # 绘制表格
        table = ax5.table(cellText=stats_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         bbox=[0.1, 0.2, 0.8, 0.6])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(stats_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f1f1f2')
                table[(i, j)].set_edgecolor('black')
                table[(i, j)].set_linewidth(1)
        
        ax5.set_title('智能体加速度统计汇总表', fontsize=18, fontweight='bold', pad=20)
        
        plt.tight_layout()
        self.save_figure(fig, '08_acceleration_curves')
    
    def plot_velocity_curves(self):
        """绘制速度曲线图"""
        selected_config = self._select_representative_config()
        if not selected_config:
            return
        
        env_data = selected_config['env_data']
        agents_data = env_data.get('agents', {})
        
        if not agents_data:
            print("未找到智能体数据")
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        fig.suptitle('智能体速度分析与运动特性评估', fontsize=24, fontweight='bold', y=0.95)
        
        # 计算速度统计
        agent_velocities = {}
        agent_velocity_components = {}
        agent_speed_profiles = {}
        
        for agent_idx, (agent_id, agent_info) in enumerate(agents_data.items()):
            history_states = agent_info.get('history_states', [])
            if not history_states:
                continue
            
            states = np.array(history_states)
            if states.ndim != 2 or states.shape[1] < 4:
                continue
            
            velocities = states[:, 2:4]
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            
            agent_velocities[agent_id] = velocity_magnitudes
            agent_velocity_components[agent_id] = velocities
            
            # 计算速度轮廓（加速、减速、匀速阶段）
            if len(velocity_magnitudes) > 2:
                speed_changes = np.diff(velocity_magnitudes)
                acceleration_phases = speed_changes > 0.01
                deceleration_phases = speed_changes < -0.01
                constant_phases = np.abs(speed_changes) <= 0.01
                
                agent_speed_profiles[agent_id] = {
                    'acceleration': np.sum(acceleration_phases),
                    'deceleration': np.sum(deceleration_phases),
                    'constant': np.sum(constant_phases)
                }
        
        if not agent_velocities:
            print("未找到有效的状态数据")
            return
        
        dt = 2.0  # 状态时间步长
        
        # 子图1：速度大小时间序列
        ax1 = fig.add_subplot(gs[0, 0])
        
        for agent_idx, (agent_id, velocities) in enumerate(agent_velocities.items()):
            time_points = np.arange(len(velocities)) * dt
            color = self.colors[agent_idx]
            
            # 主速度曲线
            ax1.plot(time_points, velocities, 
                    color=color, linewidth=4, alpha=0.9,
                    linestyle=self.line_styles[agent_idx % len(self.line_styles)],
                    label=f'智能体 {agent_id}')
            
            # 移动平均线
            if len(velocities) > 10:
                window_size = min(10, len(velocities) // 5)
                moving_avg = pd.Series(velocities).rolling(window=window_size, center=True).mean()
                ax1.plot(time_points, moving_avg, 
                        color=color, linewidth=2, linestyle='-', alpha=0.6)
            
            # 标记重要速度事件
            if len(velocities) > 5:
                # 最高速度点
                max_speed_idx = np.argmax(velocities)
                ax1.scatter(time_points[max_speed_idx], velocities[max_speed_idx], 
                           color=color, s=150, marker='*', 
                           edgecolors='black', linewidth=2, zorder=10)
                
                # 添加注释
                ax1.annotate(f'最高速度\n{velocities[max_speed_idx]:.3f}m/s', 
                           (time_points[max_speed_idx], velocities[max_speed_idx]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                                 facecolor=color, alpha=0.7))
        
        ax1.set_xlabel('仿真时间 (s)', fontsize=16, fontweight='bold')
        ax1.set_ylabel('速度大小 (m/s)', fontsize=16, fontweight='bold')
        ax1.set_title('智能体速度大小时间序列', fontsize=18, fontweight='bold')
        ax1.grid(True, alpha=0.4, linestyle='--')
        ax1.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True)
        
        # 子图2：速度分量对比
        ax2 = fig.add_subplot(gs[0, 1])
        
        for agent_idx, (agent_id, velocity_components) in enumerate(agent_velocity_components.items()):
            time_points = np.arange(len(velocity_components)) * dt
            color = self.colors[agent_idx]
            
            # X方向速度分量
            ax2.plot(time_points, velocity_components[:, 0], 
                    color=color, linewidth=3, alpha=0.8,
                    linestyle='-', label=f'{agent_id} (Vx)')
            
            # Y方向速度分量
            ax2.plot(time_points, velocity_components[:, 1], 
                    color=color, linewidth=3, alpha=0.6,
                    linestyle='--', label=f'{agent_id} (Vy)')
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax2.set_xlabel('仿真时间 (s)', fontsize=16, fontweight='bold')
        ax2.set_ylabel('速度分量 (m/s)', fontsize=16, fontweight='bold')
        ax2.set_title('智能体速度分量时间序列', fontsize=18, fontweight='bold')
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.legend(loc='upper right', fontsize=10, ncol=2)
        
        # 子图3：速度相位图（vx vs vy）
        ax3 = fig.add_subplot(gs[1, 0])
        
        for agent_idx, (agent_id, velocity_components) in enumerate(agent_velocity_components.items()):
            color = self.colors[agent_idx]
            
            # 轨迹线
            ax3.plot(velocity_components[:, 0], velocity_components[:, 1], 
                    color=color, linewidth=3, alpha=0.7, 
                    label=f'智能体 {agent_id}')
            
            # 标记起点和终点
            if len(velocity_components) > 0:
                # 起点
                ax3.scatter(velocity_components[0, 0], velocity_components[0, 1], 
                           color=color, s=150, marker='o', 
                           edgecolors='white', linewidth=3, zorder=10)
                ax3.annotate('起点', 
                           (velocity_components[0, 0], velocity_components[0, 1]),
                           xytext=(-10, -10), textcoords='offset points',
                           fontsize=10, color=color, fontweight='bold')
                
                # 终点
                ax3.scatter(velocity_components[-1, 0], velocity_components[-1, 1], 
                           color=color, s=150, marker='s', 
                           edgecolors='white', linewidth=3, zorder=10)
                ax3.annotate('终点', 
                           (velocity_components[-1, 0], velocity_components[-1, 1]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, color=color, fontweight='bold')
        
        # 添加坐标轴和象限标识
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        
        # 象限标签
        xlim = ax3.get_xlim()
        ylim = ax3.get_ylim()
        ax3.text(xlim[1]*0.8, ylim[1]*0.8, '第一象限\n(右上运动)', 
                ha='center', va='center', fontsize=11, alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
        
        ax3.set_xlabel('X方向速度 (m/s)', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Y方向速度 (m/s)', fontsize=16, fontweight='bold')
        ax3.set_title('智能体速度相位图', fontsize=18, fontweight='bold')
        ax3.grid(True, alpha=0.4, linestyle='--')
        ax3.legend(loc='upper left', fontsize=12)
        ax3.set_aspect('equal', adjustable='box')
        
        # 子图4：速度分布统计
        ax4 = fig.add_subplot(gs[1, 1])
        
        # 准备箱线图数据
        velocity_data = []
        agent_labels = []
        colors_for_box = []
        
        for agent_idx, (agent_id, velocities) in enumerate(agent_velocities.items()):
            velocity_data.append(velocities)
            agent_labels.append(agent_id)
            colors_for_box.append(self.colors[agent_idx])
        
        # 绘制箱线图
        box_plot = ax4.boxplot(velocity_data, labels=agent_labels, 
                              patch_artist=True, notch=True, 
                              showmeans=True, meanline=True,
                              flierprops=dict(marker='d', markerfacecolor='red', 
                                           markersize=6, alpha=0.6))
        
        # 设置箱线图颜色
        for patch, color in zip(box_plot['boxes'], colors_for_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
        
        # 添加核密度估计曲线
        for i, velocities in enumerate(velocity_data):
            # 简化的密度估计
            hist, bin_edges = np.histogram(velocities, bins=20, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # 在右侧添加小的密度图
            density_x = i + 1 + hist * 0.3 / np.max(hist)  # 标准化并缩放
            ax4.plot(density_x, bin_centers, color=colors_for_box[i], 
                    linewidth=2, alpha=0.8)
        
        ax4.set_ylabel('速度大小 (m/s)', fontsize=16, fontweight='bold')
        ax4.set_xlabel('智能体', fontsize=16, fontweight='bold')
        ax4.set_title('智能体速度分布统计', fontsize=18, fontweight='bold')
        ax4.grid(True, alpha=0.4, axis='y', linestyle='--')
        
        # 子图5：运动模式分析
        ax5 = fig.add_subplot(gs[2, :])
        
        if agent_speed_profiles:
            agents_list = list(agent_speed_profiles.keys())
            phase_types = ['acceleration', 'constant', 'deceleration']
            phase_labels = ['加速阶段', '匀速阶段', '减速阶段']
            phase_colors = [self.colors[0], self.colors[1], self.colors[2]]
            
            x_pos = np.arange(len(agents_list))
            width = 0.25
            
            for i, phase_type in enumerate(phase_types):
                values = [agent_speed_profiles[agent][phase_type] for agent in agents_list]
                ax5.bar(x_pos + i * width, values, width, 
                       label=phase_labels[i], color=phase_colors[i], 
                       alpha=0.8, edgecolor='black', linewidth=1)
                
                # 在柱子上添加数值标签
                for j, v in enumerate(values):
                    ax5.text(x_pos[j] + i * width, v + 0.5, str(v), 
                            ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax5.set_xlabel('智能体', fontsize=16, fontweight='bold')
            ax5.set_ylabel('时间步数', fontsize=16, fontweight='bold')
            ax5.set_title('智能体运动模式分析', fontsize=18, fontweight='bold')
            ax5.set_xticks(x_pos + width)
            ax5.set_xticklabels(agents_list, fontsize=14)
            ax5.grid(True, alpha=0.4, axis='y', linestyle='--')
            ax5.legend(loc='upper right', fontsize=14)
            ax5.yaxis.set_major_locator(MaxNLocator(integer=True))
            
        plt.tight_layout()
        self.save_figure(fig, '09_velocity_curves')
    
    def plot_utility_convergence(self):
        """绘制收益值收敛图"""
        # 从分析数据和实验数据中获取效用收敛信息
        scalability_data = self.analysis_data.get('scalability', {})
        comparison_data = self.analysis_data.get('comparison', {})
        
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        fig.suptitle('系统效用收敛分析与性能评估', fontsize=24, fontweight='bold', y=0.95)
        
        # 子图1：不同配置的效用收敛对比
        ax1 = fig.add_subplot(gs[0, :])
        
        # 选择几个代表性配置进行对比
        selected_configs = []
        
        # 从扩展性实验中选择
        for exp_type in ['scalability', 'comparison']:
            type_data = self.experiment_data.get(exp_type, {})
            for config_name, runs in type_data.items():
                if runs and len(selected_configs) < 6:
                    env_data = runs[0]  # 选择第一次运行
                    utility_history = env_data.get('history_global_utility_ksc', [])
                    if utility_history:
                        selected_configs.append((config_name, utility_history, exp_type))
        
        if selected_configs:
            for i, (config_name, utility_history, exp_type) in enumerate(selected_configs):
                color = self.colors[i % len(self.colors)]
                linestyle = self.line_styles[i % len(self.line_styles)]
                
                # KSC决策轮次
                rounds = list(range(1, len(utility_history) + 1))
                
                # 绘制效用曲线
                ax1.plot(rounds, utility_history, 
                        color=color, linewidth=4, alpha=0.9,
                        linestyle=linestyle, marker=self.markers[i % len(self.markers)],
                        markersize=8, label=f'{config_name.replace("_", " ")}')
                
                # 添加收敛指标
                if len(utility_history) > 10:
                    # 计算收敛点（效用变化小于阈值）
                    final_utility = utility_history[-1]
                    convergence_threshold = 0.05 * abs(final_utility)
                    
                    convergence_round = len(utility_history)
                    for j in range(len(utility_history) - 5, 0, -1):
                        if abs(utility_history[j] - final_utility) > convergence_threshold:
                            convergence_round = j + 5
                            break
                    
                    # 标记收敛点
                    if convergence_round < len(utility_history):
                        ax1.axvline(x=convergence_round, color=color, 
                                   linestyle=':', alpha=0.6, linewidth=2)
                        ax1.text(convergence_round, final_utility, 
                                f'收敛@{convergence_round}轮', 
                                rotation=90, ha='right', va='bottom',
                                fontsize=10, color=color, alpha=0.8)
        
        ax1.set_xlabel('KSC决策轮次', fontsize=18, fontweight='bold')
        ax1.set_ylabel('系统总效用', fontsize=18, fontweight='bold')
        ax1.set_title('不同配置效用收敛对比', fontsize=20, fontweight='bold')
        ax1.grid(True, alpha=0.4, linestyle='--')
        ax1.legend(loc='lower right', fontsize=12, ncol=2, frameon=True, fancybox=True)
        
        # 子图2：效用改进分析
        ax2 = fig.add_subplot(gs[1, 0])
        
        if selected_configs:
            improvement_data = []
            config_names = []
            
            for config_name, utility_history, _ in selected_configs:
                if len(utility_history) > 1:
                    initial_utility = utility_history[0]
                    final_utility = utility_history[-1]
                    improvement = (final_utility - initial_utility) / abs(initial_utility) * 100
                    improvement_data.append(improvement)
                    config_names.append(config_name.replace("_", " ")[:15])  # 截断长名称
            
            if improvement_data:
                colors_bar = [self.colors[i % len(self.colors)] for i in range(len(improvement_data))]
                bars = ax2.bar(range(len(config_names)), improvement_data, 
                              color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
                
                # 在柱子上添加数值标签
                for bar, improvement in zip(bars, improvement_data):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -5),
                            f'{improvement:.1f}%', ha='center', 
                            va='bottom' if height >= 0 else 'top',
                            fontsize=12, fontweight='bold')
                
                ax2.set_xticks(range(len(config_names)))
                ax2.set_xticklabels(config_names, rotation=45, ha='right', fontsize=10)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_ylabel('效用改进 (%)', fontsize=16, fontweight='bold')
        ax2.set_title('各配置效用改进对比', fontsize=18, fontweight='bold')
        ax2.grid(True, alpha=0.4, axis='y', linestyle='--')
        
        # 子图3：收敛速度分析
        ax3 = fig.add_subplot(gs[1, 1])
        
        if selected_configs:
            convergence_speeds = []
            config_labels = []
            
            for config_name, utility_history, _ in selected_configs:
                if len(utility_history) > 5:
                    # 计算收敛速度（达到90%最终值所需轮次）
                    final_utility = utility_history[-1]
                    initial_utility = utility_history[0]
                    target_utility = initial_utility + 0.9 * (final_utility - initial_utility)
                    
                    convergence_round = len(utility_history)
                    for j, utility in enumerate(utility_history):
                        if utility >= target_utility:
                            convergence_round = j + 1
                            break
                    
                    convergence_speeds.append(convergence_round)
                    config_labels.append(config_name.replace("_", " ")[:15])
            
            if convergence_speeds:
                colors_scatter = [self.colors[i % len(self.colors)] for i in range(len(convergence_speeds))]
                
                scatter = ax3.scatter(range(len(config_labels)), convergence_speeds, 
                                    c=colors_scatter, s=200, alpha=0.8, 
                                    edgecolors='black', linewidth=2)
                
                # 添加趋势线
                if len(convergence_speeds) > 2:
                    z = np.polyfit(range(len(convergence_speeds)), convergence_speeds, 1)
                    p = np.poly1d(z)
                    ax3.plot(range(len(convergence_speeds)), p(range(len(convergence_speeds))), 
                            'r--', alpha=0.8, linewidth=2, label='趋势线')
                
                ax3.set_xticks(range(len(config_labels)))
                ax3.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=10)
                
                # 添加数值标签
                for i, speed in enumerate(convergence_speeds):
                    ax3.annotate(f'{speed}轮', (i, speed), xytext=(0, 10), 
                               textcoords='offset points', ha='center', va='bottom',
                               fontsize=11, fontweight='bold')
        
        ax3.set_ylabel('收敛轮次', fontsize=16, fontweight='bold')
        ax3.set_title('效用收敛速度对比', fontsize=18, fontweight='bold')
        ax3.grid(True, alpha=0.4, axis='y', linestyle='--')
        ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 子图4：效用统计分析
        ax4 = fig.add_subplot(gs[2, :])
        
        # 从汇总统计中获取效用相关数据
        summary_stats = self.analysis_data.get('summary_stats', {})
        
        if summary_stats:
            utility_stats = {}
            
            # 收集各实验类型的效用统计
            for exp_type, type_stats in summary_stats.items():
                for config_name, config_stats in type_stats.items():
                    final_utility_data = config_stats.get('final_utility', {})
                    if 'mean' in final_utility_data:
                        utility_stats[f"{exp_type}_{config_name}"] = {
                            'mean': final_utility_data['mean'],
                            'std': final_utility_data.get('std', 0),
                            'min': final_utility_data.get('min', 0),
                            'max': final_utility_data.get('max', 0)
                        }
            
            if utility_stats:
                # 选择前8个配置进行展示
                configs_to_plot = list(utility_stats.keys())[:8]
                
                means = [utility_stats[config]['mean'] for config in configs_to_plot]
                stds = [utility_stats[config]['std'] for config in configs_to_plot]
                
                x_pos = np.arange(len(configs_to_plot))
                colors_stat = [self.colors[i % len(self.colors)] for i in range(len(configs_to_plot))]
                
                # 绘制均值柱状图并添加误差线
                bars = ax4.bar(x_pos, means, yerr=stds, capsize=8,
                              color=colors_stat, alpha=0.8, 
                              edgecolor='black', linewidth=2,
                              error_kw={'linewidth': 3, 'capthick': 3})
                
                # 添加显著性标记（简化版本）
                max_height = max(m + s for m, s in zip(means, stds))
                for i in range(len(means) - 1):
                    # 简单的显著性检验模拟
                    diff = abs(means[i] - means[i+1])
                    pooled_std = np.sqrt((stds[i]**2 + stds[i+1]**2) / 2)
                    
                    if diff > 2 * pooled_std:  # 简化的显著性标准
                        # 添加显著性标记
                        y = max_height * 1.1
                        ax4.plot([i, i+1], [y, y], 'k-', linewidth=2)
                        ax4.text((i + i+1) / 2, y + max_height * 0.02, '*', 
                                ha='center', va='bottom', fontsize=16, fontweight='bold')
                
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels([config.replace("_", " ")[:20] for config in configs_to_plot], 
                                   rotation=45, ha='right', fontsize=11)
                
                ax4.set_ylabel('最终效用值', fontsize=16, fontweight='bold')
                ax4.set_title('各配置最终效用统计对比（含误差线与显著性检验）', fontsize=18, fontweight='bold')
                ax4.grid(True, alpha=0.4, axis='y', linestyle='--')
                
                # 添加统计信息文本框
                overall_mean = np.mean(means)
                overall_std = np.std(means)
                stats_text = f'总体均值: {overall_mean:.2f}\n总体标准差: {overall_std:.2f}\n配置数量: {len(configs_to_plot)}'
                
                ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                        fontsize=12, verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', 
                                alpha=0.8, edgecolor='black'))
        
        plt.tight_layout()
        self.save_figure(fig, '10_utility_convergence')
    
    def plot_k_value_comparison(self):
        """绘制K值对比图"""
        k_value_analysis = self.analysis_data.get('k_value', {})
        
        if not k_value_analysis:
            print("未找到K值分析数据")
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        fig.suptitle('KSC算法K值参数影响分析', fontsize=24, fontweight='bold', y=0.95)
        
        k_value_data = k_value_analysis.get('k_value_data', {})
        metrics_analysis = k_value_analysis.get('metrics_analysis', {})
        
        if not k_value_data:
            print("K值数据为空")
            return
        
        k_values = sorted(k_value_data.keys())
        
        # 子图1：不同K值下的效用对比
        ax1 = fig.add_subplot(gs[0, 0])
        
        utility_means = []
        utility_stds = []
        
        for k in k_values:
            utilities = k_value_data[k].get('final_utility', [])
            if utilities:
                utility_means.append(np.mean(utilities))
                utility_stds.append(np.std(utilities))
            else:
                utility_means.append(0)
                utility_stds.append(0)
        
        # 绘制柱状图和误差线
        colors_k = [self.colors[i % len(self.colors)] for i in range(len(k_values))]
        bars = ax1.bar(k_values, utility_means, yerr=utility_stds, capsize=10,
                      color=colors_k, alpha=0.8, edgecolor='black', linewidth=2,
                      error_kw={'linewidth': 3, 'capthick': 3, 'ecolor': 'black'})
        
        # 添加数值标签
        for bar, mean, std in zip(bars, utility_means, utility_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
        
        # 标记最优K值
        optimal_k = k_value_analysis.get('optimal_k_value')
        if optimal_k and optimal_k in k_values:
            optimal_idx = k_values.index(optimal_k)
            bars[optimal_idx].set_edgecolor('red')
            bars[optimal_idx].set_linewidth(4)
            ax1.text(optimal_k, utility_means[optimal_idx] + utility_stds[optimal_idx] + 0.5,
                    '最优K值', ha='center', va='bottom', fontsize=14, 
                    fontweight='bold', color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        ax1.set_xlabel('K值', fontsize=16, fontweight='bold')
        ax1.set_ylabel('最终效用值', fontsize=16, fontweight='bold')
        ax1.set_title('不同K值的效用对比', fontsize=18, fontweight='bold')
        ax1.grid(True, alpha=0.4, axis='y', linestyle='--')
        ax1.set_xticks(k_values)
        
        # 子图2：K值对任务完成率的影响
        ax2 = fig.add_subplot(gs[0, 1])
        
        completion_means = []
        completion_stds = []
        
        for k in k_values:
            completion_rates = k_value_data[k].get('task_completion_rate', [])
            if completion_rates:
                completion_means.append(np.mean(completion_rates) * 100)  # 转换为百分比
                completion_stds.append(np.std(completion_rates) * 100)
            else:
                completion_means.append(0)
                completion_stds.append(0)
        
        ax2.errorbar(k_values, completion_means, yerr=completion_stds, 
                    marker='o', linewidth=4, markersize=12, capsize=8,
                    color=self.colors[0], markerfacecolor=self.colors[1],
                    markeredgecolor='black', markeredgewidth=2,
                    elinewidth=3, capthick=3)
        
        ax2.set_xlabel('K值', fontsize=16, fontweight='bold')
        ax2.set_ylabel('任务完成率 (%)', fontsize=16, fontweight='bold')
        ax2.set_title('K值对任务完成率的影响', fontsize=18, fontweight='bold')
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.set_xticks(k_values)
        ax2.set_ylim(0, 105)
        
        # 添加完成率数值标签
        for k, mean, std in zip(k_values, completion_means, completion_stds):
            ax2.text(k, mean + std + 2, f'{mean:.1f}%', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 子图3：K值对通信开销的影响
        ax3 = fig.add_subplot(gs[1, 0])
        
        message_means = []
        message_stds = []
        
        for k in k_values:
            messages = k_value_data[k].get('total_messages_broadcasted', [])
            if messages:
                message_means.append(np.mean(messages))
                message_stds.append(np.std(messages))
            else:
                message_means.append(0)
                message_stds.append(0)
        
        ax3.plot(k_values, message_means, marker='s', linewidth=4, markersize=12,
                color=self.colors[2], markerfacecolor=self.colors[3],
                markeredgecolor='black', markeredgewidth=2, label='总消息数')
        
        # 添加误差阴影
        message_means_array = np.array(message_means)
        message_stds_array = np.array(message_stds)
        ax3.fill_between(k_values, message_means_array - message_stds_array,
                        message_means_array + message_stds_array,
                        alpha=0.3, color=self.colors[2])
        
        ax3.set_xlabel('K值', fontsize=16, fontweight='bold')
        ax3.set_ylabel('总消息数', fontsize=16, fontweight='bold')
        ax3.set_title('K值对通信开销的影响', fontsize=18, fontweight='bold')
        ax3.grid(True, alpha=0.4, linestyle='--')
        ax3.set_xticks(k_values)
        ax3.legend(fontsize=12)
        
        # 子图4：K值对信念收敛速度的影响
        ax4 = fig.add_subplot(gs[1, 1])
        
        convergence_means = []
        convergence_stds = []
        
        for k in k_values:
            convergence_speeds = k_value_data[k].get('belief_convergence_speed', [])
            if convergence_speeds:
                convergence_means.append(np.mean(convergence_speeds))
                convergence_stds.append(np.std(convergence_speeds))
            else:
                convergence_means.append(0)
                convergence_stds.append(0)
        
        # 绘制小提琴图样式的分布
        parts = ax4.violinplot([k_value_data[k].get('belief_convergence_speed', [0]) for k in k_values],
                              positions=k_values, showmeans=True, showmedians=True)
        
        # 设置小提琴图颜色
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self.colors[i % len(self.colors)])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(2)
        
        ax4.set_xlabel('K值', fontsize=16, fontweight='bold')
        ax4.set_ylabel('信念收敛速度', fontsize=16, fontweight='bold')
        ax4.set_title('K值对信念收敛速度的影响', fontsize=18, fontweight='bold')
        ax4.grid(True, alpha=0.4, linestyle='--')
        ax4.set_xticks(k_values)
        
        # 子图5：相关性分析热图
        ax5 = fig.add_subplot(gs[2, :])
        
        if metrics_analysis:
            metrics_names = list(metrics_analysis.keys())
            correlation_matrix = np.zeros((len(metrics_names), len(metrics_names)))
            
            # 计算指标间的相关性
            for i, metric1 in enumerate(metrics_names):
                for j, metric2 in enumerate(metrics_names):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                    else:
                        # 从分析数据中获取相关性
                        corr = metrics_analysis.get(metric1, {}).get('correlation', 0)
                        correlation_matrix[i, j] = corr
                        correlation_matrix[j, i] = corr
            
            # 绘制热图
            im = ax5.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            
            # 添加数值标签
            for i in range(len(metrics_names)):
                for j in range(len(metrics_names)):
                    text = ax5.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white",
                                   fontsize=12, fontweight='bold')
            
            ax5.set_xticks(range(len(metrics_names)))
            ax5.set_yticks(range(len(metrics_names)))
            ax5.set_xticklabels([name.replace('_', ' ').title() for name in metrics_names], 
                               rotation=45, ha='right', fontsize=12)
            ax5.set_yticklabels([name.replace('_', ' ').title() for name in metrics_names], fontsize=12)
            ax5.set_title('K值影响下各性能指标相关性矩阵', fontsize=18, fontweight='bold', pad=20)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax5, pad=0.02, aspect=30)
            cbar.set_label('相关系数', fontsize=14, fontweight='bold')
            cbar.ax.tick_params(labelsize=12)
        
        # 添加统计显著性检验结果
        statistical_tests = self.analysis_data.get('statistical', {})
        if statistical_tests:
            k_anova = statistical_tests.get('k_value_anova', {})
            if k_anova:
                significance_text = f"ANOVA检验结果:\nF统计量: {k_anova.get('f_statistic', 0):.3f}\np值: {k_anova.get('p_value', 1):.3f}\n"
                significance_text += "显著性: " + ("显著" if k_anova.get('significant', False) else "不显著")
                
                fig.text(0.02, 0.02, significance_text, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', 
                                alpha=0.8, edgecolor='black'),
                        verticalalignment='bottom', horizontalalignment='left')
        
        plt.tight_layout()
        self.save_figure(fig, '11_k_value_comparison')
    
    def plot_communication_statistics(self):
        """绘制通信统计图"""
        # 从实验数据中提取通信统计
        communication_data = {}
        
        # 从不同实验类型收集通信数据
        for exp_type in ['scalability', 'comparison', 'robustness']:
            type_data = self.experiment_data.get(exp_type, {})
            for config_name, runs in type_data.items():
                if runs:
                    env_data = runs[0]  # 取第一次运行
                    
                    # 提取通信相关数据
                    messages_broadcasted = env_data.get('ksc_stats_messages_broadcasted', [])
                    messages_processed = env_data.get('ksc_stats_messages_processed', [])
                    active_links = env_data.get('ksc_stats_active_links', [])
                    
                    if messages_broadcasted or messages_processed or active_links:
                        communication_data[config_name] = {
                            'messages_broadcasted': messages_broadcasted,
                            'messages_processed': messages_processed,
                            'active_links': active_links,
                            'exp_type': exp_type
                        }
        
        if not communication_data:
            print("未找到通信统计数据")
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        fig.suptitle('系统通信统计分析与网络性能评估', fontsize=24, fontweight='bold', y=0.95)
        
        # 子图1：消息广播数量时间序列
        ax1 = fig.add_subplot(gs[0, 0])
        
        # 选择前6个配置进行可视化
        configs_to_plot = list(communication_data.keys())[:6]
        
        for i, config_name in enumerate(configs_to_plot):
            data = communication_data[config_name]
            messages = data.get('messages_broadcasted', [])
            
            if messages:
                rounds = list(range(1, len(messages) + 1))
                color = self.colors[i % len(self.colors)]
                linestyle = self.line_styles[i % len(self.line_styles)]
                
                ax1.plot(rounds, messages, 
                        color=color, linewidth=4, alpha=0.8,
                        linestyle=linestyle, marker=self.markers[i % len(self.markers)],
                        markersize=8, label=config_name.replace("_", " ")[:20])
                
                # 添加移动平均线
                if len(messages) > 5:
                    window_size = min(5, len(messages) // 3)
                    moving_avg = pd.Series(messages).rolling(window=window_size, center=True).mean()
                    ax1.plot(rounds, moving_avg, 
                            color=color, linewidth=2, alpha=0.6, linestyle='-')
        
        ax1.set_xlabel('KSC决策轮次', fontsize=16, fontweight='bold')
        ax1.set_ylabel('广播消息数', fontsize=16, fontweight='bold')
        ax1.set_title('消息广播数量时间演化', fontsize=18, fontweight='bold')
        ax1.grid(True, alpha=0.4, linestyle='--')
        ax1.legend(loc='upper right', fontsize=10, ncol=2)
        
        # 子图2：通信链接数量分析
        ax2 = fig.add_subplot(gs[0, 1])
        
        for i, config_name in enumerate(configs_to_plot):
            data = communication_data[config_name]
            links = data.get('active_links', [])
            
            if links:
                rounds = list(range(1, len(links) + 1))
                color = self.colors[i % len(self.colors)]
                
                ax2.plot(rounds, links, 
                        color=color, linewidth=4, alpha=0.8,
                        linestyle=self.line_styles[i % len(self.line_styles)],
                        marker=self.markers[i % len(self.markers)],
                        markersize=8, label=config_name.replace("_", " ")[:20])
        
        ax2.set_xlabel('KSC决策轮次', fontsize=16, fontweight='bold')
        ax2.set_ylabel('活动通信链接数', fontsize=16, fontweight='bold')
        ax2.set_title('通信网络连接性演化', fontsize=18, fontweight='bold')
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.legend(loc='upper right', fontsize=10, ncol=2)
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 子图3：通信效率对比
        ax3 = fig.add_subplot(gs[1, 0])
        
        efficiency_data = []
        config_labels = []
        
        for config_name in configs_to_plot:
            data = communication_data[config_name]
            messages_b = data.get('messages_broadcasted', [])
            messages_p = data.get('messages_processed', [])
            
            if messages_b and messages_p:
                # 计算处理效率（处理/广播比例）
                total_broadcasted = sum(messages_b)
                total_processed = sum(messages_p)
                efficiency = total_processed / total_broadcasted if total_broadcasted > 0 else 0
                
                efficiency_data.append(efficiency)
                config_labels.append(config_name.replace("_", " ")[:15])
        
        if efficiency_data:
            colors_eff = [self.colors[i % len(self.colors)] for i in range(len(efficiency_data))]
            bars = ax3.bar(range(len(config_labels)), efficiency_data,
                          color=colors_eff, alpha=0.8, edgecolor='black', linewidth=2)
            
            # 添加数值标签
            for bar, eff in zip(bars, efficiency_data):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{eff:.2f}', ha='center', va='bottom',
                        fontsize=12, fontweight='bold')
            
            ax3.set_xticks(range(len(config_labels)))
            ax3.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=10)
            ax3.set_ylabel('通信效率 (处理/广播)', fontsize=16, fontweight='bold')
            ax3.set_title('各配置通信效率对比', fontsize=18, fontweight='bold')
            ax3.grid(True, alpha=0.4, axis='y', linestyle='--')
            
            # 添加效率参考线
            avg_efficiency = np.mean(efficiency_data)
            ax3.axhline(y=avg_efficiency, color='red', linestyle='--', linewidth=2, 
                       alpha=0.7, label=f'平均效率: {avg_efficiency:.2f}')
            ax3.legend(fontsize=12)
        
        # 子图4：消息负载分布
        ax4 = fig.add_subplot(gs[1, 1])
        
        # 收集所有配置的消息数据用于分布分析
        all_message_data = []
        message_labels = []
        
        for config_name in configs_to_plot:
            data = communication_data[config_name]
            messages = data.get('messages_broadcasted', [])
            
            if messages:
                all_message_data.append(messages)
                message_labels.append(config_name.replace("_", " ")[:12])
        
        if all_message_data:
            # 绘制箱线图
            box_plot = ax4.boxplot(all_message_data, labels=message_labels,
                                  patch_artist=True, notch=True, showmeans=True,
                                  flierprops=dict(marker='o', markersize=4, alpha=0.6))
            
            # 设置颜色
            colors_box = [self.colors[i % len(self.colors)] for i in range(len(all_message_data))]
            for patch, color in zip(box_plot['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(2)
            
            ax4.set_ylabel('每轮广播消息数', fontsize=16, fontweight='bold')
            ax4.set_xlabel('配置', fontsize=16, fontweight='bold')
            ax4.set_title('消息负载分布统计', fontsize=18, fontweight='bold')
            ax4.grid(True, alpha=0.4, axis='y', linestyle='--')
            
            # 旋转x轴标签
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # 子图5：通信网络拓扑分析
        ax5 = fig.add_subplot(gs[2, :])
        
        # 计算网络拓扑指标
        topology_metrics = {}
        
        for config_name in configs_to_plot:
            data = communication_data[config_name]
            links = data.get('active_links', [])
            
            if links:
                # 计算网络密度（假设完全图的最大链接数）
                # 假设智能体数量可以从配置名称推断或使用固定值
                num_agents = 5  # 默认值，实际应该从配置中获取
                max_links = num_agents * (num_agents - 1) // 2
                
                avg_links = np.mean(links)
                network_density = avg_links / max_links if max_links > 0 else 0
                
                # 计算连接稳定性
                link_stability = 1.0 - (np.std(links) / np.mean(links)) if np.mean(links) > 0 else 0
                
                topology_metrics[config_name] = {
                    'density': network_density,
                    'stability': link_stability,
                    'avg_links': avg_links
                }
        
        if topology_metrics:
            config_names = list(topology_metrics.keys())
            densities = [topology_metrics[name]['density'] for name in config_names]
            stabilities = [topology_metrics[name]['stability'] for name in config_names]
            
            # 绘制散点图
            colors_topo = [self.colors[i % len(self.colors)] for i in range(len(config_names))]
            scatter = ax5.scatter(densities, stabilities, 
                                c=colors_topo, s=200, alpha=0.8,
                                edgecolors='black', linewidth=2)
            
            # 添加配置标签
            for i, name in enumerate(config_names):
                ax5.annotate(name.replace("_", " ")[:15], 
                           (densities[i], stabilities[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, alpha=0.8)
            
            # 添加象限分割线
            ax5.axhline(y=np.median(stabilities), color='gray', linestyle='--', alpha=0.5)
            ax5.axvline(x=np.median(densities), color='gray', linestyle='--', alpha=0.5)
            
            # 象限标签
            ax5.text(0.95, 0.95, '高密度\n高稳定', transform=ax5.transAxes,
                    ha='right', va='top', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
            
            ax5.set_xlabel('网络密度', fontsize=16, fontweight='bold')
            ax5.set_ylabel('连接稳定性', fontsize=16, fontweight='bold')
            ax5.set_title('通信网络拓扑特性分析', fontsize=18, fontweight='bold')
            ax5.grid(True, alpha=0.4, linestyle='--')
            ax5.set_xlim(0, 1.05)
            ax5.set_ylim(0, 1.05)
        
        plt.tight_layout()
        self.save_figure(fig, '12_communication_statistics')
    
    def plot_scalability_analysis(self):
        """绘制扩展性分析图"""
        scalability_data = self.analysis_data.get('scalability', {})
        
        if not scalability_data:
            print("未找到扩展性分析数据")
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        fig.suptitle('系统扩展性分析与性能预测', fontsize=24, fontweight='bold', y=0.95)
        
        agent_scalability = scalability_data.get('agent_scalability', {})
        task_scalability = scalability_data.get('task_scalability', {})
        
        # 子图1：智能体数量扩展性
        ax1 = fig.add_subplot(gs[0, 0])
        
        if agent_scalability:
            agent_nums = sorted(agent_scalability.keys())
            utilities = [agent_scalability[n]['final_utility_mean'] for n in agent_nums]
            utility_stds = [agent_scalability[n]['final_utility_std'] for n in agent_nums]
            
            # 主曲线
            ax1.errorbar(agent_nums, utilities, yerr=utility_stds,
                        marker='o', linewidth=4, markersize=12, capsize=8,
                        color=self.colors[0], markerfacecolor=self.colors[1],
                        markeredgecolor='black', markeredgewidth=2,
                        elinewidth=3, capthick=3, label='实际性能')
            
            # 拟合趋势线
            if len(agent_nums) > 2:
                z = np.polyfit(agent_nums, utilities, 2)  # 二次拟合
                p = np.poly1d(z)
                x_smooth = np.linspace(min(agent_nums), max(agent_nums), 100)
                ax1.plot(x_smooth, p(x_smooth), '--', color=self.colors[2], 
                        linewidth=3, alpha=0.8, label='趋势预测')
                
                # 预测更多智能体的性能
                future_agents = list(range(max(agent_nums) + 1, max(agent_nums) + 4))
                future_utilities = [p(n) for n in future_agents]
                ax1.plot(future_agents, future_utilities, 
                        'o', color=self.colors[3], markersize=10, 
                        alpha=0.7, label='性能预测')
            
            ax1.set_xlabel('智能体数量', fontsize=16, fontweight='bold')
            ax1.set_ylabel('平均最终效用', fontsize=16, fontweight='bold')
            ax1.set_title('智能体数量扩展性分析', fontsize=18, fontweight='bold')
            ax1.grid(True, alpha=0.4, linestyle='--')
            ax1.legend(fontsize=12)
            ax1.set_xticks(agent_nums + future_agents if len(agent_nums) > 2 else agent_nums)
        
        # 子图2：任务数量扩展性
        ax2 = fig.add_subplot(gs[0, 1])
        
        if task_scalability:
            task_nums = sorted(task_scalability.keys())
            completion_rates = [task_scalability[n]['task_completion_rate_mean'] * 100 for n in task_nums]
            completion_stds = [task_scalability[n]['task_completion_rate_std'] * 100 for n in task_nums]
            
            # 主曲线
            ax2.errorbar(task_nums, completion_rates, yerr=completion_stds,
                        marker='s', linewidth=4, markersize=12, capsize=8,
                        color=self.colors[2], markerfacecolor=self.colors[3],
                        markeredgecolor='black', markeredgewidth=2,
                        elinewidth=3, capthick=3, label='完成率')
            
            # 理论最优线（100%完成率）
            ax2.axhline(y=100, color='green', linestyle='--', linewidth=3, 
                       alpha=0.7, label='理想完成率')
            
            # 可接受性能线（80%完成率）
            ax2.axhline(y=80, color='orange', linestyle=':', linewidth=3, 
                       alpha=0.7, label='可接受水平')
            
            ax2.set_xlabel('任务数量', fontsize=16, fontweight='bold')
            ax2.set_ylabel('平均任务完成率 (%)', fontsize=16, fontweight='bold')
            ax2.set_title('任务数量扩展性分析', fontsize=18, fontweight='bold')
            ax2.grid(True, alpha=0.4, linestyle='--')
            ax2.legend(fontsize=12)
            ax2.set_ylim(0, 105)
            ax2.set_xticks(task_nums)
        
        # 子图3：扩展性效率分析
        ax3 = fig.add_subplot(gs[1, 0])
        
        if agent_scalability:
            # 计算每个智能体的平均效用贡献
            agent_efficiency = []
            agent_nums_eff = []
            
            for n in sorted(agent_scalability.keys()):
                utility_per_agent = agent_scalability[n]['final_utility_mean'] / n if n > 0 else 0
                agent_efficiency.append(utility_per_agent)
                agent_nums_eff.append(n)
            
            ax3.plot(agent_nums_eff, agent_efficiency, 
                    marker='D', linewidth=4, markersize=12,
                    color=self.colors[4], markerfacecolor=self.colors[5],
                    markeredgecolor='black', markeredgewidth=2, label='单体效率')
            
            # 添加效率下降趋势线
            if len(agent_efficiency) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(agent_nums_eff, agent_efficiency)
                trend_line = [slope * n + intercept for n in agent_nums_eff]
                ax3.plot(agent_nums_eff, trend_line, '--', 
                        color='red', linewidth=3, alpha=0.8,
                        label=f'趋势 (斜率={slope:.3f})')
            
            ax3.set_xlabel('智能体数量', fontsize=16, fontweight='bold')
            ax3.set_ylabel('单智能体平均效用', fontsize=16, fontweight='bold')
            ax3.set_title('智能体扩展效率分析', fontsize=18, fontweight='bold')
            ax3.grid(True, alpha=0.4, linestyle='--')
            ax3.legend(fontsize=12)
        
        # 子图4：通信开销扩展性
        ax4 = fig.add_subplot(gs[1, 1])
        
        if agent_scalability:
            # 通信开销（每个智能体的平均消息数）
            comm_overhead = []
            
            for n in sorted(agent_scalability.keys()):
                messages_per_agent = agent_scalability[n].get('messages_per_agent', 0)
                comm_overhead.append(messages_per_agent)
            
            ax4.bar(agent_nums_eff, comm_overhead, 
                   color=[self.colors[i % len(self.colors)] for i in range(len(agent_nums_eff))],
                   alpha=0.8, edgecolor='black', linewidth=2)
            
            # 理论通信复杂度线（O(n²)）
            if agent_nums_eff:
                theoretical = [n**2 * comm_overhead[0] / agent_nums_eff[0]**2 for n in agent_nums_eff]
                ax4.plot(agent_nums_eff, theoretical, 'r--', linewidth=3, 
                        alpha=0.8, label='理论O(n²)复杂度')
            
            ax4.set_xlabel('智能体数量', fontsize=16, fontweight='bold')
            ax4.set_ylabel('单智能体平均消息数', fontsize=16, fontweight='bold')
            ax4.set_title('通信开销扩展性分析', fontsize=18, fontweight='bold')
            ax4.grid(True, alpha=0.4, axis='y', linestyle='--')
            ax4.legend(fontsize=12)
            ax4.set_xticks(agent_nums_eff)
        
        # 子图5：综合扩展性评估
        ax5 = fig.add_subplot(gs[2, :])
        
        # 创建综合扩展性指标
        if agent_scalability and task_scalability:
            # 收集各种规模下的性能数据
            scale_data = []
            
            # 智能体扩展数据
            for n in sorted(agent_scalability.keys()):
                data = agent_scalability[n]
                scale_data.append({
                    'scale': f'{n}智能体',
                    'type': '智能体扩展',
                    'utility': data['final_utility_mean'],
                    'completion': data['task_completion_rate_mean'] * 100,
                    'efficiency': data['final_utility_mean'] / n if n > 0 else 0,
                    'size': n
                })
            
            # 任务扩展数据
            for n in sorted(task_scalability.keys()):
                data = task_scalability[n]
                scale_data.append({
                    'scale': f'{n}任务',
                    'type': '任务扩展',
                    'utility': data['final_utility_mean'],
                    'completion': data['task_completion_rate_mean'] * 100,
                    'efficiency': data['final_utility_mean'] / n if n > 0 else 0,
                    'size': n
                })
            
            # 绘制扩展性雷达图风格的对比
            if scale_data:
                # 按类型分组
                agent_data = [d for d in scale_data if d['type'] == '智能体扩展']
                task_data = [d for d in scale_data if d['type'] == '任务扩展']
                
                # 标准化数据用于对比
                all_utilities = [d['utility'] for d in scale_data]
                all_completions = [d['completion'] for d in scale_data]
                all_efficiencies = [d['efficiency'] for d in scale_data]
                
                max_utility = max(all_utilities) if all_utilities else 1
                max_completion = max(all_completions) if all_completions else 1
                max_efficiency = max(all_efficiencies) if all_efficiencies else 1
                
                # 绘制智能体扩展性能
                if agent_data:
                    sizes = [d['size'] for d in agent_data]
                    norm_utilities = [d['utility'] / max_utility for d in agent_data]
                    norm_completions = [d['completion'] / max_completion for d in agent_data]
                    norm_efficiencies = [d['efficiency'] / max_efficiency for d in agent_data]
                    
                    ax5.plot(sizes, norm_utilities, 'o-', linewidth=4, markersize=10,
                            color=self.colors[0], label='标准化效用', alpha=0.8)
                    ax5.plot(sizes, norm_completions, 's-', linewidth=4, markersize=10,
                            color=self.colors[1], label='标准化完成率', alpha=0.8)
                    ax5.plot(sizes, norm_efficiencies, '^-', linewidth=4, markersize=10,
                            color=self.colors[2], label='标准化效率', alpha=0.8)
                
                ax5.set_xlabel('系统规模', fontsize=16, fontweight='bold')
                ax5.set_ylabel('标准化性能指标', fontsize=16, fontweight='bold')
                ax5.set_title('综合扩展性性能评估', fontsize=18, fontweight='bold')
                ax5.grid(True, alpha=0.4, linestyle='--')
                ax5.legend(fontsize=14, loc='upper right')
                ax5.set_ylim(0, 1.1)
                
                # 添加性能基准线
                ax5.axhline(y=0.8, color='green', linestyle='--', linewidth=2, 
                           alpha=0.7, label='良好性能线')
                ax5.axhline(y=0.6, color='orange', linestyle='--', linewidth=2, 
                           alpha=0.7, label='可接受性能线')
        
        # 添加扩展性统计信息
        scalability_trends = scalability_data.get('agent_trends', {})
        if scalability_trends:
            stats_text = "扩展性统计:\n"
            stats_text += f"效用趋势斜率: {scalability_trends.get('utility_slope', 0):.3f}\n"
            stats_text += f"完成率趋势斜率: {scalability_trends.get('completion_slope', 0):.3f}\n"
            stats_text += f"效用相关性: {scalability_trends.get('utility_correlation', 0):.3f}"
            
            fig.text(0.02, 0.02, stats_text, fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', 
                            alpha=0.8, edgecolor='black'),
                    verticalalignment='bottom', horizontalalignment='left')
        
        plt.tight_layout()
        self.save_figure(fig, '13_scalability_analysis')
    
    def plot_performance_comparison(self):
        """绘制性能对比图"""
        summary_stats = self.analysis_data.get('summary_stats', {})
        
        if not summary_stats:
            print("未找到汇总统计数据")
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        fig.suptitle('系统性能综合对比分析', fontsize=24, fontweight='bold', y=0.95)
        
        # 收集所有实验类型的性能数据
        performance_data = {}
        
        for exp_type, type_stats in summary_stats.items():
            for config_name, config_stats in type_stats.items():
                key = f"{exp_type}_{config_name}"
                performance_data[key] = {
                    'final_utility': config_stats.get('final_utility', {}),
                    'task_completion_rate': config_stats.get('task_completion_rate', {}),
                    'total_messages_broadcasted': config_stats.get('total_messages_broadcasted', {}),
                    'belief_convergence_speed': config_stats.get('belief_convergence_speed', {}),
                    'simulation_time': config_stats.get('simulation_time', {}),
                    'exp_type': exp_type
                }
        
        # 子图1：多维性能雷达图
        ax1 = fig.add_subplot(gs[0, :], projection='polar')
        
        # 选择代表性配置进行雷达图对比
        configs_for_radar = list(performance_data.keys())[:5]
        
        if configs_for_radar:
            # 定义性能维度
            metrics = ['final_utility', 'task_completion_rate', 'belief_convergence_speed']
            metric_labels = ['最终效用', '任务完成率', '信念收敛速度']
            
            # 计算角度
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            # 为每个配置绘制雷达图
            for i, config_name in enumerate(configs_for_radar):
                config_data = performance_data[config_name]
                
                # 提取并标准化数据
                values = []
                for metric in metrics:
                    metric_data = config_data.get(metric, {})
                    value = metric_data.get('mean', 0)
                    values.append(value)
                
                # 标准化到0-1范围
                if values:
                    max_vals = [max(performance_data[c].get(m, {}).get('mean', 1) 
                                  for c in configs_for_radar) for m in metrics]
                    normalized_values = [v / max_v if max_v > 0 else 0 for v, max_v in zip(values, max_vals)]
                    normalized_values += normalized_values[:1]  # 闭合
                    
                    color = self.colors[i % len(self.colors)]
                    ax1.plot(angles, normalized_values, 'o-', linewidth=3, 
                            color=color, label=config_name.replace("_", " ")[:20], alpha=0.8)
                    ax1.fill(angles, normalized_values, alpha=0.25, color=color)
            
            # 设置雷达图
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels(metric_labels, fontsize=14)
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.4)
            ax1.set_title('多维性能雷达图对比', fontsize=18, fontweight='bold', pad=30)
            ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        # 子图2：实验类型性能对比
        ax2 = fig.add_subplot(gs[1, 0])
        
        # 按实验类型统计平均性能
        exp_type_stats = {}
        for exp_type in ['scalability', 'comparison', 'robustness']:
            type_utilities = []
            type_completions = []
            
            for key, data in performance_data.items():
                if data['exp_type'] == exp_type:
                    utility_mean = data['final_utility'].get('mean', 0)
                    completion_mean = data['task_completion_rate'].get('mean', 0)
                    
                    if utility_mean > 0:
                        type_utilities.append(utility_mean)
                    if completion_mean > 0:
                        type_completions.append(completion_mean * 100)
            
            if type_utilities or type_completions:
                exp_type_stats[exp_type] = {
                    'utility_mean': np.mean(type_utilities) if type_utilities else 0,
                    'utility_std': np.std(type_utilities) if type_utilities else 0,
                    'completion_mean': np.mean(type_completions) if type_completions else 0,
                    'completion_std': np.std(type_completions) if type_completions else 0
                }
        
        if exp_type_stats:
            exp_types = list(exp_type_stats.keys())
            exp_type_labels = {'scalability': '扩展性', 'comparison': '对比', 'robustness': '鲁棒性'}
            
            utilities_mean = [exp_type_stats[et]['utility_mean'] for et in exp_types]
            utilities_std = [exp_type_stats[et]['utility_std'] for et in exp_types]
            
            x_pos = np.arange(len(exp_types))
            colors_exp = [self.colors[i] for i in range(len(exp_types))]
            
            bars = ax2.bar(x_pos, utilities_mean, yerr=utilities_std, capsize=8,
                          color=colors_exp, alpha=0.8, edgecolor='black', linewidth=2,
                          error_kw={'linewidth': 3, 'capthick': 3})
            
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([exp_type_labels.get(et, et) for et in exp_types], fontsize=14)
            ax2.set_ylabel('平均最终效用', fontsize=16, fontweight='bold')
            ax2.set_title('不同实验类型性能对比', fontsize=18, fontweight='bold')
            ax2.grid(True, alpha=0.4, axis='y', linestyle='--')
            
            # 添加数值标签
            for bar, mean, std in zip(bars, utilities_mean, utilities_std):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                        f'{mean:.1f}±{std:.1f}', ha='center', va='bottom',
                        fontsize=12, fontweight='bold')
        
        # 子图3：性能-复杂度权衡分析
        ax3 = fig.add_subplot(gs[1, 1])
        
        # 计算性能复杂度权衡
        complexity_data = []
        performance_scores = []
        config_labels = []
        
        for config_name, data in performance_data.items():
            # 简化的复杂度计算（基于消息数量）
            complexity = data.get('total_messages_broadcasted', {}).get('mean', 0)
            
            # 综合性能得分
            utility = data.get('final_utility', {}).get('mean', 0)
            completion = data.get('task_completion_rate', {}).get('mean', 0)
            convergence = data.get('belief_convergence_speed', {}).get('mean', 0)
            
            # 标准化并加权
            performance_score = 0.4 * utility + 0.4 * completion * 100 + 0.2 * convergence
            
            if complexity > 0 and performance_score > 0:
                complexity_data.append(complexity)
                performance_scores.append(performance_score)
                config_labels.append(config_name.replace("_", " ")[:15])
        
        if complexity_data and performance_scores:
            # 按实验类型着色
            colors_complex = []
            for label in config_labels:
                if 'scalability' in label:
                    colors_complex.append(self.colors[0])
                elif 'comparison' in label:
                    colors_complex.append(self.colors[1])
                else:
                    colors_complex.append(self.colors[2])
            
            scatter = ax3.scatter(complexity_data, performance_scores, 
                                c=colors_complex, s=150, alpha=0.8,
                                edgecolors='black', linewidth=2)
            
            # 添加效率前沿线
            if len(complexity_data) > 3:
                # 找到帕累托前沿
                sorted_indices = np.argsort(complexity_data)
                pareto_x = []
                pareto_y = []
                max_performance = 0
                
                for idx in sorted_indices:
                    if performance_scores[idx] > max_performance:
                        pareto_x.append(complexity_data[idx])
                        pareto_y.append(performance_scores[idx])
                        max_performance = performance_scores[idx]
                
                if len(pareto_x) > 1:
                    ax3.plot(pareto_x, pareto_y, 'r--', linewidth=3, 
                            alpha=0.8, label='效率前沿')
            
            # 添加配置标签
            for i, label in enumerate(config_labels):
                ax3.annotate(label, (complexity_data[i], performance_scores[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
            
            ax3.set_xlabel('系统复杂度 (消息数)', fontsize=16, fontweight='bold')
            ax3.set_ylabel('综合性能得分', fontsize=16, fontweight='bold')
            ax3.set_title('性能-复杂度权衡分析', fontsize=18, fontweight='bold')
            ax3.grid(True, alpha=0.4, linestyle='--')
            
            if len(pareto_x) > 1:
                ax3.legend(fontsize=12)
        
        # 子图4：配置排名分析
        ax4 = fig.add_subplot(gs[2, :])
        
        # 计算综合排名
        ranking_data = []
        
        for config_name, data in performance_data.items():
            utility = data.get('final_utility', {}).get('mean', 0)
            completion = data.get('task_completion_rate', {}).get('mean', 0)
            messages = data.get('total_messages_broadcasted', {}).get('mean', 0)
            sim_time = data.get('simulation_time', {}).get('mean', 0)
            
            # 综合评分（效用和完成率正向，消息数和时间负向）
            score = utility * 0.4 + completion * 100 * 0.3 - messages * 0.0001 - sim_time * 0.001
            
            ranking_data.append({
                'config': config_name.replace("_", " "),
                'score': score,
                'utility': utility,
                'completion': completion * 100,
                'messages': messages,
                'time': sim_time
            })
        
        # 按得分排序
        ranking_data.sort(key=lambda x: x['score'], reverse=True)
        
        if ranking_data:
            # 取前10名
            top_configs = ranking_data[:10]
            
            config_names_rank = [item['config'][:20] for item in top_configs]
            scores = [item['score'] for item in top_configs]
            
            # 创建颜色渐变
            colors_rank = plt.cm.RdYlGn(np.linspace(0.3, 1.0, len(top_configs)))
            
            bars = ax4.barh(range(len(config_names_rank)), scores,
                           color=colors_rank, alpha=0.8, edgecolor='black', linewidth=1)
            
            # 添加排名标签
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax4.text(score + max(scores) * 0.01, bar.get_y() + bar.get_height()/2,
                        f'#{i+1}: {score:.1f}', ha='left', va='center',
                        fontsize=11, fontweight='bold')
            
            ax4.set_yticks(range(len(config_names_rank)))
            ax4.set_yticklabels(config_names_rank, fontsize=10)
            ax4.set_xlabel('综合性能得分', fontsize=16, fontweight='bold')
            ax4.set_title('配置性能综合排名（前10名）', fontsize=18, fontweight='bold')
            ax4.grid(True, alpha=0.4, axis='x', linestyle='--')
            
            # 添加得分计算说明
            score_formula = "得分计算:\n效用×0.4 + 完成率×0.3 - 消息数×0.0001 - 时间×0.001"
            ax4.text(0.98, 0.02, score_formula, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', 
                           alpha=0.8, edgecolor='gray'))
       
        plt.tight_layout()
        self.save_figure(fig, '14_performance_comparison')
   
    def plot_statistical_significance(self):
       """绘制统计显著性分析图"""
       statistical_data = self.analysis_data.get('statistical', {})
       
       if not statistical_data:
           print("未找到统计检验数据")
           return
       
       fig = plt.figure(figsize=(20, 16))
       gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
       fig.suptitle('统计显著性检验与假设验证分析', fontsize=24, fontweight='bold', y=0.95)
       
       # 子图1：K值ANOVA检验结果
       ax1 = fig.add_subplot(gs[0, 0])
       
       k_anova = statistical_data.get('k_value_anova', {})
       if k_anova:
           f_statistic = k_anova.get('f_statistic', 0)
           p_value = k_anova.get('p_value', 1)
           is_significant = k_anova.get('significant', False)
           
           # 绘制F统计量分布
           df_between = 4  # 假设K值1-5，自由度为4
           df_within = 20   # 假设总样本自由度
           
           x = np.linspace(0, max(8, f_statistic * 1.5), 1000)
           f_dist = stats.f.pdf(x, df_between, df_within)
           
           ax1.plot(x, f_dist, 'b-', linewidth=3, alpha=0.8, label='F分布')
           ax1.axvline(x=f_statistic, color='red', linestyle='--', linewidth=4,
                      label=f'观测F值 = {f_statistic:.3f}')
           
           # 临界值
           critical_value = stats.f.ppf(0.95, df_between, df_within)
           ax1.axvline(x=critical_value, color='orange', linestyle=':', linewidth=3,
                      label=f'临界值 (α=0.05) = {critical_value:.3f}')
           
           # 填充拒绝域
           x_reject = x[x >= critical_value]
           f_reject = stats.f.pdf(x_reject, df_between, df_within)
           ax1.fill_between(x_reject, 0, f_reject, alpha=0.3, color='red',
                          label='拒绝域 (α=0.05)')
           
           ax1.set_xlabel('F统计量', fontsize=16, fontweight='bold')
           ax1.set_ylabel('概率密度', fontsize=16, fontweight='bold')
           ax1.set_title(f'K值效应ANOVA检验\nF({df_between},{df_within}) = {f_statistic:.3f}, p = {p_value:.4f}', 
                        fontsize=16, fontweight='bold')
           ax1.legend(fontsize=12)
           ax1.grid(True, alpha=0.4, linestyle='--')
           
           # 添加结论文本框
           conclusion = "显著性结论:\n"
           conclusion += f"F统计量 = {f_statistic:.3f}\n"
           conclusion += f"p值 = {p_value:.4f}\n"
           conclusion += f"结果: {'显著' if is_significant else '不显著'}"
           
           ax1.text(0.02, 0.98, conclusion, transform=ax1.transAxes,
                   fontsize=12, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle="round,pad=0.5", 
                           facecolor='lightgreen' if is_significant else 'lightcoral',
                           alpha=0.8, edgecolor='black'))
       
       # 子图2：两两比较检验结果
       ax2 = fig.add_subplot(gs[0, 1])
       
       pairwise_tests = statistical_data.get('k_value_pairwise', {})
       if pairwise_tests:
           # 创建p值矩阵
           k_values = [1, 2, 3, 4, 5]  # 假设的K值范围
           p_matrix = np.ones((len(k_values), len(k_values)))
           
           # 填充p值矩阵
           for comparison, test_result in pairwise_tests.items():
               if 'K' in comparison and 'vs' in comparison:
                   parts = comparison.replace('K', '').split('_vs_K')
                   if len(parts) == 2:
                       try:
                           k1_idx = k_values.index(int(parts[0]))
                           k2_idx = k_values.index(int(parts[1]))
                           p_val = test_result.get('p_value', 1.0)
                           p_matrix[k1_idx, k2_idx] = p_val
                           p_matrix[k2_idx, k1_idx] = p_val
                       except (ValueError, IndexError):
                           continue
           
           # 绘制热图
           im = ax2.imshow(p_matrix, cmap='RdYlBu_r', vmin=0, vmax=0.1, aspect='auto')
           
           # 添加数值标签和显著性标记
           for i in range(len(k_values)):
               for j in range(len(k_values)):
                   p_val = p_matrix[i, j]
                   
                   # 确定文本颜色
                   text_color = "white" if p_val < 0.05 else "black"
                   
                   # 添加p值
                   ax2.text(j, i, f'{p_val:.3f}', ha="center", va="center", 
                           color=text_color, fontsize=11, fontweight='bold')
                   
                   # 添加显著性星号
                   if p_val < 0.001:
                       stars = "***"
                   elif p_val < 0.01:
                       stars = "**"
                   elif p_val < 0.05:
                       stars = "*"
                   else:
                       stars = "ns"
                   
                   ax2.text(j, i + 0.3, stars, ha="center", va="center", 
                           color=text_color, fontsize=10, fontweight='bold')
           
           ax2.set_xticks(range(len(k_values)))
           ax2.set_yticks(range(len(k_values)))
           ax2.set_xticklabels([f'K={k}' for k in k_values], fontsize=12)
           ax2.set_yticklabels([f'K={k}' for k in k_values], fontsize=12)
           ax2.set_title('K值两两比较t检验p值矩阵', fontsize=16, fontweight='bold')
           
           # 添加颜色条
           cbar = plt.colorbar(im, ax=ax2, pad=0.02, aspect=30)
           cbar.set_label('p值', fontsize=14, fontweight='bold')
           cbar.ax.tick_params(labelsize=12)
           
           # 添加显著性说明
           sig_text = "显著性标记:\n*** p<0.001\n** p<0.01\n* p<0.05\nns 不显著"
           ax2.text(1.15, 0.5, sig_text, transform=ax2.transAxes,
                   fontsize=11, verticalalignment='center', horizontalalignment='left',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
       
       # 子图3：扩展性相关性检验
       ax3 = fig.add_subplot(gs[1, 0])
       
       agent_corr = statistical_data.get('agent_scalability_correlation', {})
       task_corr = statistical_data.get('task_scalability_correlation', {})
       
       if agent_corr or task_corr:
           # 准备数据
           correlations = []
           p_values = []
           test_names = []
           
           if agent_corr:
               correlations.append(agent_corr.get('correlation', 0))
               p_values.append(agent_corr.get('p_value', 1))
               test_names.append('智能体数量-效用')
           
           if task_corr:
               correlations.append(task_corr.get('correlation', 0))
               p_values.append(task_corr.get('p_value', 1))
               test_names.append('任务数量-效用')
           
           # 绘制相关系数条形图
           colors_corr = [self.colors[0] if corr > 0 else self.colors[1] for corr in correlations]
           bars = ax3.bar(range(len(test_names)), correlations, 
                         color=colors_corr, alpha=0.8, edgecolor='black', linewidth=2)
           
           # 添加显著性标记
           for i, (bar, p_val) in enumerate(zip(bars, p_values)):
               height = bar.get_height()
               
               # 显著性星号
               if p_val < 0.001:
                   stars = "***"
               elif p_val < 0.01:
                   stars = "**"
               elif p_val < 0.05:
                   stars = "*"
               else:
                   stars = "ns"
               
               # 在柱子顶部添加显著性标记
               y_pos = height + 0.02 if height >= 0 else height - 0.05
               ax3.text(bar.get_x() + bar.get_width()/2, y_pos, stars,
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=14, fontweight='bold', color='red')
               
               # 添加p值标签
               ax3.text(bar.get_x() + bar.get_width()/2, height/2,
                       f'r={height:.3f}\np={p_val:.3f}',
                       ha='center', va='center', fontsize=11, fontweight='bold',
                       color='white' if abs(height) > 0.3 else 'black')
           
           ax3.set_xticks(range(len(test_names)))
           ax3.set_xticklabels(test_names, fontsize=12)
           ax3.set_ylabel('相关系数', fontsize=16, fontweight='bold')
           ax3.set_title('扩展性相关性检验', fontsize=18, fontweight='bold')
           ax3.set_ylim(-1.1, 1.1)
           ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
           ax3.grid(True, alpha=0.4, axis='y', linestyle='--')
           
           # 添加相关性强度参考线
           for threshold, label, color in [(0.7, '强相关', 'green'), 
                                          (0.3, '中等相关', 'orange'),
                                          (-0.3, '中等负相关', 'orange'),
                                          (-0.7, '强负相关', 'red')]:
               ax3.axhline(y=threshold, color=color, linestyle=':', alpha=0.7, linewidth=2)
               ax3.text(len(test_names) - 0.5, threshold + 0.05, label, 
                       fontsize=10, color=color, alpha=0.8)
       
       # 子图4：置信区间分析
       ax4 = fig.add_subplot(gs[1, 1])
       
       # 从K值分析中构建置信区间
       k_value_data = self.analysis_data.get('k_value', {}).get('k_value_data', {})
       
       if k_value_data:
           k_values_ci = sorted(k_value_data.keys())
           means = []
           ci_lower = []
           ci_upper = []
           
           for k in k_values_ci:
               utilities = k_value_data[k].get('final_utility', [])
               if utilities:
                   mean_val = np.mean(utilities)
                   std_val = np.std(utilities, ddof=1)
                   n = len(utilities)
                   
                   # 95%置信区间
                   t_critical = stats.t.ppf(0.975, df=n-1)
                   margin_error = t_critical * std_val / np.sqrt(n)
                   
                   means.append(mean_val)
                   ci_lower.append(mean_val - margin_error)
                   ci_upper.append(mean_val + margin_error)
           
           if means:
               # 绘制置信区间
               ax4.errorbar(k_values_ci, means, 
                          yerr=[np.array(means) - np.array(ci_lower),
                                np.array(ci_upper) - np.array(means)],
                          marker='o', linewidth=4, markersize=12, capsize=10,
                          color=self.colors[0], markerfacecolor=self.colors[1],
                          markeredgecolor='black', markeredgewidth=2,
                          elinewidth=3, capthick=3, label='95%置信区间')
               
               # 填充置信区间
               ax4.fill_between(k_values_ci, ci_lower, ci_upper, 
                              alpha=0.3, color=self.colors[0], label='置信区间范围')
               
               # 添加均值点
               ax4.scatter(k_values_ci, means, s=150, color=self.colors[2],
                          edgecolors='black', linewidth=2, zorder=10, label='样本均值')
               
               ax4.set_xlabel('K值', fontsize=16, fontweight='bold')
               ax4.set_ylabel('效用值', fontsize=16, fontweight='bold')
               ax4.set_title('K值效用95%置信区间', fontsize=18, fontweight='bold')
               ax4.grid(True, alpha=0.4, linestyle='--')
               ax4.legend(fontsize=12)
               ax4.set_xticks(k_values_ci)
               
               # 检查置信区间重叠
               overlap_text = "置信区间重叠分析:\n"
               for i in range(len(k_values_ci) - 1):
                   k1, k2 = k_values_ci[i], k_values_ci[i + 1]
                   if ci_upper[i] > ci_lower[i + 1]:
                       overlap_text += f"K{k1}-K{k2}: 重叠\n"
                   else:
                       overlap_text += f"K{k1}-K{k2}: 不重叠*\n"
               
               ax4.text(0.02, 0.98, overlap_text, transform=ax4.transAxes,
                       fontsize=11, verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', 
                               alpha=0.8, edgecolor='gray'))
       
       # 子图5：效应量分析
       ax5 = fig.add_subplot(gs[2, :])
       
       # 计算Cohen's d效应量
       if k_value_data and len(k_value_data) > 1:
           effect_sizes = []
           comparisons = []
           
           k_list = sorted(k_value_data.keys())
           
           # 计算所有两两比较的效应量
           for i in range(len(k_list)):
               for j in range(i + 1, len(k_list)):
                   k1, k2 = k_list[i], k_list[j]
                   
                   utilities1 = k_value_data[k1].get('final_utility', [])
                   utilities2 = k_value_data[k2].get('final_utility', [])
                   
                   if utilities1 and utilities2:
                       # 计算Cohen's d
                       mean1, mean2 = np.mean(utilities1), np.mean(utilities2)
                       std1, std2 = np.std(utilities1, ddof=1), np.std(utilities2, ddof=1)
                       n1, n2 = len(utilities1), len(utilities2)
                       
                       # 合并标准差
                       pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                       
                       if pooled_std > 0:
                           cohens_d = (mean1 - mean2) / pooled_std
                           effect_sizes.append(abs(cohens_d))
                           comparisons.append(f'K{k1} vs K{k2}')
           
           if effect_sizes:
               # 绘制效应量条形图
               colors_effect = []
               for effect in effect_sizes:
                   if effect < 0.2:
                       colors_effect.append('lightgray')  # 小效应
                   elif effect < 0.5:
                       colors_effect.append('yellow')     # 中等效应
                   elif effect < 0.8:
                       colors_effect.append('orange')     # 大效应
                   else:
                       colors_effect.append('red')        # 非常大效应
               
               bars = ax5.bar(range(len(comparisons)), effect_sizes,
                             color=colors_effect, alpha=0.8, edgecolor='black', linewidth=2)
               
               # 添加效应量标准参考线
               ax5.axhline(y=0.2, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='小效应 (d=0.2)')
               ax5.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='中等效应 (d=0.5)')
               ax5.axhline(y=0.8, color='red', linestyle='--', linewidth=2, alpha=0.7, label='大效应 (d=0.8)')
               
               # 添加数值标签
               for bar, effect in zip(bars, effect_sizes):
                   height = bar.get_height()
                   
                   # 效应量分类
                   if effect < 0.2:
                       effect_label = "小"
                   elif effect < 0.5:
                       effect_label = "中"
                   elif effect < 0.8:
                       effect_label = "大"
                   else:
                       effect_label = "极大"
                   
                   ax5.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                           f'{effect:.2f}\n({effect_label})', ha='center', va='bottom',
                           fontsize=10, fontweight='bold')
               
               ax5.set_xticks(range(len(comparisons)))
               ax5.set_xticklabels(comparisons, rotation=45, ha='right', fontsize=12)
               ax5.set_ylabel("Cohen's d (效应量)", fontsize=16, fontweight='bold')
               ax5.set_title('K值比较效应量分析', fontsize=18, fontweight='bold')
               ax5.grid(True, alpha=0.4, axis='y', linestyle='--')
               ax5.legend(loc='upper right', fontsize=12)
               
               # 添加效应量解释
               interpretation = "效应量解释:\nd < 0.2: 小效应\n0.2 ≤ d < 0.5: 中等效应\n0.5 ≤ d < 0.8: 大效应\nd ≥ 0.8: 非常大效应"
               ax5.text(0.02, 0.98, interpretation, transform=ax5.transAxes,
                       fontsize=11, verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle="round,pad=0.4", facecolor='lightcyan', 
                               alpha=0.8, edgecolor='black'))
       
       plt.tight_layout()
       self.save_figure(fig, '15_statistical_significance')

def main():
   """主函数"""
   # 实验数据目录
   experiment_dir = "results/strong_comm_belief_experiments"
   
   if not Path(experiment_dir).exists():
       print(f"实验目录不存在: {experiment_dir}")
       print("请先运行实验程序生成数据")
       return
   
   print("开始生成强通信信念自学习实验可视化...")
   
   # 创建可视化器
   visualizer = BeliefLearningVisualizer(experiment_dir)
   
   # 生成所有可视化图表
   visualizer.create_all_visualizations()
   
   print("可视化完成！")
   print(f"所有图表已保存到: {visualizer.output_dir}")

if __name__ == "__main__":
   main()