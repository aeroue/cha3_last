# experiments/strong_comm_belief_learning/data_analyzer.py
"""
强通信信念自学习实验数据分析器
负责分析实验数据并生成统计结果
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class BeliefLearningDataAnalyzer:
    """强通信信念自学习实验数据分析器"""
    
    def __init__(self, experiment_dir: str):
        """
        初始化数据分析器
        
        Args:
            experiment_dir: 实验结果目录
        """
        self.experiment_dir = Path(experiment_dir)
        self.analysis_output_dir = self.experiment_dir / "analysis"
        self.analysis_output_dir.mkdir(exist_ok=True)
        
        # 加载实验数据
        self.experiment_data = self.load_experiment_data()
        self.summary_stats = self.calculate_summary_statistics()
        
    def load_experiment_data(self) -> Dict[str, Any]:
        """加载所有实验数据"""
        print("正在加载实验数据...")
        
        experiment_data = {
            'scalability': {},
            'comparison': {}, 
            'robustness': {}
        }
        
        for exp_type in experiment_data.keys():
            exp_type_dir = self.experiment_dir / exp_type
            if not exp_type_dir.exists():
                continue
                
            for config_dir in exp_type_dir.iterdir():
                if not config_dir.is_dir():
                    continue
                    
                config_name = config_dir.name
                config_data = []
                
                # 加载该配置的所有运行结果
                for run_dir in config_dir.iterdir():
                    if not run_dir.is_dir() or not run_dir.name.startswith('run_'):
                        continue
                    
                    # 加载结果文件
                    result_file = run_dir / "results.pkl"
                    env_state_file = run_dir / "env_state.pkl"
                    
                    if result_file.exists() and env_state_file.exists():
                        try:
                            with open(result_file, 'rb') as f:
                                result_data = pickle.load(f)
                            
                            with open(env_state_file, 'rb') as f:
                                env_data = pickle.load(f)
                            
                            # 合并数据
                            combined_data = {**result_data, 'env_data': env_data}
                            config_data.append(combined_data)
                            
                        except Exception as e:
                            print(f"加载数据失败: {run_dir}, 错误: {e}")
                
                if config_data:
                    experiment_data[exp_type][config_name] = config_data
        
        print(f"数据加载完成，共加载 {sum(len(configs) for configs in experiment_data.values())} 个配置")
        return experiment_data
    
    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """计算汇总统计信息"""
        print("计算汇总统计信息...")
        
        summary_stats = {}
        
        for exp_type, type_data in self.experiment_data.items():
            type_stats = {}
            
            for config_name, config_runs in type_data.items():
                if not config_runs:
                    continue
                
                # 提取关键指标
                metrics = {
                    'final_utility': [run.get('final_utility', 0) for run in config_runs],
                    'task_completion_rate': [run.get('task_completion_rate', 0) for run in config_runs],
                    'total_messages_broadcasted': [run.get('total_messages_broadcasted', 0) for run in config_runs],
                    'belief_convergence_speed': [run.get('belief_convergence_speed', 0) for run in config_runs],
                    'final_belief_accuracy': [run.get('final_belief_accuracy', 0) for run in config_runs],
                    'belief_consensus_level': [run.get('belief_consensus_level', 0) for run in config_runs],
                    'simulation_time': [run.get('simulation_time', 0) for run in config_runs],
                    'num_agents': config_runs[0].get('num_agents', 0),
                    'num_tasks': config_runs[0].get('num_tasks', 0)
                }
                
                # 计算统计指标
                config_stats = {}
                for metric_name, values in metrics.items():
                    if isinstance(values, list) and len(values) > 0:
                        config_stats[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'median': np.median(values),
                            'q25': np.percentile(values, 25),
                            'q75': np.percentile(values, 75),
                            'values': values
                        }
                    else:
                        config_stats[metric_name] = values
                
                type_stats[config_name] = config_stats
            
            summary_stats[exp_type] = type_stats
        
        # 保存汇总统计
        stats_file = self.analysis_output_dir / "summary_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            # 处理numpy类型以便JSON序列化
            serializable_stats = self._make_json_serializable(summary_stats)
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
        
        print(f"汇总统计信息已保存到: {stats_file}")
        return summary_stats
    
    def analyze_scalability(self) -> Dict[str, Any]:
        """分析系统扩展性"""
        print("分析系统扩展性...")
        
        scalability_data = self.experiment_data.get('scalability', {})
        if not scalability_data:
            return {}
        
        # 按智能体数量和任务数量分组
        agent_scalability = {}
        task_scalability = {}
        
        for config_name, runs in scalability_data.items():
            if not runs:
                continue
            
            # 提取配置参数
            num_agents = runs[0].get('num_agents', 0)
            num_tasks = runs[0].get('num_tasks', 0)
            
            # 计算性能指标
            performance_metrics = {
                'final_utility_mean': np.mean([r.get('final_utility', 0) for r in runs]),
                'final_utility_std': np.std([r.get('final_utility', 0) for r in runs]),
                'task_completion_rate_mean': np.mean([r.get('task_completion_rate', 0) for r in runs]),
                'task_completion_rate_std': np.std([r.get('task_completion_rate', 0) for r in runs]),
                'messages_per_agent': np.mean([r.get('total_messages_broadcasted', 0) / num_agents for r in runs]) if num_agents > 0 else 0,
                'convergence_speed_mean': np.mean([r.get('belief_convergence_speed', 0) for r in runs]),
                'simulation_time_mean': np.mean([r.get('simulation_time', 0) for r in runs])
            }
            
            # 按智能体数量分组
            if 'scalability_agents' in config_name:
                agent_scalability[num_agents] = performance_metrics
            
            # 按任务数量分组
            elif 'scalability_tasks' in config_name:
                task_scalability[num_tasks] = performance_metrics
        
        scalability_analysis = {
            'agent_scalability': agent_scalability,
            'task_scalability': task_scalability
        }
        
        # 计算扩展性趋势
        if agent_scalability:
            agent_nums = sorted(agent_scalability.keys())
            utilities = [agent_scalability[n]['final_utility_mean'] for n in agent_nums]
            completion_rates = [agent_scalability[n]['task_completion_rate_mean'] for n in agent_nums]
            
            # 线性回归分析趋势
            if len(agent_nums) > 2:
                utility_slope, _, utility_r, utility_p, _ = stats.linregress(agent_nums, utilities)
                completion_slope, _, completion_r, completion_p, _ = stats.linregress(agent_nums, completion_rates)
                
                scalability_analysis['agent_trends'] = {
                    'utility_slope': utility_slope,
                    'utility_correlation': utility_r,
                    'utility_p_value': utility_p,
                    'completion_slope': completion_slope,
                    'completion_correlation': completion_r,
                    'completion_p_value': completion_p
                }
        
        if task_scalability:
            task_nums = sorted(task_scalability.keys())
            utilities = [task_scalability[n]['final_utility_mean'] for n in task_nums]
            completion_rates = [task_scalability[n]['task_completion_rate_mean'] for n in task_nums]
            
            if len(task_nums) > 2:
                utility_slope, _, utility_r, utility_p, _ = stats.linregress(task_nums, utilities)
                completion_slope, _, completion_r, completion_p, _ = stats.linregress(task_nums, completion_rates)
                
                scalability_analysis['task_trends'] = {
                    'utility_slope': utility_slope,
                    'utility_correlation': utility_r,
                    'utility_p_value': utility_p,
                    'completion_slope': completion_slope,
                    'completion_correlation': completion_r,
                    'completion_p_value': completion_p
                }
        
        return scalability_analysis
    
    def analyze_k_value_impact(self) -> Dict[str, Any]:
        """分析K值对系统性能的影响"""
        print("分析K值影响...")
        
        comparison_data = self.experiment_data.get('comparison', {})
        if not comparison_data:
            return {}
        
        k_value_data = {}
        
        # 提取K值相关的实验数据
        for config_name, runs in comparison_data.items():
            if 'k_value_comparison' in config_name and runs:
                # 从配置名中提取K值
                k_value = None
                parts = config_name.split('_')
                for part in parts:
                    if part.startswith('k') and part[1:].isdigit():
                        k_value = int(part[1:])
                        break
                
                if k_value is not None:
                    performance_metrics = {
                        'final_utility': [r.get('final_utility', 0) for r in runs],
                        'task_completion_rate': [r.get('task_completion_rate', 0) for r in runs],
                        'belief_convergence_speed': [r.get('belief_convergence_speed', 0) for r in runs],
                        'total_messages_broadcasted': [r.get('total_messages_broadcasted', 0) for r in runs],
                        'belief_consensus_level': [r.get('belief_consensus_level', 0) for r in runs],
                        'simulation_time': [r.get('simulation_time', 0) for r in runs]
                    }
                    
                    k_value_data[k_value] = performance_metrics
        
        if not k_value_data:
            return {}
        
        # 分析K值影响
        k_analysis = {'k_value_data': k_value_data}
        
        # 计算不同K值下的性能趋势
        k_values = sorted(k_value_data.keys())
        
        metrics_analysis = {}
        for metric_name in ['final_utility', 'task_completion_rate', 'belief_convergence_speed', 'total_messages_broadcasted']:
            metric_means = []
            metric_stds = []
            
            for k in k_values:
                values = k_value_data[k][metric_name]
                metric_means.append(np.mean(values))
                metric_stds.append(np.std(values))
            
            # 线性回归分析
            if len(k_values) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(k_values, metric_means)
                
                metrics_analysis[metric_name] = {
                    'means': metric_means,
                    'stds': metric_stds,
                    'slope': slope,
                    'intercept': intercept,
                    'correlation': r_value,
                    'p_value': p_value,
                    'std_error': std_err
                }
        
        k_analysis['metrics_analysis'] = metrics_analysis
        
        # 找出最优K值
        utility_means = [np.mean(k_value_data[k]['final_utility']) for k in k_values]
        best_k_idx = np.argmax(utility_means)
        k_analysis['optimal_k_value'] = k_values[best_k_idx]
        k_analysis['optimal_k_utility'] = utility_means[best_k_idx]
        
        return k_analysis
    
    def analyze_belief_convergence_detailed(self) -> Dict[str, Any]:
        """详细分析信念收敛过程"""
        print("详细分析信念收敛...")
        
        convergence_analysis = {}
        
        for exp_type, type_data in self.experiment_data.items():
            type_convergence = {}
            
            for config_name, runs in type_data.items():
                if not runs:
                    continue
                
                config_convergence = []
                
                for run in runs:
                    env_data = run.get('env_data', {})
                    belief_history = env_data.get('history_beliefs_at_ksc_step', {})
                    tasks = env_data.get('tasks', {})
                    
                    if belief_history and tasks:
                        run_convergence = self._analyze_single_run_convergence(belief_history, tasks)
                        config_convergence.append(run_convergence)
                
                if config_convergence:
                    # 汇总该配置的收敛分析
                    type_convergence[config_name] = self._aggregate_convergence_analysis(config_convergence)
            
            convergence_analysis[exp_type] = type_convergence
        
        return convergence_analysis
    
    def _analyze_single_run_convergence(self, belief_history: Dict, tasks: Dict) -> Dict[str, Any]:
        """分析单次运行的信念收敛"""
        convergence_metrics = {
            'convergence_times': {},
            'final_accuracies': {},
            'convergence_rates': {},
            'consensus_evolution': {}
        }
        
        for task_id, task_info in tasks.items():
            true_type_idx = task_info.get('true_type_idx', -1)
            if true_type_idx == -1:
                continue
            
            # 收集所有智能体对此任务的信念
            agent_beliefs = {}
            rounds = sorted(belief_history.keys())
            
            for round_num in rounds:
                round_data = belief_history[round_num]
                for agent_id, agent_data in round_data.items():
                    if task_id in agent_data and agent_data[task_id] is not None:
                        if agent_id not in agent_beliefs:
                            agent_beliefs[agent_id] = []
                        
                        belief_dist = agent_data[task_id]
                        if len(belief_dist) > true_type_idx:
                            agent_beliefs[agent_id].append({
                                'round': round_num,
                                'true_type_belief': belief_dist[true_type_idx],
                                'full_distribution': belief_dist
                            })
            
            # 分析收敛时间和最终准确性
            task_convergence = {}
            for agent_id, belief_seq in agent_beliefs.items():
                if len(belief_seq) < 3:
                    continue
                
                # 计算收敛时间（信念变化小于阈值的时间点）
                convergence_time = self._find_convergence_time(belief_seq)
                final_accuracy = belief_seq[-1]['true_type_belief']
                
                # 计算收敛速率
                convergence_rate = self._calculate_convergence_rate(belief_seq)
                
                task_convergence[agent_id] = {
                    'convergence_time': convergence_time,
                    'final_accuracy': final_accuracy,
                    'convergence_rate': convergence_rate
                }
            
            if task_convergence:
                convergence_metrics['convergence_times'][task_id] = task_convergence
                convergence_metrics['final_accuracies'][task_id] = {
                    agent_id: data['final_accuracy'] 
                    for agent_id, data in task_convergence.items()
                }
                
                # 计算智能体间的共识演化
                consensus_evolution = self._calculate_consensus_evolution(agent_beliefs)
                convergence_metrics['consensus_evolution'][task_id] = consensus_evolution
        
        return convergence_metrics
    
    def _find_convergence_time(self, belief_sequence: List[Dict], threshold: float = 0.05) -> int:
        """找到信念收敛的时间点"""
        if len(belief_sequence) < 5:
            return len(belief_sequence)
        
        # 从后往前找，寻找最后一次大幅变化
        for i in range(len(belief_sequence) - 3, 0, -1):
            current_belief = belief_sequence[i]['true_type_belief']
            next_belief = belief_sequence[i + 1]['true_type_belief']
            
            if abs(current_belief - next_belief) > threshold:
                return i + 2  # 收敛发生在大幅变化之后
        
        return 1  # 如果没有找到大幅变化，认为很快就收敛了
    
    def _calculate_convergence_rate(self, belief_sequence: List[Dict]) -> float:
        """计算信念收敛速率"""
        if len(belief_sequence) < 2:
            return 0.0
        
        # 计算信念变化的衰减率
        changes = []
        for i in range(1, len(belief_sequence)):
            prev_belief = belief_sequence[i-1]['true_type_belief']
            curr_belief = belief_sequence[i]['true_type_belief']
            changes.append(abs(curr_belief - prev_belief))
        
        if not changes:
            return 0.0
        
        # 使用指数衰减模型拟合收敛速率
        rounds = list(range(len(changes)))
        if len(rounds) > 3:
            try:
                # 简单的衰减率计算
                early_changes = np.mean(changes[:len(changes)//2])
                late_changes = np.mean(changes[len(changes)//2:])
                
                if early_changes > 0:
                    decay_rate = (early_changes - late_changes) / early_changes
                    return max(0, decay_rate)
            except:
                pass
        
        return np.mean(changes)
    
    def _calculate_consensus_evolution(self, agent_beliefs: Dict) -> List[float]:
        """计算智能体间共识的演化过程"""
        if len(agent_beliefs) < 2:
            return []
        
        # 获取所有轮次
        all_rounds = set()
        for belief_seq in agent_beliefs.values():
            for belief_point in belief_seq:
                all_rounds.add(belief_point['round'])
        
        sorted_rounds = sorted(all_rounds)
        consensus_evolution = []
        
        for round_num in sorted_rounds:
            # 获取该轮次所有智能体的信念
            round_beliefs = []
            for belief_seq in agent_beliefs.values():
                for belief_point in belief_seq:
                    if belief_point['round'] == round_num:
                        round_beliefs.append(belief_point['true_type_belief'])
                        break
            
            if len(round_beliefs) > 1:
                # 计算共识程度（1 - 标准差）
                consensus = 1.0 - np.std(round_beliefs)
                consensus_evolution.append(max(0, consensus))
            else:
                consensus_evolution.append(1.0 if round_beliefs else 0.0)
        
        return consensus_evolution
    
    def _aggregate_convergence_analysis(self, convergence_list: List[Dict]) -> Dict[str, Any]:
        """汇总多次运行的收敛分析"""
        if not convergence_list:
            return {}
        
        # 汇总最终准确性
        all_final_accuracies = []
        all_convergence_times = []
        all_convergence_rates = []
        
        for run_convergence in convergence_list:
            final_accuracies = run_convergence.get('final_accuracies', {})
            convergence_times = run_convergence.get('convergence_times', {})
            
            for task_id, agent_accuracies in final_accuracies.items():
                all_final_accuracies.extend(agent_accuracies.values())
            
            for task_id, agent_times in convergence_times.items():
                for agent_id, agent_data in agent_times.items():
                    all_convergence_times.append(agent_data['convergence_time'])
                    all_convergence_rates.append(agent_data['convergence_rate'])
        
        aggregated = {}
        if all_final_accuracies:
            aggregated['final_accuracy'] = {
                'mean': np.mean(all_final_accuracies),
                'std': np.std(all_final_accuracies),
                'min': np.min(all_final_accuracies),
                'max': np.max(all_final_accuracies)
            }
        
        if all_convergence_times:
            aggregated['convergence_time'] = {
                'mean': np.mean(all_convergence_times),
                'std': np.std(all_convergence_times),
                'min': np.min(all_convergence_times),
                'max': np.max(all_convergence_times)
            }
        
        if all_convergence_rates:
            aggregated['convergence_rate'] = {
                'mean': np.mean(all_convergence_rates),
                'std': np.std(all_convergence_rates),
                'min': np.min(all_convergence_rates),
                'max': np.max(all_convergence_rates)
            }
        
        return aggregated
    
    def perform_statistical_tests(self) -> Dict[str, Any]:
        """执行统计显著性检验"""
        print("执行统计显著性检验...")
        
        statistical_results = {}
        
        # K值影响的显著性检验
        k_value_data = self.analyze_k_value_impact().get('k_value_data', {})
        if len(k_value_data) > 2:
            k_values = sorted(k_value_data.keys())
            
            # 对不同K值的效用进行方差分析(ANOVA)
            utility_groups = [k_value_data[k]['final_utility'] for k in k_values]
            
            try:
                f_stat, p_value = stats.f_oneway(*utility_groups)
                statistical_results['k_value_anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                # 两两比较（Tukey HSD）
                pairwise_comparisons = {}
                for i, k1 in enumerate(k_values):
                    for j, k2 in enumerate(k_values[i+1:], i+1):
                        stat, p_val = stats.ttest_ind(
                            k_value_data[k1]['final_utility'],
                            k_value_data[k2]['final_utility']
                        )
                        pairwise_comparisons[f'K{k1}_vs_K{k2}'] = {
                            't_statistic': stat,
                            'p_value': p_val,
                            'significant': p_val < 0.05
                        }
                
                statistical_results['k_value_pairwise'] = pairwise_comparisons
                
            except Exception as e:
                print(f"K值统计检验失败: {e}")
        
        # 扩展性分析的显著性检验
        scalability_analysis = self.analyze_scalability()
        
        # 智能体数量影响的相关性检验
        agent_scalability = scalability_analysis.get('agent_scalability', {})
        if len(agent_scalability) > 3:
            agent_nums = sorted(agent_scalability.keys())
            utilities = [agent_scalability[n]['final_utility_mean'] for n in agent_nums]
            
            correlation, p_value = stats.pearsonr(agent_nums, utilities)
            statistical_results['agent_scalability_correlation'] = {
                'correlation': correlation,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # 任务数量影响的相关性检验
        task_scalability = scalability_analysis.get('task_scalability', {})
        if len(task_scalability) > 3:
            task_nums = sorted(task_scalability.keys())
            utilities = [task_scalability[n]['final_utility_mean'] for n in task_nums]
            
            correlation, p_value = stats.pearsonr(task_nums, utilities)
            statistical_results['task_scalability_correlation'] = {
                'correlation': correlation,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return statistical_results
    
    def generate_analysis_report(self) -> str:
        """生成分析报告"""
        print("生成分析报告...")
        
        # 执行所有分析
        scalability_analysis = self.analyze_scalability()
        k_value_analysis = self.analyze_k_value_impact()
        convergence_analysis = self.analyze_belief_convergence_detailed()
        statistical_tests = self.perform_statistical_tests()
        
        # 生成报告
        report_lines = [
            "# 强通信信念自学习实验分析报告",
            "",
            f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. 实验概览",
            ""
        ]
        
        # 添加基本统计信息
        total_configs = sum(len(configs) for configs in self.experiment_data.values())
        total_runs = sum(len(runs) for type_data in self.experiment_data.values() 
                        for runs in type_data.values())
        
        report_lines.extend([
            f"- 实验类型数量: {len(self.experiment_data)}",
            f"- 总配置数量: {total_configs}",
            f"- 总运行次数: {total_runs}",
            ""
        ])
        
        # 添加扩展性分析结果
        if scalability_analysis:
            report_lines.extend([
                "## 2. 扩展性分析",
                "",
                "### 2.1 智能体数量扩展性",
                ""
            ])
            
            agent_scalability = scalability_analysis.get('agent_scalability', {})
            if agent_scalability:
                for num_agents in sorted(agent_scalability.keys()):
                    metrics = agent_scalability[num_agents]
                    report_lines.append(
                        f"- {num_agents}个智能体: 平均效用 {metrics['final_utility_mean']:.2f}, "
                        f"任务完成率 {metrics['task_completion_rate_mean']:.2%}"
                    )
                
                agent_trends = scalability_analysis.get('agent_trends', {})
                if agent_trends:
                    report_lines.extend([
                        "",
                        f"智能体扩展性趋势:",
                        f"- 效用斜率: {agent_trends['utility_slope']:.4f} (p={agent_trends['utility_p_value']:.4f})",
                        f"- 完成率斜率: {agent_trends['completion_slope']:.4f} (p={agent_trends['completion_p_value']:.4f})",
                        ""
                    ])
        
        # 添加K值分析结果
        if k_value_analysis:
            report_lines.extend([
                "## 3. K值影响分析",
                ""
            ])
            
            optimal_k = k_value_analysis.get('optimal_k_value')
            if optimal_k is not None:
                optimal_utility = k_value_analysis.get('optimal_k_utility', 0)
                report_lines.append(f"最优K值: {optimal_k} (效用: {optimal_utility:.2f})")
                report_lines.append("")
            
            metrics_analysis = k_value_analysis.get('metrics_analysis', {})
            for metric_name, analysis in metrics_analysis.items():
                correlation = analysis.get('correlation', 0)
                p_value = analysis.get('p_value', 1.0)
                significant = "显著" if p_value < 0.05 else "不显著"
                
                report_lines.append(
                    f"- {metric_name}: 相关性 {correlation:.3f} (p={p_value:.4f}, {significant})"
                )
            
            report_lines.append("")
        
        # 添加统计检验结果
        if statistical_tests:
            report_lines.extend([
                "## 4. 统计显著性检验",
                ""
            ])
            
            # K值ANOVA结果
            k_anova = statistical_tests.get('k_value_anova', {})
            if k_anova:
                significant = "显著" if k_anova.get('significant', False) else "不显著"
                report_lines.extend([
                    f"K值效应ANOVA: F={k_anova.get('f_statistic', 0):.4f}, "
                    f"p={k_anova.get('p_value', 1):.4f} ({significant})",
                    ""
                ])
            
            # 扩展性相关性结果
            agent_corr = statistical_tests.get('agent_scalability_correlation', {})
            if agent_corr:
                significant = "显著" if agent_corr.get('significant', False) else "不显著"
                report_lines.append(
                    f"智能体数量-效用相关性: r={agent_corr.get('correlation', 0):.3f}, "
                    f"p={agent_corr.get('p_value', 1):.4f} ({significant})"
                )
            
            task_corr = statistical_tests.get('task_scalability_correlation', {})
            if task_corr:
                significant = "显著" if task_corr.get('significant', False) else "不显著"
                report_lines.append(
                    f"任务数量-效用相关性: r={task_corr.get('correlation', 0):.3f}, "
                    f"p={task_corr.get('p_value', 1):.4f} ({significant})"
                )
        
        # 保存报告
        report_content = "\n".join(report_lines)
        report_file = self.analysis_output_dir / "analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"分析报告已保存到: {report_file}")
        return report_content
    
    def _make_json_serializable(self, obj):
        """使对象可以JSON序列化"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def save_processed_data(self):
        """保存处理后的数据"""
        print("保存处理后的数据...")
        
        # 保存扩展性分析
        scalability_file = self.analysis_output_dir / "scalability_analysis.json"
        scalability_data = self.analyze_scalability()
        with open(scalability_file, 'w', encoding='utf-8') as f:
            json.dump(self._make_json_serializable(scalability_data), f, indent=2, ensure_ascii=False)
        
        # 保存K值分析
        k_value_file = self.analysis_output_dir / "k_value_analysis.json"
        k_value_data = self.analyze_k_value_impact()
        with open(k_value_file, 'w', encoding='utf-8') as f:
            json.dump(self._make_json_serializable(k_value_data), f, indent=2, ensure_ascii=False)
        
        # 保存信念收敛分析
        convergence_file = self.analysis_output_dir / "convergence_analysis.json"
        convergence_data = self.analyze_belief_convergence_detailed()
        with open(convergence_file, 'w', encoding='utf-8') as f:
            json.dump(self._make_json_serializable(convergence_data), f, indent=2, ensure_ascii=False)
        
        # 保存统计检验结果
        stats_file = self.analysis_output_dir / "statistical_tests.json"
        stats_data = self.perform_statistical_tests()
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self._make_json_serializable(stats_data), f, indent=2, ensure_ascii=False)
        
        print("所有分析数据已保存")

def main():
    """主函数"""
    experiment_dir = "results/strong_comm_belief_experiments"
    
    if not Path(experiment_dir).exists():
        print(f"实验目录不存在: {experiment_dir}")
        print("请先运行实验程序生成数据")
        return
    
    # 创建数据分析器
    analyzer = BeliefLearningDataAnalyzer(experiment_dir)
    
    # 生成分析报告
    report = analyzer.generate_analysis_report()
    
    # 保存处理后的数据
    analyzer.save_processed_data()
    
    print("数据分析完成！")

if __name__ == "__main__":
    main()