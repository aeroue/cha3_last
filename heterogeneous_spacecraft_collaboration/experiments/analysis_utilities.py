# experiments/analysis_utilities.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import pickle
import json

# 导入项目路径
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
_project_root_directory = os.path.dirname(_current_file_directory)
if _project_root_directory not in sys.path:
    sys.path.insert(0, _project_root_directory)

class ResultsAnalyzer:
    """
    实验结果分析工具类：用于聚合和分析实验结果数据
    """
    def __init__(self, results_base_dir: str = "results/experiments"):
        """
        初始化结果分析器
        
        参数:
            results_base_dir (str): 实验结果的基础目录
        """
        self.results_base_dir = results_base_dir
    
    def load_experiment_results(self, 
                               experiment_dir: str, 
                               file_pattern: str = "*_metrics.csv") -> pd.DataFrame:
        """
        加载实验结果数据
        
        参数:
            experiment_dir (str): 实验目录名称，相对于results_base_dir
            file_pattern (str): 文件模式，用于筛选结果文件
            
        返回:
            pd.DataFrame: 包含实验结果的DataFrame
        """
        import glob
        
        # 构建完整路径
        full_dir_path = os.path.join(self.results_base_dir, experiment_dir)
        
        if not os.path.exists(full_dir_path):
            print(f"警告: 实验目录 '{full_dir_path}' 不存在")
            return pd.DataFrame()
        
        # 找到所有匹配的文件
        file_pattern_path = os.path.join(full_dir_path, file_pattern)
        matching_files = glob.glob(file_pattern_path)
        
        if not matching_files:
            print(f"警告: 在目录 '{full_dir_path}' 中未找到匹配 '{file_pattern}' 的文件")
            return pd.DataFrame()
        
        # 读取并合并所有CSV文件
        dfs = []
        for file_path in matching_files:
            try:
                df = pd.read_csv(file_path)
                # 添加文件名作为来源标识
                df['source_file'] = os.path.basename(file_path)
                dfs.append(df)
            except Exception as e:
                print(f"警告: 无法读取文件 '{file_path}': {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        # 合并所有数据帧
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    
    def load_simulation_data(self, 
                           results_dir: str, 
                           scenario_name: str, 
                           run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        加载单次仿真的原始数据（从pickle文件）
        
        参数:
            results_dir (str): 结果目录，相对于results_base_dir
            scenario_name (str): 场景名称
            run_id (str, optional): 运行ID，如果未提供则查找最新的
            
        返回:
            Dict[str, Any]: 仿真结果数据
        """
        # 构建完整路径
        full_dir_path = os.path.join(self.results_base_dir, results_dir)
        
        if not os.path.exists(full_dir_path):
            print(f"警告: 结果目录 '{full_dir_path}' 不存在")
            return {}
        
        # 查找匹配的pickle文件
        import glob
        
        if run_id:
            pickle_pattern = f"full_results_{scenario_name}_{run_id}.pkl"
        else:
            pickle_pattern = f"full_results_{scenario_name}_*.pkl"
            
        matching_files = glob.glob(os.path.join(full_dir_path, pickle_pattern))
        
        if not matching_files:
            print(f"警告: 未找到匹配 '{pickle_pattern}' 的仿真结果文件")
            return {}
        
        # 如果没有指定run_id，选择最新的文件
        if not run_id:
            matching_files.sort(key=os.path.getmtime, reverse=True)
        
        # 加载pickle文件
        try:
            with open(matching_files[0], 'rb') as f:
                sim_data = pickle.load(f)
                return sim_data
        except Exception as e:
            print(f"警告: 无法加载仿真数据文件 '{matching_files[0]}': {e}")
            return {}
    
    def extract_perf_metrics(self, 
                           sim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从仿真数据中提取关键性能指标
        
        参数:
            sim_data (Dict[str, Any]): 仿真结果数据
            
        返回:
            Dict[str, Any]: 关键性能指标
        """
        if not sim_data:
            return {}
        
        # 提取关键指标
        metrics = {
            "scenario_name": sim_data.get("scenario_name", "未知"),
            "total_steps": sim_data.get("total_simulation_steps_run", 0),
            "final_time": sim_data.get("final_simulation_time_s", 0),
            "total_ksc_rounds": sim_data.get("total_ksc_decision_rounds", 0)
        }
        
        # 计算已完成任务数量
        completed_tasks = 0
        for task_id, task_data in sim_data.get("tasks_final_state", {}).items():
            if task_data.get("status") == "completed":
                completed_tasks += 1
        metrics["completed_tasks"] = completed_tasks
        
        # 提取最终KSC效用
        if "history_global_utility_ksc" in sim_data and sim_data["history_global_utility_ksc"]:
            metrics["final_ksc_utility"] = sim_data["history_global_utility_ksc"][-1]
        else:
            metrics["final_ksc_utility"] = None
        
        # 计算平均通信负载
        if "ksc_stats_messages_broadcasted" in sim_data and sim_data["ksc_stats_messages_broadcasted"]:
            metrics["avg_messages_broadcasted"] = np.mean(sim_data["ksc_stats_messages_broadcasted"])
        else:
            metrics["avg_messages_broadcasted"] = None
            
        if "ksc_stats_messages_processed" in sim_data and sim_data["ksc_stats_messages_processed"]:
            metrics["avg_messages_processed"] = np.mean(sim_data["ksc_stats_messages_processed"])
        else:
            metrics["avg_messages_processed"] = None
        
        # 计算平均活动通信链接数
        if "ksc_stats_active_links_ksc_graph" in sim_data and sim_data["ksc_stats_active_links_ksc_graph"]:
            metrics["avg_active_links"] = np.mean(sim_data["ksc_stats_active_links_ksc_graph"])
        elif "ksc_stats_active_links" in sim_data and sim_data["ksc_stats_active_links"]:
            metrics["avg_active_links"] = np.mean(sim_data["ksc_stats_active_links"])
        else:
            metrics["avg_active_links"] = None
        
        return metrics
    
    def analyze_ksc_convergence(self, 
                              sim_data: Dict[str, Any],
                              threshold_percentage: float = 0.95) -> Dict[str, Any]:
        """
        分析KSC算法的收敛特性
        
        参数:
            sim_data (Dict[str, Any]): 仿真结果数据
            threshold_percentage (float): 收敛阈值，相对于最终值的百分比
            
        返回:
            Dict[str, Any]: 收敛分析结果
        """
        if not sim_data or "history_global_utility_ksc" not in sim_data or not sim_data["history_global_utility_ksc"]:
            return {}
        
        utility_history = sim_data["history_global_utility_ksc"]
        final_utility = utility_history[-1]
        
        # 如果最终效用为零，无法计算收敛
        if abs(final_utility) < 1e-6:
            return {
                "rounds_to_converge": len(utility_history),
                "converged": False,
                "convergence_percentage": 0.0
            }
        
        # 计算达到不同百分比阈值的轮次
        thresholds = {
            "50%": 0.5 * final_utility,
            "75%": 0.75 * final_utility,
            "90%": 0.9 * final_utility,
            "95%": 0.95 * final_utility,
            "99%": 0.99 * final_utility
        }
        
        convergence_rounds = {}
        for label, threshold in thresholds.items():
            for i, value in enumerate(utility_history):
                if value >= threshold:
                    convergence_rounds[label] = i + 1
                    break
            else:
                convergence_rounds[label] = len(utility_history)
        
        # 计算收敛率（每轮的平均效用增长）
        if len(utility_history) > 1:
            utility_diffs = np.diff(utility_history)
            positive_diffs = [diff for diff in utility_diffs if diff > 0]
            if positive_diffs:
                avg_improvement_rate = np.mean(positive_diffs)
            else:
                avg_improvement_rate = 0.0
        else:
            avg_improvement_rate = 0.0
        
        return {
            "rounds_to_converge": convergence_rounds,
            "converged": convergence_rounds[f"{int(threshold_percentage*100)}%"] < len(utility_history),
            "avg_improvement_rate": avg_improvement_rate,
            "utility_history": utility_history
        }
    
    def compare_belief_convergence(self, 
                                 sim_data_list: List[Dict[str, Any]],
                                 agent_task_pairs: List[Tuple[str, str]]) -> Dict[str, Dict[str, List[float]]]:
        """
        比较不同仿真中信念收敛的速度
        
        参数:
            sim_data_list (List[Dict[str, Any]]): 仿真结果数据列表
            agent_task_pairs (List[Tuple[str, str]]): 要分析的(智能体ID, 任务ID)对列表
            
        返回:
            Dict[str, Dict[str, List[float]]]: 信念收敛分析结果
        """
        results = {}
        
        for i, sim_data in enumerate(sim_data_list):
            scenario_name = sim_data.get("scenario_name", f"Scenario_{i+1}")
            results[scenario_name] = {}
            
            history_beliefs = sim_data.get("history_beliefs_at_ksc_step", {})
            if not history_beliefs:
                continue
                
            for agent_id, task_id in agent_task_pairs:
                # 查找任务的真实类型索引
                task_def = sim_data.get("tasks_final_state", {}).get(task_id, {})
                true_type_idx = task_def.get("true_type_idx", -1)
                
                if true_type_idx < 0:
                    continue
                    
                # 提取信念历史
                belief_history = []
                rounds = sorted(history_beliefs.keys())
                for round_idx in rounds:
                    if agent_id in history_beliefs[round_idx] and task_id in history_beliefs[round_idx][agent_id]:
                        belief = history_beliefs[round_idx][agent_id][task_id]
                        if true_type_idx < len(belief):
                            belief_history.append(belief[true_type_idx])
                
                if belief_history:
                    pair_key = f"{agent_id}_{task_id}"
                    results[scenario_name][pair_key] = belief_history
        
        return results
    
    def analyze_adaptive_switching(self, 
                                 sim_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        分析自适应控制器的模式切换行为
        
        参数:
            sim_data (Dict[str, Any]): 仿真结果数据
            
        返回:
            Dict[str, Dict[str, Any]]: 模式切换分析结果
        """
        if not sim_data or "agents_history" not in sim_data:
            return {}
        
        results = {}
        
        for agent_id, agent_data in sim_data.get("agents_history", {}).items():
            if "history_decision_mode" not in agent_data or "history_csi_smooth" not in agent_data:
                continue
                
            modes = agent_data["history_decision_mode"]
            csi_values = agent_data["history_csi_smooth"]
            
            # 确保数据长度匹配
            min_len = min(len(modes), len(csi_values))
            if min_len == 0:
                continue
                
            modes = modes[:min_len]
            csi_values = csi_values[:min_len]
            
            # 计算各模式占比
            mode_counts = {}
            for mode in modes:
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                
            mode_percentages = {mode: count / len(modes) * 100 for mode, count in mode_counts.items()}
            
            # 计算模式切换次数
            mode_switches = 0
            for i in range(1, len(modes)):
                if modes[i] != modes[i-1]:
                    mode_switches += 1
            
            # 计算CSI统计信息
            csi_stats = {
                "mean": np.mean(csi_values),
                "min": np.min(csi_values),
                "max": np.max(csi_values),
                "std": np.std(csi_values)
            }
            
            results[agent_id] = {
                "mode_percentages": mode_percentages,
                "mode_switches": mode_switches,
                "csi_stats": csi_stats,
                "modes": modes,
                "csi_values": csi_values
            }
        
        return results
    
    def generate_summary_report(self, 
                              experiment_name: str,
                              output_file: Optional[str] = None) -> str:
        """
        生成实验摘要报告
        
        参数:
            experiment_name (str): 实验名称（目录名）
            output_file (str, optional): 输出文件路径，如果未提供则使用标准名称
            
        返回:
            str: 报告文本
        """
        # 加载实验结果
        df = self.load_experiment_results(experiment_name)
        
        if df.empty:
            return "无法加载实验结果数据。"
        
        # 构建报告
        report = f"# {experiment_name} 实验结果摘要\n\n"
        report += f"## 实验基本信息\n\n"
        report += f"- 实验名称: {experiment_name}\n"
        report += f"- 数据文件数: {df['source_file'].nunique()}\n"
        report += f"- 总记录数: {len(df)}\n\n"
        
        # 分析场景
        if 'scenario' in df.columns:
            scenarios = df['scenario'].unique()
            report += f"- 测试场景数: {len(scenarios)}\n"
            report += f"- 场景列表: {', '.join(scenarios)}\n\n"
        
        # 分析运行次数
        if 'run' in df.columns:
            runs = df['run'].unique()
            report += f"- 每个场景的运行次数: {len(runs)}\n\n"
        
        # 分析关键指标
        report += f"## 性能指标统计\n\n"
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['run', 'k_value', 'comm_range']:  # 排除非指标列
                stats = df[col].describe()
                report += f"### {col}\n\n"
                report += f"- 均值: {stats['mean']:.2f}\n"
                report += f"- 标准差: {stats['std']:.2f}\n"
                report += f"- 最小值: {stats['min']:.2f}\n"
                report += f"- 最大值: {stats['max']:.2f}\n"
                report += f"- 中位数: {stats['50%']:.2f}\n\n"
        
        # 如果存在分组变量，添加分组统计
        group_vars = []
        if 'scenario' in df.columns:
            group_vars.append('scenario')
        if 'k_value' in df.columns:
            group_vars.append('k_value')
        if 'comm_range' in df.columns:
            group_vars.append('comm_range')
        
        if group_vars:
            report += f"## 分组统计\n\n"
            
            for group_var in group_vars:
                report += f"### 按 {group_var} 分组\n\n"
                
                grouped = df.groupby(group_var)
                
                for metric in [col for col in numeric_columns if col not in ['run', 'k_value', 'comm_range']]:
                    report += f"#### {metric}\n\n"
                    stats = grouped[metric].agg(['mean', 'std']).round(2)
                    
                    # 转换为Markdown表格
                    report += f"| {group_var} | 均值 | 标准差 |\n"
                    report += f"|---|---|---|\n"
                    
                    for idx, row in stats.iterrows():
                        report += f"| {idx} | {row['mean']} | {row['std']} |\n"
                    
                    report += "\n"
        
        # 保存报告
        if output_file is None:
            output_file = os.path.join(self.results_base_dir, experiment_name, "summary_report.md")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"摘要报告已保存到: {output_file}")
        
        return report

# 如果直接运行此脚本，执行一个简单的测试
if __name__ == "__main__":
    analyzer = ResultsAnalyzer()
    
    # 尝试加载一些实验结果
    print("可用的实验目录:")
    
    import glob
    experiment_dirs = glob.glob(os.path.join(analyzer.results_base_dir, "*"))
    
    for i, dir_path in enumerate(experiment_dirs):
        if os.path.isdir(dir_path):
            dir_name = os.path.basename(dir_path)
            print(f"  {i+1}. {dir_name}")
    
    # 如果有实验目录，选择第一个生成报告
    if experiment_dirs:
        first_dir = os.path.basename(experiment_dirs[0])
        if os.path.isdir(os.path.join(analyzer.results_base_dir, first_dir)):
            print(f"\n生成实验 '{first_dir}' 的摘要报告...")
            analyzer.generate_summary_report(first_dir)
    else:
        print("\n无可用的实验目录。请先运行一些实验。")