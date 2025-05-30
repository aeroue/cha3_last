# experiments/strong_comm_belief_learning/experiment_runner.py
"""
强通信信念自学习实验运行器
负责批量运行实验并收集数据
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import concurrent.futures
import multiprocessing as mp
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments"))

from simulation.environment import SimulationEnvironment
from simulation.scenarios import load_scenario, SCENARIOS
from experiment_config import BeliefLearningExperimentConfig

class BeliefLearningExperimentRunner:
    """强通信信念自学习实验运行器"""
    
    def __init__(self, 
                 output_dir: str = "results/strong_comm_belief_experiments",
                 num_runs_per_config: int = 5,
                 enable_parallel: bool = True,
                 max_workers: Optional[int] = None):
        """
        初始化实验运行器
        
        Args:
            output_dir: 输出目录
            num_runs_per_config: 每个配置运行的次数
            enable_parallel: 是否启用并行运行
            max_workers: 最大工作进程数
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_runs_per_config = num_runs_per_config
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers or min(mp.cpu_count() - 1, 4)
        
        self.config_generator = BeliefLearningExperimentConfig()
        
        # 实验结果存储
        self.experiment_results = {}
        
        # 日志设置
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / "experiment_log.txt"
        self.log_file = open(log_file, 'w', encoding='utf-8')
        self.log(f"实验运行器初始化完成，输出目录: {self.output_dir}")
        self.log(f"每个配置运行次数: {self.num_runs_per_config}")
        self.log(f"并行运行: {self.enable_parallel}, 最大工作进程: {self.max_workers}")
    
    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        if hasattr(self, 'log_file'):
            self.log_file.write(log_message + "\n")
            self.log_file.flush()
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """运行所有实验"""
        self.log("开始运行所有实验...")
        
        # 获取所有实验配置
        all_configs = self.config_generator.get_all_experiment_configs()
        
        total_experiments = sum(len(configs) for configs in all_configs.values()) * self.num_runs_per_config
        self.log(f"总计需要运行 {total_experiments} 个实验")
        
        all_results = {}
        experiment_count = 0
        
        for experiment_type, configs in all_configs.items():
            self.log(f"\n开始运行 {experiment_type} 实验类型...")
            
            type_results = {}
            
            if self.enable_parallel:
                type_results = self._run_experiments_parallel(experiment_type, configs)
            else:
                type_results = self._run_experiments_sequential(experiment_type, configs)
            
            all_results[experiment_type] = type_results
            experiment_count += len(configs) * self.num_runs_per_config
            
            self.log(f"{experiment_type} 实验类型完成，已完成 {experiment_count}/{total_experiments} 个实验")
        
        # 保存实验结果汇总
        self._save_experiment_summary(all_results)
        
        self.log("所有实验运行完成！")
        return all_results
    
    def _run_experiments_parallel(self, experiment_type: str, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """并行运行实验"""
        type_results = {}
        
        # 准备所有实验任务
        tasks = []
        for config in configs:
            for run_id in range(self.num_runs_per_config):
                task = (experiment_type, config, run_id)
                tasks.append(task)
        
        # 并行执行
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._run_single_experiment_static, task): task 
                for task in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                experiment_type, config, run_id = task
                
                try:
                    result = future.result()
                    config_name = config["scenario_name"]
                    
                    if config_name not in type_results:
                        type_results[config_name] = []
                    
                    type_results[config_name].append(result)
                    
                    self.log(f"完成实验: {experiment_type}/{config_name}/run_{run_id}")
                    
                except Exception as e:
                    self.log(f"实验失败: {task}, 错误: {str(e)}")
        
        return type_results
    
    def _run_experiments_sequential(self, experiment_type: str, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """顺序运行实验"""
        type_results = {}
        
        for config in configs:
            config_name = config["scenario_name"]
            config_results = []
            
            for run_id in range(self.num_runs_per_config):
                try:
                    result = self._run_single_experiment(experiment_type, config, run_id)
                    config_results.append(result)
                    
                    self.log(f"完成实验: {experiment_type}/{config_name}/run_{run_id}")
                    
                except Exception as e:
                    self.log(f"实验失败: {experiment_type}/{config_name}/run_{run_id}, 错误: {str(e)}")
            
            type_results[config_name] = config_results
        
        return type_results
    
    @staticmethod
    def _run_single_experiment_static(task_data: Tuple[str, Dict[str, Any], int]) -> Dict[str, Any]:
        """静态方法，用于并行执行单个实验"""
        experiment_type, config, run_id = task_data
        
        # 在新进程中创建运行器实例
        runner = BeliefLearningExperimentRunner()
        return runner._run_single_experiment(experiment_type, config, run_id)
    
    def _run_single_experiment(self, experiment_type: str, config: Dict[str, Any], run_id: int) -> Dict[str, Any]:
        """运行单个实验"""
        config_name = config["scenario_name"]
        
        # 设置随机种子
        random_seed = 12345 + run_id * 1000 + hash(config_name) % 1000
        np.random.seed(random_seed)
        
        # 创建输出目录
        experiment_output_dir = self.output_dir / experiment_type / config_name / f"run_{run_id:02d}"
        experiment_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config_file = experiment_output_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表以便JSON序列化
            config_serializable = self._make_json_serializable(config)
            json.dump(config_serializable, f, indent=2, ensure_ascii=False)
        
        # 临时添加场景到SCENARIOS
        temp_scenario_name = f"temp_{config_name}_{run_id}"
        SCENARIOS[temp_scenario_name] = config
        
        try:
            # 创建仿真环境
            env = SimulationEnvironment(temp_scenario_name)
            
            # 运行仿真
            start_time = time.time()
            
            step_count = 0
            max_steps = config["simulation_params"]["max_time_steps"]
            
            while step_count < max_steps:
                should_continue = env.step()
                step_count += 1
                
                if not should_continue:
                    break
            
            end_time = time.time()
            simulation_time = end_time - start_time
            
            # 收集结果数据
            result = self._collect_experiment_results(env, config, run_id, simulation_time)
            
            # 保存详细结果
            result_file = experiment_output_dir / "results.pkl"
            with open(result_file, 'wb') as f:
                pickle.dump(result, f)
            
            # 保存仿真环境状态
            env_state_file = experiment_output_dir / "env_state.pkl"
            with open(env_state_file, 'wb') as f:
                pickle.dump({
                    'history_global_utility_ksc': env.history_global_utility_ksc,
                    'history_assignments_ksc': env.history_assignments_ksc,
                    'history_beliefs_at_ksc_step': env.history_beliefs_at_ksc_step,
                    'ksc_stats_messages_broadcasted': env.ksc_stats_messages_broadcasted,
                    'ksc_stats_messages_processed': env.ksc_stats_messages_processed,
                    'ksc_stats_active_links': env.ksc_stats_active_links,
                    'agents': {aid: {
                        'history_states': agent.history_states,
                        'history_controls': agent.history_controls,
                        'history_decision_mode': agent.history_decision_mode,
                        'history_csi_smooth': agent.history_csi_smooth
                    } for aid, agent in env.agents.items()},
                    'tasks': env.tasks,
                    'final_time': env.current_time
                }, f)
            
            return result
            
        except Exception as e:
            error_result = {
                'config_name': config_name,
                'run_id': run_id,
                'success': False,
                'error': str(e),
                'final_utility': 0.0,
                'tasks_completed': 0,
                'simulation_time': 0.0
            }
            return error_result
            
        finally:
            # 清理临时场景
            if temp_scenario_name in SCENARIOS:
                del SCENARIOS[temp_scenario_name]
    
    def _collect_experiment_results(self, env: SimulationEnvironment, config: Dict[str, Any], run_id: int, simulation_time: float) -> Dict[str, Any]:
        """收集实验结果数据"""
        
        # 基础信息
        result = {
            'config_name': config["scenario_name"],
            'run_id': run_id,
            'success': True,
            'simulation_time': simulation_time,
            'random_seed': 12345 + run_id * 1000 + hash(config["scenario_name"]) % 1000,
            
            # 场景参数
            'num_agents': config["num_agents"],
            'num_tasks': config["num_tasks"],
            'final_time': env.current_time,
            'total_steps': env.time_step,
            'ksc_rounds': env.ksc_decision_round_count,
        }
        
        # KSC性能指标
        if env.history_global_utility_ksc:
            result.update({
                'final_utility': env.history_global_utility_ksc[-1],
                'max_utility': max(env.history_global_utility_ksc),
                'utility_convergence_steps': self._calculate_convergence_steps(env.history_global_utility_ksc),
                'utility_improvement': env.history_global_utility_ksc[-1] - env.history_global_utility_ksc[0] if len(env.history_global_utility_ksc) > 1 else 0.0
            })
        else:
            result.update({
                'final_utility': 0.0,
                'max_utility': 0.0,
                'utility_convergence_steps': -1,
                'utility_improvement': 0.0
            })
        
        # 任务完成情况
        completed_tasks = sum(1 for task in env.tasks.values() if task.get('current_completion_percentage', 0) >= 100.0)
        total_value_realized = sum(task.get('value_realized', 0.0) for task in env.tasks.values())
        
        result.update({
            'tasks_completed': completed_tasks,
            'task_completion_rate': completed_tasks / config["num_tasks"] if config["num_tasks"] > 0 else 0.0,
            'total_value_realized': total_value_realized,
            'average_task_completion': np.mean([task.get('current_completion_percentage', 0.0) for task in env.tasks.values()])
        })
        
        # 通信统计
        if env.ksc_stats_messages_broadcasted:
            result.update({
                'total_messages_broadcasted': sum(env.ksc_stats_messages_broadcasted),
                'average_messages_per_round': np.mean(env.ksc_stats_messages_broadcasted),
                'max_messages_per_round': max(env.ksc_stats_messages_broadcasted),
                'communication_efficiency': completed_tasks / sum(env.ksc_stats_messages_broadcasted) if sum(env.ksc_stats_messages_broadcasted) > 0 else 0.0
            })
        else:
            result.update({
                'total_messages_broadcasted': 0,
                'average_messages_per_round': 0.0,
                'max_messages_per_round': 0,
                'communication_efficiency': 0.0
            })
        
        # 信念收敛分析
        belief_convergence_metrics = self._analyze_belief_convergence(env.history_beliefs_at_ksc_step, env.tasks)
        result.update(belief_convergence_metrics)
        
        # 智能体性能
        agent_metrics = self._analyze_agent_performance(env.agents)
        result.update(agent_metrics)
        
        # 计算K值相关指标（如果适用）
        if 'k_value' in config.get('scenario_name', ''):
            k_value = self._extract_k_value_from_config(config)
            if k_value is not None:
                result['k_value'] = k_value
        
        return result
    
    def _calculate_convergence_steps(self, utility_history: List[float], threshold: float = 0.01) -> int:
        """计算收敛步数"""
        if len(utility_history) < 10:
            return -1
        
        final_utility = utility_history[-1]
        
        # 寻找最后一次显著变化的位置
        for i in range(len(utility_history) - 10, -1, -1):
            if abs(utility_history[i] - final_utility) > threshold * abs(final_utility):
                return i + 10
        
        return len(utility_history)
    
    def _analyze_belief_convergence(self, belief_history: Dict[int, Dict[str, Dict[str, np.ndarray]]], tasks: Dict[str, Any]) -> Dict[str, Any]:
        """分析信念收敛情况"""
        if not belief_history:
            return {
                'belief_convergence_speed': 0.0,
                'final_belief_accuracy': 0.0,
                'belief_consensus_level': 0.0
            }
        
        # 计算信念变化速度
        convergence_speeds = []
        accuracy_scores = []
        consensus_scores = []
        
        for task_id, task_info in tasks.items():
            true_type_idx = task_info.get('true_type_idx', -1)
            if true_type_idx == -1:
                continue
            
            # 收集所有智能体对此任务的信念历史
            agent_beliefs = {}
            for round_num, round_data in belief_history.items():
                for agent_id, agent_data in round_data.items():
                    if task_id in agent_data and agent_data[task_id] is not None:
                        if agent_id not in agent_beliefs:
                            agent_beliefs[agent_id] = []
                        
                        belief_dist = agent_data[task_id]
                        if len(belief_dist) > true_type_idx:
                            agent_beliefs[agent_id].append(belief_dist[true_type_idx])
            
            # 计算收敛速度
            for agent_id, belief_seq in agent_beliefs.items():
                if len(belief_seq) > 5:
                    # 计算信念变化的衰减率
                    changes = [abs(belief_seq[i+1] - belief_seq[i]) for i in range(len(belief_seq)-1)]
                    if changes:
                        convergence_speeds.append(np.mean(changes))
                    
                    # 计算最终准确性
                    final_belief = belief_seq[-1]
                    accuracy_scores.append(final_belief)
            
            # 计算智能体间的共识程度
            if len(agent_beliefs) > 1:
                final_beliefs = [beliefs[-1] for beliefs in agent_beliefs.values() if beliefs]
                if final_beliefs:
                    consensus_scores.append(1.0 - np.std(final_beliefs))
        
        return {
            'belief_convergence_speed': np.mean(convergence_speeds) if convergence_speeds else 0.0,
            'final_belief_accuracy': np.mean(accuracy_scores) if accuracy_scores else 0.0,
            'belief_consensus_level': np.mean(consensus_scores) if consensus_scores else 0.0
        }
    
    def _analyze_agent_performance(self, agents: Dict[str, Any]) -> Dict[str, Any]:
        """分析智能体性能"""
        total_distance = 0.0
        total_fuel_consumption = 0.0
        num_agents = len(agents)
        
        for agent in agents.values():
            if hasattr(agent, 'history_states') and agent.history_states:
                # 计算总移动距离
                states = np.array(agent.history_states)
                if states.ndim == 2 and states.shape[0] > 1:
                    distances = np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)
                    total_distance += np.sum(distances)
            
            if hasattr(agent, 'history_controls') and agent.history_controls:
                # 估算燃料消耗（基于控制输入）
                controls = np.array(agent.history_controls)
                if controls.ndim == 2:
                    fuel_consumption = np.sum(np.linalg.norm(controls, axis=1))
                    total_fuel_consumption += fuel_consumption
        
        return {
            'average_distance_traveled': total_distance / num_agents if num_agents > 0 else 0.0,
            'average_fuel_consumption': total_fuel_consumption / num_agents if num_agents > 0 else 0.0,
            'total_distance_traveled': total_distance,
            'total_fuel_consumption': total_fuel_consumption
        }
    
    def _extract_k_value_from_config(self, config: Dict[str, Any]) -> Optional[int]:
        """从配置中提取K值"""
        scenario_name = config.get('scenario_name', '')
        if 'k_value_comparison' in scenario_name:
            # 从场景名中提取K值
            parts = scenario_name.split('_')
            for part in parts:
                if part.startswith('k') and part[1:].isdigit():
                    return int(part[1:])
        
        # 从航天器配置中提取
        if 'spacecrafts' in config:
            k_values = [sc.get('ksc_params', {}).get('k_value') for sc in config['spacecrafts']]
            k_values = [k for k in k_values if k is not None]
            if k_values:
                return int(np.mean(k_values))
        
        return None
    
    def _make_json_serializable(self, obj):
        """使对象可以JSON序列化"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _save_experiment_summary(self, all_results: Dict[str, Any]):
        """保存实验结果汇总"""
        summary_file = self.output_dir / "experiment_summary.json"
        
        # 计算汇总统计
        summary = {
            'experiment_info': {
                'total_experiment_types': len(all_results),
                'total_configurations': sum(len(configs) for configs in all_results.values()),
                'runs_per_config': self.num_runs_per_config,
                'total_runs': sum(len(configs) * self.num_runs_per_config for configs in all_results.values()),
                'completion_time': datetime.now().isoformat()
            },
            'experiment_types': {}
        }
        
        for exp_type, type_results in all_results.items():
            type_summary = {
                'num_configurations': len(type_results),
                'configurations': {}
            }
            
            for config_name, config_results in type_results.items():
                # 计算每个配置的统计信息
                if config_results:
                    successful_runs = [r for r in config_results if r.get('success', False)]
                    
                    config_stats = {
                        'total_runs': len(config_results),
                        'successful_runs': len(successful_runs),
                        'success_rate': len(successful_runs) / len(config_results) if config_results else 0.0
                    }
                    
                    if successful_runs:
                        # 性能指标统计
                        metrics = ['final_utility', 'tasks_completed', 'task_completion_rate', 
                                  'total_messages_broadcasted', 'belief_convergence_speed']
                        
                        for metric in metrics:
                            values = [r.get(metric, 0) for r in successful_runs if metric in r]
                            if values:
                                config_stats[f'{metric}_mean'] = float(np.mean(values))
                                config_stats[f'{metric}_std'] = float(np.std(values))
                                config_stats[f'{metric}_max'] = float(np.max(values))
                                config_stats[f'{metric}_min'] = float(np.min(values))
                    
                    type_summary['configurations'][config_name] = config_stats
            
            summary['experiment_types'][exp_type] = type_summary
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.log(f"实验汇总已保存到: {summary_file}")
    
    def __del__(self):
        """析构函数，关闭日志文件"""
        if hasattr(self, 'log_file'):
            self.log_file.close()

def main():
    """主函数，运行所有实验"""
    print("开始强通信信念自学习实验...")
    
    # 创建实验运行器
    runner = BeliefLearningExperimentRunner(
        output_dir="results/strong_comm_belief_experiments",
        num_runs_per_config=3,  # 每个配置运行3次
        enable_parallel=True,
        max_workers=4
    )
    
    # 运行所有实验
    results = runner.run_all_experiments()
    
    print("所有实验完成！")
    return results

if __name__ == "__main__":
    main()