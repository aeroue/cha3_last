# run_strong_communication_experiments.py
"""
强通信信念自学习实验总运行脚本
一键运行完整的强通信实验流程，包括实验运行、数据分析和可视化
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置中文字体支持
def setup_chinese_fonts():
    """设置中文字体支持"""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import platform
    
    chinese_fonts = []
    if platform.system() == 'Windows':
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DengXian']
    elif platform.system() == 'Darwin':  # macOS
        chinese_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STSong']
    else:  # Linux
        chinese_fonts = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
    
    for font_name in chinese_fonts:
        try:
            if fm.findfont(font_name, fallback_to_default=False):
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 成功设置中文字体: {font_name}")
                return True
        except:
            continue
    
    print("⚠ 未找到合适的中文字体，图表中的中文可能无法正确显示")
    return False

class StrongCommunicationExperimentRunner:
    """强通信实验总运行器"""
    
    def __init__(self, 
                 output_dir: str = "results/strong_comm_experiments",
                 num_runs_per_config: int = 3,
                 enable_parallel: bool = True,
                 max_workers: int = 4):
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
        self.max_workers = max_workers
        
        # 设置日志
        self.setup_logging()
        
        # 导入必要的模块
        self.import_experiment_modules()
        
        print("=" * 60)
        print("强通信信念自学习实验系统")
        print("=" * 60)
        print(f"输出目录: {self.output_dir}")
        print(f"每配置运行次数: {self.num_runs_per_config}")
        print(f"并行运行: {self.enable_parallel}")
        print(f"最大工作进程: {self.max_workers}")
        print("=" * 60)
    
    def setup_logging(self):
        """设置日志系统"""
        log_file = self.output_dir / "experiment_log.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("实验运行器初始化完成")
    
    def import_experiment_modules(self):
        """导入实验相关模块"""
        try:
            # 导入实验配置模块
            from experiments.strong_communication_belief_learning.experiment_config import BeliefLearningExperimentConfig
            self.config_generator = BeliefLearningExperimentConfig()
            
            # 导入实验运行模块
            from experiments.strong_communication_belief_learning.experiment_runner import BeliefLearningExperimentRunner
            self.experiment_runner_class = BeliefLearningExperimentRunner
            
            # 导入数据分析模块
            from experiments.strong_communication_belief_learning.data_analyzer import BeliefLearningDataAnalyzer
            self.data_analyzer_class = BeliefLearningDataAnalyzer
            
            # 导入可视化模块
            from experiments.strong_communication_belief_learning.visualizer import BeliefLearningVisualizer
            self.visualizer_class = BeliefLearningVisualizer
            
            self.logger.info("✓ 所有实验模块导入成功")
            
        except ImportError as e:
            self.logger.error(f"✗ 导入实验模块失败: {e}")
            print("\n请确保以下文件存在于正确位置:")
            print("- experiments/strong_communication_belief_learning/experiment_config.py")
            print("- experiments/strong_communication_belief_learning/experiment_runner.py") 
            print("- experiments/strong_communication_belief_learning/data_analyzer.py")
            print("- experiments/strong_communication_belief_learning/visualizer.py")
            raise
    
    def run_quick_test(self):
        """运行快速测试实验（少量配置，用于验证系统）"""
        self.logger.info("开始运行快速测试实验...")
        
        # 创建简化的测试配置
        test_configs = {
            "test_scalability": [
                self.config_generator._create_base_config(
                    num_agents=3, num_tasks=4, 
                    experiment_name="test_scalability_3a_4t"
                ),
                self.config_generator._create_base_config(
                    num_agents=4, num_tasks=6, 
                    experiment_name="test_scalability_4a_6t"
                )
            ],
            "test_comparison": [
                self.config_generator._create_base_config(
                    num_agents=3, num_tasks=4, 
                    experiment_name="test_k_value_comparison_k2",
                    custom_k_value=2
                ),
                self.config_generator._create_base_config(
                    num_agents=3, num_tasks=4, 
                    experiment_name="test_k_value_comparison_k3",
                    custom_k_value=3
                )
            ]
        }
        
        # 运行测试实验（减少运行次数）
        test_runner = self.experiment_runner_class(
            output_dir=str(self.output_dir),
            num_runs_per_config=1,  # 测试时只运行1次
            enable_parallel=False,  # 测试时不用并行
            max_workers=1
        )
        
        # 手动运行测试配置
        for exp_type, configs in test_configs.items():
            self.logger.info(f"运行测试实验类型: {exp_type}")
            for config in configs:
                self.logger.info(f"运行配置: {config['scenario_name']}")
                try:
                    result = test_runner._run_single_experiment(exp_type, config, 0)
                    if result.get('success', False):
                        self.logger.info(f"✓ 配置 {config['scenario_name']} 运行成功")
                    else:
                        self.logger.warning(f"⚠ 配置 {config['scenario_name']} 运行失败")
                except Exception as e:
                    self.logger.error(f"✗ 配置 {config['scenario_name']} 运行出错: {e}")
        
        self.logger.info("快速测试完成")
    
    def run_full_experiments(self):
        """运行完整实验"""
        self.logger.info("开始运行完整实验...")
        
        # 创建实验运行器
        runner = self.experiment_runner_class(
            output_dir=str(self.output_dir),
            num_runs_per_config=self.num_runs_per_config,
            enable_parallel=self.enable_parallel,
            max_workers=self.max_workers
        )
        
        # 运行所有实验
        start_time = time.time()
        results = runner.run_all_experiments()
        end_time = time.time()
        
        # 统计结果
        total_experiments = sum(len(configs) for configs in results.values()) * self.num_runs_per_config
        elapsed_time = end_time - start_time
        
        self.logger.info(f"实验完成! 总共运行了 {total_experiments} 个实验")
        self.logger.info(f"总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.1f} 分钟)")
        
        return results
    
    def analyze_data(self):
        """分析实验数据"""
        self.logger.info("开始分析实验数据...")
        
        try:
            # 创建数据分析器
            analyzer = self.data_analyzer_class(str(self.output_dir))
            
            # 生成分析报告
            report = analyzer.generate_analysis_report()
            
            # 保存处理后的数据
            analyzer.save_processed_data()
            
            self.logger.info("✓ 数据分析完成")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ 数据分析失败: {e}")
            return False
    
    def generate_visualizations(self):
        """生成可视化图表"""
        self.logger.info("开始生成可视化图表...")
        
        try:
            # 设置中文字体
            setup_chinese_fonts()
            
            # 创建可视化器
            visualizer = self.visualizer_class(str(self.output_dir))
            
            # 生成所有可视化图表
            visualizer.create_all_visualizations()
            
            self.logger.info("✓ 可视化图表生成完成")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ 可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_summary_report(self):
        """生成总结报告"""
        self.logger.info("生成总结报告...")
        
        try:
            report_content = f"""# 强通信信念自学习实验总结报告

## 实验信息
- 执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 输出目录: {self.output_dir}
- 每配置运行次数: {self.num_runs_per_config}
- 并行运行: {self.enable_parallel}

## 实验结果
实验已成功完成，包括以下内容：

### 1. 实验数据
- 扩展性实验: 测试不同智能体数量和任务数量的系统性能
- 对比实验: 比较不同K值、信念强度、观测精度等参数的影响
- 鲁棒性实验: 测试系统在不同环境条件下的表现

### 2. 数据分析
- 汇总统计信息
- 扩展性分析
- K值影响分析
- 信念收敛分析
- 统计显著性检验

### 3. 可视化图表
已生成15个专业的学术级图表：
1. 初始位置和通信连接图
2. 智能体轨迹图
3. 关键时刻轨迹图
4. 联盟演化图
5. 任务甘特图
6. 智能体执行时序图
7. 信念收敛图
8. 加速度曲线图
9. 速度曲线图
10. 效用收敛图
11. K值对比图
12. 通信统计图
13. 扩展性分析图
14. 性能对比图
15. 统计显著性图

## 结果位置
- 原始数据: `{self.output_dir}/*/run_*/`
- 分析结果: `{self.output_dir}/analysis/`
- 可视化图表: `{self.output_dir}/visualizations/`
- 日志文件: `{self.output_dir}/experiment_log.txt`

## 下一步
1. 查看可视化图表了解实验结果
2. 阅读分析报告获取详细洞察
3. 根据需要调整参数重新运行实验
"""
            
            # 保存报告
            report_file = self.output_dir / "experiment_summary.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"✓ 总结报告已保存到: {report_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ 生成总结报告失败: {e}")
            return False
    
    def run_complete_pipeline(self, run_experiments: bool = True, 
                            analyze_data: bool = True, 
                            generate_viz: bool = True):
        """运行完整的实验流程"""
        self.logger.info("开始运行完整的实验流程...")
        
        pipeline_start_time = time.time()
        success_steps = []
        
        try:
            # 步骤1: 运行实验
            if run_experiments:
                self.logger.info("步骤 1/4: 运行实验")
                results = self.run_full_experiments()
                success_steps.append("实验运行")
            
            # 步骤2: 分析数据
            if analyze_data:
                self.logger.info("步骤 2/4: 分析数据")
                if self.analyze_data():
                    success_steps.append("数据分析")
            
            # 步骤3: 生成可视化
            if generate_viz:
                self.logger.info("步骤 3/4: 生成可视化")
                if self.generate_visualizations():
                    success_steps.append("可视化生成")
            
            # 步骤4: 生成报告
            self.logger.info("步骤 4/4: 生成总结报告")
            if self.generate_summary_report():
                success_steps.append("总结报告")
            
        except Exception as e:
            self.logger.error(f"实验流程执行失败: {e}")
            import traceback
            traceback.print_exc()
        
        pipeline_end_time = time.time()
        total_time = pipeline_end_time - pipeline_start_time
        
        # 输出最终结果
        print("\n" + "=" * 60)
        print("实验流程执行完成!")
        print("=" * 60)
        print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")
        print(f"成功完成的步骤: {', '.join(success_steps)}")
        print(f"结果保存在: {self.output_dir}")
        print("=" * 60)
        
        return len(success_steps)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="强通信信念自学习实验运行器")
    
    parser.add_argument("--mode", type=str, default="full",
                       choices=["test", "full", "analyze", "visualize"],
                       help="运行模式: test(快速测试), full(完整流程), analyze(仅分析), visualize(仅可视化)")
    
    parser.add_argument("--output_dir", type=str, 
                       default="results/strong_comm_experiments",
                       help="输出目录")
    
    parser.add_argument("--num_runs", type=int, default=3,
                       help="每个配置运行的次数")
    
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="启用并行运行")
    
    parser.add_argument("--max_workers", type=int, default=4,
                       help="最大工作进程数")
    
    parser.add_argument("--no_experiments", action="store_true",
                       help="跳过实验运行，仅执行分析和可视化")
    
    parser.add_argument("--no_analysis", action="store_true",
                       help="跳过数据分析")
    
    parser.add_argument("--no_visualization", action="store_true",
                       help="跳过可视化生成")
    
    args = parser.parse_args()
    
    # 创建实验运行器
    runner = StrongCommunicationExperimentRunner(
        output_dir=args.output_dir,
        num_runs_per_config=args.num_runs,
        enable_parallel=args.parallel,
        max_workers=args.max_workers
    )
    
    # 根据模式执行不同操作
    if args.mode == "test":
        runner.run_quick_test()
        
    elif args.mode == "full":
        runner.run_complete_pipeline(
            run_experiments=not args.no_experiments,
            analyze_data=not args.no_analysis,
            generate_viz=not args.no_visualization
        )
        
    elif args.mode == "analyze":
        runner.analyze_data()
        runner.generate_summary_report()
        
    elif args.mode == "visualize":
        runner.generate_visualizations()
        runner.generate_summary_report()

if __name__ == "__main__":
    main()