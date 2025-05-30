# experiments/__init__.py
# 实验包初始化文件
# 用于简化导入路径，确保实验模块可以被识别

# 将项目根目录添加到Python路径中，确保可以导入其他模块
import os
import sys

# 获取当前文件所在目录(experiments/)
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
_project_root_directory = os.path.dirname(_current_file_directory)

# 将项目根目录添加到Python路径中，确保可以导入其他模块
if _project_root_directory not in sys.path:
    sys.path.insert(0, _project_root_directory)

    # 导出主要组件
from .experiment_manager import ExperimentManager
from .scenario_generators import ScenarioGenerator
from .benchmark_experiments import BenchmarkExperiments
from .parameter_analysis import ParameterAnalysisExperiments
from .communication_experiments import CommunicationExperiments
from .monte_carlo_experiments import MonteCarloExperiments
from .scalability_experiments import ScalabilityExperiments
from .analysis_utilities import ResultsAnalyzer