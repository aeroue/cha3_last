# experiments/scenario_generators.py
import os
import sys
import copy
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# 导入项目路径
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
_project_root_directory = os.path.dirname(_current_file_directory)
if _project_root_directory not in sys.path:
    sys.path.insert(0, _project_root_directory)

from simulation.scenarios import SCENARIOS, load_scenario

class ScenarioGenerator:
    """
    场景生成器类：用于创建和修改场景配置
    """
    def __init__(self):
        """初始化场景生成器"""
        self.base_scenarios = copy.deepcopy(SCENARIOS)
        
    def get_available_scenarios(self) -> List[str]:
        """获取所有可用场景名称"""
        return list(self.base_scenarios.keys())
    
    def get_scenario_info(self, scenario_name: str) -> Dict[str, Any]:
        """获取特定场景的基本信息"""
        scenario = self.base_scenarios.get(scenario_name)
        if not scenario:
            return {}
            
        info = {
            "名称": scenario.get("scenario_display_name", scenario_name),
            "描述": scenario.get("description", "无描述"),
            "智能体数量": scenario.get("num_agents", 0),
            "任务数量": scenario.get("num_tasks", 0),
            "任务类型": scenario.get("task_type_names", []),
            "最大步数": scenario.get("simulation_params", {}).get("max_time_steps", 0),
            "通信范围": scenario.get("simulation_params", {}).get("communication_range", "无限制"),
            "是否启用强通信": scenario.get("simulation_params", {}).get("strong_mode_components_enabled", True)
        }
        return info
    
    def create_k_variation_scenario(self, 
                                   base_scenario_name: str, 
                                   k_value: int, 
                                   new_scenario_name: Optional[str] = None) -> str:
        """
        创建K值变化场景
        
        参数:
            base_scenario_name (str): 基础场景名称
            k_value (int): 新的K值
            new_scenario_name (str, optional): 新场景名称，如果未提供则自动生成
            
        返回:
            str: 新场景名称
        """
        if base_scenario_name not in self.base_scenarios:
            raise ValueError(f"基础场景 '{base_scenario_name}' 不存在")
            
        # 创建配置深拷贝
        new_config = copy.deepcopy(self.base_scenarios[base_scenario_name])
        
        # 修改K值
        for sc_def in new_config["spacecrafts"]:
            if "ksc_params" in sc_def:
                sc_def["ksc_params"]["k_value"] = k_value
                
        # 更新场景名称
        if new_scenario_name is None:
            new_scenario_name = f"{base_scenario_name}_k{k_value}"
            
        # 更新显示名称
        base_display_name = new_config.get("scenario_display_name", base_scenario_name)
        new_config["scenario_display_name"] = f"{base_display_name} (K={k_value})"
        
        # 保存到全局场景字典
        SCENARIOS[new_scenario_name] = new_config
        
        return new_scenario_name
        
    def create_comm_range_scenario(self, 
                                  base_scenario_name: str, 
                                  comm_range: float,
                                  new_scenario_name: Optional[str] = None) -> str:
        """
        创建通信范围变化场景
        
        参数:
            base_scenario_name (str): 基础场景名称
            comm_range (float): 新的通信范围，None表示无限制
            new_scenario_name (str, optional): 新场景名称，如果未提供则自动生成
            
        返回:
            str: 新场景名称
        """
        if base_scenario_name not in self.base_scenarios:
            raise ValueError(f"基础场景 '{base_scenario_name}' 不存在")
            
        # 创建配置深拷贝
        new_config = copy.deepcopy(self.base_scenarios[base_scenario_name])
        
        # 修改通信范围
        if "simulation_params" in new_config:
            new_config["simulation_params"]["communication_range"] = comm_range
            
        # 更新场景名称
        if new_scenario_name is None:
            range_str = str(int(comm_range)) if comm_range is not None else "unlimited"
            new_scenario_name = f"{base_scenario_name}_range{range_str}"
            
        # 更新显示名称
        base_display_name = new_config.get("scenario_display_name", base_scenario_name)
        range_display = f"{comm_range}m" if comm_range is not None else "无限制"
        new_config["scenario_display_name"] = f"{base_display_name} (通信范围={range_display})"
        
        # 保存到全局场景字典
        SCENARIOS[new_scenario_name] = new_config
        
        return new_scenario_name
        
    def create_reasoning_level_scenario(self, 
                                      base_scenario_name: str, 
                                      reasoning_level: int,
                                      use_epistemic_planning: bool = False,
                                      new_scenario_name: Optional[str] = None) -> str:
        """
        创建推理级别变化场景
        
        参数:
            base_scenario_name (str): 基础场景名称
            reasoning_level (int): 新的推理级别 (0, 1, 2)
            use_epistemic_planning (bool): 是否使用认知规划
            new_scenario_name (str, optional): 新场景名称，如果未提供则自动生成
            
        返回:
            str: 新场景名称
        """
        if base_scenario_name not in self.base_scenarios:
            raise ValueError(f"基础场景 '{base_scenario_name}' 不存在")
            
        # 创建配置深拷贝
        new_config = copy.deepcopy(self.base_scenarios[base_scenario_name])
        
        # 修改推理级别
        if "aif_global_hyperparams" in new_config:
            new_config["aif_global_hyperparams"]["reasoning_level"] = reasoning_level
            new_config["aif_global_hyperparams"]["use_epistemic_planning"] = use_epistemic_planning
            
        # 更新场景名称
        if new_scenario_name is None:
            ep_str = "ep" if use_epistemic_planning else "noep"
            new_scenario_name = f"{base_scenario_name}_r{reasoning_level}_{ep_str}"
            
        # 更新显示名称
        base_display_name = new_config.get("scenario_display_name", base_scenario_name)
        ep_display = "+EP" if use_epistemic_planning else ""
        new_config["scenario_display_name"] = f"{base_display_name} (推理L{reasoning_level}{ep_display})"
        
        # 保存到全局场景字典
        SCENARIOS[new_scenario_name] = new_config
        
        return new_scenario_name
    
    def create_adaptive_threshold_scenario(self, 
                                         base_scenario_name: str, 
                                         low_threshold: float,
                                         high_threshold: float,
                                         new_scenario_name: Optional[str] = None) -> str:
        """
        创建自适应阈值变化场景
        
        参数:
            base_scenario_name (str): 基础场景名称
            low_threshold (float): 低CSI阈值 (0-1)
            high_threshold (float): 高CSI阈值 (0-1)
            new_scenario_name (str, optional): 新场景名称，如果未提供则自动生成
            
        返回:
            str: 新场景名称
        """
        if base_scenario_name not in self.base_scenarios:
            raise ValueError(f"基础场景 '{base_scenario_name}' 不存在")
            
        # 创建配置深拷贝
        new_config = copy.deepcopy(self.base_scenarios[base_scenario_name])
        
        # 修改阈值
        if "adaptive_controller_params" in new_config:
            new_config["adaptive_controller_params"]["csi_threshold_low"] = low_threshold
            new_config["adaptive_controller_params"]["csi_threshold_high"] = high_threshold
            
        # 更新场景名称
        if new_scenario_name is None:
            new_scenario_name = f"{base_scenario_name}_thresh_{int(low_threshold*10)}_{int(high_threshold*10)}"
            
        # 更新显示名称
        base_display_name = new_config.get("scenario_display_name", base_scenario_name)
        new_config["scenario_display_name"] = f"{base_display_name} (CSI阈值={low_threshold:.1f}-{high_threshold:.1f})"
        
        # 保存到全局场景字典
        SCENARIOS[new_scenario_name] = new_config
        
        return new_scenario_name
    
    def create_comparison_scenarios(self) -> Dict[str, str]:
        """
        创建算法比较场景：KSC+AIF+自适应对比纯KSC和纯AIF
        
        返回:
            Dict[str, str]: 创建的场景名称 -> 描述映射
        """
        base_scenario_names = [name for name in self.base_scenarios.keys() 
                              if "dynamic_comms" in name or "adaptive" in name]
        
        if not base_scenario_names:
            print("警告: 未找到适合的自适应场景作为基础")
            return {}
            
        base_scenario = base_scenario_names[0]
        
        comparison_scenarios = {}
        
        # 创建纯强通信场景
        strong_scenario = self.create_communication_mode_scenario(
            base_scenario, "强通信", enable_aif=False, 
            enable_adaptive=False, enable_ksc=True
        )
        comparison_scenarios[strong_scenario] = "纯强通信 (KSC)"
        
        # 创建纯弱通信场景
        weak_scenario = self.create_communication_mode_scenario(
            base_scenario, "弱通信", enable_aif=True, 
            enable_adaptive=False, enable_ksc=False
        )
        comparison_scenarios[weak_scenario] = "纯弱通信 (AIF)"
        
        # 创建混合自适应场景
        adaptive_scenario = self.create_communication_mode_scenario(
            base_scenario, "自适应", enable_aif=True, 
            enable_adaptive=True, enable_ksc=True
        )
        comparison_scenarios[adaptive_scenario] = "混合自适应 (KSC+AIF)"
        
        return comparison_scenarios
        
    def create_communication_mode_scenario(self, 
                                         base_scenario_name: str, 
                                         mode_name: str,
                                         enable_aif: bool = True,
                                         enable_adaptive: bool = True,
                                         enable_ksc: bool = True,
                                         new_scenario_name: Optional[str] = None) -> str:
        """
        创建特定通信模式场景
        
        参数:
            base_scenario_name (str): 基础场景名称
            mode_name (str): 模式名称
            enable_aif (bool): 是否启用AIF
            enable_adaptive (bool): 是否启用自适应
            enable_ksc (bool): 是否启用KSC
            new_scenario_name (str, optional): 新场景名称，如果未提供则自动生成
            
        返回:
            str: 新场景名称
        """
        if base_scenario_name not in self.base_scenarios:
            raise ValueError(f"基础场景 '{base_scenario_name}' 不存在")
            
        # 创建配置深拷贝
        new_config = copy.deepcopy(self.base_scenarios[base_scenario_name])
        
        # 修改通信模式
        if "simulation_params" in new_config:
            new_config["simulation_params"]["strong_mode_components_enabled"] = enable_ksc
            
        # 如果禁用自适应，设置CSI阈值使其固定在所需模式
        if not enable_adaptive and "adaptive_controller_params" in new_config:
            if not enable_ksc and enable_aif:  # 仅弱通信
                new_config["adaptive_controller_params"]["csi_threshold_low"] = 0.0
                new_config["adaptive_controller_params"]["csi_threshold_high"] = 0.0
            elif enable_ksc and not enable_aif:  # 仅强通信
                new_config["adaptive_controller_params"]["csi_threshold_low"] = 1.0
                new_config["adaptive_controller_params"]["csi_threshold_high"] = 1.0
                
        # 更新场景名称
        if new_scenario_name is None:
            mode_suffix = mode_name.lower().replace(" ", "_")
            new_scenario_name = f"{base_scenario_name}_{mode_suffix}_mode"
            
        # 更新显示名称
        base_display_name = new_config.get("scenario_display_name", base_scenario_name)
        new_config["scenario_display_name"] = f"{base_display_name} ({mode_name}模式)"
        
        # 保存到全局场景字典
        SCENARIOS[new_scenario_name] = new_config
        
        return new_scenario_name

# 如果直接运行此脚本，执行一些测试
if __name__ == "__main__":
    generator = ScenarioGenerator()
    print("可用场景:")
    for scenario in generator.get_available_scenarios():
        print(f"  - {scenario}")
        
    # 选择一个示例场景
    example_scenarios = [s for s in generator.get_available_scenarios() 
                        if "dynamic_comms" in s or "adaptive" in s or "strong_comms" in s]
    
    if example_scenarios:
        base_scenario = example_scenarios[0]
        print(f"\n选择基础场景: {base_scenario}")
        
        # 显示场景信息
        info = generator.get_scenario_info(base_scenario)
        print("\n场景信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
            
        # 测试创建K值变化场景
        k3_scenario = generator.create_k_variation_scenario(base_scenario, 3)
        print(f"\n创建K=3场景: {k3_scenario}")
        
        # 测试创建通信范围变化场景
        range_scenario = generator.create_comm_range_scenario(base_scenario, 200.0)
        print(f"\n创建通信范围=200m场景: {range_scenario}")
        
        # 测试创建推理级别变化场景
        reasoning_scenario = generator.create_reasoning_level_scenario(base_scenario, 2, True)
        print(f"\n创建推理级别=2+EP场景: {reasoning_scenario}")
        
        # 测试创建通信模式场景
        comparison_scenarios = generator.create_comparison_scenarios()
        print("\n创建算法比较场景:")
        for name, desc in comparison_scenarios.items():
            print(f"  - {name}: {desc}")
    else:
        print("未找到合适的示例场景")