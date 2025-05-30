# # strong_communication/ksc_coalition_formation.py
# import sys
# import os

# # --- 标准路径处理代码 ---
# def _add_project_root_to_sys_path():
#     current_file_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(os.path.dirname(current_file_dir)) # ksc_coalition_formation.py在strong_communication下
#     if project_root not in sys.path:
#         sys.path.insert(0, project_root)
# _add_project_root_to_sys_path()
# # --- 路径处理代码结束 ---

# import numpy as np
# import random
# import copy
# from typing import Dict, List, Tuple, Any, Optional, Set
# import uuid
# import time

# # 假设这些模块可以通过PYTHONPATH或上述路径修改找到
# from strong_communication.belief_manager import BeliefManager # 您的代码中有这个
# from strong_communication.utility_calculator import UtilityCalculator # 您的代码中有这个

# Assignment = Dict[str, Optional[str]]
# SwitchOperation = Tuple[str, Optional[str], Optional[str]]
# TransformationChain = List[SwitchOperation]
# IDLE_TASK_ID = "IDLE_TASK_PLACEHOLDER" # 与您代码中的定义一致
# NeighborGraph = Dict[str, List[str]]

# class KSCMessage: # 与您上传版本一致
#     def __init__(self,
#                  base_assignment: Assignment,
#                  chain: TransformationChain,
#                  originator_id: str,
#                  achieved_utility: float,
#                  k_limit_of_originator: int,
#                  message_id: Optional[str] = None,
#                  sender_id: Optional[str] = None, # 您代码中有这个
#                  iteration_created: int = -1, # 您代码中有这个
#                  path_taken: Optional[List[str]] = None): # 您代码中有这个
#         self.base_assignment = copy.deepcopy(base_assignment)
#         self.chain = copy.deepcopy(chain)
#         self.originator_id = originator_id
#         self.achieved_utility = achieved_utility
#         self.k_limit_of_originator = k_limit_of_originator
#         self.message_id = message_id if message_id else str(uuid.uuid4())
#         self.sender_id = sender_id
#         self.iteration_created = iteration_created
#         self.path_taken = path_taken if path_taken is not None else []
#         self.processed_for_extension_by: Set[str] = set() # 您代码中有这个

#     def __repr__(self):
#         chain_str = " -> ".join([f"{a}({t_old if t_old else 'Idle'}->{t_new if t_new else 'Idle'})" for a, t_old, t_new in self.chain])
#         path_str = "->".join(self.path_taken)
#         chain_display_limit = 50 # 您代码中的显示限制
#         if len(chain_str) > chain_display_limit:
#             chain_str = chain_str[:chain_display_limit] + "..."
#         return (f"KSCMsg(id={self.message_id[:4]},orig={self.originator_id},send={self.sender_id},"
#                 f"k_lim={self.k_limit_of_originator},util={self.achieved_utility:.2f},"
#                 f"len={len(self.chain)},iter={self.iteration_created},path_len={len(self.path_taken)},chain=[{chain_str}])")

#     def __eq__(self, other): # 与您代码一致
#         if not isinstance(other, KSCMessage): return NotImplemented
#         return self.message_id == other.message_id
#     def __hash__(self): return hash(self.message_id) # 与您代码一致

# class KSCAgentData: # 与您上传版本一致
#     def __init__(self, agent_id: str, k_value: int, initial_global_assignment: Assignment, initial_global_utility: float):
#         self.agent_id = agent_id
#         self.k_value = k_value
#         self.message_buffer: List[KSCMessage] = []
#         self.best_known_global_assignment: Assignment = copy.deepcopy(initial_global_assignment)
#         self.best_known_global_utility: float = initial_global_utility
#         self.max_buffer_size = 50 # 您代码中是50
#         self.messages_processed_this_iteration = 0 # 您代码中有这个
#         self.messages_generated_this_iteration = 0 # 您代码中有这个
#         self.sent_message_ids_this_iteration: Set[str] = set() # 您代码中有这个

#     def add_message_to_buffer(self, message: KSCMessage, current_agent_id: str):
#         if len(message.chain) > message.k_limit_of_originator: return
#         # 您代码中的逻辑
#         if current_agent_id in message.path_taken and message.originator_id != current_agent_id : return
#         existing_msg_idx = -1
#         for i, existing_msg in enumerate(self.message_buffer):
#             if existing_msg.message_id == message.message_id:
#                 existing_msg_idx = i; break
#         if existing_msg_idx != -1: # 如果消息已存在
#             if message.achieved_utility > self.message_buffer[existing_msg_idx].achieved_utility: # 且新消息更好
#                 self.message_buffer[existing_msg_idx] = message # 更新它
#         else: # 否则添加新消息
#             self.message_buffer.append(message)
#         self.message_buffer.sort(key=lambda m: (m.achieved_utility, -len(m.chain)), reverse=True) # 按效用降序，链长升序
#         if len(self.message_buffer) > self.max_buffer_size:
#             self.message_buffer = self.message_buffer[:self.max_buffer_size]

#     def get_messages_for_processing(self, count: int = 5) -> List[KSCMessage]: # 您代码中count=3
#         # 您代码中是 self.message_buffer[:count]，这里保持一致
#         return [copy.deepcopy(msg) for msg in self.message_buffer[:count]]


#     def get_messages_for_broadcast(self, iteration_num: int, count: int = 3) -> List[KSCMessage]: # 您代码中count=3
#         # 与您上传的KSCAgentData中的get_messages_for_broadcast逻辑保持一致
#         candidates = []
#         # 优先选择本轮新生成且效用显著提升的消息
#         for msg in self.message_buffer:
#             if msg.message_id in self.sent_message_ids_this_iteration:
#                 continue
#             is_new_this_iteration = (msg.iteration_created == iteration_num)
#             is_significantly_better = (msg.achieved_utility > self.best_known_global_utility + 1e-5) # 使用小epsilon比较浮点数

#             if is_new_this_iteration and is_significantly_better:
#                 if len(candidates) < count:
#                     candidates.append(copy.deepcopy(msg))
#                 else:
#                     break # 已经达到广播数量上限
#             elif is_significantly_better and len(candidates) < count // 2 : # 如果不是新的但效用好，也考虑广播一部分
#                  candidates.append(copy.deepcopy(msg))


#         # 如果上述选出的消息不足，再从本轮新生成的消息中补充（不一定效用显著提升，但至少是新的）
#         if len(candidates) < count:
#             for msg in self.message_buffer:
#                 if msg.message_id in self.sent_message_ids_this_iteration or msg in candidates:
#                     continue
#                 if msg.iteration_created == iteration_num and len(candidates) < count:
#                     candidates.append(copy.deepcopy(msg))
#                 elif len(candidates) >= count:
#                     break
#         return candidates

# # class KSCCoalitionFormation:
# #     def __init__(self,
# #                  agent_ids: List[str],
# #                  task_ids: List[str],
# #                  agent_k_values: Dict[str, int],
# #                  utility_calculator: UtilityCalculator,
# #                  belief_manager: BeliefManager,
# #                  task_type_names: List[str],
# #                  initial_assignment: Optional[Assignment] = None,
# #                  max_coalition_size_per_task: Optional[Dict[str, int]] = None,
# #                  min_agents_per_task: Optional[Dict[str, int]] = None,
# #                  agent_initial_states: Optional[Dict[str, np.ndarray]] = None,
# #                  communication_range: Optional[float] = None,
# #                  max_dfs_branching_factor: int = 3, # 对应您代码的DFS_BRANCHING_FACTOR
# #                  max_dfs_depth_for_others: int = 1): # 对应您代码的DFS_SEARCH_DEPTH_OTHERS

# #         self.agent_ids = agent_ids
# #         self.task_ids_actual = list(task_ids)
# #         self.task_ids_with_idle = self.task_ids_actual + [IDLE_TASK_ID]

# #         self.utility_calculator = utility_calculator
# #         self.belief_manager = belief_manager
# #         self.task_type_names = task_type_names

# #         self.max_coalition_size_by_type = max_coalition_size_per_task if max_coalition_size_per_task else {}
# #         self.min_agents_per_task_by_type = min_agents_per_task if min_agents_per_task else {}

# #         self.current_agent_physical_states = agent_initial_states if agent_initial_states else self._get_current_agent_states_placeholder(self.agent_ids)
# #         self.communication_range = communication_range
# #         self.neighbor_graph: NeighborGraph = {}
        
# #         self.stats_total_messages_broadcasted_per_iteration: List[int] = []
# #         self.stats_total_messages_processed_per_iteration: List[int] = []
# #         self.stats_active_links_per_iteration: List[int] = [] # 这个在_update_neighbor_graph中更新
        
# #         self._update_neighbor_graph() # 初始化时计算一次

# #         if initial_assignment is None:
# #             initial_assignment = {agent_id: IDLE_TASK_ID for agent_id in self.agent_ids}

# #         initial_global_utility = self._calculate_global_utility(initial_assignment, self.current_agent_physical_states)
# #         if initial_global_utility == -float('inf') and not self.min_agents_per_task_by_type : # 检查min_agents_per_task_by_type是否有内容
# #             initial_global_utility = 0.0

# #         self.agents_data: Dict[str, KSCAgentData] = {}
# #         for agent_id in self.agent_ids:
# #             k_val = agent_k_values.get(agent_id)
# #             if k_val is None: raise ValueError(f"Agent {agent_id} 的K值未提供。")
# #             self.agents_data[agent_id] = KSCAgentData(agent_id, k_val, initial_assignment, initial_global_utility)

# #         self._sim_best_assignment = copy.deepcopy(initial_assignment)
# #         self._sim_best_utility = initial_global_utility
        
# #         self.max_dfs_branching_factor = max_dfs_branching_factor
# #         self.max_dfs_depth_for_others = max_dfs_depth_for_others

# #         # DEBUGGING PRINTS
# #         print(f"[KSC DEBUG __init__] Initialized. Agents: {len(self.agent_ids)}. Tasks: {len(self.task_ids_actual)}.")
# #         print(f"  [KSC DEBUG __init__] Initial Best Utility: {self._sim_best_utility:.3f}, Assignment: {self._sim_best_assignment}")
# #         if self.stats_active_links_per_iteration:
# #             print(f"  [KSC DEBUG __init__] Initial active comm links: {self.stats_active_links_per_iteration[0]}")
# #         print(f"  [KSC DEBUG __init__] Max coalition size by type: {self.max_coalition_size_by_type}")
# #         print(f"  [KSC DEBUG __init__] Min agents per task by type: {self.min_agents_per_task_by_type}")
# #         if self.utility_calculator and hasattr(self.utility_calculator, 'task_definitions'):
# #             print(f"  [KSC DEBUG __init__] Utility Calculator Task Defs at init: {list(self.utility_calculator.task_definitions.keys())}")


# #     def _get_current_agent_states_placeholder(self, agent_ids_list: List[str]) -> Dict[str, np.ndarray]:
# #         # (保持不变)
# #         states = {}
# #         for i, agent_id in enumerate(agent_ids_list):
# #             angle = 2 * np.pi * i / len(agent_ids_list)
# #             radius = 150 + i * 30
# #             states[agent_id] = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0, 0.0])
# #         return states

# #     def _update_neighbor_graph(self):
# #         # (保持不变，但确保 self.stats_active_links_per_iteration 被正确更新)
# #         self.neighbor_graph = {agent_id: [] for agent_id in self.agent_ids}
# #         active_links_count = 0
# #         if self.communication_range is None: # 全连接
# #             for agent_id in self.agent_ids:
# #                 self.neighbor_graph[agent_id] = [other_id for other_id in self.agent_ids if other_id != agent_id]
# #             if len(self.agent_ids) > 1: active_links_count = len(self.agent_ids) * (len(self.agent_ids) -1) // 2
# #         else: # 基于范围
# #             agent_ids_list = list(self.agent_ids) # 确保是列表以便索引
# #             for i in range(len(agent_ids_list)):
# #                 for j in range(i + 1, len(agent_ids_list)):
# #                     agent1_id, agent2_id = agent_ids_list[i], agent_ids_list[j]
# #                     if agent1_id not in self.current_agent_physical_states or \
# #                        agent2_id not in self.current_agent_physical_states:
# #                         continue # 确保状态存在
# #                     pos1 = self.current_agent_physical_states[agent1_id][:2]
# #                     pos2 = self.current_agent_physical_states[agent2_id][:2]
# #                     if np.linalg.norm(pos1 - pos2) <= self.communication_range:
# #                         self.neighbor_graph[agent1_id].append(agent2_id)
# #                         self.neighbor_graph[agent2_id].append(agent1_id)
# #                         active_links_count +=1
# #         # 仅当列表为空或值改变时才添加，或者如果这是第一次（len < 2）
# #         if not self.stats_active_links_per_iteration or self.stats_active_links_per_iteration[-1] != active_links_count or len(self.stats_active_links_per_iteration) < 1:
# #             self.stats_active_links_per_iteration.append(active_links_count)


# #     def update_agent_physical_states(self, new_states: Dict[str, np.ndarray]):
# #         # (保持不变)
# #         self.current_agent_physical_states.update(new_states)
# #         self._update_neighbor_graph() # 更新邻接图
# #         # 重新评估当前最优分配的效用，因为物理状态变了
# #         current_utility_of_best_assignment = self._calculate_global_utility(self._sim_best_assignment, self.current_agent_physical_states)
# #         # print(f"  [KSC DEBUG StatesUpdate] Utility of SimBestAssignment {self._sim_best_assignment} re-evaluated to {current_utility_of_best_assignment:.2f} (was {self._sim_best_utility:.2f})")
# #         self._sim_best_utility = current_utility_of_best_assignment
# #         # 更新所有智能体的认知
# #         for agent_id in self.agent_ids:
# #             agent_data = self.agents_data[agent_id]
# #             utility_of_agent_known_best = self._calculate_global_utility(agent_data.best_known_global_assignment, self.current_agent_physical_states)
# #             agent_data.best_known_global_utility = utility_of_agent_known_best
# #             # 如果全局最优比智能体当前认知的好，则更新智能体的认知
# #             if self._sim_best_utility > agent_data.best_known_global_utility + 1e-5 : # Epsilon
# #                  # print(f"    [KSC DEBUG StatesUpdate] Agent {agent_id} updated its best known global utility to {self._sim_best_utility:.2f}")
# #                  agent_data.best_known_global_assignment = copy.deepcopy(self._sim_best_assignment)
# #                  agent_data.best_known_global_utility = self._sim_best_utility

# #     def notify_task_completion(self, task_id_completed: str, agents_in_completed_coalition: List[str]):
# #         # (保持不变)
# #         print(f"  [KSC DEBUG Event] 任务 {task_id_completed} 完成. 参与者: {agents_in_completed_coalition}.")
# #         if task_id_completed in self.task_ids_actual:
# #             self.task_ids_actual.remove(task_id_completed) # 从活动任务列表中移除
# #             self.task_ids_with_idle = self.task_ids_actual + [IDLE_TASK_ID] # 更新包含空闲的任务列表
# #             # 从 UtilityCalculator 的任务定义中移除（如果它持有副本）
# #             if self.utility_calculator and hasattr(self.utility_calculator, 'task_definitions'):
# #                  if task_id_completed in self.utility_calculator.task_definitions:
# #                     del self.utility_calculator.task_definitions[task_id_completed]
# #             # 从 BeliefManager 的任务列表中移除（如果它不自动同步）
# #             if self.belief_manager:
# #                 self.belief_manager.task_ids = list(self.task_ids_actual)
# #             # 将所有分配到已完成任务的智能体设置为空闲
# #             needs_re_eval = False
# #             for agent_id in self.agent_ids:
# #                 if agent_id in self._sim_best_assignment and self._sim_best_assignment[agent_id] == task_id_completed:
# #                     self._sim_best_assignment[agent_id] = IDLE_TASK_ID
# #                     needs_re_eval = True
# #                 if agent_id in self.agents_data: # 确保 agent_id 存在
# #                     # 更新智能体自己已知的最优分配
# #                     if agent_id in self.agents_data[agent_id].best_known_global_assignment and \
# #                        self.agents_data[agent_id].best_known_global_assignment[agent_id] == task_id_completed:
# #                         self.agents_data[agent_id].best_known_global_assignment[agent_id] = IDLE_TASK_ID
# #                     # 清理消息缓冲区中与已完成任务相关的消息
# #                     new_buffer = []
# #                     for msg in self.agents_data[agent_id].message_buffer:
# #                         # 如果消息的基础分配或链中包含已完成的任务，则考虑移除或调整
# #                         # 一个简单粗暴但有效的方法：如果链中任何部分涉及已完成任务，则丢弃
# #                         chain_involves_completed = any(op[1] == task_id_completed or op[2] == task_id_completed for op in msg.chain)
# #                         base_involves_completed = any(tid == task_id_completed for tid in msg.base_assignment.values())
# #                         if not (chain_involves_completed or base_involves_completed):
# #                             new_buffer.append(msg)
# #                     self.agents_data[agent_id].message_buffer = new_buffer

# #             if needs_re_eval: # 如果最优分配被修改，重新计算其效用
# #                  self._sim_best_utility = self._calculate_global_utility(self._sim_best_assignment, self.current_agent_physical_states)
# #                  # 更新所有智能体的认知
# #                  for ad_val in self.agents_data.values():
# #                      ad_val.best_known_global_assignment = copy.deepcopy(self._sim_best_assignment)
# #                      ad_val.best_known_global_utility = self._sim_best_utility
# #         else:
# #             print(f"  [KSC DEBUG Event] 任务 {task_id_completed} 已被移除或不存在于 task_ids_actual。")


# #     def _calculate_global_utility(self, assignment: Assignment, agent_states: Dict[str, np.ndarray]) -> float:
# #         total_utility = 0.0
# #         # DEBUG: 打印传入的参数
# #         # print(f"  [KSC DEBUG _calc_util] Calculating utility for assignment: {assignment}")
# #         # print(f"    [KSC DEBUG _calc_util] Current active tasks for KSC: {self.task_ids_actual}")
# #         # if self.utility_calculator:
# #         #     print(f"    [KSC DEBUG _calc_util] Utility calc's task defs keys: {list(self.utility_calculator.task_definitions.keys())}")


# #         assigned_agents_count_per_task: Dict[str, int] = {task_id: 0 for task_id in self.task_ids_actual}

# #         for agent_id, task_id in assignment.items():
# #             if task_id is not None and task_id != IDLE_TASK_ID:
# #                 if task_id not in self.task_ids_actual:
# #                     # print(f"    [KSC DEBUG _calc_util] Task {task_id} (for agent {agent_id}) is not in active tasks. Utility will be -inf.")
# #                     return -float('inf')
# #                 assigned_agents_count_per_task[task_id] += 1

# #         # 检查最小和最大智能体约束
# #         for task_id_check_constraints in self.task_ids_actual:
# #             # 确保从 utility_calculator 获取任务定义，它应该有最新的信息（包括类型名）
# #             if not self.utility_calculator or task_id_check_constraints not in self.utility_calculator.task_definitions:
# #                 # print(f"    [KSC DEBUG _calc_util] Task {task_id_check_constraints} def not found in utility_calc for constraint check. Assuming it's completed or invalid for current calc.")
# #                 # 如果一个任务在utility_calculator中没有了（例如刚完成），但仍在assignment中，此assignment无效
# #                 if assigned_agents_count_per_task.get(task_id_check_constraints, 0) > 0:
# #                     return -float('inf')
# #                 continue

# #             task_def_constraints = self.utility_calculator.task_definitions[task_id_check_constraints]
# #             task_type_name_constraints = task_def_constraints.get("true_type_name")
# #             count_for_this_task = assigned_agents_count_per_task.get(task_id_check_constraints, 0)

# #             if task_type_name_constraints:
# #                 min_req = self.min_agents_per_task_by_type.get(task_type_name_constraints, 0)
# #                 if 0 < count_for_this_task < min_req:
# #                     # print(f"    [KSC DEBUG _calc_util] Task {task_id_check_constraints} (Type: {task_type_name_constraints}) "
# #                     #       f"violates min_agents ({min_req}). Has: {count_for_this_task}. Utility = -inf.")
# #                     return -float('inf')
                
# #                 max_allowed = self.max_coalition_size_by_type.get(task_type_name_constraints)
# #                 if max_allowed is not None and count_for_this_task > max_allowed:
# #                     # print(f"    [KSC DEBUG _calc_util] Task {task_id_check_constraints} (Type: {task_type_name_constraints}) "
# #                     #       f"violates max_agents ({max_allowed}). Has: {count_for_this_task}. Utility = -inf.")
# #                     return -float('inf')
# #             # else:
# #                 # print(f"    [KSC DEBUG _calc_util] Task {task_id_check_constraints} has no type name, skipping type-specific min/max agent constraints for it.")


# #         # 计算总效用
# #         for agent_id, task_id in assignment.items():
# #             if task_id is not None and task_id != IDLE_TASK_ID:
# #                 if task_id not in self.task_ids_actual: continue # 再次确认

# #                 current_agent_state = agent_states.get(agent_id)
# #                 if current_agent_state is None:
# #                     # print(f"    [KSC DEBUG _calc_util] Agent {agent_id} state not found. Utility = -inf.")
# #                     return -float('inf')

# #                 if task_id not in self.belief_manager.task_ids: # 确保任务在信念管理器中
# #                     # print(f"    [KSC DEBUG _calc_util] Task {task_id} not in belief_manager for agent {agent_id}. Skipping utility for this pair.")
# #                     continue # 可能任务刚完成，从belief_manager移除了

# #                 agent_beliefs = self.belief_manager.get_expected_belief_dist_for_task(agent_id, task_id)
# #                 if agent_beliefs is None: # 如果无法获取信念
# #                     # print(f"    [KSC DEBUG _calc_util] Agent {agent_id} belief for task {task_id} is None. Utility = -inf.")
# #                     return -float('inf')

# #                 # print(f"    [KSC DEBUG _calc_util] Agent {agent_id} to Task {task_id}. Beliefs: {np.round(agent_beliefs,2)}")
# #                 utility = self.utility_calculator.get_expected_task_utility_for_agent(
# #                     agent_id, current_agent_state, task_id, agent_beliefs, self.task_type_names
# #                 )
# #                 # print(f"      Individual utility for {agent_id} on {task_id}: {utility:.2f}")
# #                 if utility == -float('inf'): return -float('inf')
# #                 total_utility += utility
        
# #         # print(f"  [KSC DEBUG _calc_util] Calculated total utility for assignment: {total_utility:.3f}")
# #         return total_utility

# #     def _is_assignment_valid(self, assignment: Assignment, task_id_to_check: Optional[str], agent_id_switched: str) -> bool:
# #         # print(f"  [KSC DEBUG _is_valid] Checking validity for task '{task_id_to_check}' by agent '{agent_id_switched}'")
# #         # print(f"    Assignment being checked: {assignment}")
# #         # print(f"    Current active KSC tasks: {self.task_ids_actual}")

# #         if task_id_to_check is None or task_id_to_check == IDLE_TASK_ID:
# #             # print(f"    Task is None or IDLE, considered valid.")
# #             return True
        
# #         if task_id_to_check not in self.task_ids_actual: # 必须是当前活动任务
# #             # print(f"    Task '{task_id_to_check}' not in active tasks {self.task_ids_actual}. Invalid.")
# #             return False

# #         # 从 utility_calculator 获取任务定义，它应该有最新的信息
# #         if not self.utility_calculator or task_id_to_check not in self.utility_calculator.task_definitions:
# #             # print(f"    Task '{task_id_to_check}' not in utility_calculator.task_definitions. Invalid.")
# #             return False # 如果任务定义都找不到了，那肯定无效
        
# #         task_def = self.utility_calculator.task_definitions[task_id_to_check]
# #         task_type_name = task_def.get("true_type_name")
# #         # print(f"    Task '{task_id_to_check}' is of type: '{task_type_name}'")

# #         if not task_type_name:
# #             # print(f"    Task '{task_id_to_check}' has no defined type name. Max size constraint (type-specific) not applied.")
# #             pass # 如果没有类型名，则不应用基于类型的最大规模约束，或者可以有一个全局默认
# #         else:
# #             max_size_for_this_task_type = self.max_coalition_size_by_type.get(task_type_name)
# #             # print(f"    Max size for type '{task_type_name}': {max_size_for_this_task_type}")
# #             if max_size_for_this_task_type is not None:
# #                 # assignment 参数应该是应用了潜在切换后的分配
# #                 current_size = sum(1 for assigned_tid_val in assignment.values() if assigned_tid_val == task_id_to_check)
# #                 # print(f"    Current size for task '{task_id_to_check}' in (potential) assignment: {current_size}")
# #                 if current_size > max_size_for_this_task_type:
# #                     # print(f"    INVALID: Task '{task_id_to_check}' (Type '{task_type_name}') violates max size ({max_size_for_this_task_type}). Current: {current_size}.")
# #                     return False
        
# #         # print(f"    Assignment for task '{task_id_to_check}' is VALID by agent '{agent_id_switched}'.")
# #         return True


# #     def _apply_chain_to_assignment(self, base_assignment: Assignment, chain: TransformationChain) -> Assignment:
# #         # (保持不变)
# #         new_assignment = copy.deepcopy(base_assignment)
# #         for agent_id, _, new_task_id in chain:
# #             new_assignment[agent_id] = new_task_id
# #         return new_assignment

# #     def _local_optimization_dfs(self,
# #                                 agent_data: KSCAgentData,
# #                                 base_assignment_for_dfs: Assignment,
# #                                 current_chain_being_built: TransformationChain,
# #                                 current_assignment_in_dfs: Assignment,
# #                                 iteration_num: int,
# #                                 visited_assignments_in_dfs: Set[Tuple[Tuple[str, Optional[str]], ...]]
# #                                ) -> None:
# #         # (保持不变，除了添加更多DEBUG打印)
# #         assignment_tuple = tuple(sorted(current_assignment_in_dfs.items()))
# #         if assignment_tuple in visited_assignments_in_dfs: return
# #         visited_assignments_in_dfs.add(assignment_tuple)

# #         # print(f"  [DFS {agent_data.agent_id}] Depth: {len(current_chain_being_built)}, k_limit: {agent_data.k_value}")
# #         # print(f"    Base Assign DFS: {base_assignment_for_dfs}")
# #         # print(f"    Current Chain DFS: {current_chain_being_built}")
# #         # print(f"    Current Assign in DFS: {current_assignment_in_dfs}")

# #         if len(current_chain_being_built) >= agent_data.k_value:
# #             # print(f"    DFS: Max depth reached for agent {agent_data.agent_id}.")
# #             return

# #         agents_to_consider_for_next_switch = []
# #         if not current_chain_being_built:
# #             agents_to_consider_for_next_switch = [agent_data.agent_id]
# #         else:
# #             # 根据您上传的代码，这里是固定 DFS_SEARCH_DEPTH_OTHERS = 1，意味着链的后续步骤必须由发起者自己执行
# #             # if len(current_chain_being_built) < self.max_dfs_depth_for_others:
# #             #     # 允许其他人，但您的版本中这个深度是1，所以只有发起者能扩展
# #             #     pass
# #             # 确保至少发起者可以继续
# #             if agent_data.agent_id not in agents_to_consider_for_next_switch:
# #                  agents_to_consider_for_next_switch.append(agent_data.agent_id)


# #         # print(f"    DFS: Agents to consider for next switch: {agents_to_consider_for_next_switch}")

# #         for next_agent_to_switch in agents_to_consider_for_next_switch:
# #             original_task_of_next = current_assignment_in_dfs.get(next_agent_to_switch)
# #             sorted_target_tasks = sorted([tid for tid in self.task_ids_with_idle if tid != original_task_of_next], key=lambda x: x == IDLE_TASK_ID)

# #             for target_task in sorted_target_tasks:
# #                 # print(f"      DFS: Agent {next_agent_to_switch} considering switch from '{original_task_of_next}' to '{target_task}'")
# #                 new_switch: SwitchOperation = (next_agent_to_switch, original_task_of_next, target_task)
# #                 potential_assignment_after_this_switch = copy.deepcopy(current_assignment_in_dfs)
# #                 potential_assignment_after_this_switch[next_agent_to_switch] = target_task

# #                 if not self._is_assignment_valid(potential_assignment_after_this_switch, target_task, next_agent_to_switch):
# #                     # print(f"        DFS: Switch for {next_agent_to_switch} to {target_task} is INVALID (constraints).")
# #                     continue

# #                 extended_chain = current_chain_being_built + [new_switch]
# #                 assignment_after_extended_chain = self._apply_chain_to_assignment(base_assignment_for_dfs, extended_chain)
# #                 # print(f"        DFS: Extended chain: {extended_chain}")
# #                 # print(f"        DFS: Assignment after extended chain: {assignment_after_extended_chain}")

# #                 utility_after_extended_chain = self._calculate_global_utility(assignment_after_extended_chain, self.current_agent_physical_states)
# #                 # print(f"        DFS: Utility after extended chain: {utility_after_extended_chain:.3f}")

# #                 if utility_after_extended_chain > agent_data.best_known_global_utility + 1e-5:
# #                     # print(f"        DFS: SUCCESS! New best by {agent_data.agent_id}. Util: {utility_after_extended_chain:.3f} > {agent_data.best_known_global_utility:.3f}")
# #                     agent_data.best_known_global_assignment = copy.deepcopy(assignment_after_extended_chain)
# #                     agent_data.best_known_global_utility = utility_after_extended_chain
# #                     agent_data.messages_generated_this_iteration +=1
# #                     new_message = KSCMessage(
# #                         base_assignment_for_dfs, extended_chain, agent_data.agent_id,
# #                         utility_after_extended_chain, agent_data.k_value,
# #                         iteration_created=iteration_num, path_taken=[agent_data.agent_id]
# #                     )
# #                     agent_data.add_message_to_buffer(new_message, agent_data.agent_id)

# #                 if utility_after_extended_chain > -float('inf') and len(extended_chain) < agent_data.k_value:
# #                     self._local_optimization_dfs(agent_data, base_assignment_for_dfs, extended_chain, potential_assignment_after_this_switch, iteration_num, visited_assignments_in_dfs)
# #                 # else:
# #                     # print(f"        DFS: Not recursing. Utility invalid or max depth for chain.")


# #     def _process_message_buffer(self, agent_data: KSCAgentData, iteration_num: int):
# #         # (与您上传的代码中的逻辑基本一致，但现在会调用修正后的内部方法)
# #         # 增加 self.max_dfs_branching_factor (对应您代码中的 DFS_BRANCHING_FACTOR)
# #         messages_to_attempt_extension = agent_data.get_messages_for_processing(count=self.max_dfs_branching_factor )
# #         processed_count_this_agent = 0
# #         # print(f"  [MSG_PROC {agent_data.agent_id}] Processing {len(messages_to_attempt_extension)} messages from buffer.")

# #         for msg_from_buffer in messages_to_attempt_extension:
# #             # print(f"    [MSG_PROC {agent_data.agent_id}] Considering message: {msg_from_buffer}")
# #             # 您的代码中没有 processed_for_extension_by，这里为了避免重复处理，保持这个逻辑
# #             if agent_data.agent_id in msg_from_buffer.processed_for_extension_by:
# #                 # print(f"      Already processed this message for extension by {agent_data.agent_id}.")
# #                 continue
            
# #             processed_count_this_agent +=1
# #             msg_from_buffer.processed_for_extension_by.add(agent_data.agent_id)

# #             if len(msg_from_buffer.chain) >= msg_from_buffer.k_limit_of_originator:
# #                 # print(f"      Chain length meets/exceeds k_limit. Cannot extend.")
# #                 continue

# #             assignment_after_buffer_chain = self._apply_chain_to_assignment(msg_from_buffer.base_assignment, msg_from_buffer.chain)
# #             original_task_of_current_agent = assignment_after_buffer_chain.get(agent_data.agent_id)

# #             sorted_target_tasks_msg_proc = sorted([tid for tid in self.task_ids_with_idle if tid != original_task_of_current_agent], key=lambda x: x == IDLE_TASK_ID)

# #             for target_task in sorted_target_tasks_msg_proc:
# #                 # print(f"      [MSG_PROC {agent_data.agent_id}] Trying to extend chain by switching self to '{target_task}'")
# #                 new_switch: SwitchOperation = (agent_data.agent_id, original_task_of_current_agent, target_task)
                
# #                 if len(msg_from_buffer.chain) + 1 > msg_from_buffer.k_limit_of_originator:
# #                     continue

# #                 potential_assignment_after_new_switch = copy.deepcopy(assignment_after_buffer_chain)
# #                 potential_assignment_after_new_switch[agent_data.agent_id] = target_task

# #                 if not self._is_assignment_valid(potential_assignment_after_new_switch, target_task, agent_data.agent_id):
# #                     # print(f"        Switch for {agent_data.agent_id} to {target_task} is INVALID (constraints).")
# #                     continue
                
# #                 extended_chain_from_buffer_msg = msg_from_buffer.chain + [new_switch]
# #                 assignment_after_extended = self._apply_chain_to_assignment(msg_from_buffer.base_assignment, extended_chain_from_buffer_msg)
# #                 utility_after_extended = self._calculate_global_utility(assignment_after_extended, self.current_agent_physical_states)
# #                 # print(f"        Extended chain utility: {utility_after_extended:.3f}")

# #                 if utility_after_extended > agent_data.best_known_global_utility + 1e-5:
# #                     # print(f"        SUCCESS! Extended msg chain results in better utility for {agent_data.agent_id}: {utility_after_extended:.3f} > {agent_data.best_known_global_utility:.3f}")
# #                     agent_data.best_known_global_assignment = copy.deepcopy(assignment_after_extended)
# #                     agent_data.best_known_global_utility = utility_after_extended
# #                     agent_data.messages_generated_this_iteration +=1
                    
# #                     new_path = list(msg_from_buffer.path_taken)
# #                     if agent_data.agent_id not in new_path: new_path.append(agent_data.agent_id)

# #                     new_extended_message = KSCMessage(
# #                         msg_from_buffer.base_assignment, extended_chain_from_buffer_msg, msg_from_buffer.originator_id,
# #                         utility_after_extended, msg_from_buffer.k_limit_of_originator,
# #                         iteration_created=iteration_num, path_taken=new_path)
# #                     agent_data.add_message_to_buffer(new_extended_message, agent_data.agent_id)
# #         agent_data.messages_processed_this_iteration += processed_count_this_agent

# #     def run_iteration(self, iteration_num: int) -> Tuple[Assignment, float, Dict[str, Any]]:
# #         # (与您最新上传的代码中的该方法基本一致，但现在会调用修正后的内部方法)
# #         # 主要确保 DFS_BRANCHING_FACTOR (即 self.max_dfs_branching_factor) 和
# #         # DFS_SEARCH_DEPTH_OTHERS (这里简化为只允许发起者扩展链的本地DFS，或通过消息扩展)
# #         # 的使用与您版本一致。
# #         # 您的版本中 _local_optimization_dfs 的 agents_to_consider_for_next_switch 逻辑更复杂，
# #         # 这里我使用了一个简化的版本，但核心是调用修正后的 _is_assignment_valid 和 _calculate_global_utility。

# #         iter_start_time = time.time()
# #         self._update_neighbor_graph()

# #         total_messages_broadcasted_this_iter = 0
# #         total_messages_processed_by_all_agents = 0

# #         for agent_id_loop in self.agent_ids: # 重置迭代统计
# #             self.agents_data[agent_id_loop].messages_generated_this_iteration = 0
# #             self.agents_data[agent_id_loop].messages_processed_this_iteration = 0
# #             self.agents_data[agent_id_loop].sent_message_ids_this_iteration = set()

# #         # 1. Local Optimization and Message Buffer Processing
# #         for agent_id_loop_loc_opt in self.agent_ids:
# #             agent_data_loc_opt = self.agents_data[agent_id_loop_loc_opt]
# #             # print(f"  [KSC Iter {iteration_num}] Agent {agent_id_loop_loc_opt} starting optimization phase...")
# #             base_assign_dfs = copy.deepcopy(agent_data_loc_opt.best_known_global_assignment)
# #             visited_dfs: Set[Tuple[Tuple[str, Optional[str]], ...]] = set()
# #             self._local_optimization_dfs(agent_data_loc_opt, base_assign_dfs, [], base_assign_dfs, iteration_num, visited_dfs)
# #             self._process_message_buffer(agent_data_loc_opt, iteration_num)
# #             total_messages_processed_by_all_agents += agent_data_loc_opt.messages_processed_this_iteration

# #         # 2. Message Propagation
# #         messages_to_deliver: Dict[str, List[KSCMessage]] = {aid: [] for aid in self.agent_ids}
# #         for sender_id_loop in self.agent_ids:
# #             sender_data_loop = self.agents_data[sender_id_loop]
# #             neighbors_loop = self.neighbor_graph.get(sender_id_loop, [])
# #             if not neighbors_loop: continue
# #             msgs_to_broadcast_loop = sender_data_loop.get_messages_for_broadcast(iteration_num, count=3) # 您代码是3
# #             for msg_loop in msgs_to_broadcast_loop:
# #                 sender_data_loop.sent_message_ids_this_iteration.add(msg_loop.message_id)
# #                 for neighbor_id_loop in neighbors_loop:
# #                     if neighbor_id_loop not in msg_loop.path_taken:
# #                         msg_for_neighbor_loop = copy.deepcopy(msg_loop)
# #                         msg_for_neighbor_loop.sender_id = sender_id_loop
# #                         new_path_for_msg = list(msg_for_neighbor_loop.path_taken)
# #                         if sender_id_loop not in new_path_for_msg : new_path_for_msg.append(sender_id_loop)
# #                         msg_for_neighbor_loop.path_taken = new_path_for_msg
# #                         messages_to_deliver[neighbor_id_loop].append(msg_for_neighbor_loop)
# #                         total_messages_broadcasted_this_iter += 1
# #         for recipient_id_loop, received_msgs_loop in messages_to_deliver.items():
# #             recipient_data_loop = self.agents_data[recipient_id_loop]
# #             for r_msg_loop in received_msgs_loop:
# #                 recipient_data_loop.add_message_to_buffer(r_msg_loop, recipient_id_loop)

# #         # 3. OAS (Optimal Assignment Sharing) and RAR (Re-evaluation and Adjustment Rule)
# #         # OAS Part
# #         current_iter_best_util = self._sim_best_utility # Start with previous global best
# #         current_iter_best_assign = copy.deepcopy(self._sim_best_assignment)
# #         for agent_id_oas_chk in self.agent_ids: # Find the best utility known by any agent
# #             agent_data_oas_chk = self.agents_data[agent_id_oas_chk]
# #             if agent_data_oas_chk.best_known_global_utility > current_iter_best_util + 1e-5:
# #                 current_iter_best_util = agent_data_oas_chk.best_known_global_utility
# #                 current_iter_best_assign = copy.deepcopy(agent_data_oas_chk.best_known_global_assignment)
        
# #         if current_iter_best_assign is not None and current_iter_best_util > self._sim_best_utility + 1e-5:
# #             # print(f"  [KSC Iter {iteration_num}] Global best updated via OAS: {self._sim_best_utility:.2f} -> {current_iter_best_util:.2f}")
# #             self._sim_best_utility = current_iter_best_util
# #             self._sim_best_assignment = copy.deepcopy(current_iter_best_assign)

# #         # RAR Part: All agents adopt the new global best and re-evaluate their message buffers
# #         for agent_id_oas_rar in self.agent_ids:
# #             agent_data_oas_rar = self.agents_data[agent_id_oas_rar]
# #             if self._sim_best_assignment is not None and \
# #                (self._sim_best_utility > agent_data_oas_rar.best_known_global_utility + 1e-5 or \
# #                 agent_data_oas_rar.best_known_global_assignment != self._sim_best_assignment):

# #                 agent_data_oas_rar.best_known_global_assignment = copy.deepcopy(self._sim_best_assignment)
# #                 agent_data_oas_rar.best_known_global_utility = self._sim_best_utility
                
# #                 new_buffer_rar = []
# #                 for msg_rar in agent_data_oas_rar.message_buffer:
# #                     # Re-evaluate message based on the NEW best_known_global_assignment as its base
# #                     re_eval_assign_rar = self._apply_chain_to_assignment(agent_data_oas_rar.best_known_global_assignment, msg_rar.chain)
# #                     re_eval_util_rar = self._calculate_global_utility(re_eval_assign_rar, self.current_agent_physical_states)

# #                     if re_eval_util_rar > -float('inf'):
# #                         updated_msg_rar = KSCMessage(
# #                             base_assignment=copy.deepcopy(agent_data_oas_rar.best_known_global_assignment),
# #                             chain=copy.deepcopy(msg_rar.chain), originator_id=msg_rar.originator_id,
# #                             achieved_utility=re_eval_util_rar, k_limit_of_originator=msg_rar.k_limit_of_originator,
# #                             message_id=msg_rar.message_id, sender_id=msg_rar.sender_id,
# #                             iteration_created=msg_rar.iteration_created, path_taken=copy.deepcopy(msg_rar.path_taken)
# #                         )
# #                         updated_msg_rar.processed_for_extension_by = set() # Allow re-processing
# #                         new_buffer_rar.append(updated_msg_rar)
                
# #                 agent_data_oas_rar.message_buffer = new_buffer_rar
# #                 agent_data_oas_rar.message_buffer.sort(key=lambda m: (m.achieved_utility, -len(m.chain)), reverse=True)
# #                 if len(agent_data_oas_rar.message_buffer) > agent_data_oas_rar.max_buffer_size:
# #                      agent_data_oas_rar.message_buffer = agent_data_oas_rar.message_buffer[:agent_data_oas_rar.max_buffer_size]

# #         self.stats_total_messages_broadcasted_per_iteration.append(total_messages_broadcasted_this_iter)
# #         self.stats_total_messages_processed_per_iteration.append(total_messages_processed_by_all_agents)
        
# #         stats_this_iteration = {
# #             "sim_best_utility": self._sim_best_utility,
# #             "messages_broadcasted": total_messages_broadcasted_this_iter,
# #             "messages_processed_by_agents": total_messages_processed_by_all_agents,
# #             "active_communication_links": self.stats_active_links_per_iteration[-1] if self.stats_active_links_per_iteration else 0
# #         }
# #         # print(f"  [KSC Iter {iteration_num}] End. Best util: {self._sim_best_utility:.2f}. Assignment: {self._sim_best_assignment}")
# #         # print(f"    Iter Time: {time.time() - iter_start_time:.3f}s")
# #         return self._sim_best_assignment, self._sim_best_utility, stats_this_iteration
# class KSCCoalitionFormation:
#     def __init__(self,
#                  agent_ids: List[str],
#                  task_ids: List[str], # These are the initial active task IDs
#                  agent_k_values: Dict[str, int],
#                  utility_calculator: UtilityCalculator,
#                  belief_manager: BeliefManager,
#                  task_type_names: List[str],
#                  initial_assignment: Optional[Assignment] = None,
#                  max_coalition_size_per_task: Optional[Dict[str, int]] = None,
#                  min_agents_per_task: Optional[Dict[str, int]] = None,
#                  agent_initial_states: Optional[Dict[str, np.ndarray]] = None,
#                  communication_range: Optional[float] = None,
#                  max_dfs_branching_factor: int = 3, # Default from ksc_coalition_formation.py
#                  max_dfs_depth_for_others: int = 1): # Default from ksc_coalition_formation.py

#         self.agent_ids = agent_ids
#         self.task_ids_actual = list(task_ids) # Current list of active tasks for KSC
#         self.task_ids_with_idle = self.task_ids_actual + [IDLE_TASK_ID]

#         self.utility_calculator = utility_calculator # Should be updated with current self.tasks by Environment
#         self.belief_manager = belief_manager     # Should be updated with current self.task_ids_actual by self
#         self.task_type_names = task_type_names

#         self.max_coalition_size = max_coalition_size_per_task if max_coalition_size_per_task else {}
#         self.min_agents_per_task = min_agents_per_task if min_agents_per_task else {}

#         self.current_agent_physical_states = agent_initial_states if agent_initial_states else self._get_current_agent_states_placeholder(self.agent_ids)
        
#         self.communication_range = communication_range
#         self.neighbor_graph: NeighborGraph = {}
        
#         self.stats_total_messages_broadcasted_per_iteration: List[int] = []
#         self.stats_total_messages_processed_per_iteration: List[int] = []
#         self.stats_active_links_per_iteration: List[int] = []
        
#         self._update_neighbor_graph() 

#         if initial_assignment is None:
#             initial_assignment = {agent_id: IDLE_TASK_ID for agent_id in self.agent_ids}

#         # Initial utility is 0 for all idle, unless min_agents_per_task makes it -inf
#         initial_global_utility = self._calculate_global_utility(initial_assignment, self.current_agent_physical_states)
#         # If it's -inf due to constraints but no tasks are assigned yet, it should be 0.
#         is_all_idle = all(tid == IDLE_TASK_ID for tid in initial_assignment.values())
#         if initial_global_utility == -float('inf') and is_all_idle and not self.min_agents_per_task:
#             initial_global_utility = 0.0
#         elif initial_global_utility == -float('inf') and is_all_idle and self.min_agents_per_task and not any(m > 0 for m in self.min_agents_per_task.values()): # if all min_req are 0
#             initial_global_utility = 0.0


#         self.agents_data: Dict[str, KSCAgentData] = {}
#         for agent_id in self.agent_ids:
#             k_val = agent_k_values.get(agent_id)
#             if k_val is None: raise ValueError(f"Agent {agent_id} 的K值未提供。")
#             self.agents_data[agent_id] = KSCAgentData(agent_id, k_val, initial_assignment, initial_global_utility)

#         self._sim_best_assignment = copy.deepcopy(initial_assignment)
#         self._sim_best_utility = initial_global_utility
        
#         self.max_dfs_branching_factor = max_dfs_branching_factor
#         self.max_dfs_depth_for_others = max_dfs_depth_for_others

#         # print(f"[KSC DEBUG __init__] Initialized. Agents: {len(self.agent_ids)}. Tasks: {len(self.task_ids_actual)}.")
#         # print(f"  [KSC DEBUG __init__] Initial Best Utility: {self._sim_best_utility:.3f}, Assignment: {self._sim_best_assignment}")
#         # if self.stats_active_links_per_iteration: print(f"  [KSC DEBUG __init__] Initial active comm links: {self.stats_active_links_per_iteration[0]}")

#     def _get_current_agent_states_placeholder(self, agent_ids_list: List[str]) -> Dict[str, np.ndarray]:
#         states = {}; angle_step = 2 * np.pi / len(agent_ids_list) if agent_ids_list else 0
#         for i, agent_id in enumerate(agent_ids_list):
#             angle = i * angle_step; radius = 150 + i * 30
#             states[agent_id] = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0, 0.0])
#         return states

#     def _update_neighbor_graph(self):
#         self.neighbor_graph = {agent_id: [] for agent_id in self.agent_ids}
#         active_links_count = 0
#         if not self.agent_ids: self.stats_active_links_per_iteration.append(0); return
#         if self.communication_range is None: 
#             for agent_id in self.agent_ids: self.neighbor_graph[agent_id] = [other_id for other_id in self.agent_ids if other_id != agent_id]
#             if len(self.agent_ids) > 1: active_links_count = len(self.agent_ids) * (len(self.agent_ids) -1) // 2 # Each link counted once
#         else:
#             agent_ids_list = list(self.agent_ids) 
#             for i in range(len(agent_ids_list)):
#                 for j in range(i + 1, len(agent_ids_list)):
#                     agent1_id, agent2_id = agent_ids_list[i], agent_ids_list[j]
#                     if agent1_id not in self.current_agent_physical_states or agent2_id not in self.current_agent_physical_states: continue 
#                     pos1 = self.current_agent_physical_states[agent1_id][:2]; pos2 = self.current_agent_physical_states[agent2_id][:2]
#                     if np.linalg.norm(pos1 - pos2) <= self.communication_range:
#                         self.neighbor_graph[agent1_id].append(agent2_id); self.neighbor_graph[agent2_id].append(agent1_id)
#                         active_links_count += 1 
#         self.stats_active_links_per_iteration.append(active_links_count)

#     def update_agent_physical_states(self, new_states: Dict[str, np.ndarray]):
#         self.current_agent_physical_states.update(new_states)
#         self._update_neighbor_graph() 
#         current_utility_of_best_assignment = self._calculate_global_utility(self._sim_best_assignment, self.current_agent_physical_states)
#         self._sim_best_utility = current_utility_of_best_assignment
#         for agent_id in self.agent_ids:
#             agent_data = self.agents_data[agent_id]
#             utility_of_agent_known_best = self._calculate_global_utility(agent_data.best_known_global_assignment, self.current_agent_physical_states)
#             agent_data.best_known_global_utility = utility_of_agent_known_best
#             if self._sim_best_utility > agent_data.best_known_global_utility + 1e-5 :
#                  agent_data.best_known_global_assignment = copy.deepcopy(self._sim_best_assignment)
#                  agent_data.best_known_global_utility = self._sim_best_utility
    
#     def notify_task_completion(self, task_id_completed: str, agents_in_completed_coalition: List[str]):
#         # print(f"  [KSC NOTIFY_TASK_COMPLETION] Task '{task_id_completed}' completed by {agents_in_completed_coalition}.")
#         if task_id_completed in self.task_ids_actual:
#             self.task_ids_actual.remove(task_id_completed) 
#             self.task_ids_with_idle = self.task_ids_actual + [IDLE_TASK_ID]
            
#             # Update UtilityCalculator's view (it should operate on a copy or be refreshed by Environment)
#             # The Environment's step loop should pass the updated self.tasks to utility_calculator
#             # So, KSC doesn't need to directly manipulate utility_calculator.task_definitions if env does it.
#             # However, if KSC's utility_calculator holds its own copy, it must be updated.
#             if self.utility_calculator and hasattr(self.utility_calculator, 'task_definitions'):
#                  if task_id_completed in self.utility_calculator.task_definitions:
#                     # print(f"    [KSC NOTIFY] Removing {task_id_completed} from utility_calculator.task_definitions")
#                     del self.utility_calculator.task_definitions[task_id_completed]
            
#             if self.belief_manager: # Update BeliefManager's view of active tasks
#                 self.belief_manager.task_ids = list(self.task_ids_actual)
            
#             needs_re_evaluation = False
#             for agent_id_notify in self.agent_ids: 
#                 if agent_id_notify in self._sim_best_assignment and self._sim_best_assignment[agent_id_notify] == task_id_completed:
#                     self._sim_best_assignment[agent_id_notify] = IDLE_TASK_ID
#                     needs_re_evaluation = True
#                 if agent_id_notify in self.agents_data:
#                     agent_data_notify = self.agents_data[agent_id_notify]
#                     if agent_id_notify in agent_data_notify.best_known_global_assignment and \
#                        agent_data_notify.best_known_global_assignment[agent_id_notify] == task_id_completed:
#                         agent_data_notify.best_known_global_assignment[agent_id_notify] = IDLE_TASK_ID
#                     new_buffer = []
#                     for msg in agent_data_notify.message_buffer:
#                         chain_involves_completed = any(op[1] == task_id_completed or op[2] == task_id_completed for op in msg.chain)
#                         base_involves_completed = any(tid == task_id_completed for tid in msg.base_assignment.values())
#                         # More precise: check if the *new_task_id* in the chain is the completed task
#                         # or if the base assignment assigns to the completed task
#                         # For now, the broader removal is safer.
#                         if not (chain_involves_completed or base_involves_completed):
#                             new_buffer.append(msg)
#                         # else:
#                             # print(f"    [KSC NOTIFY] Agent {agent_id_notify} removing msg (ID: {msg.message_id[:4]}) due to task {task_id_completed} completion.")
#                     agent_data_notify.message_buffer = new_buffer
            
#             if needs_re_evaluation: 
#                  self._sim_best_utility = self._calculate_global_utility(self._sim_best_assignment, self.current_agent_physical_states)
#                  for ad_val_notify in self.agents_data.values(): 
#                      # Ensure all agents are aware of the new best utility after re-evaluation
#                      if self._sim_best_utility > ad_val_notify.best_known_global_utility or \
#                         ad_val_notify.best_known_global_assignment != self._sim_best_assignment:
#                          ad_val_notify.best_known_global_assignment = copy.deepcopy(self._sim_best_assignment)
#                          ad_val_notify.best_known_global_utility = self._sim_best_utility
#         # else:
#             # print(f"  [KSC NOTIFY_TASK_COMPLETION] Task '{task_id_completed}' was not in active KSC task list {self.task_ids_actual}.")


#     def _calculate_global_utility(self, assignment: Assignment, agent_states: Dict[str, np.ndarray]) -> float:
#         total_utility = 0.0
#         # print(f"  [KSC _calc_util] Assignment: {assignment}, KSC Active Tasks: {self.task_ids_actual}")
        
#         # Check if all agents are idle AND there are no active tasks for KSC to consider
#         all_agents_idle_in_assignment = all(task_id is None or task_id == IDLE_TASK_ID for task_id in assignment.values())
#         if all_agents_idle_in_assignment and not self.task_ids_actual: # No active tasks left for KSC
#             # print(f"    [KSC _calc_util] All agents idle and no active tasks. Utility = 0.0 (Base case).")
#             return 0.0

#         assigned_agents_count_per_task: Dict[str, int] = {task_id: 0 for task_id in self.task_ids_actual}        

#         for agent_id_iter, task_id_iter in assignment.items():
#             if task_id_iter is not None and task_id_iter != IDLE_TASK_ID:
#                 if task_id_iter not in self.task_ids_actual: # Task must be in KSC's current list of active tasks
#                     # print(f"    [KSC _calc_util CRITICAL] Assigned task {task_id_iter} for agent {agent_id_iter} not in KSC's current active tasks {self.task_ids_actual}. Invalid assignment, Utility = -inf.")
#                     return -float('inf')
#                 assigned_agents_count_per_task[task_id_iter] +=1

#         if not self.utility_calculator: return -float('inf') # Should not happen if initialized correctly

#         for task_id_check_constraints in self.task_ids_actual: # Iterate only KSC's active tasks
#             task_def_constraints = self.utility_calculator.task_definitions.get(task_id_check_constraints)
#             # It's possible utility_calculator.task_definitions was pruned by notify_task_completion
#             # while self.task_ids_actual (KSC's list) might not be pruned yet if calc_util is called mid-iteration.
#             # However, Environment should update utility_calculator.task_definitions before KSC runs.
#             if not task_def_constraints: # If task no longer in utility calculator's list (e.g. completed and removed)
#                 if assigned_agents_count_per_task.get(task_id_check_constraints, 0) > 0:
#                     # print(f"    [KSC _calc_util CRITICAL] Task {task_id_check_constraints} is assigned but not in utility_calculator definitions. Invalid assignment. Utility = -inf.")
#                     return -float('inf') 
#                 continue # Task is not in utility calc, so it won't contribute utility and constraints are moot.

#             task_type_name_constraints = task_def_constraints.get("true_type_name")
#             count_for_this_task = assigned_agents_count_per_task.get(task_id_check_constraints, 0)

#             if task_type_name_constraints: # Only apply if type name is valid for constraint lookup
#                 min_req = self.min_agents_per_task.get(task_type_name_constraints, 0) 
#                 if 0 < count_for_this_task < min_req:
#                     # print(f"    [KSC _calc_util CONSTRAINT] Task {task_id_check_constraints} (Type: {task_type_name_constraints}) violates min_agents ({min_req}). Has: {count_for_this_task}. Utility = -inf.")
#                     return -float('inf')
                
#                 max_allowed = self.max_coalition_size.get(task_type_name_constraints)
#                 if max_allowed is not None and count_for_this_task > max_allowed:
#                     # print(f"    [KSC _calc_util CONSTRAINT] Task {task_id_check_constraints} (Type: {task_type_name_constraints}) violates max_agents ({max_allowed}). Has: {count_for_this_task}. Utility = -inf.")
#                     return -float('inf')
#             # else:
#                 # print(f"    [KSC _calc_util WARNING] Task {task_id_check_constraints} has no 'true_type_name' in utility_calculator's def. Min/max constraints for this task type might not apply as expected.")


#         for agent_id, task_id in assignment.items():
#             if task_id is not None and task_id != IDLE_TASK_ID:
#                 # At this point, task_id should be in self.task_ids_actual and also in self.utility_calculator.task_definitions
#                 # (unless there's a sync issue, which the constraint check above tries to catch)
#                 current_agent_state = agent_states.get(agent_id)
#                 if current_agent_state is None: return -float('inf') 

#                 if not self.belief_manager: return -float('inf')
                
#                 # Belief manager's task list (self.belief_manager.task_ids) is also updated by notify_task_completion
#                 # to match KSC's self.task_ids_actual.
#                 agent_beliefs = self.belief_manager.get_expected_belief_dist_for_task(agent_id, task_id)
#                 # get_expected_task_utility_for_agent should handle agent_beliefs being None
                
#                 utility = self.utility_calculator.get_expected_task_utility_for_agent(
#                     agent_id, current_agent_state, task_id, agent_beliefs, self.task_type_names
#                 )
#                 if utility == -float('inf'): return -float('inf')
#                 total_utility += utility
        
#         # print(f"  [KSC _calc_util RESULT] Assignment: {assignment}, Final Total Utility: {total_utility:.3f}")
#         return total_utility

#     def _is_assignment_valid(self, assignment: Assignment, task_id_to_check: Optional[str], agent_id_switched: str) -> bool:
#         if task_id_to_check is None or task_id_to_check == IDLE_TASK_ID: return True
        
#         # Task must be in KSC's current list of active tasks
#         if task_id_to_check not in self.task_ids_actual: 
#             # print(f"    [KSC _is_valid DEBUG] Task '{task_id_to_check}' not in KSC's active tasks {self.task_ids_actual}. Invalid.")
#             return False 

#         # Task definition must exist in the utility calculator (which should have an up-to-date list from env)
#         if not self.utility_calculator or task_id_to_check not in self.utility_calculator.task_definitions:
#             # print(f"    [KSC _is_valid DEBUG] Task '{task_id_to_check}' not in utility_calculator definitions. Invalid.")
#             return False 
        
#         task_def = self.utility_calculator.task_definitions[task_id_to_check]
#         task_type_name = task_def.get("true_type_name")

#         if task_type_name and self.max_coalition_size: # Check only if type and constraints exist
#             max_size_for_this_task_type = self.max_coalition_size.get(task_type_name)
#             if max_size_for_this_task_type is not None:
#                 current_size = sum(1 for assigned_tid_val in assignment.values() if assigned_tid_val == task_id_to_check)
#                 if current_size > max_size_for_this_task_type:
#                     # print(f"    [KSC _is_valid DEBUG] Task '{task_id_to_check}' (Type '{task_type_name}') violates max size ({max_size_for_this_task_type}). Current: {current_size}. Invalid.")
#                     return False
#         return True

#     def _apply_chain_to_assignment(self, base_assignment: Assignment, chain: TransformationChain) -> Assignment:
#         new_assignment = copy.deepcopy(base_assignment)
#         for agent_id, _, new_task_id in chain: # Original old_task_id from chain is not used here
#             new_assignment[agent_id] = new_task_id
#         return new_assignment

#     # _local_optimization_dfs, _process_message_buffer, run_iteration:
#     # These methods heavily rely on _calculate_global_utility and _is_assignment_valid.
#     # The key is that if these utility/validity checks consistently make non-IDLE assignments unattractive or invalid,
#     # KSC will not find better assignments.
#     # The logic within these methods from your ksc_coalition_formation.py (assumed to be the latest/target) seems plausible
#     # given correct utility and validity inputs.
#     # (For brevity, their code from your file is assumed here if no specific changes are needed beyond what _calculate_global_utility provides)
#     def _local_optimization_dfs(self, agent_data: KSCAgentData, base_assignment_for_dfs: Assignment, current_chain_being_built: TransformationChain, current_assignment_in_dfs: Assignment, iteration_num: int, visited_assignments_in_dfs: Set[Tuple[Tuple[str, Optional[str]], ...]]):
#         assignment_tuple = tuple(sorted(current_assignment_in_dfs.items()));
#         if assignment_tuple in visited_assignments_in_dfs: return
#         visited_assignments_in_dfs.add(assignment_tuple)
#         if len(current_chain_being_built) >= agent_data.k_value: return
#         agents_to_consider_for_next_switch = []
#         if not current_chain_being_built: agents_to_consider_for_next_switch = [agent_data.agent_id]
#         else: # Simplified: only originator can extend in local DFS unless branching factor allows more
#             if self.max_dfs_branching_factor > 1 or not current_chain_being_built: # Allow branching or first step
#                 possible_others = [aid for aid in self.agent_ids if aid != agent_data.agent_id and aid not in [op[0] for op in current_chain_being_built]] # Avoid agent switching multiple times in one chain explored by one agent
#                 random.shuffle(possible_others)
#                 agents_to_consider_for_next_switch = [agent_data.agent_id] + possible_others[:max(0, self.max_dfs_branching_factor -1)]
#             else: # Only originator can continue the chain locally beyond depth 1 for others
#                 agents_to_consider_for_next_switch = [agent_data.agent_id]
#             agents_to_consider_for_next_switch = list(set(agents_to_consider_for_next_switch)) # Unique

#         for next_agent_to_switch in agents_to_consider_for_next_switch:
#             original_task_of_next = current_assignment_in_dfs.get(next_agent_to_switch, IDLE_TASK_ID)
#             # Iterate through KSC's current view of active tasks + IDLE
#             sorted_target_tasks = sorted([tid for tid in self.task_ids_with_idle if tid != original_task_of_next], key=lambda x: x == IDLE_TASK_ID)
#             for target_task in sorted_target_tasks:
#                 new_switch: SwitchOperation = (next_agent_to_switch, original_task_of_next, target_task) # Corrected order
#                 potential_assignment_after_this_switch = copy.deepcopy(current_assignment_in_dfs)
#                 potential_assignment_after_this_switch[next_agent_to_switch] = target_task
#                 if not self._is_assignment_valid(potential_assignment_after_this_switch, target_task, next_agent_to_switch): continue
#                 extended_chain = current_chain_being_built + [new_switch]
#                 assignment_after_extended_chain = self._apply_chain_to_assignment(base_assignment_for_dfs, extended_chain)
#                 utility_after_extended_chain = self._calculate_global_utility(assignment_after_extended_chain, self.current_agent_physical_states)
#                 # Add a small positive bias for non-idle assignments if utility is exactly 0, to explore.
#                 # Or ensure IDLE has a tiny negative utility if all other options are truly zero.
#                 # For now, stick to strict utility.
#                 if utility_after_extended_chain > agent_data.best_known_global_utility + 1e-5: # Epsilon
#                     agent_data.best_known_global_assignment = copy.deepcopy(assignment_after_extended_chain)
#                     agent_data.best_known_global_utility = utility_after_extended_chain
#                     agent_data.messages_generated_this_iteration +=1
#                     new_message = KSCMessage(base_assignment_for_dfs, extended_chain, agent_data.agent_id, utility_after_extended_chain, agent_data.k_value, iteration_created=iteration_num, path_taken=[agent_data.agent_id])
#                     agent_data.add_message_to_buffer(new_message, agent_data.agent_id)
#                 if utility_after_extended_chain > -float('inf') and len(extended_chain) < agent_data.k_value:
#                     self._local_optimization_dfs(agent_data, base_assignment_for_dfs, extended_chain, potential_assignment_after_this_switch, iteration_num, visited_assignments_in_dfs)

#     def _process_message_buffer(self, agent_data: KSCAgentData, iteration_num: int):
#         messages_to_attempt_extension = agent_data.get_messages_for_processing(count=self.max_dfs_branching_factor + 1)
#         processed_count_this_agent = 0
#         for msg_from_buffer in messages_to_attempt_extension:
#             if agent_data.agent_id in msg_from_buffer.processed_for_extension_by: continue
#             processed_count_this_agent +=1
#             msg_from_buffer.processed_for_extension_by.add(agent_data.agent_id)
#             if len(msg_from_buffer.chain) >= msg_from_buffer.k_limit_of_originator: continue
#             assignment_after_buffer_chain = self._apply_chain_to_assignment(msg_from_buffer.base_assignment, msg_from_buffer.chain)
#             original_task_of_current_agent = assignment_after_buffer_chain.get(agent_data.agent_id, IDLE_TASK_ID)
#             sorted_target_tasks_msg_proc = sorted([tid for tid in self.task_ids_with_idle if tid != original_task_of_current_agent], key=lambda x: x == IDLE_TASK_ID)
#             for target_task in sorted_target_tasks_msg_proc:
#                 new_switch: SwitchOperation = (agent_data.agent_id, original_task_of_current_agent, target_task) # Corrected
#                 if len(msg_from_buffer.chain) + 1 > msg_from_buffer.k_limit_of_originator: continue
#                 potential_assignment_after_new_switch = copy.deepcopy(assignment_after_buffer_chain)
#                 potential_assignment_after_new_switch[agent_data.agent_id] = target_task
#                 if not self._is_assignment_valid(potential_assignment_after_new_switch, target_task, agent_data.agent_id): continue
#                 extended_chain_from_buffer_msg = msg_from_buffer.chain + [new_switch]
#                 assignment_after_extended = self._apply_chain_to_assignment(msg_from_buffer.base_assignment, extended_chain_from_buffer_msg)
#                 utility_after_extended = self._calculate_global_utility(assignment_after_extended, self.current_agent_physical_states)
#                 if utility_after_extended > agent_data.best_known_global_utility + 1e-5: # Epsilon
#                     agent_data.best_known_global_assignment = copy.deepcopy(assignment_after_extended)
#                     agent_data.best_known_global_utility = utility_after_extended
#                     agent_data.messages_generated_this_iteration +=1
#                     new_path = list(msg_from_buffer.path_taken);
#                     if agent_data.agent_id not in new_path: new_path.append(agent_data.agent_id)
#                     new_extended_message = KSCMessage(msg_from_buffer.base_assignment, extended_chain_from_buffer_msg, msg_from_buffer.originator_id, utility_after_extended, msg_from_buffer.k_limit_of_originator, iteration_created=iteration_num, path_taken=new_path)
#                     agent_data.add_message_to_buffer(new_extended_message, agent_data.agent_id)
#         agent_data.messages_processed_this_iteration += processed_count_this_agent

#     def run_iteration(self, iteration_num: int) -> Tuple[Assignment, float, Dict[str, Any]]:
#         iter_start_time = time.time(); self._update_neighbor_graph()
#         total_messages_broadcasted_this_iter = 0; total_messages_processed_by_all_agents = 0 
#         for agent_id_iter_init in self.agent_ids: 
#             self.agents_data[agent_id_iter_init].messages_generated_this_iteration = 0
#             self.agents_data[agent_id_iter_init].messages_processed_this_iteration = 0 
#             self.agents_data[agent_id_iter_init].sent_message_ids_this_iteration = set() 
#         for agent_id_local_opt in self.agent_ids:
#             agent_data_local_opt = self.agents_data[agent_id_local_opt]
#             base_assignment_for_dfs_local = copy.deepcopy(agent_data_local_opt.best_known_global_assignment)
#             visited_dfs_set_local: Set[Tuple[Tuple[str, Optional[str]], ...]] = set()
#             self._local_optimization_dfs(agent_data_local_opt, base_assignment_for_dfs_local, [], base_assignment_for_dfs_local, iteration_num, visited_dfs_set_local)
#             self._process_message_buffer(agent_data_local_opt, iteration_num)
#             total_messages_processed_by_all_agents += agent_data_local_opt.messages_processed_this_iteration
#         messages_to_be_delivered_next: Dict[str, List[KSCMessage]] = {aid: [] for aid in self.agent_ids}
#         for agent_id_sender_prop in self.agent_ids:
#             agent_data_sender_prop = self.agents_data[agent_id_sender_prop]
#             neighbors_prop = self.neighbor_graph.get(agent_id_sender_prop, [])
#             if not neighbors_prop: continue
#             msgs_to_broadcast_prop = agent_data_sender_prop.get_messages_for_broadcast(iteration_num, count=3) # User had count=3
#             for msg_broadcast_item in msgs_to_broadcast_prop:
#                 agent_data_sender_prop.sent_message_ids_this_iteration.add(msg_broadcast_item.message_id)
#                 for neighbor_id_receiver_prop in neighbors_prop:
#                     if neighbor_id_receiver_prop not in msg_broadcast_item.path_taken: 
#                         msg_for_neighbor_prop = copy.deepcopy(msg_broadcast_item); msg_for_neighbor_prop.sender_id = agent_id_sender_prop
#                         new_path_for_msg_prop = list(msg_for_neighbor_prop.path_taken) 
#                         if agent_id_sender_prop not in new_path_for_msg_prop : new_path_for_msg_prop.append(agent_id_sender_prop)
#                         msg_for_neighbor_prop.path_taken = new_path_for_msg_prop
#                         messages_to_be_delivered_next[neighbor_id_receiver_prop].append(msg_for_neighbor_prop)
#                         total_messages_broadcasted_this_iter += 1
#         for recipient_id_prop, received_msgs_list_prop in messages_to_be_delivered_next.items():
#             recipient_data_prop = self.agents_data[recipient_id_prop]
#             for r_msg_item_prop in received_msgs_list_prop: recipient_data_prop.add_message_to_buffer(r_msg_item_prop, recipient_id_prop)
        
#         current_iteration_best_utility_oas = -float('inf')
#         current_iteration_best_assignment_oas = self._sim_best_assignment # Start with current global best
#         if self._sim_best_utility > current_iteration_best_utility_oas : # Ensure current_iter_best_util is at least sim_best_utility
#             current_iteration_best_utility_oas = self._sim_best_utility


#         for agent_id_oas_find in self.agent_ids: 
#             agent_data_oas_find = self.agents_data[agent_id_oas_find]
#             if agent_data_oas_find.best_known_global_utility > current_iteration_best_utility_oas + 1e-5: # Epsilon
#                 current_iteration_best_utility_oas = agent_data_oas_find.best_known_global_utility
#                 current_iteration_best_assignment_oas = copy.deepcopy(agent_data_oas_find.best_known_global_assignment)
        
#         if current_iteration_best_utility_oas > self._sim_best_utility + 1e-5: # Epsilon
#             self._sim_best_utility = current_iteration_best_utility_oas
#             self._sim_best_assignment = copy.deepcopy(current_iteration_best_assignment_oas)

#         for agent_id_oas_sync in self.agent_ids: 
#             agent_data_oas_sync = self.agents_data[agent_id_oas_sync]
#             if self._sim_best_utility > agent_data_oas_sync.best_known_global_utility + 1e-5: # Epsilon
#                 agent_data_oas_sync.best_known_global_assignment = copy.deepcopy(self._sim_best_assignment)
#                 agent_data_oas_sync.best_known_global_utility = self._sim_best_utility
#                 new_buffer_after_rar_sync = []
#                 for msg_in_buf_sync in agent_data_oas_sync.message_buffer:
#                     re_evaluated_assignment_sync = self._apply_chain_to_assignment(agent_data_oas_sync.best_known_global_assignment, msg_in_buf_sync.chain)
#                     re_evaluated_utility_sync = self._calculate_global_utility(re_evaluated_assignment_sync, self.current_agent_physical_states)
#                     if re_evaluated_utility_sync > -float('inf') and \
#                        (re_evaluated_utility_sync > self._sim_best_utility * 0.5 or re_evaluated_utility_sync > msg_in_buf_sync.achieved_utility * 0.8): 
#                         updated_msg_sync = KSCMessage(agent_data_oas_sync.best_known_global_assignment, msg_in_buf_sync.chain, msg_in_buf_sync.originator_id, re_evaluated_utility_sync, msg_in_buf_sync.k_limit_of_originator, iteration_created=msg_in_buf_sync.iteration_created, path_taken=copy.deepcopy(msg_in_buf_sync.path_taken), message_id=msg_in_buf_sync.message_id)
#                         updated_msg_sync.processed_for_extension_by = set() 
#                         new_buffer_after_rar_sync.append(updated_msg_sync)
#                 agent_data_oas_sync.message_buffer = new_buffer_after_rar_sync
#                 agent_data_oas_sync.message_buffer.sort(key=lambda m: (m.achieved_utility, -len(m.chain)), reverse=True)
#                 if len(agent_data_oas_sync.message_buffer) > agent_data_oas_sync.max_buffer_size:
#                      agent_data_oas_sync.message_buffer = agent_data_oas_sync.message_buffer[:agent_data_oas_sync.max_buffer_size]
        
#         self.stats_total_messages_broadcasted_per_iteration.append(total_messages_broadcasted_this_iter)
#         self.stats_total_messages_processed_per_iteration.append(total_messages_processed_by_all_agents)
#         stats_iter = {"sim_best_utility": self._sim_best_utility, "messages_broadcasted": total_messages_broadcasted_this_iter, "messages_processed_by_agents": total_messages_processed_by_all_agents, "active_communication_links": self.stats_active_links_per_iteration[-1] if self.stats_active_links_per_iteration else 0}
#         return self._sim_best_assignment, self._sim_best_utility, stats_iter
# strong_communication/ksc_coalition_formation.py
import sys
import os
# --- 统一的导入路径管理 ---
_ksc_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir_ksc = os.path.dirname(_ksc_dir) 
if _project_root_dir_ksc not in sys.path:
    sys.path.insert(0, _project_root_dir_ksc)
# --- 路径管理结束 ---
import numpy as np
import random
import copy
from typing import Dict, List, Tuple, Any, Optional, Set
import uuid
import time
# import matplotlib.pyplot as plt # Not typically needed in core algorithm file
# import matplotlib
# import seaborn as sns # Not typically needed

from strong_communication.belief_manager import BeliefManager 
from strong_communication.utility_calculator import UtilityCalculator

# (KSCMessage, KSCAgentData 类定义保持与您之前确认的版本一致)
Assignment = Dict[str, Optional[str]]
SwitchOperation = Tuple[str, Optional[str], Optional[str]]
TransformationChain = List[SwitchOperation]
IDLE_TASK_ID = "IDLE_TASK_PLACEHOLDER" 
NeighborGraph = Dict[str, List[str]]

class KSCMessage: 
    def __init__(self,
                 base_assignment: Assignment, chain: TransformationChain, originator_id: str,
                 achieved_utility: float, k_limit_of_originator: int,
                 message_id: Optional[str] = None, sender_id: Optional[str] = None,
                 iteration_created: int = -1, path_taken: Optional[List[str]] = None):
        self.base_assignment = copy.deepcopy(base_assignment)
        self.chain = copy.deepcopy(chain)
        self.originator_id = originator_id
        self.achieved_utility = achieved_utility
        self.k_limit_of_originator = k_limit_of_originator
        self.message_id = message_id if message_id else str(uuid.uuid4())
        self.sender_id = sender_id
        self.iteration_created = iteration_created
        self.path_taken = path_taken if path_taken is not None else []
        self.processed_for_extension_by: Set[str] = set()

    def __repr__(self):
        chain_str = " -> ".join([f"{a}({t_old if t_old else 'Idle'}->{t_new if t_new else 'Idle'})" for a, t_old, t_new in self.chain])
        path_str = "->".join(self.path_taken)
        chain_display_limit = 50 
        if len(chain_str) > chain_display_limit: chain_str = chain_str[:chain_display_limit] + "..."
        return (f"KSCMsg(id={self.message_id[:4]},orig={self.originator_id},send={self.sender_id},"
                f"k_lim={self.k_limit_of_originator},util={self.achieved_utility:.2f},"
                f"len={len(self.chain)},iter={self.iteration_created},path_len={len(self.path_taken)},chain=[{chain_str}])")

    def __eq__(self, other): 
        if not isinstance(other, KSCMessage): return NotImplemented
        return self.message_id == other.message_id
    def __hash__(self): return hash(self.message_id) 

class KSCAgentData: 
    def __init__(self, agent_id: str, k_value: int, initial_global_assignment: Assignment, initial_global_utility: float):
        self.agent_id = agent_id
        self.k_value = k_value
        self.message_buffer: List[KSCMessage] = []
        self.best_known_global_assignment: Assignment = copy.deepcopy(initial_global_assignment)
        self.best_known_global_utility: float = initial_global_utility
        self.max_buffer_size = 50 
        self.messages_processed_this_iteration = 0 
        self.messages_generated_this_iteration = 0 
        self.sent_message_ids_this_iteration: Set[str] = set() 

    def add_message_to_buffer(self, message: KSCMessage, current_agent_id: str):
        if len(message.chain) > message.k_limit_of_originator: return
        if current_agent_id in message.path_taken and message.originator_id != current_agent_id : return
        existing_msg_idx = -1
        for i, existing_msg in enumerate(self.message_buffer):
            if existing_msg.message_id == message.message_id: existing_msg_idx = i; break
        if existing_msg_idx != -1: 
            if message.achieved_utility > self.message_buffer[existing_msg_idx].achieved_utility: 
                self.message_buffer[existing_msg_idx] = message 
        else: self.message_buffer.append(message)
        self.message_buffer.sort(key=lambda m: (m.achieved_utility, -len(m.chain)), reverse=True) 
        if len(self.message_buffer) > self.max_buffer_size:
            self.message_buffer = self.message_buffer[:self.max_buffer_size]

    def get_messages_for_processing(self, count: int = 3) -> List[KSCMessage]: # Match Visualizer call if it uses count=5
        return [copy.deepcopy(msg) for msg in self.message_buffer[:count]]

    def get_messages_for_broadcast(self, iteration_num: int, count: int = 3) -> List[KSCMessage]:
        candidates = []
        for msg in self.message_buffer:
            if msg.message_id in self.sent_message_ids_this_iteration: continue
            is_new_this_iteration = (msg.iteration_created == iteration_num)
            is_significantly_better = (msg.achieved_utility > self.best_known_global_utility + 1e-5) 
            if is_new_this_iteration and is_significantly_better:
                if len(candidates) < count: candidates.append(copy.deepcopy(msg))
                else: break
            elif is_significantly_better and len(candidates) < count // 2 :
                 candidates.append(copy.deepcopy(msg))
        if len(candidates) < count:
            for msg in self.message_buffer:
                if msg.message_id in self.sent_message_ids_this_iteration or msg in candidates: continue
                if msg.iteration_created == iteration_num and len(candidates) < count: candidates.append(copy.deepcopy(msg))
                elif len(candidates) >= count: break
        return candidates

class KSCCoalitionFormation:
    def __init__(self,
                 agent_ids: List[str], task_ids: List[str], agent_k_values: Dict[str, int],
                 utility_calculator: UtilityCalculator, belief_manager: BeliefManager, task_type_names: List[str],
                 initial_assignment: Optional[Assignment] = None,
                 max_coalition_size_per_task: Optional[Dict[str, int]] = None,
                 min_agents_per_task: Optional[Dict[str, int]] = None,
                 agent_initial_states: Optional[Dict[str, np.ndarray]] = None,
                 communication_range: Optional[float] = None,
                 max_dfs_branching_factor: int = 2, # Keep defaults from your ksc_coalition_formation.py
                 max_dfs_depth_for_others: int = 1):

        self.agent_ids = agent_ids
        self.task_ids_actual = list(task_ids) # Store the original list of active tasks
        self.task_ids_with_idle = self.task_ids_actual + [IDLE_TASK_ID]

        self.utility_calculator = utility_calculator
        self.belief_manager = belief_manager
        self.task_type_names = task_type_names

        self.max_coalition_size = max_coalition_size_per_task if max_coalition_size_per_task else {}
        self.min_agents_per_task = min_agents_per_task if min_agents_per_task else {}

        self.current_agent_physical_states = agent_initial_states if agent_initial_states \
            else self._get_current_agent_states_placeholder(self.agent_ids)
        
        self.communication_range = communication_range
        self.neighbor_graph: NeighborGraph = {}
        
        self.stats_total_messages_broadcasted_per_iteration: List[int] = []
        self.stats_total_messages_processed_per_iteration: List[int] = []
        self.stats_active_links_per_iteration: List[int] = []
        
        self._update_neighbor_graph() 

        if initial_assignment is None:
            initial_assignment = {agent_id: IDLE_TASK_ID for agent_id in self.agent_ids}

        # Initial utility calculation requires self.utility_calculator to have task_definitions set.
        # This is usually done by Environment before KSC is used.
        # If KSC is init'd first, utility_calculator might not have tasks yet.
        # Let's assume utility_calculator.task_definitions is up-to-date by now.
        initial_global_utility = self._calculate_global_utility(initial_assignment, self.current_agent_physical_states)
        if initial_global_utility == -float('inf') and not self.min_agents_per_task :
            initial_global_utility = 0.0 # Default utility for all idle if no constraints making it invalid

        self.agents_data: Dict[str, KSCAgentData] = {}
        for agent_id in self.agent_ids:
            k_val = agent_k_values.get(agent_id)
            if k_val is None: raise ValueError(f"Agent {agent_id} 的K值未提供。")
            self.agents_data[agent_id] = KSCAgentData(agent_id, k_val, initial_assignment, initial_global_utility)

        self._sim_best_assignment = copy.deepcopy(initial_assignment)
        self._sim_best_utility = initial_global_utility
        
        self.max_dfs_branching_factor = max_dfs_branching_factor
        self.max_dfs_depth_for_others = max_dfs_depth_for_others

    def _get_current_agent_states_placeholder(self, agent_ids_list: List[str]) -> Dict[str, np.ndarray]:
        states = {}; angle_step = 2 * np.pi / len(agent_ids_list) if agent_ids_list else 0
        for i, agent_id in enumerate(agent_ids_list):
            angle = i * angle_step; radius = 150 + i * 30
            states[agent_id] = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0, 0.0])
        return states

    def _update_neighbor_graph(self): # (Unchanged from your version)
        self.neighbor_graph = {agent_id: [] for agent_id in self.agent_ids}
        active_links_count = 0
        if self.communication_range is None: 
            for agent_id in self.agent_ids:
                self.neighbor_graph[agent_id] = [other_id for other_id in self.agent_ids if other_id != agent_id]
            if len(self.agent_ids) > 1: active_links_count = len(self.agent_ids) * (len(self.agent_ids) -1) // 2
        else:
            agent_ids_list = list(self.agent_ids) 
            for i in range(len(agent_ids_list)):
                for j in range(i + 1, len(agent_ids_list)):
                    agent1_id, agent2_id = agent_ids_list[i], agent_ids_list[j]
                    if agent1_id not in self.current_agent_physical_states or \
                       agent2_id not in self.current_agent_physical_states: continue 
                    pos1 = self.current_agent_physical_states[agent1_id][:2] 
                    pos2 = self.current_agent_physical_states[agent2_id][:2]
                    if np.linalg.norm(pos1 - pos2) <= self.communication_range:
                        self.neighbor_graph[agent1_id].append(agent2_id)
                        self.neighbor_graph[agent2_id].append(agent1_id)
                        active_links_count +=1 
        if not self.stats_active_links_per_iteration or self.stats_active_links_per_iteration[-1] != active_links_count:
            self.stats_active_links_per_iteration.append(active_links_count)
        elif not self.stats_active_links_per_iteration and active_links_count == 0 and len(self.agent_ids)>0 : # First time, no links
            self.stats_active_links_per_iteration.append(0)


    def update_agent_physical_states(self, new_states: Dict[str, np.ndarray]): # (Unchanged)
        self.current_agent_physical_states.update(new_states)
        self._update_neighbor_graph() 
        current_utility_of_best_assignment = self._calculate_global_utility(self._sim_best_assignment, self.current_agent_physical_states)
        self._sim_best_utility = current_utility_of_best_assignment
        for agent_id in self.agent_ids:
            agent_data = self.agents_data[agent_id]
            utility_of_agent_known_best = self._calculate_global_utility(agent_data.best_known_global_assignment, self.current_agent_physical_states)
            agent_data.best_known_global_utility = utility_of_agent_known_best
            if self._sim_best_utility > agent_data.best_known_global_utility + 1e-5 :
                 agent_data.best_known_global_assignment = copy.deepcopy(self._sim_best_assignment)
                 agent_data.best_known_global_utility = self._sim_best_utility
    
    def notify_task_completion(self, task_id_completed: str, agents_in_completed_coalition: List[str]):
        # print(f"  [KSC NOTIFY DEBUG] Task {task_id_completed} completed. Current KSC active tasks: {self.task_ids_actual}")
        if task_id_completed in self.task_ids_actual:
            self.task_ids_actual.remove(task_id_completed) 
            self.task_ids_with_idle = self.task_ids_actual + [IDLE_TASK_ID]
            
            if self.utility_calculator and hasattr(self.utility_calculator, 'task_definitions'):
                 if task_id_completed in self.utility_calculator.task_definitions:
                    # print(f"    [KSC NOTIFY DEBUG] Removing {task_id_completed} from utility_calculator.task_definitions")
                    # It's better if utility_calculator's task_definitions is updated by the environment
                    # to avoid KSC directly deleting from a shared dict if that's the case.
                    # For now, assume KSC manages its view or UtilityCalculator is robust to missing tasks.
                    # If UtilityCalculator gets its task_definitions from env.tasks which is updated, this is fine.
                    pass # Environment should update UtilityCalculator's task_definitions if it's shared.
            
            if self.belief_manager: # BeliefManager should also be updated by Environment ideally.
                self.belief_manager.task_ids = list(self.task_ids_actual) 
                # print(f"    [KSC NOTIFY DEBUG] BeliefManager tasks updated to: {self.belief_manager.task_ids}")
            
            needs_re_evaluation = False
            for agent_id in self.agent_ids: 
                if agent_id in self._sim_best_assignment and self._sim_best_assignment[agent_id] == task_id_completed:
                    self._sim_best_assignment[agent_id] = IDLE_TASK_ID
                    needs_re_evaluation = True
                if agent_id in self.agents_data:
                    if agent_id in self.agents_data[agent_id].best_known_global_assignment and \
                       self.agents_data[agent_id].best_known_global_assignment[agent_id] == task_id_completed:
                        self.agents_data[agent_id].best_known_global_assignment[agent_id] = IDLE_TASK_ID
                    new_buffer = []
                    for msg in self.agents_data[agent_id].message_buffer:
                        chain_involves_completed = any(op[1] == task_id_completed or op[2] == task_id_completed for op in msg.chain)
                        base_involves_completed = any(tid == task_id_completed for tid in msg.base_assignment.values())
                        # More robust: check if the *resulting* assignment from the chain assigns to the completed task
                        # temp_assigned_after_chain = self._apply_chain_to_assignment(msg.base_assignment, msg.chain)
                        # resulting_involves_completed = any(tid_res == task_id_completed for tid_res in temp_assigned_after_chain.values())

                        if not (chain_involves_completed or base_involves_completed):
                            new_buffer.append(msg)
                        # else:
                            # print(f"    [KSC NOTIFY DEBUG] Agent {agent_id} removed msg (orig: {msg.originator_id}, chain: {msg.chain}) due to task {task_id_completed} completion.")
                    self.agents_data[agent_id].message_buffer = new_buffer
            
            if needs_re_evaluation: 
                 self._sim_best_utility = self._calculate_global_utility(self._sim_best_assignment, self.current_agent_physical_states)
                 # print(f"    [KSC NOTIFY DEBUG] Sim best utility re-evaluated to {self._sim_best_utility:.2f} after task completion.")
                 for agent_data_val in self.agents_data.values(): 
                     agent_data_val.best_known_global_assignment = copy.deepcopy(self._sim_best_assignment)
                     agent_data_val.best_known_global_utility = self._sim_best_utility
        # else:
            # print(f"  [KSC NOTIFY DEBUG] Task {task_id_completed} was already removed or not in KSC's active list.")


    def _calculate_global_utility(self, assignment: Assignment, agent_states: Dict[str, np.ndarray]) -> float:
        total_utility = 0.0
        # print(f"  [KSC _calc_util DEBUG] Calculating utility for assignment: {assignment}")
        # print(f"    KSC active tasks: {self.task_ids_actual}")
        # if self.utility_calculator: print(f"    UtilityCalc tasks: {list(self.utility_calculator.task_definitions.keys())}")

        assigned_agents_count_per_task: Dict[str, int] = {task_id: 0 for task_id in self.task_ids_actual}

        for agent_id_iter, task_id_iter in assignment.items():
            if task_id_iter is not None and task_id_iter != IDLE_TASK_ID:
                if task_id_iter not in self.task_ids_actual:
                    # print(f"    [KSC _calc_util CRITICAL] Task '{task_id_iter}' (for agent '{agent_id_iter}') not in KSC's current active tasks {self.task_ids_actual}. Returning -inf.")
                    return -float('inf')
                assigned_agents_count_per_task[task_id_iter] = assigned_agents_count_per_task.get(task_id_iter, 0) + 1
        
        if not self.utility_calculator: return -float('inf') # Should be caught by Environment

        for task_id_check_constraints in self.task_ids_actual: # Iterate KSC's view of active tasks
            # Task definitions for constraints should come from utility_calculator, which gets updated by environment
            task_def_constraints = self.utility_calculator.task_definitions.get(task_id_check_constraints)
            
            if not task_def_constraints: # If task is active for KSC but not in util_calc (e.g., just completed and env updated util_calc)
                if assigned_agents_count_per_task.get(task_id_check_constraints, 0) > 0:
                    # print(f"    [KSC _calc_util WARNING] Task '{task_id_check_constraints}' active in KSC but not in util_calc. Assigned agents exist. Returning -inf.")
                    return -float('inf') 
                continue # No agents assigned to this (now non-existent in util_calc) task, so no penalty.

            task_type_name_constraints = task_def_constraints.get("true_type_name")
            count_for_this_task = assigned_agents_count_per_task.get(task_id_check_constraints, 0)

            if task_type_name_constraints: # Only apply type-based constraints if type name exists
                min_req = self.min_agents_per_task.get(task_type_name_constraints, 0)
                if 0 < count_for_this_task < min_req:
                    # print(f"    [KSC _calc_util INFO] Task '{task_id_check_constraints}' (Type: {task_type_name_constraints}) violates min_agents ({min_req}). Has: {count_for_this_task}. Utility = -inf.")
                    return -float('inf')
                max_allowed = self.max_coalition_size.get(task_type_name_constraints)
                if max_allowed is not None and count_for_this_task > max_allowed:
                    # print(f"    [KSC _calc_util INFO] Task '{task_id_check_constraints}' (Type: {task_type_name_constraints}) violates max_agents ({max_allowed}). Has: {count_for_this_task}. Utility = -inf.")
                    return -float('inf')
            # else:
                # print(f"    [KSC _calc_util DEBUG] Task '{task_id_check_constraints}' has no type_name in util_calc defs. Min/max constraints by type not applied.")


        for agent_id, task_id in assignment.items():
            if task_id is not None and task_id != IDLE_TASK_ID:
                if task_id not in self.task_ids_actual: continue 

                current_agent_state = agent_states.get(agent_id)
                if current_agent_state is None: return -float('inf')

                if not self.belief_manager: return -float('inf')
                
                agent_beliefs = self.belief_manager.get_expected_belief_dist_for_task(agent_id, task_id)
                # UtilityCalculator will handle if agent_beliefs is None (e.g. task completed and removed from belief manager's list)
                # print(f"      [KSC _calc_util DEBUG] Getting utility for Agent {agent_id} to Task {task_id}. Beliefs: {np.round(agent_beliefs,2) if agent_beliefs is not None else 'None'}")

                utility = self.utility_calculator.get_expected_task_utility_for_agent(
                    agent_id, current_agent_state, task_id, agent_beliefs, self.task_type_names
                )
                # print(f"        Individual utility for {agent_id} to {task_id}: {utility:.3f}")
                if utility == -float('inf'): return -float('inf')
                total_utility += utility
        
        # print(f"  [KSC _calc_util INFO] Assignment: {assignment}, Calculated Total Utility: {total_utility:.3f}")
        # A small positive number to prefer any assignment over pure idle if idle utility is 0.0
        # This helps break ties if all assignments yield 0 utility (e.g. if rewards are too low or costs too high)
        # but only if the assignment is not all idle.
        if total_utility == 0.0 and any(tid != IDLE_TASK_ID for tid in assignment.values()):
             return 1e-6 
        return total_utility


    def _is_assignment_valid(self, assignment: Assignment, task_id_to_check: Optional[str], agent_id_switched: str) -> bool: # (Unchanged)
        if task_id_to_check is None or task_id_to_check == IDLE_TASK_ID: return True
        if task_id_to_check not in self.task_ids_actual: return False 
        if not self.utility_calculator or task_id_to_check not in self.utility_calculator.task_definitions: return False
        task_def = self.utility_calculator.task_definitions[task_id_to_check]
        task_type_name = task_def.get("true_type_name")
        if not task_type_name: return True 
        max_size_for_this_task_type = self.max_coalition_size.get(task_type_name)
        if max_size_for_this_task_type is not None:
            current_size = sum(1 for assigned_tid_val in assignment.values() if assigned_tid_val == task_id_to_check)
            if current_size > max_size_for_this_task_type: return False
        return True

    def _apply_chain_to_assignment(self, base_assignment: Assignment, chain: TransformationChain) -> Assignment: # (Unchanged)
        new_assignment = copy.deepcopy(base_assignment)
        for agent_id, _, new_task_id in chain: new_assignment[agent_id] = new_task_id
        return new_assignment

    def _local_optimization_dfs(self, # (Unchanged from your version, assuming DFS branching logic is intended)
                                agent_data: KSCAgentData, base_assignment_for_dfs: Assignment,
                                current_chain_being_built: TransformationChain, current_assignment_in_dfs: Assignment,
                                iteration_num: int, visited_assignments_in_dfs: Set[Tuple[Tuple[str, Optional[str]], ...]]):
        assignment_tuple = tuple(sorted(current_assignment_in_dfs.items()))
        if assignment_tuple in visited_assignments_in_dfs: return
        visited_assignments_in_dfs.add(assignment_tuple)
        if len(current_chain_being_built) >= agent_data.k_value: return
        agents_to_consider_for_next_switch = []
        if not current_chain_being_built: agents_to_consider_for_next_switch = [agent_data.agent_id]
        else:
            possible_others = [aid for aid in self.agent_ids if aid != agent_data.agent_id]
            random.shuffle(possible_others)
            candidates_for_switch = [agent_data.agent_id] + possible_others[:max(0, self.max_dfs_branching_factor -1)]
            agents_to_consider_for_next_switch = random.sample(candidates_for_switch, min(len(candidates_for_switch), self.max_dfs_branching_factor))
        for next_agent_to_switch in agents_to_consider_for_next_switch:
            original_task_of_next = current_assignment_in_dfs.get(next_agent_to_switch)
            sorted_target_tasks = sorted([tid for tid in self.task_ids_with_idle if tid != original_task_of_next], key=lambda x: x == IDLE_TASK_ID)
            for target_task in sorted_target_tasks:
                new_switch: SwitchOperation = (next_agent_to_switch, original_task_of_next, target_task)
                potential_assignment_after_this_switch = copy.deepcopy(current_assignment_in_dfs)
                potential_assignment_after_this_switch[next_agent_to_switch] = target_task
                if not self._is_assignment_valid(potential_assignment_after_this_switch, target_task, next_agent_to_switch): continue
                extended_chain = current_chain_being_built + [new_switch]
                assignment_after_extended_chain = self._apply_chain_to_assignment(base_assignment_for_dfs, extended_chain)
                utility_after_extended_chain = self._calculate_global_utility(assignment_after_extended_chain, self.current_agent_physical_states)
                if utility_after_extended_chain > agent_data.best_known_global_utility + 1e-5: # Added epsilon
                    agent_data.best_known_global_assignment = copy.deepcopy(assignment_after_extended_chain)
                    agent_data.best_known_global_utility = utility_after_extended_chain
                    agent_data.messages_generated_this_iteration +=1
                    new_message = KSCMessage(
                        base_assignment_for_dfs, extended_chain, agent_data.agent_id,
                        utility_after_extended_chain, agent_data.k_value,
                        iteration_created=iteration_num, path_taken=[agent_data.agent_id]
                    )
                    agent_data.add_message_to_buffer(new_message, agent_data.agent_id)
                if utility_after_extended_chain > -float('inf') and len(extended_chain) < agent_data.k_value:
                    self._local_optimization_dfs(agent_data, base_assignment_for_dfs, extended_chain, potential_assignment_after_this_switch, iteration_num, visited_assignments_in_dfs)

    def _process_message_buffer(self, agent_data: KSCAgentData, iteration_num: int): # (Unchanged from your version)
        messages_to_attempt_extension = agent_data.get_messages_for_processing(count=self.max_dfs_branching_factor + 1 ) # Match your code
        processed_count_this_agent = 0
        for msg_from_buffer in messages_to_attempt_extension:
            if agent_data.agent_id in msg_from_buffer.processed_for_extension_by: continue
            processed_count_this_agent +=1
            msg_from_buffer.processed_for_extension_by.add(agent_data.agent_id)
            if len(msg_from_buffer.chain) >= msg_from_buffer.k_limit_of_originator: continue
            assignment_after_buffer_chain = self._apply_chain_to_assignment(msg_from_buffer.base_assignment, msg_from_buffer.chain)
            original_task_of_current_agent = assignment_after_buffer_chain.get(agent_data.agent_id)
            sorted_target_tasks_msg_proc = sorted([tid for tid in self.task_ids_with_idle if tid != original_task_of_current_agent], key=lambda x: x == IDLE_TASK_ID)
            for target_task in sorted_target_tasks_msg_proc:
                new_switch: SwitchOperation = (agent_data.agent_id, original_task_of_current_agent, target_task)
                if len(msg_from_buffer.chain) + 1 > msg_from_buffer.k_limit_of_originator: continue
                potential_assignment_after_new_switch = copy.deepcopy(assignment_after_buffer_chain)
                potential_assignment_after_new_switch[agent_data.agent_id] = target_task
                if not self._is_assignment_valid(potential_assignment_after_new_switch, target_task, agent_data.agent_id): continue
                extended_chain_from_buffer_msg = msg_from_buffer.chain + [new_switch]
                assignment_after_extended = self._apply_chain_to_assignment(msg_from_buffer.base_assignment, extended_chain_from_buffer_msg)
                utility_after_extended = self._calculate_global_utility(assignment_after_extended, self.current_agent_physical_states)
                if utility_after_extended > agent_data.best_known_global_utility + 1e-5: # Added epsilon
                    agent_data.best_known_global_assignment = copy.deepcopy(assignment_after_extended)
                    agent_data.best_known_global_utility = utility_after_extended
                    agent_data.messages_generated_this_iteration +=1
                    new_path = list(msg_from_buffer.path_taken)
                    if agent_data.agent_id not in new_path: new_path.append(agent_data.agent_id)
                    new_extended_message = KSCMessage(
                        msg_from_buffer.base_assignment, extended_chain_from_buffer_msg, msg_from_buffer.originator_id,
                        utility_after_extended, msg_from_buffer.k_limit_of_originator,
                        iteration_created=iteration_num, path_taken=new_path)
                    agent_data.add_message_to_buffer(new_extended_message, agent_data.agent_id)
        agent_data.messages_processed_this_iteration += processed_count_this_agent

    def run_iteration(self, iteration_num: int) -> Tuple[Assignment, float, Dict[str, Any]]: # (Unchanged from your version)
        iter_start_time = time.time()
        self._update_neighbor_graph() 
        total_messages_broadcasted_this_iter = 0
        total_messages_processed_by_all_agents = 0 
        for agent_id_iter_init in self.agent_ids: 
            self.agents_data[agent_id_iter_init].messages_generated_this_iteration = 0
            self.agents_data[agent_id_iter_init].messages_processed_this_iteration = 0 
            self.agents_data[agent_id_iter_init].sent_message_ids_this_iteration = set() 
        for agent_id_local_opt in self.agent_ids:
            agent_data_local_opt = self.agents_data[agent_id_local_opt]
            base_assignment_for_dfs_local = copy.deepcopy(agent_data_local_opt.best_known_global_assignment)
            visited_dfs_set_local: Set[Tuple[Tuple[str, Optional[str]], ...]] = set()
            self._local_optimization_dfs(
                agent_data=agent_data_local_opt, base_assignment_for_dfs=base_assignment_for_dfs_local,
                current_chain_being_built=[], current_assignment_in_dfs=base_assignment_for_dfs_local,
                iteration_num=iteration_num, visited_assignments_in_dfs=visited_dfs_set_local)
            self._process_message_buffer(agent_data_local_opt, iteration_num)
            total_messages_processed_by_all_agents += agent_data_local_opt.messages_processed_this_iteration
        messages_to_be_delivered_next: Dict[str, List[KSCMessage]] = {aid: [] for aid in self.agent_ids}
        for agent_id_sender_prop in self.agent_ids:
            agent_data_sender_prop = self.agents_data[agent_id_sender_prop]
            neighbors_prop = self.neighbor_graph.get(agent_id_sender_prop, [])
            if not neighbors_prop: continue
            msgs_to_broadcast_prop = agent_data_sender_prop.get_messages_for_broadcast(iteration_num, count=3) # Your count=3
            for msg_broadcast_item in msgs_to_broadcast_prop:
                agent_data_sender_prop.sent_message_ids_this_iteration.add(msg_broadcast_item.message_id)
                for neighbor_id_receiver_prop in neighbors_prop:
                    if neighbor_id_receiver_prop not in msg_broadcast_item.path_taken: 
                        msg_for_neighbor_prop = copy.deepcopy(msg_broadcast_item)
                        msg_for_neighbor_prop.sender_id = agent_id_sender_prop
                        new_path_for_msg_prop = list(msg_for_neighbor_prop.path_taken) 
                        if agent_id_sender_prop not in new_path_for_msg_prop : new_path_for_msg_prop.append(agent_id_sender_prop)
                        msg_for_neighbor_prop.path_taken = new_path_for_msg_prop
                        messages_to_be_delivered_next[neighbor_id_receiver_prop].append(msg_for_neighbor_prop)
                        total_messages_broadcasted_this_iter += 1
        for recipient_id_prop, received_msgs_list_prop in messages_to_be_delivered_next.items():
            recipient_data_prop = self.agents_data[recipient_id_prop]
            for r_msg_item_prop in received_msgs_list_prop:
                recipient_data_prop.add_message_to_buffer(r_msg_item_prop, recipient_id_prop)
        current_iteration_best_utility_oas = -float('inf')
        current_iteration_best_assignment_oas = copy.deepcopy(self._sim_best_assignment) 
        for agent_id_oas_find in self.agent_ids: 
            agent_data_oas_find = self.agents_data[agent_id_oas_find]
            if agent_data_oas_find.best_known_global_utility > current_iteration_best_utility_oas + 1e-5: # Epsilon
                current_iteration_best_utility_oas = agent_data_oas_find.best_known_global_utility
                current_iteration_best_assignment_oas = copy.deepcopy(agent_data_oas_find.best_known_global_assignment)
        if current_iteration_best_utility_oas > self._sim_best_utility + 1e-5: # Epsilon
            self._sim_best_utility = current_iteration_best_utility_oas
            self._sim_best_assignment = copy.deepcopy(current_iteration_best_assignment_oas)
        for agent_id_oas_sync in self.agent_ids: 
            agent_data_oas_sync = self.agents_data[agent_id_oas_sync]
            if self._sim_best_utility > agent_data_oas_sync.best_known_global_utility + 1e-5: # Epsilon
                agent_data_oas_sync.best_known_global_assignment = copy.deepcopy(self._sim_best_assignment)
                agent_data_oas_sync.best_known_global_utility = self._sim_best_utility
                new_buffer_after_rar_sync = []
                for msg_in_buf_sync in agent_data_oas_sync.message_buffer:
                    re_evaluated_assignment_sync = self._apply_chain_to_assignment(agent_data_oas_sync.best_known_global_assignment, msg_in_buf_sync.chain)
                    re_evaluated_utility_sync = self._calculate_global_utility(re_evaluated_assignment_sync, self.current_agent_physical_states)
                    if re_evaluated_utility_sync > -float('inf') and \
                       (re_evaluated_utility_sync > self._sim_best_utility * 0.5 or \
                        re_evaluated_utility_sync > msg_in_buf_sync.achieved_utility * 0.8): 
                        updated_msg_sync = KSCMessage(
                            base_assignment=agent_data_oas_sync.best_known_global_assignment,
                            chain=msg_in_buf_sync.chain, originator_id=msg_in_buf_sync.originator_id,
                            achieved_utility=re_evaluated_utility_sync, k_limit_of_originator=msg_in_buf_sync.k_limit_of_originator,
                            iteration_created=msg_in_buf_sync.iteration_created, path_taken=copy.deepcopy(msg_in_buf_sync.path_taken),
                            message_id=msg_in_buf_sync.message_id )
                        updated_msg_sync.processed_for_extension_by = set() 
                        new_buffer_after_rar_sync.append(updated_msg_sync)
                agent_data_oas_sync.message_buffer = new_buffer_after_rar_sync
                agent_data_oas_sync.message_buffer.sort(key=lambda m: (m.achieved_utility, -len(m.chain)), reverse=True)
                if len(agent_data_oas_sync.message_buffer) > agent_data_oas_sync.max_buffer_size:
                     agent_data_oas_sync.message_buffer = agent_data_oas_sync.message_buffer[:agent_data_oas_sync.max_buffer_size]
        self.stats_total_messages_broadcasted_per_iteration.append(total_messages_broadcasted_this_iter)
        self.stats_total_messages_processed_per_iteration.append(total_messages_processed_by_all_agents)
        stats_iter = { "sim_best_utility": self._sim_best_utility, "messages_broadcasted": total_messages_broadcasted_this_iter,
            "messages_processed_by_agents": total_messages_processed_by_all_agents,
            "active_communication_links": self.stats_active_links_per_iteration[-1] if self.stats_active_links_per_iteration else 0}
        return self._sim_best_assignment, self._sim_best_utility, stats_iter

# ... (模块自测试 test_ksc_coalition_formation 保持不变, 但应使用修改后的 UtilityCalculator 和场景参数进行测试)