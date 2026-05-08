"""
简化版同构MCTS算法
纯暴力搜索，无RL依赖
"""

from __future__ import annotations
import random
import math
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from data_model import (
    FloorPlanRO, PinAssignment,
    Segment, SegmentUsage
)
from state_adapter import MultiNetSearchAdapter
from simple_reward_function import SimpleRewardFunction

logger = logging.getLogger(__name__)

@dataclass
class SimplifiedMCTSNode:
    """简化MCTS节点 - 纯暴力搜索"""
    
    state: MultiNetSearchAdapter
    parent: Optional['SimplifiedMCTSNode'] = None
    action: Optional[int] = None  # 导致该节点的动作
    
    # 基本MCTS统计
    visits: int = 0
    value_sum: float = 0.0
    children: List['SimplifiedMCTSNode'] = None
    
    # 基础信息
    depth: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def q_value(self) -> float:
        """Q值 - 平均奖励"""
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
    @property
    def ucb_score(self) -> float:
        """标准UCB评分"""
        if self.parent is None or self.parent.visits == 0:
            return float('inf') if self.visits == 0 else 0.0
        
        # 标准UCB公式
        exploration_constant = 1.414
        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / (1 + self.visits))
        
        return exploitation + exploration
    
    def is_fully_expanded(self) -> bool:
        """检查是否完全扩展"""
        # 获取当前状态的所有合法动作
        legal_actions = self.state.get_legal_segments()
        
        # 检查是否每个动作都有对应的子节点
        expanded_actions = {child.action for child in self.children}
        return len(expanded_actions) >= len(legal_actions)

@dataclass
class SearchResult:
    """搜索结果"""
    best_state: MultiNetSearchAdapter
    best_reward: float
    simulation_count: int
    search_time: float
    search_statistics: Dict[str, Any]

class SimplifiedIsomorphicMCTS:
    """简化版同构MCTS - 纯暴力搜索，无RL依赖"""
    
    def __init__(self, 
                 floorplan: FloorPlanRO,
                 reward_function: SimpleRewardFunction,
                 exploration_constant: float = 1.414):
        
        self.floorplan = floorplan
        self.reward_function = reward_function
        self.exploration_constant = exploration_constant
        
        # 基础统计
        self.total_simulations = 0
        self.best_reward_found = float('-inf')
        self.best_state_found = None
    
    def search(self, 
               initial_state: MultiNetSearchAdapter,
               num_simulations: int = 100,
               time_limit: float = 30.0) -> SearchResult:
        """执行简化MCTS搜索 - 纯暴力搜索"""
        
        logger.info(f"开始MCTS搜索: simulations={num_simulations}, time_limit={time_limit}s")
        logger.info(f"初始状态信息: is_terminal={initial_state.is_terminal()}, current_pin={initial_state.get_current_pin()}")
        
        # 1. 创建根节点（简化版）
        root = SimplifiedMCTSNode(
            state=initial_state
        )
        
        # 2. 执行模拟
        start_time = time.time()
        
        for simulation_id in range(num_simulations):
            
            # 执行单次模拟 - 添加详细日志
            # logger.debug(f"开始第{simulation_id + 1}次模拟")
            self._execute_simple_simulation(root)
            self.total_simulations += 1
            
            # 定期日志
            if (simulation_id + 1) % 10 == 0:  # 改为每10次记录一次，更频繁
                logger.debug(f"完成{simulation_id + 1}次模拟，最佳奖励: {self.best_reward_found:.4f}")
        
        # 3. 选择最佳结果（基于访问次数）
        best_node = self._select_best_node_simple(root)
        
        # 4. 返回结果
        search_time = time.time() - start_time
        logger.info(f"MCTS搜索完成: 模拟{self.total_simulations}次，用时{search_time:.2f}s，最佳奖励{best_node.q_value:.4f}")
        
        return SearchResult(
            best_state=best_node.state,
            best_reward=best_node.q_value,
            simulation_count=self.total_simulations,
            search_time=search_time,
            search_statistics=self._generate_simple_statistics()
        )
    
    def _execute_simple_simulation(self, root: SimplifiedMCTSNode):
        """执行单次简化MCTS模拟 - 纯UCB选择 + 随机模拟"""
        
        try:
            # 1. 选择阶段 - 标准UCB
            selected_node = self._select_node_simple(root)
            
            # 2. 扩展阶段 - 扩展第一个未扩展的合法动作
            if not selected_node.state.is_terminal():
                expanded_node = self._expand_node_simple(selected_node)
            else:
                expanded_node = selected_node
            
            # 3. 模拟阶段 - 随机模拟到终止
            simulation_reward = self._simulate_randomly(expanded_node.state)
            
            # 4. 回溯阶段 - 更新访问统计
            self._backpropagate_simple(expanded_node, simulation_reward)
            
            # 5. 更新最佳记录
            if simulation_reward > self.best_reward_found:
                self.best_reward_found = simulation_reward
                self.best_state_found = expanded_node.state
                logger.debug(f"更新最佳记录: 新最佳奖励={self.best_reward_found:.4f}")
                
        except Exception as e:
            logger.error(f"单次模拟执行失败: {e}", exc_info=True)
            raise
    
    def _select_node_simple(self, root: SimplifiedMCTSNode) -> SimplifiedMCTSNode:
        """简化节点选择 - 纯UCB"""
        current = root
        
        while current.children and not current.state.is_terminal():
            # 如果有未完全扩展的节点，优先选择
            if not current.is_fully_expanded():
                # 选择访问次数最少的子节点
                unexpanded_children = [child for child in current.children if child.visits == 0]
                if unexpanded_children:
                    current = unexpanded_children[0]
                    break
            
            # 否则选择UCB最高的子节点
            current = max(current.children, key=lambda c: c.ucb_score)
        
        return current
    
    def _expand_node_simple(self, parent: SimplifiedMCTSNode) -> SimplifiedMCTSNode:
        """简化节点扩展 - 完善节点扩展逻辑"""
        
        # 获取当前pin
        current_pin = parent.state.get_current_pin()
        if current_pin is None:
            return parent
        
        # 检查是否是现在是关键同构Pin
        if parent.state.current_isomorphic_group and current_pin in parent.state.current_isomorphic_group.pin_ids:
            logger.debug(f"当前pin{current_pin}是关键同构Pin，设置动作-1")
            # 创建新状态，动作为-1表示跳过
            new_state = self._create_next_state_simple(parent.state, action=-1)
            
            # 创建子节点
            child_node = SimplifiedMCTSNode(
                state=new_state,
                parent=parent,
                action=-1,
                depth=parent.depth + 1
            )
            
            parent.children.append(child_node)
            return child_node
        
        # 检查当前pin是否已完成分配
        if parent.state.is_pin_allocated(current_pin):
            logger.debug(f"当前pin{current_pin}已完成分配，设置动作-1")
            # 创建新状态，动作为-1表示跳过
            new_state = self._create_next_state_simple(parent.state, action=-1)
            
            # 创建子节点
            child_node = SimplifiedMCTSNode(
                state=new_state,
                parent=parent,
                action=-1,
                depth=parent.depth + 1
            )
            
            parent.children.append(child_node)
            return child_node
        
        # 检查是否存在复数Pins的同构Pin（非当前关键同构组）
        current_group = parent.state.current_isomorphic_group
        if current_group:
            pin_isomorphic_group = self.floorplan.get_pin_isomorphic_group(current_pin)
            if (pin_isomorphic_group and
                len(pin_isomorphic_group.pin_ids) > 1 and  # 确保是同构组（多个Pin）
                pin_isomorphic_group.group_id != current_group.group_id and
                current_pin in pin_isomorphic_group.pin_ids):
                logger.debug(f"当前pin{current_pin}是复用同构Pin（组{pin_isomorphic_group.group_id}），设置动作-1")
                # 创建新状态，动作为-1表示后续特殊处理
                new_state = self._create_next_state_simple(parent.state, action=-1)
                
                # 创建子节点
                child_node = SimplifiedMCTSNode(
                    state=new_state,
                    parent=parent,
                    action=-1,
                    depth=parent.depth + 1
                )
                
                parent.children.append(child_node)
                return child_node
        
        # 获取合法动作
        legal_actions = self._get_legal_actions_simple(parent.state)
        
        if not legal_actions:
            return parent
        
        # 找到第一个未扩展的动作
        expanded_actions = {child.action for child in parent.children}
        for action in legal_actions:
            if action not in expanded_actions:
                # 创建新状态
                new_state = self._create_next_state_simple(parent.state, action)
                
                # 创建子节点
                child_node = SimplifiedMCTSNode(
                    state=new_state,
                    parent=parent,
                    action=action,
                    depth=parent.depth + 1
                )
                
                parent.children.append(child_node)
                return child_node
        
        # 所有动作都已扩展，返回父节点
        return parent
    
    def _simulate_randomly(self, state: MultiNetSearchAdapter) -> float:
        """MCTS兼容的智能模拟 - 使用临时状态副本，不影响原始节点"""
        
        # 创建临时状态副本，避免修改原始节点状态
        temp_state = state.clone()
        simulation_steps = 0
        max_simulation_steps = 100  # 适当增加步数限制，允许更完整的探索
        
        logger.debug(f"MCTS兼容模拟开始: 初始状态terminal={temp_state.is_terminal()}")
        
        # 获取当前同构组信息
        current_group = temp_state.current_isomorphic_group
        if not current_group:
            logger.warning("没有当前同构组信息，使用基础模拟")
            return self._simulate_randomly_basic(temp_state)
        
        logger.debug(f"处理同构组: {current_group.group_id}, pins={len(current_group.pin_ids)}")
        
        # MCTS兼容策略：在单次模拟中尽可能推进，避免无限循环
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while not temp_state.is_terminal() and simulation_steps < max_simulation_steps:
            simulation_steps += 1
            
            # 获取当前pin信息
            current_pin = temp_state.get_current_pin()
            if not current_pin:
                logger.debug(f"步骤{simulation_steps}: 无法获取当前pin，终止模拟")
                break
            
            # 获取当前网表和pin信息
            current_net = temp_state.floorplan.get_net(temp_state.base_state.current_net_id)
            if not current_net:
                logger.debug(f"步骤{simulation_steps}: 无法获取当前网表，终止模拟")
                break
            
            current_pin_id = current_net.pin_ids[temp_state.base_state.current_pin_idx]
            
            # 检查是否是关键同构pin
            if current_pin_id in current_group.pin_ids:
                logger.debug(f"步骤{simulation_steps}: 遇到关键同构pin={current_pin_id}，跳过")
                new_state = self._create_next_state_simple(temp_state, action=-1)
                temp_state = new_state
                continue
            
            # 检查pin是否已分配
            if temp_state.is_pin_allocated(current_pin_id):
                logger.debug(f"步骤{simulation_steps}: pin已分配={current_pin_id}，推进状态")
                new_state = self._advance_to_next_valid_state(temp_state)
                temp_state = new_state
                continue
            
            # 检查该Pin是否是同构Pin（要复用但不是当前关键同构组里的Pin）
            # 只处理包含多个Pin的同构组，因为单个Pin的同构组不需要特殊处理
            pin_isomorphic_group = self.floorplan.get_pin_isomorphic_group(current_pin_id)
            if (pin_isomorphic_group and
                len(pin_isomorphic_group.pin_ids) > 1 and  # 确保是同构组（多个Pin）
                pin_isomorphic_group.group_id != current_group.group_id and
                current_pin_id in pin_isomorphic_group.pin_ids):
                logger.debug(f"步骤{simulation_steps}: pin={current_pin_id}是同构Pin（复用Pin，组{pin_isomorphic_group.group_id}），跳过当前处理")
                new_state = self._create_next_state_simple(temp_state, action=-1)
                temp_state = new_state
                continue
            
            # 处理这个普通pin
            legal_actions = self._get_legal_actions_simple(temp_state)
            logger.debug(f"步骤{simulation_steps}: 处理普通pin={current_pin_id}, 合法动作={len(legal_actions)}")
            
            if not legal_actions:
                logger.debug(f"步骤{simulation_steps}: 无合法动作，推进状态")
                new_state = self._create_next_state_simple(temp_state, action=-1)
                temp_state = new_state
                continue
            
            # 随机选择并执行动作
            action = random.choice(legal_actions)
            try:
                new_state = self._create_next_state_simple(temp_state, action)
                # 检查是否真正推进了
                if new_state.base_state.current_pin_idx == temp_state.base_state.current_pin_idx:
                    # 创建新状态但没有推进，可能是推进逻辑问题
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.debug(f"连续{consecutive_failures}次状态创建未推进，终止模拟")
                        break
                else:
                    consecutive_failures = 0
                    temp_state = new_state
                    logger.debug(f"步骤{simulation_steps}: 分配成功")
            except Exception as e:
                logger.error(f"步骤{simulation_steps}: 分配失败: {e}")
                new_state = self._create_next_state_simple(temp_state, action=-1)
                temp_state = new_state
        
        logger.debug(f"MCTS兼容模拟完成: 步数={simulation_steps}, terminal={temp_state.is_terminal()}")
        
        # 计算奖励 - 使用临时状态，不影响原始节点
        reward = self.reward_function.compute_reward(temp_state.net_assignments[temp_state.base_state.current_net_id])
        logger.debug(f"奖励: {reward:.4f}")
        return reward
    
    def _simulate_randomly_basic(self, state: MultiNetSearchAdapter) -> float:
        """基础模拟 - 作为后备方案"""
        current_state = state
        simulation_steps = 0
        max_simulation_steps = 100
        
        while not current_state.is_terminal() and simulation_steps < max_simulation_steps:
            simulation_steps += 1
            
            legal_actions = self._get_legal_actions_simple(current_state)
            if not legal_actions:
                break
                
            action = random.choice(legal_actions)
            try:
                current_state = self._create_next_state_simple(current_state, action)
            except Exception:
                break
        
        return self.reward_function.compute_reward(current_state.base_state.net_assignments[current_state.base_state.current_net_id])
    
    def _advance_to_next_valid_state(self, state: MultiNetSearchAdapter) -> MultiNetSearchAdapter:
        """推进到下一个有效状态"""
        
        new_state = state.clone()
        new_state.advance_to_next_task()
        
        return new_state
    
    def _backpropagate_simple(self, node: SimplifiedMCTSNode, reward: float):
        """简化回溯 - 只更新访问统计"""
        current = node
        
        while current is not None:
            current.visits += 1
            current.value_sum += reward
            current = current.parent
    
    def _select_best_node_simple(self, root: SimplifiedMCTSNode) -> SimplifiedMCTSNode:
        """简化最佳节点选择 - 按照UCB分数选择，确保选择叶节点"""
        
        def is_leaf_node(node: SimplifiedMCTSNode) -> bool:
            """检查是否为叶节点 - 完成了当前net的所有pin分配"""
            return node.state.is_terminal()
        
        def find_best_leaf_by_ucb(node: SimplifiedMCTSNode) -> SimplifiedMCTSNode:
            """递归地按照UCB分数找到最佳叶节点"""
            # 如果当前节点是叶节点，返回它
            if is_leaf_node(node):
                return node
            
            # 如果没有子节点，返回当前节点
            if not node.children:
                return node
            
            # 在所有子节点中找到UCB分数最高的
            best_child = max(node.children, key=lambda c: c.ucb_score)
            
            # 递归地在最佳子节点中寻找叶节点
            return find_best_leaf_by_ucb(best_child)
        
        # 从根节点开始，按照UCB分数选择最佳叶节点
        best_leaf = find_best_leaf_by_ucb(root)
        
        logger.debug(f"选择最佳叶节点: depth={best_leaf.depth}, visits={best_leaf.visits}, "
                    f"q_value={best_leaf.q_value:.4f}, is_terminal={best_leaf.state.is_terminal()}")
        
        return best_leaf
    
    def _select_best_node_simple_alternative(self, root: SimplifiedMCTSNode) -> SimplifiedMCTSNode:
        """备选方案：使用DFS找到所有叶节点，然后选择Q值最高的"""
        
        def collect_leaf_nodes(node: SimplifiedMCTSNode, leaves: List[SimplifiedMCTSNode]):
            """收集所有叶节点"""
            if node.state.is_terminal():
                leaves.append(node)
                return
            
            if not node.children:
                # 如果没有子节点且不是terminal，也作为候选
                leaves.append(node)
                return
            
            for child in node.children:
                collect_leaf_nodes(child, leaves)
        
        # 收集所有叶节点
        leaf_nodes = []
        collect_leaf_nodes(root, leaf_nodes)
        
        if not leaf_nodes:
            logger.warning("没有找到叶节点，返回根节点")
            return root
        
        # 选择Q值最高的叶节点
        best_leaf = max(leaf_nodes, key=lambda n: n.q_value)
        
        logger.debug(f"从{len(leaf_nodes)}个叶节点选择最佳: q_value={best_leaf.q_value:.4f}, "
                    f"visits={best_leaf.visits}, depth={best_leaf.depth}")
        
        return best_leaf
    
    def _get_legal_actions_simple(self, state: MultiNetSearchAdapter) -> List[int]:
        """获取合法动作 - 应用同构约束"""
        
        # 获取当前pin
        current_pin = state.get_current_pin()
        if current_pin is None:
            return []
        
        # 获取基本合法segments
        pin_obj = self.floorplan.get_pin(current_pin)
        if not pin_obj:
            return []
        
        basic_segments = self.floorplan.get_block_segments(pin_obj.block_id)
        
        # 过滤容量约束
        capacity_legal = []
        for seg_id in basic_segments:
            usage = state.get_segment_usage(seg_id)
            if usage.can_assign(current_pin, self.floorplan):
                capacity_legal.append(seg_id)
        
        return capacity_legal
    
    def _create_next_state_simple(self, state: MultiNetSearchAdapter, action: int) -> MultiNetSearchAdapter:
        """创建下一个状态 - 简化版"""
        
        # 获取当前pin
        current_pin = state.get_current_pin()
        if current_pin is None:
            logger.warning(f"无法获取当前pin，返回克隆状态以避免无限循环")
            return state.clone()  # 返回克隆状态而不是原状态
        
        # 创建新的分配
        if not action == -1:
            segment = self.floorplan.get_segment(action)
            if not segment:
                logger.warning(f"无法找到segment {action}，返回克隆状态")
                return state.clone()  # 返回克隆状态而不是原状态
        
        # 获取当前pin的net_id
        pin_obj = self.floorplan.get_pin(current_pin)
        net_id = pin_obj.net_id if pin_obj else 0
        
        new_assignment = PinAssignment(
            pin_id=current_pin,
            seg_id=action,
            net_id=net_id
        )
        
        # 更新状态
        new_state = state.clone()
        new_state.add_assignment(new_assignment)
        
        # # 更新同构组约束
        # pin_group = self.floorplan.get_pin_isomorphic_group(current_pin)
        # if pin_group is not None:
        #     new_state.update_group_assignment(pin_group.group_id, action)
        
        # 推进到下一个任务
        new_state.advance_to_next_task()
        
        return new_state
    
    def _generate_simple_statistics(self) -> Dict[str, Any]:
        """生成简化统计信息"""
        return {
            "total_simulations": self.total_simulations,
            "best_reward_found": self.best_reward_found,
            "exploration_constant": self.exploration_constant
        }

# 简化配置
@dataclass
class SimpleMCTSConfig:
    """简化MCTS配置"""
    num_simulations: int = 1000
    time_limit: float = 30.0
    exploration_constant: float = 1.414
    max_depth: int = 50

# 工厂函数
def create_simple_isomorphic_mcts(floorplan: FloorPlanRO,
                                reward_function: SimpleRewardFunction,
                                exploration_constant: float = 1.414) -> SimplifiedIsomorphicMCTS:
    """创建简化MCTS搜索器"""
    return SimplifiedIsomorphicMCTS(
        floorplan=floorplan,
        reward_function=reward_function,
        exploration_constant=exploration_constant
    )