"""
简化奖励函数 - 无RL依赖
专注于线长优化和同构一致性
"""

from __future__ import annotations
import math
import logging
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from data_model import FloorPlanRO, PinAssignment, SegmentUsage
from state_adapter import MultiNetSearchAdapter

from PlaceDB import PlaceDB

logger = logging.getLogger(__name__)

@dataclass
class RewardResult:
    """奖励计算结果"""
    total_reward: float
    wirelength_reward: float
    consistency_reward: float
    capacity_reward: float
    components: Dict[str, float]

class SimpleRewardFunction:
    """简化奖励函数 - 纯MCTS暴力搜索"""
    
    def __init__(self, floorplan: FloorPlanRO, ftpred_session=None, placedb=None):
        self.floorplan = floorplan
        self.ftpred_session = ftpred_session  # 外部传入的 feedthrough 会话
        self.placedb = placedb  # 用于 feedthrough 计算
        self.wirelength_cache = {}
    
    # TODO: 容量reward完善
    def compute_reward(self, net_assignments:List[PinAssignment]) -> float:
        """计算综合奖励"""
        
        # 1. 线长奖励
        wirelength_reward = self._compute_wirelength_reward(net_assignments)
        # wirelength_reward = 0
        
        # 2. feedthrough 奖励
        feedthrough_reward = self._compute_feedthrough_reward(net_assignments)
        
        # 3. 容量奖励
        # capacity_reward = self._compute_capacity_reward(net_segment_usages)
        
        # 4. 综合奖励（加权）
        total_reward = (
            0.9 * wirelength_reward +
            0.1 * 0
        )
        
        logger.debug(f"综合奖励: 线长={wirelength_reward:.4f}, feedthrough={feedthrough_reward:.4f}, 总奖励={total_reward:.4f}")
        
        return total_reward
    
    def _compute_wirelength_reward(self, net_assignments: List[PinAssignment]) -> float:
        """计算线长奖励 - 新的归一化方案"""
        
        total_actual_hpwl = 0.0
        total_baseline_hpwl = 0.0
        valid_nets = 0
        

        if len(net_assignments) < 2:
            return 0.5  # 默认中等奖励
            
        # 计算实际分配的HPWL
        actual_hpwl = self._calculate_net_hpwl(net_assignments, use_block_centroid=True)
            
        # 计算所有pin都在block质心时的baseline HPWL
        baseline_hpwl = self._calculate_net_baseline_hpwl(net_assignments)
            
        if actual_hpwl > 0 and baseline_hpwl > 0:
            total_actual_hpwl += actual_hpwl
            total_baseline_hpwl += baseline_hpwl
            valid_nets += 1
        
        if valid_nets == 0:
            return 0.5  # 默认中等奖励
        
        # 计算归一化线长
        if total_baseline_hpwl > 0:
            normalized_wirelength = total_actual_hpwl / total_baseline_hpwl
        else:
            normalized_wirelength = 1.0
        
        # 使用指数函数计算奖励，alpha默认为1
        alpha = 1.0
        wirelength_reward = math.exp(-alpha * (normalized_wirelength - 1))
        wirelength_reward = -total_actual_hpwl  # 直接使用负的实际HPWL作为奖励
        
        logger.debug(f"线长奖励计算: 实际HPWL={total_actual_hpwl:.2f}, 基线HPWL={total_baseline_hpwl:.2f}, "
                    f"归一化={normalized_wirelength:.4f}, 奖励={wirelength_reward:.4f}")
        
        # return max(0.0, min(1.0, wirelength_reward))
        return wirelength_reward
    
    def _calculate_net_hpwl(self, assignments: List[PinAssignment], use_block_centroid: bool = False) -> float:
        """计算单个net的HPWL - 支持使用block质心处理seg_id=-1的情况"""
        
        if len(assignments) < 2:
            return 0.0
        
        # 获取所有分配的坐标
        x_coords = []
        y_coords = []
        
        for assignment in assignments:
            if assignment.seg_id == -1 and use_block_centroid:
                # seg_id为-1时使用block质心
                pin = self.floorplan.get_pin(assignment.pin_id)
                if pin:
                    block = self.floorplan.get_block(pin.block_id)
                    if block and block.position:
                        x_coords.append(block.position[0])
                        y_coords.append(block.position[1])
                        logger.debug(f"Pin{assignment.pin_id}使用block质心: ({block.position[0]}, {block.position[1]})")
                    else:
                        logger.warning(f"无法获取pin{assignment.pin_id}的block质心")
                        return 0.0
                else:
                    logger.warning(f"无法找到pin{assignment.pin_id}")
                    return 0.0
            else:
                # 正常情况使用SegmentInst实际坐标
                seg_inst = self.floorplan.get_segment_inst_for_pin(assignment.pin_id, assignment.seg_id)
                if seg_inst:
                    x_coords.append(seg_inst.cx)
                    y_coords.append(seg_inst.cy)
                else:
                    logger.error(f"无法找到pin{assignment.pin_id}在segment{assignment.seg_id}上的SegmentInst")
                    raise ValueError(f"无法找到pin{assignment.pin_id}在segment{assignment.seg_id}上的SegmentInst")
        
        if not x_coords or not y_coords:
            return 0.0
        
        # HPWL = (max_x - min_x) + (max_y - min_y)
        hpwl = (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
        return hpwl
    
    def _calculate_net_baseline_hpwl(self, assignments: List[PinAssignment]) -> float:
        """计算所有pin都在block质心时的baseline HPWL"""
        
        if len(assignments) < 2:
            return 0.0
        
        # 获取所有pin的block质心坐标
        x_coords = []
        y_coords = []
        
        for assignment in assignments:
            pin = self.floorplan.get_pin(assignment.pin_id)
            if pin:
                block = self.floorplan.get_block(pin.block_id)
                if block and block.position:
                    x_coords.append(block.position[0])
                    y_coords.append(block.position[1])
                else:
                    logger.warning(f"无法获取pin{assignment.pin_id}的block信息")
                    return 0.0
            else:
                logger.warning(f"无法找到pin{assignment.pin_id}")
                return 0.0
        
        if not x_coords or not y_coords:
            return 0.0
        
        # HPWL = (max_x - min_x) + (max_y - min_y)
        baseline_hpwl = (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
        
        logger.debug(f"Baseline HPWL计算: pins={len(assignments)}, coords={list(zip(x_coords, y_coords))}, HPWL={baseline_hpwl}")
        
        return baseline_hpwl
    
    def _compute_capacity_reward(self, net_segments: List[SegmentUsage]) -> float:
        """计算容量奖励"""
        
        total_score = 0.0
        segment_count = 0
        
        for seg_id, usage in net_segments:
            segment = self.floorplan.get_segment(seg_id)
            if not segment:
                continue
            
            # 计算利用率
            utilization = usage.used_capacity / max(1e-6, segment.max_capacity)
            
            # 理想的利用率在60-80%之间
            if 0.6 <= utilization <= 0.8:
                score = 1.0
            elif utilization < 0.6:
                score = utilization / 0.6  # 线性增长
            else:
                score = max(0.0, (1.0 - utilization) / 0.2)  # 线性下降
            
            total_score += score
            segment_count += 1
        
        return total_score / max(1, segment_count)
    
    def get_reward_analytics(self) -> Dict[str, any]:
        """获取奖励函数分析数据"""
        return {
            "wirelength_cache_size": len(self.wirelength_cache),
            "reward_weights": {
                "wirelength": 0.6,
                "consistency": 0.3,
                "capacity": 0.1
            }
        }
    
    def _compute_feedthrough_reward(self, net_assignments: List[PinAssignment]) -> float:
        """计算 feedthrough 奖励 - 基于 feedthrough 数量的归一化
        
        与线长奖励类似：
        - 已分配的 pin 使用 SegmentInst 中点坐标
        - 未分配的 pin 使用 block 质心坐标
        - 使用 feedthrough 计算器获取实际的 feedthrough 数量
        """
        if self.ftpred_session is None or self.placedb is None:
            logger.debug("Feedthrough 计算器未初始化，跳过 feedthrough reward")
            return 0.5  # 默认中等奖励
        
        if len(net_assignments) < 2:
            return 0.5  # 默认中等奖励
        
        try:
            # 获取 net_id（所有 assignments 应该属于同一个 net）
            net_id = None
            for assignment in net_assignments:
                pin = self.floorplan.get_pin(assignment.pin_id)
                if pin:
                    net_id = pin.net_id
                    break
            
            if net_id is None:
                logger.warning("无法获取 net_id，跳过 feedthrough reward")
                return 0.5
            
            # 在 placedb 中找到对应的 net
            target_net = None
            for net in self.placedb.nets_list:
                if getattr(net, 'id', None) == net_id or getattr(net, 'net_id', None) == net_id:
                    target_net = net
                    break
            
            if target_net is None:
                logger.warning(f"无法找到 net {net_id}，跳过 feedthrough reward")
                return 0.5
            
            # 设置 pin 坐标（已分配的使用 SegmentInst 中点，未分配的使用 block 质心）
            for assignment in net_assignments:
                pin = self.floorplan.get_pin(assignment.pin_id)
                if not pin:
                    continue
                
                # 找到 placedb 中对应的 pin
                placedb_pin = None
                for net_pin in getattr(target_net, 'pins', []):
                    if getattr(net_pin, 'id', None) == assignment.pin_id:
                        placedb_pin = net_pin
                        break
                
                if placedb_pin is None:
                    continue
                
                if assignment.seg_id == -1:
                    # 未分配，使用 block 质心
                    block = self.floorplan.get_block(pin.block_id)
                    if block and block.position:
                        placedb_pin.x = block.position[0]
                        placedb_pin.y = block.position[1]
                else:
                    # 已分配，使用 SegmentInst 中点
                    seg_inst = self.floorplan.get_segment_inst_for_pin(assignment.pin_id, assignment.seg_id)
                    if seg_inst:
                        placedb_pin.x = seg_inst.cx
                        placedb_pin.y = seg_inst.cy
                    else:
                        # 回退到 block 质心
                        block = self.floorplan.get_block(pin.block_id)
                        if block and block.position:
                            placedb_pin.x = block.position[0]
                            placedb_pin.y = block.position[1]
            
            # 计算实际的 feedthrough
            actual_feedthrough = self.ftpred_session.run_one_net(self.placedb, target_net)
            
            # 计算基线 feedthrough（所有 pin 都在 block 质心）
            # 先保存当前坐标
            saved_coords = {}
            for net_pin in getattr(target_net, 'pins', []):
                saved_coords[getattr(net_pin, 'id', None)] = (getattr(net_pin, 'x', 0), getattr(net_pin, 'y', 0))
            
            # 设置所有 pin 为 block 质心
            for assignment in net_assignments:
                pin = self.floorplan.get_pin(assignment.pin_id)
                if not pin:
                    continue
                
                placedb_pin = None
                for net_pin in getattr(target_net, 'pins', []):
                    if getattr(net_pin, 'id', None) == assignment.pin_id:
                        placedb_pin = net_pin
                        break
                
                if placedb_pin:
                    block = self.floorplan.get_block(pin.block_id)
                    if block and block.position:
                        placedb_pin.x = block.position[0]
                        placedb_pin.y = block.position[1]
            
            # 计算基线 feedthrough
            baseline_feedthrough = self.ftpred_session.run_one_net(self.placedb, target_net)
            
            # 恢复坐标
            for pin_id, (x, y) in saved_coords.items():
                for net_pin in getattr(target_net, 'pins', []):
                    if getattr(net_pin, 'id', None) == pin_id:
                        net_pin.x = x
                        net_pin.y = y
                        break
            
            # 计算 feedthrough 奖励（feedthrough 越小越好）
            # 使用基线 feedthrough 作为参考点
            if baseline_feedthrough > 0:
                # 当 actual < baseline 时，reward > 0.5（更好）
                # 当 actual > baseline 时，reward < 0.5（更差）
                # 使用指数衰减：reward = exp(-actual/baseline)
                feedthrough_reward = math.exp(-actual_feedthrough / baseline_feedthrough)
            else:
                # 基线为0的情况：actual=0 时 reward=1，否则递减
                if actual_feedthrough == 0:
                    feedthrough_reward = 1.0
                else:
                    feedthrough_reward = math.exp(-actual_feedthrough)
            
            logger.debug(f"Feedthrough 奖励计算: 实际={actual_feedthrough}, 基线={baseline_feedthrough}, "
                        f"归一化={normalized_feedthrough:.4f}, 奖励={feedthrough_reward:.4f}")
            
            return max(0.0, min(1.0, feedthrough_reward))
            
        except Exception as e:
            logger.error(f"计算 feedthrough reward 失败: {e}")
            return 0.5  # 出错时返回默认奖励

    def clear_cache(self):
        """清除缓存"""
        self.wirelength_cache.clear()

# 工厂函数
def create_simple_reward_function(floorplan: FloorPlanRO) -> SimpleRewardFunction:
    """创建简化奖励函数"""
    return SimpleRewardFunction(floorplan=floorplan)
