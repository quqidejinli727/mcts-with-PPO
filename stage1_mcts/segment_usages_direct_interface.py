"""
Segment Usages Direct Interface
直接从MCTS结果中的global_state.segment_usages获取数据
去除id为-1的部分，从floorplan获得相应seg信息，从assigned_pins里的id直接获得对应的pin
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SegmentInstAssignmentInfo:
    """SegmentInst分配信息，包含block级别的详细信息"""
    segment_inst_id: int
    segment_id: int
    segment_info: Dict[str, Any]  # Segment对象的序列化信息（母模块）
    block_id: int
    block_name: str
    assigned_pins: List[Dict[str, Any]]  # Pin对象的序列化信息列表
    used_capacity: float
    remaining_capacity: float
    coordinates: Tuple[float, float, float, float]  # (x1, y1, x2, y2) 实际坐标
    center_point: Tuple[float, float]
    direction: int
    edge_id: int
    max_capacity: float
    length: float


@dataclass
class SegmentAssignmentInfo:
    """Segment分配信息，按母模块聚合，内部包含各block的SegmentInst信息"""
    segment_id: int
    segment_info: Dict[str, Any]  # Segment对象的序列化信息（母模块）
    segment_insts: Dict[int, SegmentInstAssignmentInfo]  # {block_id: SegmentInstAssignmentInfo}
    total_used_capacity: float
    total_remaining_capacity: float
    direction: int
    edge_id: int
    max_capacity: float
    length: float


class SegmentUsagesDirectInterface:
    """SegmentUsages直接接口类 - 从MCTS结果直接读取segment_usages"""
    
    def __init__(self, floorplan):
        """
        初始化接口
        
        Args:
            floorplan: FloorPlanRO实例，提供segment和pin的详细信息
        """
        self.floorplan = floorplan
        logger.info("SegmentUsagesDirectInterface初始化完成")
    
    def convert_from_mcts_segment_usages(self, mcts_segment_usages: Dict[str, Dict[str, Any]]) -> Dict[int, SegmentAssignmentInfo]:
        """
        直接从MCTS结果的segment_usages转换数据
        
        Args:
            mcts_segment_usages: MCTS结果中的segment_usages字典，格式为 {segment_id_str: {"used_capacity": float, "assigned_pins": List[int]}}
            
        Returns:
            转换后的字典，格式为 {segment_id: SegmentAssignmentInfo}
        """
        result = {}
        
        for segment_id_str, usage_data in mcts_segment_usages.items():
            # 跳过id为-1的部分
            if segment_id_str == "-1":
                logger.debug(f"跳过segment_id为-1的条目")
                continue
            
            try:
                segment_id = int(segment_id_str)
                seg_info = self._convert_single_segment_usage(segment_id, usage_data)
                if seg_info:
                    result[segment_id] = seg_info
                else:
                    logger.warning(f"转换segment {segment_id} 失败，跳过")
            except ValueError:
                logger.error(f"无效的segment_id: {segment_id_str}")
                continue
            except Exception as e:
                logger.error(f"转换segment {segment_id_str} 时发生错误: {e}")
                continue
        
        logger.info(f"成功转换 {len(result)} 个segment_usages")
        return result
    
    def _convert_single_segment_usage(self, segment_id: int, usage_data: Dict[str, Any]) -> Optional[SegmentAssignmentInfo]:
        """
        转换单个segment_usage，按block分组创建SegmentInst信息
        
        Args:
            segment_id: segment ID（母模块级别）
            usage_data: MCTS结果中的usage数据，包含used_capacity和assigned_pins
            
        Returns:
            SegmentAssignmentInfo对象，如果转换失败则返回None
        """
        try:
            # 从floorplan获取segment基本信息（母模块）
            segment_info = self.floorplan.get_segment(segment_id)
            
            # 获取使用数据
            used_capacity = usage_data.get('used_capacity', 0.0)
            assigned_pin_ids = usage_data.get('assigned_pins', [])
            
            # 按block_id分组pins
            pins_by_block: Dict[int, List[Dict[str, Any]]] = {}
            for pin_id in assigned_pin_ids:
                if pin_id == -1:  # 去除id为-1的部分
                    continue
                    
                try:
                    pin_info = self.floorplan.get_pin(pin_id)
                    block_id = pin_info.block_id
                    
                    # 将Pin对象转换为字典格式
                    pin_dict = {
                        'id': pin_info.id,
                        'name': pin_info.name,
                        'block_id': block_id,
                        'width': pin_info.width,
                        'net_id': pin_info.net_id,
                        'isomorphic_group_id': pin_info.isomorphic_group_id
                    }
                    
                    if block_id not in pins_by_block:
                        pins_by_block[block_id] = []
                    pins_by_block[block_id].append(pin_dict)
                    
                except Exception as e:
                    logger.warning(f"获取pin {pin_id} 详细信息失败: {e}")
            
            # 处理模拟Pin还原：将segment中的模拟Pin根据其原始pin归一到原始形态
            DUMMY_SEPARATOR = "#DUMMY#"
            
            def extract_original_pin_name(full_name: str) -> str:
                """从模拟Pin的全名中提取原始Pin名称
                格式: "parent_inst.pin_name#DUMMY#index#parent_module"
                返回: "parent_inst.pin_name"
                """
                if DUMMY_SEPARATOR not in full_name:
                    return full_name
                # 找到第一个#DUMMY#之前的部分
                return full_name.split(DUMMY_SEPARATOR)[0]
            
            for block_id in list(pins_by_block.keys()):
                pins = pins_by_block[block_id]
                
                # 按原始Pin名称分组（处理模拟Pin的合并）
                # key: 原始Pin名称, value: 该原始Pin对应的所有pin（包括正常pin和模拟pin）
                original_pin_groups: Dict[str, List[Dict[str, Any]]] = {}
                
                for pin in pins:
                    pin_name = pin.get('name', '')
                    # 获取原始pin名称（模拟pin会去掉#DUMMY#后缀）
                    original_name = extract_original_pin_name(pin_name)
                    
                    if original_name not in original_pin_groups:
                        original_pin_groups[original_name] = []
                    original_pin_groups[original_name].append(pin)
                
                # 合并结果
                merged_pins = []
                
                # 处理每个原始Pin组
                for original_name, pin_list in original_pin_groups.items():
                    # 检查该组内是否有正常pin（非模拟pin）
                    normal_pins = [p for p in pin_list if DUMMY_SEPARATOR not in p.get('name', '')]
                    dummy_pins = [p for p in pin_list if DUMMY_SEPARATOR in p.get('name', '')]
                    
                    if normal_pins:
                        # 有正常pin，优先使用正常pin（直接保留）
                        merged_pins.extend(normal_pins)
                    elif dummy_pins:
                        # 全是模拟pin，合并为一个原始pin
                        first_dummy = dummy_pins[0]
                        
                        merged_pin = {
                            'id': first_dummy['id'],
                            'name': original_name,  # 恢复原始全名
                            'block_id': first_dummy['block_id'],
                            'width': first_dummy['width'],
                            'net_id': first_dummy['net_id'],
                            'isomorphic_group_id': first_dummy.get('isomorphic_group_id'),
                            'is_merged_dummy': True,
                            'merged_from': [p['id'] for p in dummy_pins]
                        }
                        merged_pins.append(merged_pin)
                
                # 更新该block的pins
                pins_by_block[block_id] = merged_pins
            
            # 构建segment信息字典（母模块）
            segment_dict = {
                'id': segment_info.id,
                'edge_id': segment_info.edge_id,
                'block_id': segment_info.block_id,
                'x1': segment_info.x1,
                'y1': segment_info.y1,
                'x2': segment_info.x2,
                'y2': segment_info.y2,
                'length': segment_info.length,
                'max_capacity': segment_info.max_capacity,
                'direction': segment_info.direction,
                'cx': segment_info.cx,
                'cy': segment_info.cy,
                'midpoint': segment_info.midpoint
            }
            
            # 为每个block创建SegmentInstAssignmentInfo
            segment_insts: Dict[int, SegmentInstAssignmentInfo] = {}
            total_used_capacity = 0.0
            
            for block_id, block_pins in pins_by_block.items():
                # 计算该block的容量使用
                block_used_capacity = sum(pin['width'] for pin in block_pins)
                total_used_capacity += block_used_capacity
                
                # 获取block名称
                try:
                    block = self.floorplan.get_block(block_id)
                    block_name = block.name if block else f"block_{block_id}"
                except:
                    block_name = f"block_{block_id}"
                
                # 获取该block的SegmentInst
                seg_inst = self.floorplan.get_segment_inst(segment_id, block_id)
                
                if seg_inst:
                    # 使用SegmentInst的实际坐标
                    coordinates = (seg_inst.x1, seg_inst.y1, seg_inst.x2, seg_inst.y2)
                    center_point = (seg_inst.cx, seg_inst.cy)
                    length = seg_inst.length
                    direction = seg_inst.direction
                    segment_inst_id = seg_inst.id
                else:
                    # 如果找不到SegmentInst，使用母模块坐标
                    coordinates = (segment_info.x1, segment_info.y1, segment_info.x2, segment_info.y2)
                    center_point = segment_info.midpoint
                    length = segment_info.length
                    direction = segment_info.direction
                    segment_inst_id = -1
                
                segment_inst_info = SegmentInstAssignmentInfo(
                    segment_inst_id=segment_inst_id,
                    segment_id=segment_id,
                    segment_info=segment_dict,
                    block_id=block_id,
                    block_name=block_name,
                    assigned_pins=block_pins,
                    used_capacity=block_used_capacity,
                    remaining_capacity=segment_info.max_capacity - block_used_capacity,
                    coordinates=coordinates,
                    center_point=center_point,
                    direction=direction,
                    edge_id=segment_info.edge_id,
                    max_capacity=segment_info.max_capacity,
                    length=length
                )
                
                segment_insts[block_id] = segment_inst_info
            
            # 如果没有分配到任何block，创建一个空的SegmentAssignmentInfo
            if not segment_insts:
                total_used_capacity = used_capacity
            
            total_remaining_capacity = segment_info.max_capacity * len(segment_insts) - total_used_capacity if segment_insts else segment_info.max_capacity - total_used_capacity
            
            return SegmentAssignmentInfo(
                segment_id=segment_id,
                segment_info=segment_dict,
                segment_insts=segment_insts,
                total_used_capacity=total_used_capacity,
                total_remaining_capacity=total_remaining_capacity,
                direction=segment_info.direction,
                edge_id=segment_info.edge_id,
                max_capacity=segment_info.max_capacity,
                length=segment_info.length
            )
            
        except Exception as e:
            logger.error(f"转换segment {segment_id} 失败: {e}")
            return None
    
    def load_mcts_segment_usages(self, mcts_results_file: str) -> Dict[str, Dict[str, Any]]:
        """
        从MCTS结果文件加载segment_usages数据
        
        Args:
            mcts_results_file: MCTS结果文件路径
            
        Returns:
            segment_usages字典
        """
        try:
            with open(mcts_results_file, 'r', encoding='utf-8') as f:
                mcts_data = json.load(f)
            
            if 'global_state' in mcts_data and 'segment_usages' in mcts_data['global_state']:
                segment_usages = mcts_data['global_state']['segment_usages']
                logger.info(f"从MCTS结果文件加载了 {len(segment_usages)} 个segment_usages")
                return segment_usages
            else:
                logger.error("MCTS结果文件中未找到global_state.segment_usages")
                return {}
                
        except Exception as e:
            logger.error(f"加载MCTS结果文件失败: {e}")
            return {}
    
    def get_segment_info_from_floorplan(self, segment_id: int) -> Optional[Dict[str, Any]]:
        """
        直接从floorplan获取segment信息
        
        Args:
            segment_id: segment ID
            
        Returns:
            segment信息的字典格式
        """
        try:
            segment_info = self.floorplan.get_segment(segment_id)
            return {
                'id': segment_info.id,
                'edge_id': segment_info.edge_id,
                'block_id': segment_info.block_id,
                'x1': segment_info.x1,
                'y1': segment_info.y1,
                'x2': segment_info.x2,
                'y2': segment_info.y2,
                'length': segment_info.length,
                'max_capacity': segment_info.max_capacity,
                'direction': segment_info.direction,
                'cx': segment_info.cx,
                'cy': segment_info.cy,
                'midpoint': segment_info.midpoint
            }
        except Exception as e:
            logger.error(f"从floorplan获取segment {segment_id} 信息失败: {e}")
            return None
    
    def get_pin_info_from_floorplan(self, pin_id: int) -> Optional[Dict[str, Any]]:
        """
        直接从floorplan获取pin信息
        
        Args:
            pin_id: pin ID
            
        Returns:
            pin信息的字典格式
        """
        try:
            pin_info = self.floorplan.get_pin(pin_id)
            return {
                'id': pin_info.id,
                'name': pin_info.name,
                'block_id': pin_info.block_id,
                'width': pin_info.width,
                'net_id': pin_info.net_id,
                'isomorphic_group_id': pin_info.isomorphic_group_id
            }
        except Exception as e:
            logger.error(f"从floorplan获取pin {pin_id} 信息失败: {e}")
            return None
    
    def print_segment_assignment_info(self, seg_info: SegmentAssignmentInfo) -> None:
        """打印segment分配信息的详细内容"""
        print(f"\n=== Segment {seg_info.segment_id} 分配信息 ===")
        print(f"方向: {seg_info.direction}")
        print(f"长度: {seg_info.length}")
        print(f"最大容量: {seg_info.max_capacity}")
        print(f"总已用容量: {seg_info.total_used_capacity:.3f}")
        print(f"总剩余容量: {seg_info.total_remaining_capacity:.3f}")
        print(f"所属edge: {seg_info.edge_id}")
        
        print(f"\nSegmentInsts ({len(seg_info.segment_insts)}个):")
        for block_id, seg_inst in seg_info.segment_insts.items():
            print(f"\n  Block {block_id} ({seg_inst.block_name}):")
            print(f"    SegmentInst ID: {seg_inst.segment_inst_id}")
            print(f"    坐标: {seg_inst.coordinates}")
            print(f"    中心点: {seg_inst.center_point}")
            print(f"    已用容量: {seg_inst.used_capacity:.3f}")
            print(f"    剩余容量: {seg_inst.remaining_capacity:.3f}")
            print(f"    已分配Pins ({len(seg_inst.assigned_pins)}个):")
            for i, pin_info in enumerate(seg_inst.assigned_pins, 1):
                print(f"      {i}. Pin ID: {pin_info['id']}, Name: {pin_info['name']}")
                print(f"         Width: {pin_info['width']}, Net ID: {pin_info['net_id']}")
    
    def export_segment_assignments(self, segment_assignments: Dict[int, SegmentAssignmentInfo], output_file: str) -> None:
        """
        导出segment分配信息到JSON文件
        
        Args:
            segment_assignments: 转换后的segment分配信息字典
            output_file: 输出文件路径
        """
        export_data = {
            'total_segments': len(segment_assignments),
            'segment_assignments': {}
        }
        
        # 导出每个segment的详细信息
        for segment_id, seg_info in segment_assignments.items():
            export_data['segment_assignments'][segment_id] = asdict(seg_info)
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Segment分配信息已导出到: {output_file}")


def create_segment_usages_direct_interface(floorplan) -> SegmentUsagesDirectInterface:
    """
    工厂函数，创建SegmentUsagesDirectInterface实例
    
    Args:
        floorplan: FloorPlanRO实例
        
    Returns:
        SegmentUsagesDirectInterface实例
    """
    return SegmentUsagesDirectInterface(floorplan)


# 使用示例
if __name__ == "__main__":
    # 这里需要提供实际的floorplan实例
    # from data_model import FloorPlanRO
    # 
    # # 假设已经有floorplan实例
    # floorplan = FloorPlanRO(...)
    # interface = create_segment_usages_direct_interface(floorplan)
    # 
    # # 从MCTS结果文件加载segment_usages
    # mcts_file = "mcts_assignment_results_1777367335.json"
    # segment_usages = interface.load_mcts_segment_usages(mcts_file)
    # 
    # # 转换数据
    # result = interface.convert_from_mcts_segment_usages(segment_usages)
    # 
    # # 打印结果
    # for seg_id, seg_info in list(result.items())[:5]:  # 只打印前5个
    #     interface.print_segment_assignment_info(seg_info)
    # 
    # # 导出到文件
    # interface.export_segment_assignments(result, "direct_segment_assignments.json")
    
    pass