#!/usr/bin/env python3
"""
线长计算脚本
从分配结果和网表连接关系计算线长

输入:
- 分配结果: mcts_result/final_result/segment_assignments_1777969377.json
- 网表连接关系: test_case_from_huawei/pingroup_case2.json
- floorplanRO: 从PlaceDB加载

输出:
- 每个net的HPWL线长
- 找不到的pin列表
- 总线长统计
"""

import json
import sys
import logging
import os
import random
import copy
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path

# 导入 feedthrough 相关模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'src_version_v6(case2)', 'feedthrough'))
from feedthrough import ftpred_loader
from PlaceDB import PlaceDB

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 路径配置
# ASSIGNMENT_FILE = "mcts_result/final_result/segment_assignments_1778003034.json" #0.1 feedthrough + 0.9 wirelength
# ASSIGNMENT_FILE = "mcts_result/final_result/segment_assignments_1777999349.json" #wirelength only
# ASSIGNMENT_FILE = "mcts_result/final_result/segment_assignments_1778031543.json" #feedthrogh only
ASSIGNMENT_FILE = "mcts_result/final_result/segment_assignments_1778152142.json" #wirelength update only
PINGROUP_FILE = "test_case_from_huawei/pingroup_case2.json"
BLOCK_FILE = "test_case_from_huawei/block_case2.json"

# feedthrough 计算器路径
FTPRED_PATH = os.path.join(os.path.dirname(__file__),  "feedthrough", "build", "ftpred.exe" if os.name == "nt" else "ftpred")


@dataclass
class PinLocation:
    """Pin位置信息"""
    pin_name: str
    x: float
    y: float
    source: str  # "segment_inst" or "block_centroid"
    segment_inst_id: Optional[int] = None
    block_name: Optional[str] = None


def load_json(filepath: str) -> dict:
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_pingroup_nets(pingroup_data: list) -> List[List[Dict]]:
    """
    解析pingroup数据，返回nets列表
    每个net是一个pingroup链
    """
    return pingroup_data


def extract_pin_name_from_pingroup(pingroup: Dict) -> str:
    """
    从pingroup信息构建pin全名
    格式: parent_inst.pingroup_name
    """
    parent_inst = pingroup.get("parent_inst", "")
    pingroup_name = pingroup.get("pingroup_name", "")
    return f"{parent_inst}.{pingroup_name}"


def parse_assignments(assignment_data: Dict) -> Dict[int, Dict]:
    """
    解析分配结果，构建pin_id到segment_inst的映射
    
    返回: {
        pin_id: {
            "segment_id": int,
            "segment_inst_id": int,
            "block_id": int,
            "block_name": str,
            "midpoint": [x, y],
            "pin_name": str
        }
    }
    """
    pin_assignments = {}
    segment_assignments = assignment_data.get("segment_assignments", {})
    
    for seg_id, seg_data in segment_assignments.items():
        segment_insts = seg_data.get("segment_insts", {})
        segment_info = seg_data.get("segment_info", {})
        
        for block_id, inst_data in segment_insts.items():
            assigned_pins = inst_data.get("assigned_pins", [])
            segment_inst_id = inst_data.get("segment_inst_id")
            center_point = inst_data.get("center_point", [0, 0])
            block_name = inst_data.get("block_name", "")
            
            for pin in assigned_pins:
                pin_id = pin.get("id")
                if pin_id is not None:
                    pin_assignments[pin_id] = {
                        "segment_id": int(seg_id),
                        "segment_inst_id": segment_inst_id,
                        "block_id": int(block_id),
                        "block_name": block_name,
                        "midpoint": center_point,
                        "pin_name": pin.get("name", ""),
                        "net_id": pin.get("net_id")
                    }
    
    logger.info(f"从分配结果解析到 {len(pin_assignments)} 个pin")
    return pin_assignments


def parse_block_vertices(block_data: Dict) -> Dict[int, Dict]:
    """
    解析block文件，获取block的顶点信息
    
    返回: {
        block_id: {
            "name": str,
            "centroid": [x, y],
            "vertices": [[x1,y1], [x2,y2], ...]
        }
    }
    """
    blocks_info = {}
    
    # block文件结构可能是module -> instances列表
    for module_name, module_data in block_data.items():
        if isinstance(module_data, dict):
            instances = module_data.get("instances", [])
            for inst in instances:
                block_id = inst.get("id")
                if block_id is not None:
                    vertices = inst.get("vertex", [])
                    # 计算质心
                    if vertices:
                        xs = [v[0] for v in vertices]
                        ys = [v[1] for v in vertices]
                        centroid = [sum(xs) / len(xs), sum(ys) / len(ys)]
                    else:
                        centroid = [0, 0]
                    
                    blocks_info[block_id] = {
                        "name": inst.get("name", ""),
                        "centroid": centroid,
                        "vertices": vertices,
                        "module": module_name
                    }
    
    logger.info(f"从block文件解析到 {len(blocks_info)} 个block")
    return blocks_info


def build_pin_name_to_id_map(pin_assignments: Dict[int, Dict]) -> Dict[str, int]:
    """构建pin名称到ID的映射"""
    name_to_id = {}
    for pin_id, pin_data in pin_assignments.items():
        pin_name = pin_data.get("pin_name", "")
        if pin_name:
            name_to_id[pin_name] = pin_id
    return name_to_id


def find_pin_location(
    pin_name: str,
    pin_assignments: Dict[int, Dict],
    pin_name_to_id: Dict[str, int],
    blocks_info: Dict[int, Dict],
    missing_pins: Set[str]
) -> Optional[PinLocation]:
    """
    查找pin的位置
    
    优先级:
    1. 从分配结果中找，使用所在segment_inst的中点
    2. 从block信息中找，使用所在block的质心
    
    返回: PinLocation对象，如果找不到返回None
    """
    # 1. 从分配结果中查找
    pin_id = pin_name_to_id.get(pin_name)
    if pin_id is not None:
        pin_data = pin_assignments.get(pin_id)
        if pin_data:
            midpoint = pin_data.get("midpoint", [0, 0])
            return PinLocation(
                pin_name=pin_name,
                x=midpoint[0],
                y=midpoint[1],
                source="segment_inst",
                segment_inst_id=pin_data.get("segment_inst_id"),
                block_name=pin_data.get("block_name")
            )
    
    # 2. 从block信息中查找
    # 解析pin名称获取block路径
    # pin名称格式: TOP.U_XXX.U_YYY.pin_name
    parts = pin_name.rsplit('.', 1)
    if len(parts) == 2:
        block_path = parts[0]
        # 在blocks_info中查找匹配的block
        for block_id, block_data in blocks_info.items():
            if block_data.get("name") == block_path:
                centroid = block_data.get("centroid", [0, 0])
                return PinLocation(
                    pin_name=pin_name,
                    x=centroid[0],
                    y=centroid[1],
                    source="block_centroid",
                    block_name=block_path
                )
    
    # 记录找不到的pin
    missing_pins.add(pin_name)
    return None


def calculate_hpwl(locations: List[PinLocation]) -> float:
    """
    计算Half-Perimeter Wirelength (HPWL)
    HPWL = (max_x - min_x) + (max_y - min_y)
    """
    if not locations:
        return 0.0
    
    xs = [loc.x for loc in locations]
    ys = [loc.y for loc in locations]
    
    hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
    return hpwl


def init_feedthrough_calculator(block_file: str, pingroup_file: str) -> Tuple[Optional[PlaceDB], Optional[ftpred_loader.FtpredBinSession]]:
    """
    初始化 feedthrough 计算器
    
    返回: (placedb, ftpred_session) 或 (None, None) 如果初始化失败
    """
    try:
        # 加载 PlaceDB
        placedb = PlaceDB(block_file, pingroup_file)
        logger.info(f"PlaceDB 加载完成: {len(placedb.all_modules_list)} 模块, {len(placedb.nets_list)} 网表")
        
        # 构建 modules
        modules = ftpred_loader.build_modules_text(placedb)
        
        # 创建常驻会话
        session = ftpred_loader.FtpredBinSession(FTPRED_PATH, modules)
        logger.info(f"Feedthrough 计算器初始化成功")
        
        return placedb, session
    except Exception as e:
        logger.error(f"Feedthrough 计算器初始化失败: {e}")
        return None, None


def calculate_net_feedthrough(
    net_pingroup_chain: List[Dict],
    locations: List[PinLocation],
    placedb: PlaceDB,
    ftpred_session: ftpred_loader.FtpredBinSession
) -> int:
    """
    计算单个 net 的 feedthrough 数量
    
    参数:
        net_pingroup_chain: net 的 pingroup 链
        locations: pin 的位置列表
        placedb: PlaceDB 实例
        ftpred_session: feedthrough 计算器会话
    
    返回: feedthrough 数量，如果计算失败返回 -1
    """
    try:
        # 找到对应的 net
        if not net_pingroup_chain:
            return 0
        
        first_pingroup = net_pingroup_chain[0]
        parent_inst = first_pingroup.get("parent_inst", "")
        pingroup_name = first_pingroup.get("pingroup_name", "")
        
        # 在 placedb 中找到对应的 net
        target_net = None
        for net in placedb.nets_list:
            net_pins = getattr(net, 'pins', [])
            for pin in net_pins:
                if (getattr(pin, 'parent_inst', '') == parent_inst and
                    getattr(pin, 'pingroup_name', '') == pingroup_name):
                    target_net = net
                    break
            if target_net:
                break
        
        if target_net is None:
            logger.warning(f"找不到对应的 net: {parent_inst}.{pingroup_name}")
            return -1
        
        # 设置 pin 坐标
        pin_name_to_location = {loc.pin_name: loc for loc in locations}
        
        for pin in getattr(target_net, 'pins', []):
            pin_full_name = f"{getattr(pin, 'parent_inst', '')}.{getattr(pin, 'pingroup_name', '')}"
            if pin_full_name in pin_name_to_location:
                loc = pin_name_to_location[pin_full_name]
                pin.x = loc.x
                pin.y = loc.y
            else:
                # 如果找不到位置，使用默认坐标
                pin.x = 0.0
                pin.y = 0.0
        
        # 计算 feedthrough
        feedthrough = ftpred_session.run_one_net(placedb, target_net)
        return int(feedthrough)
        
    except Exception as e:
        logger.error(f"计算 feedthrough 失败: {e}")
        return -1


def fix_missing_pins(
    placedb: PlaceDB,
    assignment_data: Dict,
    missing_pins: Set[str],
    pin_name_to_id: Dict[str, int],
    output_file: str
) -> Dict:
    """
    修复缺失的pin分配
    
    步骤:
    1. 遍历所有missing pin，根据pingroup_name查找其可分配的segment，随机选择一个segment_inst进行分配
       如果不存在可分配的segment_inst，则根据floorplan创建新的segment_inst
    2. 检查所有单net的pin，根据复用关系，将其分配到其复用pin已经分配的结果上
    
    参数:
        placedb: PlaceDB实例
        assignment_data: 原始分配数据
        missing_pins: 缺失的pin名称集合
        pin_name_to_id: pin名称到ID的映射
        output_file: 修复后的分配结果输出文件路径
    
    返回:
        修复后的分配数据
    """
    logger.info("\n" + "=" * 60)
    logger.info("开始修复缺失的pin分配...")
    logger.info("=" * 60)
    
    # 深拷贝原始数据
    fixed_assignment_data = copy.deepcopy(assignment_data)
    segment_assignments = fixed_assignment_data.get("segment_assignments", {})
    
    # 转换为FloorPlanRO获取完整的segment和block信息
    try:
        floorplan = placedb.convert_to_floorplan_ro()
        logger.info(f"FloorPlanRO转换完成: {floorplan.num_segments} segments, {floorplan.num_blocks} blocks")
    except Exception as e:
        logger.error(f"FloorPlanRO转换失败: {e}")
        floorplan = None
    
    # 构建segment_id到segment_inst列表的映射
    seg_id_to_insts = {}
    for seg_id, seg_data in segment_assignments.items():
        segment_insts = seg_data.get("segment_insts", {})
        seg_id_to_insts[int(seg_id)] = list(segment_insts.keys())
    
    # 获取下一个可用的segment_inst_id
    next_seg_inst_id = 0
    for seg_id, seg_data in segment_assignments.items():
        for block_id, inst_data in seg_data.get("segment_insts", {}).items():
            seg_inst_id = inst_data.get("segment_inst_id", 0)
            next_seg_inst_id = max(next_seg_inst_id, seg_inst_id + 1)
    
    # 辅助函数：获取或创建segment_inst
    def get_or_create_segment_inst(block_name: str, block_id: int):
        """获取或创建指定block的segment_inst，返回(seg_id, inst_data)"""
        nonlocal next_seg_inst_id
        
        # 首先尝试查找现有的segment_inst
        for seg_id, seg_data in segment_assignments.items():
            segment_insts = seg_data.get("segment_insts", {})
            for existing_block_id, inst_data in segment_insts.items():
                if int(existing_block_id) == block_id:
                    return int(seg_id), inst_data
        
        # 如果不存在，需要从floorplan创建新的segment_inst
        if floorplan is None:
            return None, None
        
        # 找到该block对应的所有segments
        block = floorplan.get_block(block_id)
        if not block:
            logger.warning(f"在FloorPlanRO中找不到block: {block_id} ({block_name})")
            return None, None
        
        # 获取母模块的segments
        master_block = floorplan.get_master_block(block.master_id)
        if not master_block:
            logger.warning(f"找不到block {block_id} 的母模块 {block.master_id}")
            return None, None
        
        # 创建新的segment和segment_inst
        # 获取该block的所有segment IDs
        block_segments = floorplan.get_block_segments(block_id)
        
        if not block_segments:
            logger.warning(f"Block {block_id} ({block_name}) 没有可用的segments")
            return None, None
        
        # 随机选择一个segment来创建segment_inst
        chosen_seg_id = random.choice(list(block_segments))
        segment = floorplan.get_segment(chosen_seg_id)
        
        if not segment:
            return None, None
        
        # 确保segment在segment_assignments中存在
        seg_id_str = str(chosen_seg_id)
        if seg_id_str not in segment_assignments:
            # 创建新的segment entry
            segment_assignments[seg_id_str] = {
                "segment_id": chosen_seg_id,
                "segment_info": {
                    "id": segment.id,
                    "edge_id": getattr(segment, 'edge_id', 0),
                    "block_id": segment.block_id,
                    "x1": segment.x1,
                    "y1": segment.y1,
                    "x2": segment.x2,
                    "y2": segment.y2,
                    "length": getattr(segment, 'length', 0),
                    "max_capacity": getattr(segment, 'max_capacity', 0),
                    "direction": getattr(segment, 'direction', 0),
                    "cx": getattr(segment, 'cx', (segment.x1 + segment.x2) / 2),
                    "cy": getattr(segment, 'cy', (segment.y1 + segment.y2) / 2),
                    "midpoint": [
                        getattr(segment, 'cx', (segment.x1 + segment.x2) / 2),
                        getattr(segment, 'cy', (segment.y1 + segment.y2) / 2)
                    ]
                },
                "segment_insts": {}
            }
        
        # 计算transform后的坐标（简化处理，使用block的position）
        import math
        dx = block.position[0] if block.position else 0
        dy = block.position[1] if block.position else 0
        
        # 创建新的segment_inst
        new_inst_id = next_seg_inst_id
        next_seg_inst_id += 1
        
        new_segment_inst = {
            "segment_inst_id": new_inst_id,
            "segment_id": chosen_seg_id,
            "segment_info": segment_assignments[seg_id_str]["segment_info"],
            "block_id": block_id,
            "block_name": block_name,
            "assigned_pins": [],
            "used_capacity": 0.0,
            "remaining_capacity": getattr(segment, 'max_capacity', 0),
            "coordinates": [segment.x1 + dx, segment.y1 + dy, segment.x2 + dx, segment.y2 + dy],
            "center_point": [getattr(segment, 'cx', (segment.x1 + segment.x2) / 2) + dx,
                           getattr(segment, 'cy', (segment.y1 + segment.y2) / 2) + dy],
            "direction": getattr(segment, 'direction', 0),
            "edge_id": getattr(segment, 'edge_id', 0),
            "max_capacity": getattr(segment, 'max_capacity', 0),
            "length": math.sqrt((segment.x2 - segment.x1)**2 + (segment.y2 - segment.y1)**2)
        }
        
        # 添加到segment_assignments
        segment_assignments[seg_id_str]["segment_insts"][str(block_id)] = new_segment_inst
        
        logger.info(f"为block {block_name} (id={block_id}) 创建了新的segment_inst {new_inst_id} (seg_id={chosen_seg_id})")
        
        return chosen_seg_id, new_segment_inst
    
    # Step 1: 修复missing pins
    fixed_missing_count = 0
    for pin_name in missing_pins:
        # 解析pin名称获取parent_inst和pingroup_name
        parts = pin_name.rsplit('.', 1)
        if len(parts) != 2:
            continue
        parent_inst, pingroup_name = parts
        
        # 在placedb中查找对应的pin信息
        target_pin = None
        target_net = None
        for net in placedb.nets_list:
            for pin in getattr(net, 'pins', []):
                pin_full_name = f"{getattr(pin, 'parent_inst', '')}.{getattr(pin, 'pingroup_name', '')}"
                if pin_full_name == pin_name:
                    target_pin = pin
                    target_net = net
                    break
            if target_pin:
                break
        
        if not target_pin:
            logger.warning(f"在PlaceDB中找不到pin: {pin_name}")
            continue
        
        # 查找该pin所属的block
        parent_module = getattr(target_pin, 'parent_inst', '')
        
        # 找到对应的block_id
        block_id = None
        if floorplan:
            for bid in range(floorplan.num_blocks):
                block = floorplan.get_block(bid)
                if block and block.name == parent_module:
                    block_id = bid
                    break
        
        if block_id is None:
            logger.warning(f"找不到pin {pin_name} 所属的block: {parent_module}")
            continue
        
        # 获取或创建segment_inst
        chosen_seg_id, chosen_inst_data = get_or_create_segment_inst(parent_module, block_id)
        
        if chosen_inst_data is None:
            logger.warning(f"无法为pin {pin_name} 创建或找到segment_inst")
            continue
        
        # 添加pin到该segment_inst的assigned_pins
        new_pin = {
            "id": pin_name_to_id.get(pin_name, -1),
            "name": pin_name,
            "net_id": getattr(target_net, 'id', -1) if target_net else -1
        }
        
        if "assigned_pins" not in chosen_inst_data:
            chosen_inst_data["assigned_pins"] = []
        chosen_inst_data["assigned_pins"].append(new_pin)
        
        fixed_missing_count += 1
        logger.info(f"修复missing pin: {pin_name} -> segment_inst {chosen_inst_data.get('segment_inst_id')}")
    
    logger.info(f"修复了 {fixed_missing_count}/{len(missing_pins)} 个missing pins")
    
    # Step 2: 处理复用关系
    # 新逻辑：遍历所有segment，找到segment_inst最多的作为标杆，
    # 其他segment中pin数量少于标杆的，用标杆的inst补齐所有pin
    fixed_reuse_count = 0
    
    # 辅助函数：从pin name中提取block名称（最后一个.前的部分）
    def get_block_name_from_pin(pin_name: str) -> str:
        """从pin name中提取block名称（最后一个.前的部分）"""
        parts = pin_name.rsplit('.', 1)
        return parts[0] if len(parts) > 1 else ""
    
    # 辅助函数：替换pin name中的block名称
    def replace_block_name_in_pin(original_pin_name: str, new_block_name: str) -> str:
        """替换pin name中的block名称"""
        parts = original_pin_name.rsplit('.', 1)
        if len(parts) < 2:
            return original_pin_name
        return f"{new_block_name}.{parts[1]}"
    
    # 遍历所有segment进行修复
    for seg_id, seg_data in segment_assignments.items():
        segment_insts = seg_data.get("segment_insts", {})
        if not segment_insts:
            continue
        
        int_seg_id = int(seg_id)
        
        # Step 2.1: 从floorplan获取该segment对应的所有segment_inst，补充缺失的
        if floorplan is not None:
            # 从floorplan获取该segment对应的所有segment_inst
            floorplan_seg_insts = []
            for seg_inst in floorplan._segment_insts:
                if seg_inst.segment_id == int_seg_id:
                    floorplan_seg_insts.append(seg_inst)
            
            # 获取当前已有的block_id集合
            existing_block_ids = {int(bid) for bid in segment_insts.keys()}
            
            # 补充缺失的segment_inst
            for seg_inst in floorplan_seg_insts:
                if seg_inst.block_id not in existing_block_ids:
                    # 从floorplan获取block信息
                    block = floorplan.get_block(seg_inst.block_id)
                    block_name = block.name if block else f"block_{seg_inst.block_id}"
                    
                    # 从floorplan获取segment信息
                    segment = floorplan.get_segment(int_seg_id)
                    
                    # 创建新的segment_inst
                    new_inst_id = next_seg_inst_id
                    next_seg_inst_id += 1
                    
                    new_segment_inst = {
                        "segment_inst_id": new_inst_id,
                        "segment_id": int_seg_id,
                        "segment_info": seg_data.get("segment_info", {}),
                        "block_id": seg_inst.block_id,
                        "block_name": block_name,
                        "assigned_pins": [],
                        "used_capacity": 0.0,
                        "remaining_capacity": getattr(segment, 'max_capacity', 0) if segment else 0,
                        "coordinates": [seg_inst.x1, seg_inst.y1, seg_inst.x2, seg_inst.y2],
                        "center_point": [getattr(seg_inst, 'cx', (seg_inst.x1 + seg_inst.x2) / 2),
                                       getattr(seg_inst, 'cy', (seg_inst.y1 + seg_inst.y2) / 2)],
                        "direction": getattr(seg_inst, 'direction', 0),
                        "edge_id": getattr(seg_inst, 'edge_id', 0),
                        "max_capacity": getattr(seg_inst, 'max_capacity', 0),
                        "length": getattr(seg_inst, 'length', 0)
                    }
                    
                    segment_insts[str(seg_inst.block_id)] = new_segment_inst
                    logger.debug(f"补充segment_inst: seg_id={seg_id}, block_id={seg_inst.block_id}, "
                                f"block_name={block_name}, inst_id={new_inst_id}")
        
        # Step 2.2: 找到该segment中pin数量最多的segment_inst作为标杆
        benchmark_inst = None
        benchmark_pin_count = 0
        benchmark_pins = []
        
        for block_id, inst_data in segment_insts.items():
            pins = inst_data.get("assigned_pins", [])
            if len(pins) > benchmark_pin_count:
                benchmark_pin_count = len(pins)
                benchmark_inst = inst_data
                benchmark_pins = pins.copy()
        
        if not benchmark_inst or benchmark_pin_count == 0:
            continue
        
        # Step 2.3: 遍历该segment的所有segment_inst，补齐pin数量
        for block_id, inst_data in segment_insts.items():
            current_pins = inst_data.get("assigned_pins", [])
            current_pin_names = {p.get("name", "") for p in current_pins}
            
            # 如果当前inst的pin数量已经等于标杆，不需要补齐
            if len(current_pins) >= benchmark_pin_count:
                continue
            
            # 获取当前block的名称
            current_block_name = inst_data.get("block_name", "")
            if not current_block_name:
                # 尝试从floorplan获取
                if floorplan is not None:
                    block = floorplan.get_block(int(block_id))
                    if block:
                        current_block_name = block.name
            
            # 从标杆inst的pins中，为当前inst补齐缺失的pin
            for benchmark_pin in benchmark_pins:
                benchmark_pin_name = benchmark_pin.get("name", "")
                
                # 构造当前block对应的pin name
                if current_block_name:
                    new_pin_name = replace_block_name_in_pin(benchmark_pin_name, current_block_name)
                else:
                    new_pin_name = benchmark_pin_name
                
                # 检查该pin是否已存在
                if new_pin_name in current_pin_names:
                    continue
                
                # 创建新的pin，保持其他信息与标杆一致
                new_pin = {
                    "id": pin_name_to_id.get(new_pin_name, -1),
                    "name": new_pin_name,
                    "net_id": benchmark_pin.get("net_id", -1),
                    "block_id": inst_data.get("block_id", benchmark_pin.get("block_id", -1)),
                    "width": benchmark_pin.get("width", 0.04),
                    "isomorphic_group_id": benchmark_pin.get("isomorphic_group_id", None)
                }
                
                if "assigned_pins" not in inst_data:
                    inst_data["assigned_pins"] = []
                inst_data["assigned_pins"].append(new_pin)
                current_pin_names.add(new_pin_name)
                
                fixed_reuse_count += 1
                logger.debug(f"复用补齐: {new_pin_name} -> segment_inst {inst_data.get('segment_inst_id')} "
                            f"(复用自标杆pin {benchmark_pin_name})")
    
    logger.info(f"修复了 {fixed_reuse_count} 个复用pin的分配")
    
    # 保存修复后的结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_assignment_data, f, indent=2, ensure_ascii=False)
    logger.info(f"\n修复后的分配结果已保存到: {output_file}")
    
    return fixed_assignment_data


def calculate_net_wirelength(
    net_pingroup_chain: List[Dict],
    pin_assignments: Dict[int, Dict],
    pin_name_to_id: Dict[str, int],
    blocks_info: Dict[int, Dict],
    missing_pins: Set[str],
    skip_single_pin: bool = True
) -> Tuple[float, List[PinLocation], bool]:
    """
    计算单个net的线长
    
    遍历pingroup链中的所有pin，获取它们的位置，然后计算HPWL
    
    参数:
        skip_single_pin: 如果为True，单pin net返回skipped=True
    
    返回: (hpwl, found_locations, skipped)
    """
    # 跳过单pin net
    if skip_single_pin and len(net_pingroup_chain) <= 1:
        return 0.0, [], True
    
    locations = []
    
    for pingroup in net_pingroup_chain:
        pin_name = extract_pin_name_from_pingroup(pingroup)
        location = find_pin_location(
            pin_name,
            pin_assignments,
            pin_name_to_id,
            blocks_info,
            missing_pins
        )
        if location:
            locations.append(location)
    
    hpwl = calculate_hpwl(locations)
    return hpwl, locations, False


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("线长和 Feedthrough 计算脚本启动")
    logger.info("=" * 60)
    
    # 1. 加载数据
    logger.info("\n1. 加载数据文件...")
    
    try:
        assignment_data = load_json(ASSIGNMENT_FILE)
        logger.info(f"   ✓ 分配结果: {ASSIGNMENT_FILE}")
    except Exception as e:
        logger.error(f"   ✗ 加载分配结果失败: {e}")
        return 1
    
    try:
        pingroup_data = load_json(PINGROUP_FILE)
        logger.info(f"   ✓ 网表连接关系: {PINGROUP_FILE}")
    except Exception as e:
        logger.error(f"   ✗ 加载网表连接关系失败: {e}")
        return 1
    
    try:
        block_data = load_json(BLOCK_FILE)
        logger.info(f"   ✓ Block文件: {BLOCK_FILE}")
    except Exception as e:
        logger.error(f"   ✗ 加载Block文件失败: {e}")
        return 1
    
    # 1.5 初始化 feedthrough 计算器
    logger.info("\n1.5 初始化 feedthrough 计算器...")
    placedb, ftpred_session = init_feedthrough_calculator(BLOCK_FILE, PINGROUP_FILE)
    if placedb is None or ftpred_session is None:
        logger.warning("   ! Feedthrough 计算器初始化失败，将跳过 feedthrough 计算")
    
    # 2. 解析数据
    logger.info("\n2. 解析数据...")
    
    pin_assignments = parse_assignments(assignment_data)
    blocks_info = parse_block_vertices(block_data)
    pin_name_to_id = build_pin_name_to_id_map(pin_assignments)
    
    logger.info(f"   - Pin名称到ID映射: {len(pin_name_to_id)} 个")
    
    # 3. 计算每个net的线长和 feedthrough
    logger.info("\n3. 计算线长和 feedthrough...")
    
    nets = parse_pingroup_nets(pingroup_data)
    missing_pins = set()
    net_results = []
    total_hpwl = 0.0
    total_feedthrough = 0
    successful_nets = 0
    failed_nets = 0
    skipped_single_pin_nets = 0
    feedthrough_failed_nets = 0
    
    for net_idx, net_pingroup_chain in enumerate(nets):
        hpwl, locations, skipped = calculate_net_wirelength(
            net_pingroup_chain,
            pin_assignments,
            pin_name_to_id,
            blocks_info,
            missing_pins,
            skip_single_pin=True
        )
        
        # 获取net名称（使用第一个pingroup的successors或pingroup_name）
        net_name = ""
        if net_pingroup_chain:
            first_pingroup = net_pingroup_chain[0]
            successors = first_pingroup.get("successors", [])
            if successors:
                net_name = successors[0]
            else:
                net_name = first_pingroup.get("pingroup_name", f"net_{net_idx}")
        
        if skipped:
            # 单pin net，跳过
            skipped_single_pin_nets += 1
            net_results.append({
                "net_index": net_idx,
                "net_name": net_name,
                "pingroup_count": len(net_pingroup_chain),
                "found_pins": 0,
                "hpwl": 0.0,
                "skipped": True,
                "skip_reason": "single_pin_net",
                "locations": []
            })
        elif locations:
            total_hpwl += hpwl
            successful_nets += 1
            
            # 计算 feedthrough
            feedthrough = -1
            if ftpred_session is not None and placedb is not None:
                feedthrough = calculate_net_feedthrough(
                    net_pingroup_chain, locations, placedb, ftpred_session
                )
                if feedthrough >= 0:
                    total_feedthrough += feedthrough
                else:
                    feedthrough_failed_nets += 1
            
            net_results.append({
                "net_index": net_idx,
                "net_name": net_name,
                "pingroup_count": len(net_pingroup_chain),
                "found_pins": len(locations),
                "hpwl": hpwl,
                "feedthrough": feedthrough if feedthrough >= 0 else None,
                "skipped": False,
                "locations": [
                    {
                        "pin_name": loc.pin_name,
                        "x": loc.x,
                        "y": loc.y,
                        "source": loc.source,
                        "block_name": loc.block_name
                    }
                    for loc in locations
                ]
            })
        else:
            failed_nets += 1
            net_results.append({
                "net_index": net_idx,
                "net_name": net_name,
                "pingroup_count": len(net_pingroup_chain),
                "found_pins": 0,
                "hpwl": 0.0,
                "feedthrough": None,
                "skipped": False,
                "locations": []
            })
    
    # 关闭 feedthrough 会话
    if ftpred_session is not None:
        try:
            ftpred_session.close()
            logger.info("\nFeedthrough 会话已关闭")
        except Exception as e:
            logger.error(f"关闭 feedthrough 会话失败: {e}")
    
    # 4. 输出结果
    logger.info("\n4. 线长和 Feedthrough 计算结果")
    logger.info("=" * 60)
    
    valid_nets = successful_nets + failed_nets
    logger.info(f"\n统计信息:")
    logger.info(f"   - 总net数: {len(nets)}")
    logger.info(f"   - 单pin net (跳过): {skipped_single_pin_nets}")
    logger.info(f"   - 有效net数 (>=2 pins): {valid_nets}")
    logger.info(f"   - 成功计算线长的net: {successful_nets}")
    logger.info(f"   - 未找到任何pin的net: {failed_nets}")
    logger.info(f"   - 找不到的pin数: {len(missing_pins)}")
    logger.info(f"   - 总HPWL线长: {total_hpwl:.2f}")
    logger.info(f"   - 平均HPWL线长: {total_hpwl / successful_nets if successful_nets > 0 else 0:.2f}")
    
    # Feedthrough 统计
    if ftpred_session is not None:
        successful_feedthrough_nets = successful_nets - feedthrough_failed_nets
        logger.info(f"   - Feedthrough 计算成功net数: {successful_feedthrough_nets}")
        logger.info(f"   - Feedthrough 计算失败net数: {feedthrough_failed_nets}")
        logger.info(f"   - 总Feedthrough数: {total_feedthrough}")
        logger.info(f"   - 平均Feedthrough数: {total_feedthrough / successful_feedthrough_nets if successful_feedthrough_nets > 0 else 0:.2f}")
    
    # 保存详细结果
    output_file = "wirelength_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_nets": len(nets),
                "successful_nets": successful_nets,
                "failed_nets": failed_nets,
                "missing_pins_count": len(missing_pins),
                "total_hpwl": total_hpwl,
                "average_hpwl": total_hpwl / successful_nets if successful_nets > 0 else 0,
                "total_feedthrough": total_feedthrough if ftpred_session else None,
                "average_feedthrough": (total_feedthrough / (successful_nets - feedthrough_failed_nets)
                                       if ftpred_session and (successful_nets - feedthrough_failed_nets) > 0 else None)
            },
            "net_results": net_results,
            "missing_pins": sorted(list(missing_pins))
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"\n详细结果已保存到: {output_file}")
    
    # 输出找不到的pin列表
    if missing_pins:
        missing_pins_file = "missing_pins.txt"
        with open(missing_pins_file, 'w', encoding='utf-8') as f:
            f.write(f"找不到的Pin列表 (共{len(missing_pins)}个):\n")
            f.write("=" * 60 + "\n")
            for pin_name in sorted(missing_pins):
                f.write(f"{pin_name}\n")
        logger.info(f"找不到的pin列表已保存到: {missing_pins_file}")
    
    # 输出一些net的详细线长和 feedthrough 信息（前20个有实际线长的）
    logger.info("\n前20个net的线长和 feedthrough 详情:")
    logger.info("-" * 100)
    if ftpred_session:
        logger.info(f"{'Net名称':<50} {'Pins':>6} {'HPWL':>12} {'Feedthrough':>12}")
    else:
        logger.info(f"{'Net名称':<50} {'Pins':>6} {'HPWL':>12}")
    logger.info("-" * 100)
    
    count = 0
    for result in net_results:
        if result["hpwl"] > 0 and count < 20:
            feedthrough_str = ""
            if ftpred_session and result.get("feedthrough") is not None:
                feedthrough_str = f"{result['feedthrough']:>12}"
            
            if ftpred_session:
                logger.info(f"{result['net_name'][:49]:<50} {result['found_pins']:>6} {result['hpwl']:>12.2f} {feedthrough_str}")
            else:
                logger.info(f"{result['net_name'][:49]:<50} {result['found_pins']:>6} {result['hpwl']:>12.2f}")
            count += 1
    
    logger.info("=" * 60)
    logger.info("线长计算完成!")
    
    # 5. 修复缺失的pin分配
    # if missing_pins and placedb is not None:
    #     logger.info("\n5. 开始修复缺失的pin分配...")
    #     fixed_output_file = ASSIGNMENT_FILE.replace(".json", "_fixed.json")
    #     fixed_assignment_data = fix_missing_pins(
    #         placedb=placedb,
    #         assignment_data=assignment_data,
    #         missing_pins=missing_pins,
    #         pin_name_to_id=pin_name_to_id,
    #         output_file=fixed_output_file
    #     )
    #     logger.info(f"修复完成，结果已保存到: {fixed_output_file}")
    logger.info("\n5. 开始修复缺失的pin分配...")
    fixed_output_file = ASSIGNMENT_FILE.replace(".json", "_fixed.json")
    fixed_assignment_data = fix_missing_pins(
        placedb=placedb,
        assignment_data=assignment_data,
        missing_pins=missing_pins,
        pin_name_to_id=pin_name_to_id,
        output_file=fixed_output_file
    )
    logger.info(f"修复完成，结果已保存到: {fixed_output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
