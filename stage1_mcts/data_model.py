from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Set
import math
import logging

# 配置日志
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segment:
    """边的一段，具有独立容量和几何信息 - 存储母模块形状（rotation=0）"""
    id: int                    # 全局唯一的segment ID
    edge_id: int              # 所属边的ID
    block_id: int             # 所属模块的ID
    x1: float                 # 母模块起点x坐标（rotation=0）
    y1: float                 # 母模块起点y坐标（rotation=0）
    x2: float                 # 母模块终点x坐标（rotation=0）
    y2: float                 # 母模块终点y坐标（rotation=0）
    length: float             # 段长度
    max_capacity: float         # 最大容量（可容纳的pin长度，默认与length相同）
    direction: int            # 方向（0: 水平, 1: 垂直）
    
    @property
    def cx(self) -> float:
        """中心点x坐标"""
        return (self.x1 + self.x2) / 2.0
    
    @property
    def cy(self) -> float:
        """中心点y坐标"""
        return (self.y1 + self.y2) / 2.0
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        """中心点坐标"""
        return (self.cx, self.cy)
    
    def __repr__(self) -> str:
        return f"Segment(id={self.id}, block={self.block_id}, cap={self.max_capacity})"


@dataclass(frozen=True)
class SegmentInst:
    """实例级别的segment，记录真实的两端点坐标和所属信息"""
    id: int                    # 实例唯一的segment ID
    segment_id: int           # 母模块segment ID
    block_id: int             # 所属block实例ID
    x1: float                 # 真实起点x坐标
    y1: float                 # 真实起点y坐标
    x2: float                 # 真实终点x坐标
    y2: float                 # 真实终点y坐标
    length: float             # 实例长度
    direction: int            # 方向（0: 水平, 1: 垂直）
    
    @property
    def cx(self) -> float:
        """中心点x坐标"""
        return (self.x1 + self.x2) / 2.0
    
    @property
    def cy(self) -> float:
        """中心点y坐标"""
        return (self.y1 + self.y2) / 2.0
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        """中心点坐标"""
        return (self.cx, self.cy)
    
    def __repr__(self) -> str:
        return f"SegmentInst(id={self.id}, seg={self.segment_id}, block={self.block_id}, pos=({self.x1:.1f},{self.y1:.1f})-({self.x2:.1f},{self.y2:.1f}))"


# 为保证feature_extractor提取形状合理，segements应为顺序排列
@dataclass(frozen=True)
class MasterBlock:
    """母模块定义（存储不变的几何信息）"""
    id: int                      # 母模块ID
    name: str                    # 模块名称
    segments_ids: Tuple[int, ...] # 包含的segment ID列表
    position: Tuple[float, float] # 该类母模块第一个实例的第一个点的坐标
    
    def __repr__(self) -> str:
        return f"MasterBlock(id={self.id}, name={self.name}, segments={len(self.segments_ids)}, position={self.position})"


@dataclass(frozen=True)
class Block:
    """模块实例定义（存储可变的实例信息）"""
    id: int                      # 实例ID
    name: str                    # 实例名称
    master_id: int              # 母模块ID
    parent_id: Optional[int]    # 上层模块ID
    rotation: int               # 旋转信息 参照 rotate_information.txt
    is_mirrored: int         # 是否镜像 0: 否, 1: 是
    vertex: List[List[float]] = None  # 模块顶点坐标列表 [[x1,y1],[x2,y2],...]
    childeren_ids: Optional[List[int]] = None # 下层模块ID
    position: Tuple[float, float] = None #  (x, y)
    
    def __repr__(self) -> str:
        return f"Block(id={self.id}, name={self.name}, master={self.master_id}, pos={self.position}, rot={self.rotation})"


@dataclass(frozen=True)
class IsomorphicPinGroup:
    """同构Pin组定义"""
    group_id: int              # 同构组唯一ID
    pin_ids: Tuple[int, ...]   # 组内所有Pin ID（有序）
    master_block_id: int       # 母模块ID（组内所有Pin所属模块的母模块必须相同）
    
    @property
    def size(self) -> int:
        """组大小"""
        return len(self.pin_ids)
    
    def __repr__(self) -> str:
        return f"IsomorphicPinGroup(id={self.group_id}, pins={len(self.pin_ids)}, master={self.master_block_id})"


@dataclass(frozen=True)
class Pin:
    """引脚定义（增强版）"""
    id: int                    # 全局唯一的pin ID
    name: str                  # 引脚名称
    block_id: int              # 所属模块ID
    width: float               # 引脚宽度
    net_id: int                # 所属网表ID
    isomorphic_group_id: Optional[int] = None  # 同构组ID（新增）
    
    def __repr__(self) -> str:
        return f"Pin(id={self.id}, name={self.name}, block={self.block_id}, group={self.isomorphic_group_id}， capacity={self.width})"


@dataclass(frozen=True)
class Net:
    """网表定义"""
    id: int                   # 全局唯一的net ID
    name: str                 # 网表名称
    pin_ids: Tuple[int, ...]  # 包含的pin ID列表（有序）
    
    @property
    def degree(self) -> int:
        """网表的度数（pin数量）"""
        return len(self.pin_ids)
    
    def __repr__(self) -> str:
        return f"Net(id={self.id}, name={self.name}, degree={self.degree})"


@dataclass(frozen=True)
class SegmentIndexMapping:
    """Segment在母模块中的索引映射（辅助类）"""
    segment_id: int
    master_block_id: int
    index: int                 # 在母模块segments_ids列表中的索引
    
    def __repr__(self) -> str:
        return f"SegmentIndexMapping(seg={self.segment_id}, master={self.master_block_id}, idx={self.index})"


class FloorPlanRO:
    """
    只读数据层，存储所有静态信息
    搜索过程中绝不修改，通过索引引用
    """
    __slots__ = ('_segments', '_segment_insts', '_master_blocks', '_blocks', '_pins', '_nets',
                 '_master_block_name2id', '_block_name2id', '_pin_name2id', '_net_name2id',
                 '_master_block_segments', '_block_segments', '_net_pins',
                 '_master_to_instances', '_pin_to_net',
                 '_isomorphic_groups', '_pin_to_group', '_segment_index_map',
                 '_segment_inst_map', '_chip_length')
    
    def __init__(self,
                 segments: List[Segment],
                 segment_insts: List[SegmentInst],
                 master_blocks: List[MasterBlock],
                 blocks: List[Block],
                 pins: List[Pin],
                 nets: List[Net],
                 isomorphic_pin_groups: List[IsomorphicPinGroup],
                 chip_length:float) -> None:
        # 存储只读数据
        self._segments = tuple(segments)
        self._segment_insts = tuple(segment_insts)
        self._master_blocks = tuple(master_blocks)
        self._blocks = tuple(blocks)
        self._pins = tuple(pins)
        self._nets = tuple(nets)
        self._chip_length = chip_length
        
        # 构建名称到ID的映射
        self._master_block_name2id = {mb.name: mb.id for mb in master_blocks}
        self._block_name2id = {block.name: block.id for block in blocks}
        self._pin_name2id = {pin.name: pin.id for pin in pins}
        self._net_name2id = {net.name: net.id for net in nets}
        
        # 构建母模块到segment的映射，得到的是 segment ID 列表
        self._master_block_segments: Dict[int, Tuple[int, ...]] = {}
        for mb in master_blocks:
            self._master_block_segments[mb.id] = mb.segments_ids
        
        # 构建实例到segment的映射（通过母模块）
        self._block_segments: Dict[int, Tuple[int, ...]] = {}
        for block in blocks:
            master_segments = self._master_block_segments.get(block.master_id, ())
            self._block_segments[block.id] = master_segments
        
        # 构建网表到pin的映射
        self._net_pins: Dict[int, Tuple[int, ...]] = {}
        for net in nets:
            self._net_pins[net.id] = net.pin_ids

        # 构建pin到网表的映射（反向索引）
        self._pin_to_net: Dict[int, int] = {}
        for pin in pins:
            self._pin_to_net[pin.id] = pin.net_id

        # 构建母模块到实例的映射
        self._master_to_instances: Dict[int, List[int]] = {}
        for block in blocks:
            self._master_to_instances.setdefault(block.master_id, []).append(block.id)
        
        # 构建pin到网表的映射（反向索引）
        self._pin_to_net: Dict[int, int] = {}
        for pin in pins:
            self._pin_to_net[pin.id] = pin.net_id
        
        # 同构组管理（新增）
        self._isomorphic_groups: Dict[int, IsomorphicPinGroup] = {}
        self._pin_to_group: Dict[int, int] = {}  # pin_id -> group_id
        
        # 填充同构组数据
        for group in isomorphic_pin_groups:
            self._isomorphic_groups[group.group_id] = group
            for pin_id in group.pin_ids:
                self._pin_to_group[pin_id] = group.group_id
        
        # Segment索引映射（预计算）
        self._segment_index_map: Dict[Tuple[int, int], int] = {}  # (seg_id, master_id) -> index
        
        # SegmentInst映射（预计算）: (segment_id, block_id) -> SegmentInst
        self._segment_inst_map: Dict[Tuple[int, int], SegmentInst] = {}
        
        # 构建Segment索引映射
        self._build_segment_index_map()
        
        # 构建SegmentInst映射
        self._build_segment_inst_map()
    
    def _build_segment_index_map(self) -> None:
        """构建Segment在母模块中的索引映射"""
        for master_block in self._master_blocks:
            master_id = master_block.id
            for idx, seg_id in enumerate(master_block.segments_ids):
                self._segment_index_map[(seg_id, master_id)] = idx
    
    def _build_segment_inst_map(self) -> None:
        """构建SegmentInst映射: (segment_id, block_id) -> SegmentInst"""
        for seg_inst in self._segment_insts:
            key = (seg_inst.segment_id, seg_inst.block_id)
            self._segment_inst_map[key] = seg_inst
    
    # 只读访问方法
    def get_segment(self, seg_id: int) -> Segment:
        """通过ID获取segment"""
        return self._segments[seg_id]
    
    def get_segment_inst(self, segment_id: int, block_id: int) -> Optional[SegmentInst]:
        """通过segment_id和block_id获取对应的SegmentInst"""
        key = (segment_id, block_id)
        return self._segment_inst_map.get(key)
    
    def get_segment_inst_for_pin(self, pin_id: int, segment_id: int) -> Optional[SegmentInst]:
        """通过pin_id和segment_id获取对应的SegmentInst
        
        Args:
            pin_id: 引脚ID
            segment_id: 分配的母模块segment ID
            
        Returns:
            对应的SegmentInst，如果找不到则返回None
        """
        # 获取pin所属的block
        pin = self._pins[pin_id]
        if pin is None:
            return None
        
        block_id = pin.block_id
        return self.get_segment_inst(segment_id, block_id)
    
    def get_master_block(self, master_id: int) -> MasterBlock:
        """通过ID获取母模块"""
        return self._master_blocks[master_id]
    
    def get_block(self, block_id: int) -> Block:
        """通过ID获取block实例"""
        return self._blocks[block_id]
    
    def get_pin(self, pin_id: int) -> Pin:
        """通过ID获取pin"""
        return self._pins[pin_id]
    
    def get_net(self, net_id: int) -> Net:
        """通过ID获取net"""
        return self._nets[net_id]
    
    def get_master_block_by_name(self, name: str) -> MasterBlock:
        """通过名称获取母模块"""
        master_id = self._master_block_name2id[name]
        return self._master_blocks[master_id]
    
    def get_block_by_name(self, name: str) -> Block:
        """通过名称获取block实例"""
        block_id = self._block_name2id[name]
        return self._blocks[block_id]
    
    def get_pin_by_name(self, name: str) -> Pin:
        """通过名称获取pin"""
        pin_id = self._pin_name2id[name]
        return self._pins[pin_id]
    
    def get_net_by_name(self, name: str) -> Net:
        """通过名称获取net"""
        net_id = self._net_name2id[name]
        return self._nets[net_id]
    
    def get_master_block_segments(self, master_id: int) -> Tuple[int, ...]:
        """获取母模块包含的所有segment ID"""
        return self._master_block_segments[master_id]
    
    def get_block_segments(self, block_id: int) -> Tuple[int, ...]:
        """获取block实例包含的所有segment ID（通过母模块）"""
        return self._block_segments[block_id]
    
    def get_net_pins(self, net_id: int) -> Tuple[int, ...]:
        """获取net包含的所有pin ID"""
        return self._net_pins[net_id]
    
    def get_pin_block(self, pin_id: int) -> Block:
        """获取pin所属的block实例"""
        pin = self._pins[pin_id]
        return self._blocks[pin.block_id]
    
    def get_pin_net(self, pin_id: int) -> int:
        """获取pin所属的net ID"""
        return self._pin_to_net[pin_id]
    
    def get_pin_segments(self, pin_id: int) -> Tuple[int, ...]:
        """获取pin所属block实例的所有segment ID"""
        pin = self._pins[pin_id]
        return self._block_segments[pin.block_id]
    
    def get_block_instances(self, master_id: int) -> List[Block]:
        """获取母模块的所有实例"""
        instance_ids = self._master_to_instances.get(master_id, [])
        return [self._blocks[iid] for iid in instance_ids]
    
    # 同构组管理接口（新增）
    def get_isomorphic_group(self, group_id: int) -> Optional[IsomorphicPinGroup]:
        """获取同构组"""
        return self._isomorphic_groups.get(group_id)

    def get_pin_isomorphic_group(self, pin_id: int) -> Optional[IsomorphicPinGroup]:
        """获取Pin所属的同构组"""
        group_id = self._pin_to_group.get(pin_id)
        if group_id is not None:
            return self._isomorphic_groups.get(group_id)
        return None

    def get_isomorphic_pins(self, pin_id: int) -> List[int]:
        """获取同构Pin列表（包括自己）"""
        group = self.get_pin_isomorphic_group(pin_id)
        if group is None:
            return [pin_id]
        return list(group.pin_ids)

    def get_isomorphic_group_pins(self, group_id: int) -> List[int]:
        """获取同构组内的所有Pin"""
        group = self._isomorphic_groups.get(group_id)
        if group is None:
            return []
        return list(group.pin_ids)
    
    def get_segment_index_in_master(self, seg_id: int, master_id: int) -> Optional[int]:
        """
        获取Segment在母模块中的索引位置
        返回None表示该Segment不属于此母模块
        """
        return self._segment_index_map.get((seg_id, master_id))

    def get_segment_by_index(self, master_id: int, index: int) -> Optional[int]:
        """
        根据母模块和索引获取Segment ID
        """
        master_block = self._master_blocks[master_id]
        if 0 <= index < len(master_block.segments_ids):
            return master_block.segments_ids[index]
        return None
    
    def get_corresponding_segment(self, pin_id: int, target_pin_id: int,
                                 seg_id: int) -> Optional[int]:
        """
        核心功能：获取对应的Segment
        
        参数:
            pin_id: 已分配Pin的ID（参考Pin）
            target_pin_id: 目标Pin的ID（要查询的Pin）
            seg_id: 已分配Pin的Segment ID
        
        返回:
            目标Pin应该分配的对应Segment ID，或None（无对应关系）
        """
        # 1. 验证两个Pin是否同构
        group = self.get_pin_isomorphic_group(pin_id)
        if group is None or target_pin_id not in group.pin_ids:
            return None
        
        # 2. 获取已分配Segment在母模块中的索引
        pin = self._pins[pin_id]
        pin_block = self._blocks[pin.block_id]
        pin_master_id = pin_block.master_id
        
        seg_index = self.get_segment_index_in_master(seg_id, pin_master_id)
        if seg_index is None:
            return None  # Segment不属于该母模块
        
        # 3. 获取目标Pin的母模块
        target_pin = self._pins[target_pin_id]
        target_block = self._blocks[target_pin.block_id]
        target_master_id = target_block.master_id
        
        # 4. 验证母模块是否相同（同构Pin必须属于相同母模块）
        if pin_master_id != target_master_id:
            return None
        
        # 5. 返回对应索引的Segment
        return self.get_segment_by_index(target_master_id, seg_index)
    
    # 属性访问
    @property
    def num_segments(self) -> int:
        return len(self._segments)
    
    @property
    def num_master_blocks(self) -> int:
        return len(self._master_blocks)
    
    @property
    def num_blocks(self) -> int:
        return len(self._blocks)
    
    @property
    def num_pins(self) -> int:
        return len(self._pins)
    
    @property
    def num_nets(self) -> int:
        return len(self._nets)
    
    @property
    def all_segments(self) -> Tuple[Segment, ...]:
        return self._segments
    
    @property
    def all_master_blocks(self) -> Tuple[MasterBlock, ...]:
        return self._master_blocks
    
    @property
    def all_blocks(self) -> Tuple[Block, ...]:
        return self._blocks
    
    @property
    def all_pins(self) -> Tuple[Pin, ...]:
        return self._pins
    
    @property
    def all_nets(self) -> Tuple[Net, ...]:
        return self._nets
    
    @property
    def chip_length(self) -> float:
        return self._chip_length
    
    def __repr__(self) -> str:
        return (f"FloorPlanRO(segments={self.num_segments}, master_blocks={self.num_master_blocks}, "
                f"blocks={self.num_blocks}, pins={self.num_pins}, nets={self.num_nets})")
    

# 辅助函数：从原始数据构建FloorPlanRO
# ==================== 搜索状态层 ====================

@dataclass
class IsomorphicGroupState:
    """同构组分配状态（用于状态隔离）"""
    is_assigned: bool = False
    segment_assigned: Optional[int] = None


@dataclass
class SegmentUsage:
    """Segment使用状态，可变"""
    seg_id: int                      # segment ID
    used_capacity: float = 0.0       # 已使用容量（引脚宽度的累加）
    assigned_pins: List[int] = None  # 已分配的pin ID列表
    
    def __post_init__(self):
        if self.assigned_pins is None:
            self.assigned_pins = []
    
    def can_assign(self, pin_id: int, floorplan: 'FloorPlanRO') -> bool:
        """
        检查是否可以分配引脚
        判断条件：已用容量 + 引脚宽度 <= segment最大容量
        """
        segment = floorplan.get_segment(self.seg_id)
        pin_width = floorplan.get_pin(pin_id).width
        
        # 检查容量限制
        if self.used_capacity + pin_width > segment.max_capacity - 1e-6:  # 增加微小误差限制
            return False
        
        return True
    
    def assign_pin(self, pin_id: int, floorplan: 'FloorPlanRO') -> bool:
        """分配引脚 - 增强版，检查重复分配和同构组约束"""
        # if not self.can_assign(pin_id, floorplan):
        #     logger.debug(f"无法分配pin{pin_id}到segment{self.seg_id}，容量不足")
        #     return False
        
        # 1. 检查是否已存在该pin的分配（避免重复添加）
        if pin_id in self.assigned_pins:
            logger.debug(f"pin{pin_id}已存在于segment{self.seg_id}的分配列表中，无需重复添加")
            return True
        
        # 2. 获取pin信息
        pin = floorplan.get_pin(pin_id)
        if not pin:
            logger.warning(f"无法找到pin{pin_id}")
            return False
        
        pin_width = pin.width
        
        # 3. 检查同构组约束 - 如果同构组中已有pin分配到该segment，无需增加容量
        if hasattr(pin, 'isomorphic_group_id') and pin.isomorphic_group_id is not None:
            # 检查同构组中是否已有pin分配到该segment
            for assigned_pin_id in self.assigned_pins:
                assigned_pin = floorplan.get_pin(assigned_pin_id)
                if (assigned_pin and
                    hasattr(assigned_pin, 'isomorphic_group_id') and
                    assigned_pin.isomorphic_group_id == pin.isomorphic_group_id):
                    # 同构组中已有pin分配到该segment，无需增加容量占用
                    logger.debug(f"同构组{pin.isomorphic_group_id}中已有pin分配到segment{self.seg_id}，无需增加容量")
                    self.assigned_pins.append(pin_id)
                    return True
        
        # 4. 正常分配 - 增加容量并添加到列表
        self.used_capacity += pin_width
        self.assigned_pins.append(pin_id)
        
        logger.debug(f"分配pin{pin_id}到segment{self.seg_id}，容量增加{pin_width}")
        return True
    
    
    def get_remaining_capacity(self, floorplan: 'FloorPlanRO') -> float:
        """获取剩余容量"""
        segment = floorplan.get_segment(self.seg_id)
        return segment.max_capacity - self.used_capacity
    
    def copy(self) -> SegmentUsage:
        """创建副本（用于状态转移）"""
        return SegmentUsage(
            seg_id=self.seg_id,
            used_capacity=self.used_capacity,
            assigned_pins=self.assigned_pins.copy()
        )


@dataclass
class PinAssignment:
    """Pin分配状态"""
    pin_id: int          # 引脚ID
    seg_id: int          # 分配的segment ID
    net_id: int          # 所属网表ID
    
    def __repr__(self) -> str:
        return f"PinAssignment(pin={self.pin_id}, seg={self.seg_id}, net={self.net_id})"


# 当前版本 PinAssignment 只存储当前 net 的分配状态，PPO 阶段可能需要修改
@dataclass
# TODO:针对同构Pin和segement的改进
class NetAwareSearchState:
    """
    网表感知的搜索状态
    - 网表间串行，网表内根节点存储完整字典
    - 回溯深度限制在单个网表内
    """
    # TODO: 当前方案考虑所有复用Pin在第一次出现时分配，后续可改进为动态分配，不调整则 current_pin_idx 需要考虑间隔分配
    current_net_id: int = 0                     # 当前网表ID
    current_pin_idx: int = 0                 # 当前网表内的pin索引
    # 根节点存储完整字典，子节点存储增量
    segment_usages: Optional[Dict[int, SegmentUsage]] = None
    parent: Optional[NetAwareSearchState] = None
    floorplan: Optional[FloorPlanRO] = None
    net_assignments: Optional[Dict[int, List[PinAssignment]]] = None  # 根节点存储完整net_assignments字典，子节点不存储
    
    def is_root_of_net(self) -> bool:
        """判断是否是当前网表的根节点"""
        return self.parent is None or self.parent.current_net_id != self.current_net_id
    
    def is_terminal(self) -> bool:
        """检查是否完成当前网表"""
        if self.floorplan is None:
            raise ValueError("floorplan未设置")
        
        net = self.floorplan.get_net(self.current_net_id)
        return self.current_pin_idx >= len(net.pin_ids)

