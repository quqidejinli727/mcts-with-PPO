# Segment Assignments 结果文件说明

本文件为 MCTS 布线算法的最终 Segment 分配结果，记录了每个 Pin 被分配到的具体 Segment 及其位置信息。

## 文件结构

```json
{
  "total_segments": 223,           // 总共涉及的 segment 数量
  "segment_assignments": {         // segment 分配详情，key 为 segment_id
    "segment_id": { ... },
    ...
  }
}
```

## 关键字段说明

### 顶层字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `total_segments` | int | 总共涉及的 segment 数量 |
| `segment_assignments` | dict | segment 分配详情字典，key 为 segment_id |

### Segment 层级（以 segment_id 为 key）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `segment_id` | int | Segment 的母模块级别 ID（Master Segment ID） |
| `segment_info` | object | Segment 的母模块几何信息（原始坐标、长度等） |
| `segment_insts` | dict | 该 segment 在各个 block 中的实例分配情况，key 为 block_id |
| `total_used_capacity` | float | 该 segment 在所有 block 实例中的总使用容量 |
| `total_remaining_capacity` | float | 该 segment 在所有 block 实例中的总剩余容量 |
| `direction` | int | 方向（0: 水平, 1: 垂直） |
| `edge_id` | int | 所属边的 ID |
| `max_capacity` | float | 最大容量限制 |
| `length` | float | segment 长度 |

### Segment Info（母模块几何信息，没有实际意义）

### Segment Inst 层级（以 block_id 为 key）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `segment_inst_id` | int | Segment 实例 ID（全局唯一） |
| `segment_id` | int | 所属母模块 segment ID |
| `segment_info` | object | 母模块 segment 信息（同上） |
| `block_id` | int | 该实例所在的 block ID |
| `block_name` | str | 该实例所在的 block 名称（如 "TOP.U_N_0.U_RA_2"） |
| `assigned_pins` | array | 分配到该 segment 实例的 pin 列表 |
| `used_capacity` | float | 该实例已使用的容量 |
| `remaining_capacity` | float | 该实例剩余容量 |
| `coordinates` | [x1, y1, x2, y2] | 该 segment 实例的实际坐标（变换后） |
| `center_point` | [x, y] | 中点坐标 |
| `direction` | int | 方向 |
| `edge_id` | int | 所属边 ID |
| `max_capacity` | float | 最大容量 |
| `length` | float | 长度 |

### Assigned Pin（分配的 Pin 信息）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `id` | int | Pin ID |
| `name` | str | Pin 全名（格式: "parent_inst.pin_name"） |
| `block_id` | int | 所属 block ID |
| `width` | float | Pin 宽度（占用容量） |
| `net_id` | int | 所属 Net ID |
| `isomorphic_group_id` | int | 同构组 ID（同构 Pin 共享相同的 group） |
| `is_merged_dummy` | bool |不用关注 |
| `merged_from` | array | 不用关注|
