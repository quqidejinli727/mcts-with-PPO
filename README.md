# PinAssignFlow

芯片 EDA 物理设计中的 **Pin Assignment**（引脚分配）完整流程工具。给定 floorplan 和网表连接关系，自动确定每个 pin 在模块轮廓上的精确物理坐标，使得整体互联线长最小化。

## 算法概述

PinAssignFlow 将引脚分配问题分解为三个阶段串行求解：

```
block.json + pingroup.json
            │
            ▼
┌───────────────────────────────┐
│  Stage 1: MCTS Segment分配    │  确定每个 pin 应放置在哪个 segment 上
│  (蒙特卡洛树搜索 + 同构约束)   │
└───────────────────────────────┘
            │ segment_assignments.json
            ▼
┌───────────────────────────────┐
│  Stage 2: NonLinearPlace      │  在 segment 内做连续坐标优化
│  (梯度下降, WL + Density)     │  最小化线长并控制 pin 密度
└───────────────────────────────┘
            │ result.json
            ▼
┌───────────────────────────────┐
│  Stage 3: QP Legalization     │  二次规划合法化
│  (消除重叠, 同构硬约束)        │  彻底消除 pin 间重叠
└───────────────────────────────┘
            │ result_legalized.json
            ▼
      最终 pin 坐标输出
```

### Stage 1: MCTS Segment 分配

- 基于蒙特卡洛树搜索（MCTS），考虑同构模块复用约束
- 使用 Feedthrough 预测器评估布线质量
- 输出每个 pin 所属的 segment 及该 segment 的几何信息

### Stage 2: NonLinearPlace 连续优化

- 将每个 segment 上的 pin 排列建模为一维连续优化问题
- 使用 Adam 优化器，最小化加权平均线长（Weighted-Average Wirelength）
- 引入电势密度模型防止 pin 过于聚集
- 支持复用模块的 pin 分布同步（reuse constraint）

### Stage 3: QP Legalization 合法化

- 基于二次规划（QP）的全局合法化
- 严格消除所有 pin-to-pin 重叠
- 维持同构模块间 pin 分配的一致性（master-follower 模型）
- 使用 HPWL 门控机制控制 successor 对齐约束的引入

## 输入文件

### block.json

描述模块的层级结构、几何形状和位置信息：

```json
{
  "name": "TOP",
  "module_name": "TOP",
  "vertex": [[0, 0], [20000, 0], [20000, 20000], [0, 20000]],
  "direction": 0,
  "color": "#ffffff",
  "children": [
    {
      "name": "TOP.U_A",
      "module_name": "A",
      "vertex": [[1000, 1000], [5000, 1000], [5000, 4000], [1000, 4000]],
      "direction": 0,
      "color": "#ffff00",
      "children": [...]
    }
  ]
}
```

| 字段 | 说明 |
|------|------|
| `name` | 模块实例全名（层级路径），如 `TOP.U_A.U_A1` |
| `module_name` | 模块类型名称，如 `A1` |
| `vertex` | 顶点坐标列表，定义模块轮廓多边形 |
| `direction` | 旋转方向（0-7），对应 R0/R180/R90/R270/MY/MX/MX90/MY90 |
| `color` | 颜色标识（十六进制） |
| `children` | 子模块列表（递归结构） |

**复用模块**：形状相同的模块实例，它们的 pin 相对分布必须一致。通过 `direction` 字段表示旋转/镜像变换。

### pingroup.json

描述 pin 之间的网表连接关系：

```json
[
  [
    {
      "parent_inst": "TOP.U_A.U_A1",
      "parent_module": "A1",
      "pingroup_name": "src_a1_to_b1",
      "scope": [],
      "successors": ["TOP.U_B.U_B1.src_a1_to_b1"],
      "width": 13.88
    },
    {
      "parent_inst": "TOP.U_B.U_B1",
      "parent_module": "B1",
      "pingroup_name": "src_a1_to_b1",
      "scope": [],
      "successors": [],
      "width": 13.88
    }
  ]
]
```

顶层为数组，每个元素代表一张网（net），网内的 pin 互相连接。

| 字段 | 说明 |
|------|------|
| `parent_inst` | pin 所属模块的实例全名 |
| `parent_module` | pin 所属模块的类型名 |
| `pingroup_name` | pin 名称 |
| `scope` | 输入时为空，输出时填入 `[x, y]` 坐标 |
| `successors` | 直接相连的下一跳 pin（格式: `parent_inst.pingroup_name`） |
| `width` | pin 的物理宽度 |

## 输出文件

### result_legalized.json（最终输出）

结构与 `pingroup.json` 完全一致，区别在于每个 pin 的 `scope` 字段被填入了最终的二维坐标：

```json
[
  [
    {
      "parent_inst": "TOP.U_A.U_A1",
      "parent_module": "A1",
      "pingroup_name": "src_a1_to_b1",
      "scope": [2970.35, 1299.6],
      "successors": ["TOP.U_B.U_B1.src_a1_to_b1"],
      "width": 13.88
    }
  ]
]
```

### 中间文件

| 文件 | 产生阶段 | 说明 |
|------|---------|------|
| `segment_assignments.json` | Stage 1 | pin 到 segment 的分配，含 segment 几何和复用分组 |
| `result.json` | Stage 2 | 连续优化后的 pin 坐标（可能存在轻微重叠） |
| `result_legalized.json` | Stage 3 | 最终合法化后的 pin 坐标（无重叠） |

所有中间文件保存在 `--output` 指定的目录中，方便调试和断点恢复。

## 使用方式

### 运行完整流程

```bash
python run.py --case benchmark/case2 --output output/case2
```

### 跳过 MCTS，使用已有的 segment 分配结果

```bash
python run.py --case benchmark/case2 --output output/case2 \
    --skip-mcts --segment-assignments path/to/segment_assignments.json
```

### 只运行合法化

```bash
python run.py --case benchmark/case2 --output output/case2 \
    --skip-mcts --skip-nlplace --result path/to/result.json
```

### 跳过合法化（只做 MCTS + 连续优化）

```bash
python run.py --case benchmark/case2 --output output/case2 --skip-legalization
```

## 参数配置

### Stage 1: MCTS 参数

| 参数 | 命令行 | 默认值 | 说明 |
|------|--------|--------|------|
| MCTS 模拟次数 | `--num-simulations` | 1000 | 每个处理单元的模拟次数，越大搜索越充分 |
| 时间限制 | `--time-limit` | 30.0 | 每个处理单元的时间限制（秒） |

### Stage 2: NonLinearPlace 参数

| 参数 | 命令行 | 默认值 | 说明 |
|------|--------|--------|------|
| 优化迭代次数 | `--nlplace-iterations` | 600 | Adam 优化器迭代轮数 |
| 初始密度权重 | `--nlplace-density-weight` | 100.0 | 密度目标函数的初始权重系数 |
| 参数文件 | `--nlplace-params` | 内置默认 | 自定义 params.json 路径 |
| 保存图表 | `--enable-plot` | False | 保存优化过程的指标图 |

`params.json` 格式：

```json
{
    "gpu": false,
    "random_seed": 42,
    "num_threads": 4,
    "target_density": 0.8,
    "num_bins_x": 64,
    "num_bins_y": 64
}
```

| 字段 | 说明 |
|------|------|
| `gpu` | 是否使用 CUDA GPU 加速 |
| `random_seed` | 随机种子 |
| `num_threads` | PyTorch CPU 线程数 |
| `target_density` | 目标 pin 密度（0-1） |
| `num_bins_x/y` | 密度估计的网格分辨率 |

### Stage 3: Legalization 参数

| 参数 | 命令行 | 默认值 | 说明 |
|------|--------|--------|------|
| 最小间距 | `--keepout` | 0.0 | 相邻 pin 间的最小物理间距 |
| HPWL 阈值 | `--hpwl-thresh` | 500.0 | successor 对齐约束的 HPWL 门控阈值 |
| 最大外层迭代 | `--max-outer-iter` | 30 | QP 外层迭代次数上限 |

环境变量（高级配置）：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `ISO_MASTER_REFERENCE` | 1 | 启用 master-follower 同构策略 |
| `MOVE_ANCHOR_WEIGHT` | 0.00001 | 初始位置锚定权重（越小越自由） |
| `STRICT_PAIR_HPWL_GATE` | 1 | 严格 HPWL 门控 |
| `SOLVER_PROGRESS_LOG` | 1 | 显示求解器进度日志 |

## 项目结构

```
PinAssignFlow/
├── run.py                          # 顶层入口脚本
├── config.py                       # 全局配置和预设
├── README.md
├── common/
│   └── PlaceDB.py                  # 共用数据模型定义
├── stage1_mcts/                    # Stage 1: MCTS segment 分配
│   ├── __init__.py                 # 对外接口: run_mcts()
│   ├── complete_mcts_pipeline.py   # MCTS 主流程
│   ├── PlaceDB.py                  # MCTS 专用 PlaceDB
│   ├── data_model.py              # 数据结构定义
│   ├── final_topology.py          # 拓扑构建
│   ├── multi_net_isomorphic_mcts.py
│   ├── simple_isomorphic_mcts.py
│   ├── simple_reward_function.py
│   ├── segment_usages_direct_interface.py
│   ├── state_adapter.py
│   ├── calculate_wirelength.py
│   └── feedthrough/               # Feedthrough C++ 预测器
│       ├── CMakeLists.txt
│       ├── ftpred.cpp
│       ├── flute.cpp / flute.h
│       ├── ftpred_loader.py
│       └── build/                 # 编译产物 (ftpred.exe)
├── stage2_nlplace/                 # Stage 2: 连续坐标优化
│   ├── __init__.py                 # 对外接口: run_nlplace()
│   ├── NonLinearPlace.py           # 优化主逻辑
│   ├── EdgePlace.py               # 单 edge 的 pin 放置
│   ├── PlaceObj.py                # 目标函数（WL + Density）
│   ├── PlaceDB.py                 # 数据结构
│   ├── Params.py                  # 参数管理
│   └── params.json                # 默认参数文件
├── stage3_legalization/            # Stage 3: QP 合法化
│   ├── __init__.py                 # 对外接口: run_legalization()
│   ├── pin_legalizer.py           # QP 求解器核心
│   ├── run_pin_legalizer.py       # 独立运行脚本
│   └── PlaceDB.py                 # 合法化专用 PlaceDB
├── benchmark/                      # 测试用例
│   ├── case1/
│   │   ├── block.json
│   │   └── pingroup.json
│   └── case2/
│       ├── block.json
│       └── pingroup.json
└── output/                         # 运行结果输出目录
```

## 环境要求

### Python 版本

- Python >= 3.9

### Python 依赖

**核心依赖（所有阶段）：**

```
numpy
```

**Stage 1 (MCTS) 额外依赖：**

```
shapely          # 多边形几何计算
```

Stage 1 还需要编译 Feedthrough 预测器（C++ 组件）：

```bash
cd stage1_mcts/feedthrough
mkdir build && cd build
cmake ..
make          # Linux/macOS
# 或 cmake --build . --config Release    # Windows
```

编译后需确保 `stage1_mcts/feedthrough/build/ftpred.exe`（Windows）或 `ftpred`（Linux）可执行。

**Stage 2 (NonLinearPlace) 额外依赖：**

```
torch            # PyTorch (CPU 或 CUDA)
shapely
matplotlib       # 可视化（可选，仅 --enable-plot 时需要）
Pillow           # 图像处理（可选，仅 --enable-plot 时需要）
```

**Stage 3 (Legalization) 额外依赖：**

```
scipy
cvxpy            # 凸优化建模
gurobipy         # Gurobi 求解器 (需要许可证)
shapely
```

### 安装依赖

```bash
pip install numpy torch shapely matplotlib Pillow scipy cvxpy gurobipy
```

> **注意**：Stage 3 使用 Gurobi 作为 QP 求解器，需要有效的 Gurobi 许可证。学术用户可以申请免费的学术许可证：https://www.gurobi.com/academia/academic-program-and-licenses/

### 硬件建议

- **CPU**：多核处理器（Stage 2 支持多线程）
- **内存**：>= 8 GB（大规模 case 建议 16 GB+）
- **GPU**（可选）：NVIDIA GPU + CUDA（Stage 2 可使用 GPU 加速，在 params.json 中设置 `"gpu": true`）

## Python API 调用

除命令行外，也可以在 Python 代码中直接调用各阶段：

```python
from stage1_mcts import run_mcts
from stage2_nlplace import run_nlplace
from stage3_legalization import run_legalization

# Stage 1
seg_path = run_mcts(
    block_json="benchmark/case2/block.json",
    pingroup_json="benchmark/case2/pingroup.json",
    output_dir="output/case2",
    num_simulations=1000,
    time_limit=30.0,
)

# Stage 2
result_path = run_nlplace(
    segment_assignments_path=seg_path,
    pingroup_path="benchmark/case2/pingroup.json",
    output_dir="output/case2",
    max_iterations=600,
)

# Stage 3
final_path = run_legalization(
    block_json="benchmark/case2/block.json",
    pingroup_json="benchmark/case2/pingroup.json",
    result_json=result_path,
    output_dir="output/case2",
    max_outer_iter=30,
)
```

也可以使用顶层封装函数：

```python
from run import run_full_flow

final_path = run_full_flow(
    case_dir="benchmark/case2",
    output_dir="output/case2",
    num_simulations=1000,
    nlplace_max_iterations=600,
    max_outer_iter=30,
)
```

## 预设配置

`config.py` 中提供了三种预设：

| 预设名 | 场景 | MCTS 模拟数 | NLPlace 迭代 | Legal 迭代 |
|--------|------|-------------|-------------|-----------|
| `fast` | 快速验证 | 200 | 300 | 15 |
| `default` | 日常使用 | 1000 | 600 | 30 |
| `quality` | 高质量 | 3000 | 1200 | 50 |
