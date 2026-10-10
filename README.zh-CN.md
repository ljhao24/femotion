# femotion

**为动画而做的有限元与仿真结果。** 用成熟的开源求解器计算，写成紧凑的结果文件，再逐帧画出来：变形、应力、
力流线、振型、拓扑优化的每一轮、桥面绕流。

[English](README.md)

femotion 为工程动画而做：画面上每一处变形、每一种应力颜色、每一个旋涡都要**真算出来**，不能随手画。
它围绕三个原则组织：

1. **重活交给成熟求解器。** OpenSees、scikit-fem、gmsh、pyamg、SciPy 负责计算；femotion 提供统一的建模格式、
   衔接代码，以及一个和 OpenSees 交叉验证过的线性杆系快速求解器。
2. **计算和画面之间只有一种结果文件。** 求解器写 `fem-result/1`（JSON 描述 + 小端二进制，规范见
   [docs/result-format.md](docs/result-format.md)），JavaScript 包读它，两边互不依赖。
3. **保留动画需要的东西。** 拓扑优化的每一轮、按拼装次序的刚度矩阵图案、重采样到帧率的时程、折线形式的力流线。

> 状态：alpha（0.1.0.dev0），接口可能还会变。

## 安装

```bash
pip install femotion                 # 只依赖 numpy + scipy：线性杆系、拓扑优化、结果文件
pip install "femotion[continuum]"    # + scikit-fem、pyamg：平面和实体弹性
pip install "femotion[mesh]"         # + gmsh：非结构网格、四边形、拉伸和结构化六面体
pip install "femotion[tetwild]"      # + fTetWild：从不干净的表面模型（Blender、下载的模型、扫描）生成四面体
pip install "femotion[opensees]"     # + OpenSeesPy：杆系非线性
pip install "femotion[cfd]"          # + MLX：二维格子玻尔兹曼绕流（仅苹果芯片）
pip install "femotion[io]"           # + meshio：读写约 30 种网格格式，结果导出给 ParaView
pip install "femotion[sections]"     # + sectionproperties：任意形状截面的参数和纤维截面
pip install "femotion[all]"
```

## 和其他软件交换

- **`femotion.io`**（meshio）：读 gmsh、Abaqus、Ansys、Nastran、VTK 等格式的网格，gmsh 的物理分组直接变成分组；
  `export_result` 把结果文件转成 VTU，ParaView 直接打开，逐帧结果转成 `.pvd` 序列，按播放即可。
- **`femotion.frame.sections`**（sectionproperties）：任意形状截面（箱梁、工字钢、空心墩、组合截面）算出面积、惯性矩、
  扭转常数和每延米质量，交给杆系求解器；同一形状也能直接生成 OpenSees 纤维截面，做非线性分析。

## JavaScript 包

| 包 | 内容 | 依赖 |
|---|---|---|
| [`@femotion/core`](js/core) | 解析 `fem-result/1`、取场和折线；展示几何（三角形、棱、边界环、三维外表面与可见棱、等值线、出场次序）；二维 / 正交 / 透视投影；配色；可以逐帧现算的二维刚架 | 无 |
| [`@femotion/remotion`](js/remotion) | Remotion 组件：`useFemResult`、`FieldCanvas`（逐像素上色，带逐块显现、分档、深度缓冲、网格线）、`MeshLines`、`Polylines`、`ColorBar`、`MatrixGrid` | `@femotion/core`；对等依赖 `react`、`remotion` |

Python 写出、JavaScript 读回，两边用同一份标准样例（`tests/data/golden_v1.*`）测试。

## 杆系用哪个求解器

| 问题 | 用 | 理由 |
|---|---|---|
| 线性、要反复算（车队逐帧、影响线、索力优化、振型） | `crosscheck` 通过后用 `frame.linear` | 分解一次，每个工况约 0.2 ms，比 OpenSees 快约 20 倍 |
| 非线性：屈服、塑性铰、屈曲、大变形、拉索松弛、断开后倒塌、地震损伤 | `frame.opensees` | 成熟，验证充分 |
| 线性但只算一两次 | 都行，拿不准就用 OpenSees | 速度无所谓 |

用法示例、模块说明和验证项目见英文 [README](README.md)。

接了哪些工具、哪些等用到再接、哪些不接，见 [docs/integrations.md](docs/integrations.md)。

## 开发

`pytest` 跑 Python 测试，`cd js && npm ci && npm test` 跑 JavaScript 测试。代表算例的结果钉在
`tests/data/regression_v1.npz` 里，依赖升级改动了数值就会报错。`pyproject.toml` 里的最低版本都实测过。
持续集成在最低版本和最新版本上都跑，每周一再用最新发布的版本跑一次。详见英文 README。

## 许可

Apache-2.0（见 [LICENSE](LICENSE)、[NOTICE](NOTICE)）。可选后端按各自的许可另行安装：gmsh 为 GPL-2.0 或更高版本；
OpenSees 教学、研究、非营利用途免费，商业使用需向加州大学董事会取得许可；MLX 只能在苹果芯片上运行；
sectionproperties 用的网格剖分器 Triangle 只允许非商业使用。
