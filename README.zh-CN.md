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

Linux 上 gmsh 需要几个系统库，Python 3.10 / 3.11 上的 OpenSeesPy（Linux 版 3.7）还需要 BLAS 和 LAPACK：
`sudo apt install libglu1-mesa libxcursor1 libxft2 libxinerama1 libblas3 liblapack3`（Debian / Ubuntu）。

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

## 按中国规范验算（`femotion.codes.jtg`，alpha）

钢筋混凝土和预应力混凝土（体内有黏结）公路桥的承载能力极限状态、正常使用极限状态和应力验算：

- JTG D60-2015：车道荷载、车辆荷载、冲击系数、车道数和横纵向折减、基本组合、频遇组合、准永久组合、标准值组合；
- JTG 3362-2018：材料强度、ξb、正截面抗弯（矩形、T 形、I 形、箱形、圆形）、斜截面抗剪、轴心受压和偏心受压（含 η）；
  预应力钢筋、张拉控制应力、各项预应力损失（含附录 C、G、H）、σpe 和 σp0、开裂弯矩和 9.1.13 最小配筋；
  正截面和斜截面抗裂（主应力）、裂缝宽度（受弯、偏压含圆形截面、受拉、B 类构件）、刚度和挠度、预拱度、
  使用阶段应力（含 B 类开裂截面，附录 J）、施工阶段应力。
- 预加力次效应：`femotion.frame.prestress` 由钢筋线形算等效荷载，得总效应、主效应和次效应。

内力来自 `femotion.frame.influence`：分解一次就得到所有截面的影响线，车道荷载、车辆轴列的最不利加载，自重按一致荷载
精确算到单元端。完整流程见 `examples/jtg_rc_girder.py`（两跨连续钢筋混凝土 T 梁）和 `examples/jtg_pc_girder.py`（两跨连续后张法预应力 I 梁：
钢筋线形 → 损失 → 有效预应力 → 次弯矩 → 组合 → 承载能力、抗裂、应力、挠度验算）。

规范里的数值、公式和条文号已经逐条对照交通运输部公开的规范原文核对过，程序用到的全部条文、公式、表格和参数按规范分文件整理在 [docs/codes/](docs/codes/)（[JTG D60—2015](docs/codes/JTG_D60-2015.md)、[JTG 3362—2018](docs/codes/JTG_3362-2018.md)），每项注明原文页码和对应函数，出计算书时可以直接引用。体外预应力、抗扭、局部承压还没有做。

用法示例、模块说明和验证项目见英文 [README](README.md)。

接了哪些工具、哪些等用到再接、哪些不接，见 [docs/integrations.md](docs/integrations.md)。

## 开发

`pytest` 跑 Python 测试，`cd js && npm ci && npm test` 跑 JavaScript 测试。代表算例的结果钉在
`tests/data/regression_v1.npz` 里，依赖升级改动了数值就会报错。`pyproject.toml` 里的最低版本都实测过。
持续集成在最低版本和最新版本上都跑，每周一再用最新发布的版本跑一次。详见英文 README。

## 许可

GNU Affero 通用公共许可证第 3 版或更高版本（AGPL-3.0-or-later，见 [LICENSE](LICENSE)、[NOTICE](NOTICE)）。允许商用；分发 femotion、修改版或基于它的软件，或者让用户通过网络使用它时，必须按同一许可公开相应的源代码。提交 `1be6a41` 及以前的版本以 Apache-2.0 发布，仍可按 Apache-2.0 使用。

可选后端按各自的许可另行安装：gmsh 为 GPL-2.0 或更高版本；
OpenSees 教学、研究、非营利用途免费，商业使用需向加州大学董事会取得许可；MLX 只能在苹果芯片上运行；
sectionproperties 用的网格剖分器 Triangle 只允许非商业使用。
