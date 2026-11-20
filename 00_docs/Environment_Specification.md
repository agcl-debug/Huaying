# 环境规范

> 本文件说明复现本研究所需的软件环境。**换一台电脑复现时，请先按本文件配置。**
>
> 生成：2026-10-04（采集自实际运行本研究的机器）

---

## 一、基础环境

| 项 | 值 |
|---|---|
| 操作系统 | `Windows-11-10.0.26200-SP0` |
| 架构 | `AMD64` |
| **Python** | **3.14.7** |
| 解释器路径（本机） | `C:\Python314\python.exe` |
| 磁盘 | 建议预留 **≥ 1 TB**（数据约 624 GB + 中间产出） |
| 内存 | 建议 **≥ 16 GB**（30 m 全域单幅约 1.3 GB，峰值约 8 GB） |
| 位数 | **64 位必须**（栅格尺寸 22313×15265，32 位会溢出） |

## 二、Python 依赖

### 2.1 直接安装（推荐）

```bash
pip install -r requirements.txt
```

### 2.2 版本清单（本机实测）

| 包 | 版本 | 用途 |
|---|---|---|
| `PyMuPDF` | `1.28.2` | PDF 公式提取 |
| `geopandas` | `1.1.4` | 矢量读写 |
| `matplotlib` | `3.11.1` | 成图 |
| `numpy` | `2.5.2` | 数组运算 |
| `openpyxl` | `3.1.5` | Excel 读写 |
| `pandas` | `3.0.5` | 表格与统计汇总 |
| `pillow` | `12.3.0` | 图像处理（WMF/TIFF 转换） |
| `pyproj` | `3.7.2` | 投影转换 |
| `rasterio` | `1.5.1` | 栅格读写与聚合（核心） |
| `requests` | `2.34.2` | 下载（可选） |
| `scikit-learn` | `1.9.0` | Jenks 自然断点（KMeans 近似） |
| `scipy` | `1.18.1` | 统计检验（MK、相关） |
| `shapely` | `2.1.2` | 矢量几何 |
| `tqdm` | `4.70.0` | 进度条（可选） |

### 2.3 已确认**不可用/未安装**的包

本研究的流程**不依赖**它们；若你的环境有，也不影响：

- `osgeo`
- `gdal`
- `arcpy`
- `xlrd`
- `pdfplumber`
- `PyPDF2`
- `tabulate`

> **特别说明**：
> - **不需要 ArcGIS / MATLAB** —— 全部空间运算用 `rasterio` + `numpy` 完成
> - **不需要 GDAL 命令行** —— 用 `rasterio` 的 Python API
> - `arcpy`（ArcGIS 的 Python 包）**仅绘图工程 `.mxd` 需要**，
>   而复现图件只需本流程的 `matplotlib` 脚本，**不需要 ArcGIS**

## 三、字体（成图必需）

本流程的图件使用 **中文标签**，缺字体会显示为方框。

| 平台 | 需要的字体 | 获取方式 |
|---|---|---|
| Windows | **Microsoft YaHei（微软雅黑）** 或 SimHei（黑体） | 系统自带 |
| Linux | `fonts-wqy-zenhei` / `fonts-noto-cjk` | `apt install fonts-wqy-zenhei` |
| macOS | **PingFang SC** 或 `Songti SC` | 系统自带 |

代码中已按 `Microsoft YaHei → SimHei → SimSun` 顺序回退；
若在 Linux 上运行，请把字体清单改成 `Noto Sans CJK SC` 或 `WenQuanYi Zen Hei`。

## 四、投影依赖

| 项 | 值 |
|---|---|
| 投影 | **Krasovsky_1940_Albers** |
| 中央经线 | 105°E |
| 标准纬线 | 25°N / 47°N |
| 椭球 | Krasovsky 1940（a = 6378245.0，1/f = 298.3） |
| 基准网格 | 30 m，22313 列 × 15265 行，transform `(30, 0, -815998.125, 0, -30, 3876143.0)` |
| 工作网格 | 300 m，2232 × 1527，transform `(300, 0, -815998.125, 0, -300, 3876143.0)` |

> **重要**：`rasterio` 读取时需要 **PROJ 数据文件**（安装 `rasterio` 时会自动带上）。
> 若报 `CRS not found`，设环境变量 `PROJ_LIB` 指向 `rasterio/proj_data`。

## 五、环境自检

复制好环境后，**先运行自检脚本**：

```bash
python check_env.py
```

它会逐项检查 Python 版本、依赖包、字体、投影，并给出是否可开始复现的结论。

## 六、耗时预估（本机 12 核 / 32 GB 实测）

| 阶段 | 耗时 |
|---|---|
| 阶段1 数据预处理（30 m → 300 m 聚合） | 约 3 小时 |
| 阶段2 RUSLE 侵蚀与保土 | 约 25 分钟 |
| 阶段3 InVEST 四类服务 | 约 14 分钟 |
| 阶段4 趋势与持续性 | 约 6 分钟 |
| 阶段5 地理探测器 | 约 90 分钟 |
| 阶段6 景观生态风险 | 约 20 分钟 |
| 阶段7 生态网络 | 约 15 分钟 |
| 阶段8 ESCI | 约 4 分钟 |
| 阶段9 SDGs 修复路径 | 约 2 分钟 |
| 阶段10 图件与表格 | 约 5 分钟 |
| **合计** | **约 6 小时**（不含 30 m 全量数据解压） |
