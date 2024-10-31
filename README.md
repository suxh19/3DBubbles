# 3DBubbles
遵循MIT协议的公开的气液两相流数据集（3DBubbles），提供高精度气泡流3D、2D数据及统计信息。数据来自高空间分辨率X射线CT测量设备扫描静态气液流场模体数据。

A publicly available gas-liquid two-phase flow dataset (3DBubbles) following the MIT protocol provides high-precision bubble flow 3D and 2D data and statistical information. The data come from high spatial resolution X-ray CT measurement equipment scanning static gas-liquid flow field model data.

## 文件结构：  
3DBubbles  
| ———— parameters.csv 3D气泡和2D投影图像的结构参数及几何信息  
| ———— reconstruction_characterization.csv 球谐重建阶数1-20的重建指标与结构参数  
| ———— mesh 10823个stl格式的气泡mesh文件  
----| ———— 00001.stl  
----| ———— 00002.stl  
----| ———— ……  
----\\ ———— 10823.stl  
| ———— projection 10823×26个投影图像  
----| ———— 00001  
--------| ———— Sphere_0.00_0.00_1.00_scale=39.png  
--------| ———— Sphere_0.00_0.00_-1.00_scale=39.png  
--------| ———— ……  
--------\\ ———— Sphere_-1.00_0.00_0.00_scale=39.png  
----| ———— 00002  
----| ———— ……  
----\\ ———— 10823  
\\ ———— SH_coefficient 10823×26个投影图像  
----| ———— 00001  
----| ———— Sphere_0.00_0.00_1.00_scale=39.png  
--------| ———— N=1.npy  
--------| ———— N=2.npy  
--------| ———— ……  
--------\\ ———— N=20.npy  
----| ———— 00002  
----| ———— ……  
----\\ ———— 10823  

## Coming soon

### 气泡流
1. 三维静态结构（气含率0.5%、1%、2%，mesh信息，STL格式）
2. 二维图像（4个角度的虚拟投影）
3. 气泡流统计信息（气泡尺寸分布BSD，气泡角度分布BRD，气泡位置分布BPD等）
4. 全部气泡的边界框（bounding box）和类别（class）
5. 全部气泡的掩码（mask）信息


## 联系
作者：Baodi Yu，e-mail: yubaodi20@ipe.ac.cn
