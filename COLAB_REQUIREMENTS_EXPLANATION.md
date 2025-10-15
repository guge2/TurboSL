# Colab Requirements 精简说明

## 为什么 requirements-colab.txt 比 requirements.txt 少了很多包？

### ✅ 保留的核心包（56个）
这些是代码实际使用的运行时依赖：

```
pytorch-lightning, omegaconf, nerfacc, opencv-python, imageio,
numpy, scipy, matplotlib, trimesh, PyMCubes, tensorboard, etc.
```

### ❌ 移除的包及原因

#### 1. **Colab 预装包**（会自动可用）
```python
torch, torchvision          # Colab 已预装最新版本
numpy, scipy                # Colab 已有（虽然版本可能不同）
matplotlib, pandas          # Colab 已预装
```

#### 2. **本地文件路径引用**（无法在 Colab 安装）
```python
mkl-random @ file:///home/builder/...    # ❌ 本地路径不存在
mkl-fft @ file:///...                    # ❌ 本地路径不存在
numpy @ file:///croot/...                # ❌ 本地路径不存在
six @ file:///tmp/...                    # ❌ 本地路径不存在
```
**影响**: 无，这些是 Intel MKL 加速库，Colab 有替代方案

#### 3. **开发/调试工具**（训练时不需要）
```python
ipython, jupyter_core, jupyterlab-widgets  # Colab notebook 环境自带
epdb==0.15.1                               # Python 调试器
graphviz==0.20.1                           # 可视化工具（非必需）
```

#### 4. **Web 应用框架**（训练时不需要）
```python
dash, Flask, Werkzeug                # Web dashboard（非必需）
aiohttp, aiosignal                   # 异步 HTTP（训练不用）
```

#### 5. **Blender 相关**（3D 渲染软件，训练不需要）
```python
bpy==3.4.0                           # Blender Python API
```

#### 6. **间接依赖**（会被自动安装）
```python
certifi, charset-normalizer, idna    # requests 的依赖
click, Jinja2                        # Flask 的依赖
protobuf, grpcio                     # tensorboard 的依赖
```
这些包会在安装主包时自动安装

#### 7. **特殊编译包**（在 notebook 中单独安装）
```python
tinycudann @ git+https://...         # 需要编译，在 notebook 中单独处理
```

---

## 📊 包数量对比

| 类别 | 原始 requirements.txt | requirements-colab.txt |
|------|---------------------|----------------------|
| 总数 | 140 个包 | 56 个包 + tinycudann(单独安装) |
| 无法安装 | 5 个（本地路径） | 0 个 |
| 不必要 | ~50 个 | 已移除 |
| **核心依赖** | **~85 个** | **56 个** |

---

## 🔍 验证方法

可以通过以下方式验证是否缺少关键包：

```python
# 1. 检查实际使用的 import
grep -rh "^import \|^from " code/ --include="*.py" | \
  sed 's/from \([^ ]*\).*/\1/' | \
  sed 's/import \([^ ,]*\).*/\1/' | \
  sort -u

# 2. 主要的外部依赖
cv2                  # opencv-python ✅
imageio              # imageio ✅
nerfacc              # nerfacc ✅
numpy                # numpy ✅
omegaconf            # omegaconf ✅
PIL                  # Pillow ✅
pytorch_lightning    # pytorch-lightning ✅
scipy                # scipy ✅
tinycudann           # 单独安装 ✅
torch                # Colab 预装 ✅
```

---

## ⚠️ 注意事项

### 1. tinycudann 必须单独安装
```bash
# 在 notebook 中这样安装：
!pip install git+https://github.com/NVlabs/tiny-cuda-nn/@212104...
```

### 2. 版本兼容性
- `nerfacc==0.3.5`（必须，0.5.x API 不兼容）
- `pytorch-lightning==1.9.5`（推荐，代码基于此版本）
- `open3d>=0.17.0`（0.17.0 不可用，用 0.19.0）

### 3. 首次运行会编译
- `nerfacc` 首次运行需要编译 CUDA 扩展（2-5 分钟）
- 需要 `ninja` 构建工具

---

## 总结

**requirements-colab.txt 是安全的**，因为：

1. ✅ 包含了所有代码实际 import 的包
2. ✅ 移除的都是 Colab 预装或训练不需要的包
3. ✅ 避免了本地文件路径导致的安装失败
4. ✅ 间接依赖会自动安装

如果训练时遇到 `ModuleNotFoundError`，说明确实缺少某个包，再单独添加即可。
