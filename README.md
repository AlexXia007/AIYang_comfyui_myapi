# AIYang ComfyUI API Nodes

基于ComfyUI的自定义节点，支持多种AI图像生成API的并发调用。采用模块化架构，按并发场景拆分为专门的节点。

## 🚀 功能特性

- **模块化架构**: 按并发场景拆分为专门的节点，避免功能混乱
- **三种并发模式**:
  - 多组图 + 单模型并发 (Banana2Node)
  - 多组图 + 豆包模型并发 (DoubaoBatchNode)
  - 单组图 + 多模型比较 (ModelCompareNode)
- **智能任务调度**: 自动识别有效任务，支持文生图、图生图
- **完善的错误处理**: 支持超时重试，详细的状态反馈
- **ComfyUI原生支持**: 完全兼容ComfyUI的工作流系统

## 📦 安装方法

1. 将整个项目文件夹复制到ComfyUI的`custom_nodes`目录下：
   ```
   ComfyUI/
   ├── custom_nodes/
   │   └── AIYang_comfyui_api/
   │       ├── __init__.py
   │       ├── banana2_node.py
   │       ├── doubao_batch_node.py
   │       ├── model_compare_node.py
   │       ├── config.py
   │       ├── doubao_guide.md
   │       └── README.md
   ```

2. 配置API密钥：
   - 编辑`config.py`文件，设置你的Comfly API密钥
   - 或在节点参数中直接输入API密钥

3. **完全重启ComfyUI**（重要！），节点将自动加载到"AIYang_myapi"分类中

## 🎯 节点使用指南

### 节点总览

| 节点名称 | 功能场景 | 并发模式 | 支持模型 |
|---------|---------|---------|---------|
| **Banana2** | 多组图 + 单Comfly模型 | 10组任务并发 | nano-banana系列 |
| **DoubaoBatch** | 多组图 + 单豆包模型 | 10组任务并发 | doubao-seedream-4 |
| **ModelCompare** | 单组图 + 多模型比较 | 双模型并发 | banana + doubao |

### 节点位置
在ComfyUI中搜索对应节点名称，或在 "AIYang_myapi" 分类下找到节点

### 输入参数

#### 图像输入 (40个插槽)
- `image_1.1` 到 `image_10.4`: 每组最多4张参考图像
- 支持PNG、JPEG等常见图像格式

#### 文本输入 (10个插槽)
- `prompt_1` 到 `prompt_10`: 每组的文本提示词

#### 配置参数 (12个控件)
- `provider`: API供应商 (默认: "comfly")
- `base_url`: API基础地址 (默认: "https://ai.comfly.chat")
- `api_key`: API密钥 (支持环境变量或直接输入)
- `model`: 模型选择 (nano-banana系列、doubao-seedream-4-0-250828)
- `aspect_ratio`: 图像宽高比 (1:1, 16:9等，仅nano-banana系列支持)
- `response_format`: 响应格式 (url/b64_json)
- `timeout`: 单次请求超时时间(秒) (默认: 200)
- `retry_count`: 失败重试次数 (默认: 0)
- `node_enabled`: 节点开关 (默认: True)
- `n`: 生成图片数量 (1-4，仅即梦4支持，默认: 1)
- `size`: 图片尺寸 ("1K"/"2K"，仅即梦4支持，默认: "2K")
- `watermark`: 是否添加水印 (仅即梦4支持，默认: False)
- `stream`: 是否流式响应 (仅即梦4支持，默认: False)

### 输出结果

#### 合并输出 (3个)
- `images`: 所有成功生成的图像 (IMAGE类型)
- `urls`: 各组图像URL链接 (JSON字符串)
- `responses`: 各组执行状态 (JSON数组: 1=成功, 2=失败, 0=未执行)

#### 独立组输出 (30个)
- `group1_image` 到 `group10_image`: 每组生成的图像
- `group1_url` 到 `group10_url`: 每组图像URL
- `group1_response` 到 `group10_response`: 每组状态码

#### 统计输出 (1个)
- `stats`: 执行统计信息 `(有效任务:X, 成功任务:Y)`

### 使用示例

#### Banana2 (多组图 + 单Comfly模型)
**适用场景**: 批量生成多个不同主题的图像

1. 配置模型: 选择 nano-banana 系列模型
2. 输入多组提示词: prompt_1, prompt_2, prompt_3...
3. 可选添加参考图像: image_1.1, image_2.1 等
4. 执行节点，获取10组并发结果

#### DoubaoBatch (多组图 + 单豆包模型)
**适用场景**: 使用豆包模型批量生成高清图像

1. 配置豆包参数: n=2, size="2K", watermark=false
2. 输入多组提示词: prompt_1, prompt_2, prompt_3...
3. 可选添加参考图像 (需先上传获取URL)
4. 执行节点，获取豆包风格的批量结果

#### ModelCompare (单组图 + 多模型比较)
**适用场景**: 比较不同模型的生成效果

1. 输入一张参考图和提示词
2. 配置两个API密钥 (Comfly + 豆包)
3. 选择不同的模型进行比较
4. 执行节点，同时获得两个模型的生成结果

#### 即梦4使用示例
1. 选择模型: `doubao-seedream-4-0-250828`
2. 设置参数:
   - `n`: 3 (生成3张图片)
   - `size`: "2K" (2K分辨率)
   - `watermark`: False (不添加水印)
3. 输入prompt，执行节点
4. 节点将生成指定数量的图片

## ⚙️ 工作原理

### 架构设计
采用**职责单一原则**，按并发场景拆分节点：
- **Banana2Node**: 专门处理Comfly banana系列的批量并发
- **DoubaoBatchNode**: 专门处理豆包模型的批量并发
- **ModelCompareNode**: 专门处理单任务多模型比较

### 任务执行流程
1. **输入解析**: 根据节点类型解析对应的输入参数
2. **并发执行**: 使用asyncio实现不同粒度的并发控制
3. **API适配**: 针对不同供应商使用对应的API格式和参数
4. **结果合并**: 按节点类型返回对应的输出格式

### 异步模式说明

#### Comfly Banana系列 (Banana2Node)
- **触发条件**: 自动检测 nano-banana* 模型
- **执行方式**: async=true参数 + task_id轮询
- **查询间隔**: 每5秒检查一次状态
- **超时控制**: 最长等待5分钟

#### 豆包系列 (DoubaoBatchNode)
- **执行方式**: 异步task_id轮询模式 (与Comfly相同)
- **API调用**: POST /v1/images/generations?async=true
- **状态查询**: GET /v1/images/tasks/{task_id} (每5秒)
- **图片处理**: 支持URL引用，使用config中的测试图片
- **参数丰富**: 支持n/size/watermark等专业参数

#### 模型比较 (ModelCompareNode)
- **并发策略**: 同时调用两个模型的API
- **结果合并**: 并排显示不同模型的生成结果

### 有效任务判断
- 至少包含一张有效图像或一段有效提示词
- 空值判断：图像为None/空tensor，文本为空字符串

### API适配
- **文生图**: POST /v1/images/generations (JSON格式)
- **图生图**: POST /v1/images/edits (Multipart格式)
- **异步模式**: Comfly banana模型自动使用异步task_id查询
- **查询间隔**: 每5秒查询一次任务状态，最多等待5分钟

## 🔧 技术实现

### 依赖要求
- Python 3.8+
- torch
- requests
- Pillow (PIL)
- numpy

### 架构特点
- **模块化设计**: 可扩展支持更多API供应商
- **异步处理**: 不阻塞ComfyUI主线程
- **内存优化**: 合理的图像处理和内存管理
- **错误隔离**: 单任务失败不影响其他任务

## 🚨 注意事项

### 安全提醒
- API密钥请妥善保管，避免明文存储
- 网络请求涉及安全风险，注意防火墙设置
- 大文件上传可能消耗较多带宽

### 性能考虑
- 单次请求超时建议不超过300秒
- 并发数量限制为5，避免服务过载
- 大批量任务建议分批执行

### 故障排除
- **节点未显示**: 检查__init__.py和节点注册
- **API调用失败**: 确认API密钥和网络连接
- **图像格式错误**: 确保输入图像为有效格式
- **内存不足**: 减少并发数量或输入图像分辨率

## 📝 更新日志

### v1.0.0
- ✅ 基础框架搭建
- ✅ Comfly API集成 (nano-banana模型)
- ✅ 文生图、图生图、多图生图支持
- ✅ 并发异步处理
- ✅ 重试机制
- ✅ 完善的输入输出系统

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 项目文件结构
```
AIYang_comfyui_api/
├── __init__.py             # 节点注册和集成
├── banana2_node.py         # Comfly banana批量并发节点
├── doubao_batch_node.py    # 豆包批量并发节点
├── model_compare_node.py   # 模型比较节点
├── config.py               # API配置和测试图片
├── doubao_guide.md         # 豆包专属使用指南
├── README.md               # 详细使用文档
├── requirements.txt        # 依赖说明
└── 需求及文档/             # 原始需求和API文档
```

### 扩展开发
- 添加新API供应商: 创建新的专用节点类
- 添加新模型: 在对应节点的模型列表中注册
- 优化性能: 改进并发控制和内存管理

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。
