# rust-sys1

一个可替换决策模型的 System One 推理库：提供进程内 SDK 与 TypeSafe 兼容的 API 服务。

## 协议与厂商

**System One model（决策模型）**：
一类为软件做快速、结构化决策的 AI 模型；输入 state 与 questions，输出带概率的类型化答案，而不是生成文本。
_Avoid_: Classifier、LLM

**System One API**：
System One 模型的服务协议，官方端点只有 `POST /v1/systemone` 与 `GET /v1/models`。
_Avoid_: TypeSafe API、Jev API

**TypeSafe**：
提供 Jev 的厂商与服务（typesafe.ai），是厂商名而非协议名。
_Avoid_: System One、Jev（作为厂商名时）

**Jev**：
TypeSafe 的旗舰 System One 模型，闭源、以远程服务形式提供。
_Avoid_: System One

**sys1 0.0.3**：
crates.io 上现有的 Rust 版 System One 推理服务，本项目的参考实现之一。
_Avoid_: 本项目、rust-sys1

## 接口形态

**In-process inference API（进程内推理接口）**：
在调用方进程内直接加载模型并执行推理的 Rust 接口。
_Avoid_: SDK、local server

**Inference engine（推理引擎）**：
把模型获取、manifest、后端加载、模型目录与类型化推理组合起来的高层入口；调用方通过它加载、卸载和运行模型。
_Avoid_: Model、registry、server

**Telemetry sink（遥测出口）**：
接收结构化请求、模型、batch、缓存事件的替换点；用于日志、审计或外部追踪系统，不参与推理结果。
_Avoid_: Logger（仅指文本日志时）、metrics（指标是另一类聚合数据）

**Trace context（追踪上下文）**：
一次请求的关联标识，包含 128-bit trace id、server span id 与 request id，用于把日志、事件与响应串联起来。
_Avoid_: Request id（仅指单个请求 id 时）、session

**Client SDK**：
通过 HTTP 调用 System One API 的客户端库，例如 jev-sdk-rs。
_Avoid_: In-process inference API

**Cross-request batching（跨请求批处理）**：
把同一模型、同一时刻到达的多个独立请求合并为一次前向，以提高吞吐；对调用方仍是一问一答。
_Avoid_: Model batch（模型内部的 batch 维度）、request coalescing

## 决策语义

**State**：
决策模型作答所依据的上下文，可以是字符串、对象或数组。
_Avoid_: Prompt、input、context

**Question**：
针对 state 提出的一个带类型的问题，官方类型为 choice、score、noul 与 box。
_Avoid_: Query、prompt

**Choice**：
从一组命名选项中选出一个；答案为选中项、完整概率分布与 confidence。
_Avoid_: Classification、label

**Score**：
按有序档位评分；答案为概率加权期望值，可落在两个档位之间，并带档位图例、概率分布与 confidence。
_Avoid_: Rating、regression

**Noul**：
对一个是/否命题给出 yes 的概率；没有独立的 confidence。
_Avoid_: Boolean、yes/no answer

**Box**：
在图片中定位目标；答案为归一化的上、下、左、右边界与 confidence。
_Avoid_: Bounding box、region

**Asset**：
随请求发送、并通过占位符在 state、instructions 或 criteria 中引用的图片资源；box 问题依赖它。
_Avoid_: Image、attachment

**Answer**：
与 Question 一一对应的结果，类型与问题类型一致；官方类型为 choice、score、noul 与 box。
_Avoid_: Prediction、result

**Confidence**：
答案所依据概率分布的峰值程度，用于判断自动执行还是升级处理；Noul 没有独立 confidence。
_Avoid_: Probability

**Answer confidence**：
答案概率的最大值（`max(p)`），是 abstention gate 与校准使用的量；与归一化熵的
`confidence` 不同，不能混用。
_Avoid_: Confidence（指熵置信度时）

**Abstention**：
按 `min_confidence` 阈值决定答案是 `passed`、`abstained` 还是 `unevaluated` 的策略；
它不修改答案本身，只附带 gate 结果与阈值。
_Avoid_: Confidence filter、thresholding（仅指计算动作时）

**Response**：
一次 System One API 调用的完整结果，由 model、answers、usage 组成。
_Avoid_: Answer

**Act head（动作头）**：
部分 checkpoint 附带的可选分类头，基于 pool 后的表示与答案分布特征预测 `action.act_probability`。
它只出现在进程内 `DecisionOutput.actions` 中，不属于官方 wire Response。
_Avoid_: Action、policy head（与决策语义混淆）

**Model family（模型族）**：
共享同一推理架构与权重格式的一组决策模型，如 Laya、Kev。
_Avoid_: Model type、architecture

**Laya**：
开源的 System One 模型族，包含 english、multilingual、typed-decisions 三个 checkpoint。
_Avoid_: Laya 模型（冗余）

**NLI cross-encoder（NLI 交叉编码器）**：
把 state 与选项 hypothesis 拼成一对送入编码器、用 entailment logit 作为选项分数的模型族。
_Avoid_: 文本分类器（仅指普通单文本分类时）

**GLiClass**：
Knowledgator 的 zero-shot 分类模型族；label 与文本共享一条序列，通过 label span 池化与 MLP 打分。
_Avoid_: NLI（实现与训练目标不同）

**Laya Router**：
在 Laya 变体之间按语言或 typed-decisions 工作流签名选择 checkpoint 的组件。
_Avoid_: Model selector

**Long-context scanning（长上下文扫描）**：
把超过上下文窗口的 state 按 token 切成重叠窗口、逐窗打分并按问题类型聚合的进程内能力。
它与单纯的 local-window attention 不同：后者优化单次前向，前者跨多次前向覆盖整篇文档。
_Avoid_: Chunking（仅指切分动作时）、truncation

## 模型供给

**Model source（模型源）**：
模型权重与配套文件的获取来源，如本地路径、Hugging Face、ModelScope 或自定义模型中心。
_Avoid_: Downloader、backend

**Model catalog（模型目录）**：
服务进程当前可选择的模型集合，包含规范名称、别名与 manifest；可在运行时增删。
_Avoid_: Model list（仅指 `/v1/models` 输出时）、registry（指注册表组件时）

**Model resolver（模型引用解析器）**：
把外部模型引用（本地路径、hub id、manifest URL）解析为可加载 manifest，并让模型加载器认识该引用的组件。
_Avoid_: Model source、downloader

**Model residency（模型常驻状态）**：
模型目录项对应的权重当前是否已加载到内存或显存；与磁盘缓存无关。
_Avoid_: Cache（磁盘缓存）、loaded（口语）

**Cache quota（缓存配额）**：
允许 `<cache_dir>/models` 使用的最大字节数；超出时按 LRU 淘汰非驻留条目。
_Avoid_: Cache limit（与单模型加载上限混淆）、disk full policy

**Reclaimable space（可回收空间）**：
当前可被配额淘汰立即释放的字节数，即所有 `evictable` 缓存条目大小之和；常驻模型与进行中的下载不计入。
_Avoid_: Free space（磁盘剩余空间）、available space

**Inference backend（推理后端）**：
执行模型前向计算的引擎，如 candle、ONNX Runtime。
_Avoid_: Model、engine（作为整体系统时）

**Execution provider（执行提供器）**：
ONNX Runtime 内部实际执行算子图的设备后端，如 CPU、CUDA、TensorRT、DirectML、CoreML；由 Cargo feature 启用。
_Avoid_: Backend（与 candle/ORT 这一层混淆）

**Scoring contract（打分契约）**：
模型 family 与 ONNX Runtime 之间的前向接口：`input_ids`、`attention_mask`、`type_ids`、`markers` 输入，逐 marker 的 float32 logits 输出。
_Avoid_: ONNX API、tokenizer contract

**ONNX export（ONNX 导出）**：
与某个 checkpoint 共用 tokenizer、`rl_agent_config.json` 和答案语义，但以前向计算图（`model.onnx`）替代 safetensors 权重的模型目录。
_Avoid_: ONNX checkpoint（易与 safetensors checkpoint 混淆，但可接受）

**ONNX exporter（ONNX 导出器）**：
把 safetensors 权重和模型配置翻译成 scoring contract 计算图的离线工具；当前实现为 `python/tools` 中的 `sys1-export-onnx`。
_Avoid_: Converter（未说明方向）、compiler
