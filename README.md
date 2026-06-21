# Quant Factor Agents Team

> 基于多Agent LLM架构的自动化量化因子挖掘系统，通过智能协作实现"想法生成 - 表达式构建 - 回测验证 - 改进优化"的完整闭环。

## 项目背景

在量化投资领域，Alpha因子的挖掘是一项高度依赖专业知识和经验的工作。传统方法需要研究员手动设计因子、编写代码、回测验证、反复迭代优化，整个过程耗时且效率低下。

本项目通过构建**多Agent协作系统**，将LLM的创造力与量化研究的专业流程相结合，实现了：
- **自动化创意生成**：基于数据特征和操作符库，LLM自动生成因子创意
- **智能表达式构建**：模板展开机制，一次调用生成上百条表达式
- **高效回测验证**：并行回测，自动筛选优质因子
- **迭代优化改进**：精英池、循环检测、多轮反思，持续提升因子质量

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Phase 1: Data Classification (一次性执行)         │    │
│  │                                                                     │    │
│  │    CSV文件 ──> DataClassifierAgent ──> 分类结果 (field_mapping)      │    │
│  │                    │                                                │    │
│  │                    └──> 自审机制 (迭代评审，最多3轮)                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Main Loop (主循环 - 三Agent循环)                  │    │
│  │                                                                     │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐    │    │
│  │   │    Idea      │───>│  Expression  │───>│    Backtest       │    │    │
│  │   │  Generator   │    │  Generator   │    │   Coordinator     │    │    │
│  │   │    Agent     │    │    Agent     │    │ (含相关性检查)     │    │    │
│  │   └──────────────┘    └──────────────┘    └───────────────────┘    │    │
│  │         │                    │                      │              │    │
│  │         │                    │                      │              │    │
│  │         ▼                    ▼                      ▼              │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐    │    │
│  │   │  RAG Dedup   │    │   Template   │    │  优质因子输出     │    │    │
│  │   │   Service    │    │   Expansion  │    │  (Sharpe > 阈值)  │    │    │
│  │   └──────────────┘    └──────────────┘    └───────────────────┘    │    │
│  │                                                                     │    │
│  │   循环逻辑: 每轮生成新创意 → 表达式 → 回测，持续积累优质因子       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                 Phase 3: Improvement (独立运行)                      │    │
│  │                                                                     │    │
│  │   回测结果 ──> ImprovementAgent ──> 改进后的因子                    │    │
│  │                    │                                                │    │
│  │                    ├──> 精英池维护 (Top-5 Sharpe)                   │    │
│  │                    ├──> 多轮迭代改进 (含LLM反思)                   │    │
│  │                    ├──> 循环检测与恢复                              │    │
│  │                    └──> 验证重试机制                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Shared Services (共享服务层)                   │  │
│  │                                                                       │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐  │  │
│  │  │ LLM Service │ │ RAG Dedup   │ │  Metrics    │ │   Backtest      │  │  │
│  │  │             │ │   Service   │ │  Service    │ │  Coordinator    │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘  │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    Correlation Checker                          │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Infrastructure (基础设施层)                       │  │
│  │                                                                       │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐  │  │
│  │  │   Config    │ │    LLM      │ │   Prompt    │ │    ACE Lib      │  │  │
│  │  │   Loader    │ │   Client    │ │  Templates  │ │  (BRAIN API)    │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 核心流程详解

系统采用**三阶段流水线架构**，分为初始化阶段、主循环阶段和独立优化阶段：

---

### Phase 1: 数据分类（一次性执行）

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ CSV文件 │────>│  LLM    │────>│ 自审    │────>│ 分类    │
│         │     │  分类   │     │ 评审   │     │ 结果    │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                                    │
                                    ▼
                             发现问题则重新分类
                             (最多迭代3轮)
```

**执行时机**：系统启动时执行一次，结果缓存供后续使用

**核心特性：**
- **智能分类**：基于字段ID和描述，LLM自动识别经济含义并分类（价格类、成交量类、基本面类等）
- **迭代自审**：分类完成后，另一个LLM角色（评审员）对结果进行批判性评审
- **反馈修正**：如果评审发现问题，系统会根据反馈重新分类，直到评审通过或达到最大迭代次数
- **批量处理**：对于大型数据集（>500字段），采用增量分类策略，使用已有类别指导新批次分类

---

### Main Loop: 主循环（三Agent循环）

```
┌─────────────────────────────────────────────────────────────────┐
│                        主循环 (持续运行)                         │
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐ │
│   │    Idea      │───>│  Expression  │───>│    Backtest       │ │
│   │  Generator   │    │  Generator   │    │   Coordinator     │ │
│   └──────────────┘    └──────────────┘    └───────────────────┘ │
│         │                    │                      │           │
│         ▼                    ▼                      ▼           │
│   生成因子创意         生成表达式              并行回测验证      │
│   (含语义去重)        (模板展开)              (含相关性检查)     │
│                                                                 │
│   输出: unique ideas  输出: expressions      输出: qualified    │
│                                               alphas           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                     优质因子累积输出 (Sharpe > 阈值)
```

**执行时机**：Phase 1完成后持续循环运行

**循环逻辑**：
1. **创意生成**：基于分类结果和历史记录，生成新的因子创意（去重后）
2. **表达式生成**：将创意转化为可执行的表达式（模板展开，一次生成数十条）
3. **回测验证**：并行回测所有表达式，筛选优质因子（Sharpe > 阈值）
4. **循环迭代**：回到步骤1，生成更多创意，持续积累优质因子

#### 2.1 创意生成阶段 (IdeaGeneratorAgent)

```
数据分类 + 操作符库 + 历史记录 → LLM生成创意 → 语义去重 → 独特创意
                                         ↓
                              使用MiniLM-L6-v2向量化
```

**核心特性：**
- **上下文感知**：结合数据分类结果和操作符描述，生成有针对性的因子创意
- **语义去重**：使用`all-MiniLM-L6-v2`模型将创意编码为向量，通过余弦相似度过滤重复创意（默认阈值0.85）
- **历史感知**：加载历史创意记录，避免生成重复内容
- **质量验证**：验证创意中使用的数据类别和操作符是否有效，过滤幻觉内容
- **自动重试**：如果生成的创意全部无效，自动重新生成

#### 2.2 表达式生成阶段 (ExpressionGeneratorAgent)

```
因子创意 → LLM生成模板 → 模板验证 → 变量验证 → 笛卡尔积展开 → 表达式列表
                                        ↓
                              一次调用生成上百条表达式
```

**核心特性：**
- **模板展开**：LLM生成带`{placeholder}`的模板 + 变量字典，通过`itertools.product`笛卡尔积展开，一次调用可生成上百条表达式
- **操作符验证**：验证模板中的操作符是否在`operators_desc.json`中存在，自动移除无效操作符
- **变量验证**：根据变量名识别类型（field/operator/number/group），验证字段名有效性，自动纠正常见错误
- **字段纠正**：自动修正常见字段名错误（如`ret` → `returns`，`vol` → `volume`）
- **模板缓存**：缓存已生成的模板，提高复用效率

#### 2.3 回测验证阶段 (BacktestAgent + BacktestCoordinator)

```
表达式列表 → 并行回测 → 指标提取 → 相关性检查 → 筛选结果
                ↓
        线程池管理，每个线程独立会话
```

**核心特性：**
- **并行回测**：使用线程池（最多4个worker）并行处理多个创意的回测任务
- **独立会话**：每个worker线程拥有独立的BRAIN API会话，避免冲突
- **指标提取**：提取Sharpe、收益、回撤、换手率等关键指标
- **阈值筛选**：按配置的阈值（默认Sharpe > 0.8）筛选优质因子
- **相关性检查**：对筛选后的因子进行相关性检测（阈值0.7），过滤高度相关的Alpha
- **智能排序**：按Sharpe排序，正Sharpe降序，负Sharpe升序

---

### Phase 3: 改进优化（独立运行）

```
┌─────────────────────────────────────────────────────────────────┐
│                  ImprovementAgent (独立运行)                     │
│                                                                 │
│   回测结果 ──> 选择待改进因子 ──> 多轮迭代改进                   │
│                      │                                          │
│                      ▼                                          │
│              ┌──────────────┐                                   │
│              │  精英池维护  │                                   │
│              │  (Top-5)     │                                   │
│              └──────────────┘                                   │
│                      │                                          │
│                      ▼                                          │
│   每轮循环:                                                      │
│   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  │
│   │ 选择   │─>│ LLM   │─>│ 生成   │─>│ 回测   │─>│ 反思   │  │
│   │ 精英   │  │ 改进   │  │ 变体   │  │ 验证   │  │ 分析   │  │
│   └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  │
│                                                                 │
│   终止条件: 达到目标Sharpe / 最大轮数 / 循环检测                 │
└─────────────────────────────────────────────────────────────────┘
```

**执行时机**：独立于主循环运行，基于回测结果进行深度优化

**核心特性：**
- **精英池机制**：维护Top-5 Sharpe表达式，每轮从精英池选择最优进行改进
- **多轮迭代**：支持配置最大改进轮数，每轮生成多个变体并回测
- **LLM反思**：每轮结束后，LLM分析回测数据，总结成功/失败原因，指导下一轮改进
- **循环检测**：检测表达式是否陷入循环（重复生成相似内容），进入恢复模式
- **验证重试**：表达式验证失败时，自动将错误信息反馈给LLM修正（最多3次）
- **提前终止**：达到目标Sharpe时提前结束改进
- **维度分析**：从经济逻辑一致性、交易成本、过拟合风险、参数敏感性等多维度评估

## 技术亮点

### 1. 模板展开机制

传统方法需要逐条生成表达式，效率低下。本系统采用**模板展开**策略：

```python
# LLM生成模板
template = "rank(reverse(subtract({op}({field},{window}), group_median({op}({field},{window}), {group}))))"
variables = {
    "op": ["ts_sum", "ts_mean"],
    "field": ["close"],
    "window": [5, 10, 20],
    "group": ["sector"]
}

# 笛卡尔积展开：2×1×3×1 = 6条表达式
expressions = itertools.product(*variables.values())
```

**优势**：
- 一次LLM调用生成数十到上百条表达式
- 保证表达式结构一致性，便于对比分析
- 大幅降低LLM调用成本

### 2. 语义去重机制

使用`all-MiniLM-L6-v2`模型进行语义相似度检测：

```python
# 计算创意向量
embedding = model.encode(idea_content)

# 余弦相似度检测
similarity = cosine_similarity(new_embedding, existing_embeddings)

# 过滤重复（阈值0.85）
if similarity < 0.85:
    unique_ideas.append(idea)
```

**优势**：
- 识别语义相似但表述不同的创意
- 避免重复劳动，提高创意多样性
- 模型轻量（~80MB），推理速度快

### 3. 精英池机制

```python
# 精英池维护
elite_pool = []  # 最多5个

# 每轮更新
top_alphas = select_best(expressions, backtest_results, top_n=3)
elite_pool = update_elite_pool(elite_pool, top_alphas)
elite_pool.sort(by_sharpe, descending=True)
elite_pool = elite_pool[:5]

# 选择下一轮改进目标
current_expr = select_from_elite_pool(elite_pool, strategy="best")
```

**优势**：
- 始终保留最优表达式，避免丢失好结果
- 引导改进方向，向最优解收敛
- 平衡探索与利用

### 4. 循环检测与恢复

```python
# 检测循环
def detect_loop(expr_code, history):
    normalized_new = normalize(expr_code)
    for h in history:
        normalized_hist = normalize(h["expression_code"])
        if normalized_new == normalized_hist:
            return True, "Loop detected"
    return False, ""

# 恢复模式
if loop_detected:
    loop_recovery_mode = True
    loop_recovery_count += 1
    # 添加警告信息，要求LLM生成完全不同的方案
    last_reflection["loop_recovery_warning"] = "CRITICAL: You are in loop recovery mode..."
```

**优势**：
- 避免陷入无效循环
- 强制LLM探索新方向
- 提高改进效率

### 5. 并行回测协调

```python
# 线程池并行执行
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_idea, idea): idea for idea in ideas}
    for future in as_completed(futures):
        result = future.result()
        # 立即回调下游处理
        if downstream_callback:
            downstream_callback(result)
```

**优势**：
- 充分利用多核CPU
- 每个线程独立会话，避免冲突
- 支持立即回调，实时处理结果

## 目录结构

```
quant_factor_agents/
├── main.py                        # 主入口，循环调度所有Agent
├── requirements.txt               # Python依赖
├── core/                          # 核心框架
│   ├── agent_base.py              # Agent抽象基类
│   ├── supervisor.py              # Agent调度器（重试、统计）
│   └── state_manager.py           # 持久化状态管理
├── agents/                        # Agent实现
│   ├── data_classifier.py         # 数据字段分类（含自审机制）
│   ├── idea_generator.py          # 因子创意生成（含语义去重）
│   ├── expr_generator.py          # 表达式生成（模板展开）
│   ├── backtest_agent.py          # BRAIN API回测
│   ├── improvement_agent.py       # 多轮迭代改进（精英池+循环检测）
│   └── logger.py                  # 日志记录
├── services/                      # 共享服务
│   ├── llm_service.py             # LLM高层封装
│   ├── rag_dedup_service.py       # 语义去重（Embedding）
│   ├── metrics_service.py         # 执行统计
│   ├── backtest_coordinator.py    # 并行回测协调器
│   └── correlation_checker.py     # Alpha相关性过滤
├── infrastructure/                # 基础设施
│   ├── config_loader.py           # YAML配置加载（支持环境变量覆盖）
│   ├── llm_client.py              # 多LLM提供方客户端
│   ├── prompt_templates.py        # 全部LLM Prompt模板（~1500行）
│   └── ace_lib.py                 # BRAIN API封装
├── models/                        # 数据模型
│   ├── idea.py                    # 创意模型
│   ├── expression.py              # 表达式模型
│   └── backtest_result.py         # 回测结果模型
├── config/                        # 配置文件
│   ├── settings.yaml.example      # 配置模板
│   ├── operators.json             # 操作符完整文档
│   └── operators_desc.json        # 操作符简要描述
├── tests/                         # 测试
│   └── test_agents/               # Agent测试
└── 知识库/                        # 因子研究知识库
```

## 配置说明

配置文件为 `config/settings.yaml`

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `brain.email/password` | 平台登录凭证 | 必填 |
| `llm.api_key` | LLM API密钥 | 必填 |
| `llm.model` | 模型名称 | `deepseek-v4-flash` |
| `llm.base_url` | API地址 | `https://api.deepseek.com` |
| `backtest.settings.region` | 市场区域 | `ASI` |
| `backtest.settings.universe` | 股票池 | `MINVOL1M` |
| `backtest.filter_threshold.min_sharpe` | 最低保留Sharpe | `0.8` |
| `backtest.max_workers` | 并行回测线程数 | `4` |
| `idea_generation.generate_num` | 每轮创意数 | `4` |
| `improvement.max_rounds_per_expr` | 改进轮数 | `1` |
| `improvement.target_sharpe` | 目标Sharpe | `1.25` |
| `improvement.elite_pool_size` | 精英池大小 | `5` |
| `classification.max_review_iterations` | 分类自审最大迭代 | `3` |
| `main_loop.max_iterations` | 最大迭代次数 | `null`（无限） |

## 输出产物

| 路径 | 说明 |
|------|------|
| `logs/good_factors.jsonl` | 优质因子记录 |
| `logs/bad_factors.jsonl` | 劣质因子记录 |
| `logs/rounds.jsonl` | 每轮统计 |
| `logs/metrics_summary.json` | 汇总指标 |
| `cache/state.json` | 持久化状态（支持断点续跑） |
| `cache/classification/` | 字段分类缓存 |
| `cache/ideas/` | 创意历史与Embedding |
| `cache/templates/` | 表达式模板缓存 |
| `output/backtest_results_*.json` | 完整回测结果 |
| `output/overall_reflection_*.json` | LLM改进反思 |
| `output/validation_errors_*.json` | 验证错误记录 |

## 快速开始

### 1. 克隆并安装依赖

```bash
git clone <repo-url>
cd quant_factor_agents
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

### 2. 创建配置文件

```bash
cp config/settings.yaml.example config/settings.yaml
```

编辑 `settings.yaml`，填入：
- BRAIN平台账号密码（`brain.email` / `brain.password`）
- LLM API Key（`llm.api_key`）
- 数据文件路径（`data.raw_file`）

### 3. 准备数据文件

将数据CSV文件放到项目目录，文件名格式须为 `region_universe_dataset_type.csv`，例如：

```
ASI_MINVOL1M_other455_MATRIX.csv
```

CSV内容须包含 `id` 和 `description` 列（列名可在配置中修改）。

### 4. 下载ML模型（可选，用于语义去重）

首次运行时程序会自动从HuggingFace下载。如果网络不佳，可手动下载：

```bash
pip install huggingface_hub

# 国际网络
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir models/all-MiniLM-L6-v2

# 国内镜像
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir models/all-MiniLM-L6-v2
```

### 5. 运行

```bash
# 运行主循环（三Agent循环模式）
python tests/test_agents/test_pipeline_stage.py
```

**执行流程**：
1. **Phase 1**：数据分类（一次性执行，结果缓存）
2. **Main Loop**：创意生成 → 表达式构建 → 回测验证（持续循环）
3. **Phase 3**：改进优化（独立运行，基于回测结果）

按 `Ctrl+C` 安全停止，状态自动保存，下次启动恢复。

**注意**：由于当前项目仍在测试阶段，主循环已完成 数据分类 → 创意生成 → 构建表达式 → 创建回测 的闭环。改进优化阶段（ImprovementAgent）可独立运行。

更多功能实现，敬请期待！

## 技术栈

- **Python 3.8+**
- **LLM**：DeepSeek（默认）/ OpenAI / 其他兼容API
- **Embedding**：sentence-transformers/all-MiniLM-L6-v2
- **向量计算**：NumPy
- **数据处理**：Pandas
- **并行处理**：concurrent.futures.ThreadPoolExecutor
- **配置管理**：PyYAML
- **BRAIN API**：ACE Library

## 设计理念

### 1. 三阶段流水线架构

系统采用清晰的阶段划分，各阶段职责明确：
- **Phase 1 (初始化)**：数据分类，一次性执行，结果缓存复用
- **Main Loop (主循环)**：创意生成 → 表达式构建 → 回测验证，持续循环积累优质因子
- **Phase 3 (优化)**：独立运行，深度改进因子质量

### 2. Agent协作模式

每个Agent专注于单一职责，通过标准化的输入输出接口协作：
- **输入验证**：严格验证必需字段
- **状态返回**：所有结果包含`status`字段
- **错误处理**：异常不扩散，优雅降级

### 3. LLM驱动的创造力

每个创造性/分析性步骤都由LLM完成：
- 字段分类、创意生成、表达式模板、改进策略、反思分析
- 所有Prompt集中在`prompt_templates.py`中管理（约1500行）
- 支持多轮对话，逐步优化结果

### 4. 迭代自改进

系统具备自我反思和改进能力：
- **分类自审**：分类结果由独立LLM评审，发现问题自动修正
- **改进反思**：每轮改进后，LLM分析成功/失败原因，指导下一轮
- **精英池**：保留最优结果，引导改进方向
- **循环检测**：识别无效循环，强制探索新方向

### 5. 鲁棒性设计

系统具备完善的容错和恢复机制：
- **自动重试**：Agent执行失败自动重试（最多3次）
- **验证重试**：表达式验证失败自动反馈LLM修正
- **断点续跑**：状态持久化，支持中断后恢复
- **优雅降级**：单个因子失败不影响整体流程

## 许可证

[待定]

## 致谢

感谢所有为本项目贡献思路和代码的团队成员。
