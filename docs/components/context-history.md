# 历史整理与证据恢复

letcode 使用内部 `historian` 专家增量整理会话历史。设计参考 Magic Context 的分段历史、多档表述、稳定基线与增量更新；letcode 使用三档，并沿用自己的 transcript、运行时投影与协议适配器。

## 三档历史

Historian 每次处理一段尚未整理的、协议完整的历史前缀，按工作目标生成历史段。每段一次生成三种表述：

- **Detailed**：目标、重要过程、失败原因、关键纠正及定位信息。
- **Compact**：结果、决定、约束及必要定位信息。
- **Anchor**：结果与检索线索；标题足以识别时，正文可以为空。

档位选择是确定性的，不会再次调用模型改写旧摘要。历史预算缩小时可选择更简短的档位或归档；归档只改变活跃表示，不删除 transcript。扩大预算后可以从已保存产物重新选择档位。

三档是同一历史段的不同表述，不是“摘要、证据、原文”三个存储层。模型摘要可能遗漏或误解信息，应按来源回查，不应将重要度或提炼结果视为正确性保证。

## 准备、发布与应用

`HistoryPublished` 保存经过来源覆盖校验的历史段；旧日志中的事实字段继续兼容读取。`HistoryApplied` 选择哪些历史段和档位进入当前上下文，并退休已被覆盖的原始前缀。发布本身不删除原始消息。应用后会追加一条内部续接消息，重述当前未完成 todo 与 auto-continue 状态：这些是运行时快照字段而非 transcript 文本，只靠历史总结会丢失。

活跃历史分为稳定基线、增量区和近期原始尾部。常规请求复用已选择的表述；增量更新尽量保持基线不变。预算压力、显式压缩或增量过大时重新物化基线。重新物化不意味着 provider 缓存已失效，也不保证某种费用节省。

后台任务绑定会话、分支和 checkout revision。任务执行期间主会话可以追加消息；应用在最新投影上进行，不安装旧快照覆盖新工作。取消和发布共享提交边界：已提交的产物保留，取消先于发布时不会晚写入产物。

`/compact` 使用相同的三档机制。旧会话检查点仍可读取并作为明确的 legacy baseline 保留；不会自动付费重建完整旧历史，也不会在新机制失败时切换回滚动摘要生成。

## Historian 模型

未配置时继承主模型，也可以像其它系统专家一样设置单独路由：

```toml
[agents.historian]
provider = "your-provider"
model = "your-model"
```

模型必须已经在对应 provider 中配置。Historian 使用统一 ModelRuntime 的无工具 one-shot 请求，静态任务约束进入协议原生高权威字段，不进入主 Agent 的工具循环，不递归委派。处理范围会按 Historian 路由的请求预算预检。

后台调用会产生额外用量。子视图报告 provider 返回的 usage/cache 事件；缺失用量时明确显示未报告，不伪造数字。事件可能是累计更新，不能直接全部相加。

## Evidence 与项目记忆

工具观察、子代理结果和来源引用继续进入会话内的 EvidenceRecord。Evidence 参与当前会话的选择与审计，但不再兼任跨会话项目记忆库。旧日志中已经发布和应用的事实仍按原有投影恢复，session JSONL 不迁移、不改写。

新的项目记忆由独立后台 worker 在完整 turn 持久化后增量提取，复用 Historian 的无工具 one-shot 路由，但使用独立任务 prompt。数据按规范化 workspace 路径隔离，保存在配置目录的 SQLite 读模型中；首次启用只登记当前 journal frontier，不扫描或重新提炼旧 session。session undo、checkout 或删除不会自动撤销已经记录的项目知识。

普通请求不自动注入整份项目记忆。模型仅在任务需要时调用 `memory__recall`，按关键词、代码路径、类型和状态检索；结果保留来源 session、branch 和 `raw:N` ID，以便通过 history 工具回查。记忆可能不完整或过时，不是执行授权，也不能覆盖当前代码、用户要求和高权威配置。

记忆库明确报告 `missing`、`synchronizing`、`failed`、`empty` 或 `ready`，索引缺失或失败时不会静默回退到 session 全量重放。

## 搜索和展开

- `context__search`：检索原始会话、归档历史段及 evidence，返回有界预览和来源 ID。
- `context__expand`：按 ID 读取历史段对应原始消息，或 evidence 摘录。分页 offset/limit 以字符计；搜索分页以结果条数计。
- `memory__recall`：按关键词、路径、类型和状态查询当前工作区的独立项目记忆库；不扫描 session JSONL，也不会把全部记忆自动注入普通请求。

默认作用于当前 session/branch/leaf。指定其它会话或分支时，结果明确带上该来源。展开是只读，不恢复旧 runtime、不重复执行历史工具。Evidence detail 是摘录，不等同完整工具正文。

新产生的内置大工具输出在进入持久记录前，会将折叠正文保存在 sessions 目录下的 `artifacts/<session-id>/`，内容寻址并先完成持久化再返回路径。模型可通过结果中的路径使用 `fs__read` / `search__rg` 读取正文。旧临时文件缺失时不能从摘要还原原文。

## TUI

后台工作只显示 footer 图标，不插入主时间线工具卡、不改变主任务忙碌状态。失败图标提示查看详情。

通过已有子会话导航可查看会话压缩 Historian 的结构化报告：概览、分段正文、模型、耗时和用量；旧报告仍可能包含事实字段。默认显示精简版；在 Historian 子视图且命令输入为空时，按 `1` 查看精简、`2` 查看详细、`3` 查看线索、`0` 查看完整报告 JSON、`s` 切换完整来源 ID。切换只改变查看方式，不改变主模型采用的档位。来源是可复制的引用，不是直接打开原文的按钮。项目记忆 worker 不插入主时间线工具卡。

报告显示“产物已生成”，不据此推断主会话已发布或应用。用量与缓存分别显示各自最近一次 provider 更新，不累加；缺失值显示“未报告”，完整更新保存在报告 JSON 中。旧 Markdown 报告仍按原文显示。内部整理报告不作为普通项目 Decision evidence 再次注入主模型。

手动 `/compact` 或主请求必须等待整理时，footer 显示扫描动画与“正在整理上下文…”；完成后显示短暂提示，不显示虚构完成百分比。新整理过程不会在主时间线创建空摘要块或分隔线，旧日志中已有的摘要正文仍保留展示。

## 代码入口

- `src/context_history.rs`：历史产物、应用选择、档位与来源查询。
- `src/historian.rs`：专用 prompt、结构化解析和工作报告。
- `src/agent/history_runtime.rs`：前缀准备、请求边界应用与预算处理。
- `src/session/historian.rs`：内部专家池任务、取消和发布。
- `src/transcript/transcript_projection/`：分支恢复、来源校验与活跃投影。
- `src/evidence.rs`：会话内观察、来源与旧事实兼容投影。
- `src/project_memory/`：workspace 隔离的 SQLite store、增量提取和后台 worker。
- `src/memory.rs`、`src/tool/memory.rs`：项目记忆查询参数与只读工具入口。
- `src/tool/context_history.rs`、`src/tool/fold_artifact.rs`：原文查询与大输出保存。
