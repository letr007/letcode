# 存储拓扑与数据持久化

letcode 采用文件驱动的存储架构。系统在本地磁盘中保存配置、会话日志、工具产物、缓存索引与跨会话项目记忆，确保会话数据具备完备的审计溯源与故障恢复能力。

## 磁盘目录拓扑

系统涉及的本地磁盘存储路径分布如下：

```text
~/.config/letcode/                     # 用户全局配置目录
├── letcode.toml                       # 主配置文件
├── AGENTS.md                          # 全局行为指令
├── skills/                            # 用户全局技能目录
└── memory/                            # 跨会话项目记忆库
    └── <workspace_sha256>/
        ├── memory.sqlite              # 项目记忆 SQLite 数据库（WAL 模式）
        ├── memory.sqlite-wal
        └── memory.sqlite-shm

{sessions_dir}/                        # 会话数据存储目录（默认配置同级 sessions/）
├── <session_id>.jsonl                 # 主会话日志（append-only，schema v2）
├── <session_id>.jsonl.lock            # 主会话跨进程独占写锁
├── sessions-index.json                # 会话快速发现索引（基于 mtime 派生）
├── artifacts/                         # 大体积工具输出暂存区
│   └── <session_id>/
│       └── <content_sha256>           # 基于内容哈希寻址的完整文本
└── children/                          # 子代理会话存储目录
    ├── <child_session_id>.jsonl       # 子代理会话日志
    └── <child_session_id>.jsonl.lock  # 子代理写锁

{logs_dir}/                            # 系统运行日志目录
└── combined.log                       # 诊断与跟踪日志文件
```

## 会话日志与独占写锁

每个会话以独立文件 `<session_id>.jsonl` 保存，采用只追加写入（append-only）的行式 JSON 格式。

数据落盘与一致性保护机制包括：

1. **跨进程文件锁**：记录器在打开会话时获取 `<session_id>.jsonl.lock` 的非阻塞独占文件锁。单个会话同一时刻仅允许一个活动进程写入，防止多实例并发篡改。若实例崩溃或异常退出，操作系统自动释放该文件锁；
2. **原子事务写入**：需要保持原子性的多个事件由事务封装，每条记录携带事务标识与连续索引，并在末尾追加 `transaction_commit` 提交行。只有包含合法提交行的事务，在恢复时才视为有效数据；
3. **安全数据刷盘**：常规记录执行缓冲写入与 `flush`；对于影响会话恢复的关键状态事件（如分支创建或配置变更），系统追加调用操作系统的 `sync_data` 确保数据真正持久化到磁盘；
4. **子会话分离**：委派给子代理的任务独立保存在 `children/<child_session_id>.jsonl` 中，拥有独立的日志与锁生命周期，避免子任务的高频输出污染主会话文件。

## 大体积产物折叠转存

为了防止单次工具的大输出膨胀主日志，系统内置了大文本折叠保存机制：

1. 当工具输出超过安全阈值时，系统计算输出正文的 SHA-256 哈希值；
2. 完整输出正文被单独写入 `{sessions_dir}/artifacts/<session_id>/<hash>`；
3. 日志记录中仅保留输出摘要与产物文件的绝对路径；
4. 模型在后续回合中，可按需调用 `fs__read` 或 `search__rg` 读取该文件。

该机制兼顾了主日志的精简与原始事实的完整性。

## 会话发现索引

为了在启动或切换会话时快速呈现会话列表，系统在会话目录下维护 `sessions-index.json`：

- **增量扫描**：索引记录每个会话文件的最后修改时间（mtime）与文件大小（size）。系统在列举会话时比对元数据，仅对发生变动的 JSONL 执行增量重解析；
- **派生缓存属性**：该文件属于辅助派生索引。若索引文件缺失或损坏，系统会自动重新扫描目录并重建索引，不影响会话日志本身的完整性。

## 跨会话项目记忆库

独立于各会话的短期历史，系统在 `~/.config/letcode/memory/` 下为每个工作区维护持久化的项目记忆：

- **工作区隔离**：系统对规范化后的工作区根路径进行 SHA-256 哈希，每个项目拥有独立的存储目录，目录权限在 Unix 环境下设为私有的 `0700`；
- **SQLite 存储**：记忆数据保存在 `memory.sqlite` 中，启用 WAL（Write-Ahead Logging）预写日志模式，支持高效并发读写；
- **异步提取**：后台任务在完整交互回合持久化后，异步提取项目关键决策、架构结论与避坑经验入库，普通会话操作无需等待写入完成。

## 闲置归档与维护

系统提供长期运行的自动维护策略：

- **闲置会话归档**：在配置中开启 `[global.session_archive]` 后，闲置天数超过 `older_than_days` 的历史会话由后台异步压缩，节省磁盘占用。归档会话在恢复（Resume）时自动解压并加载；
- **空会话自动清理**：在会话退出或切换时，若检测到未产生实际对话的空日志文件，系统会自动移除该文件与伴生锁，保持会话目录整洁。

## 源码索引

- `src/transcript/recorder.rs`：实现基于追加写模式的日志写入、事务提交与跨进程文件锁。
- `src/transcript/journal.rs`：负责日志读取、模式版本校验与文件完整性指纹计算。
- `src/transcript/session_index.rs`：维护 `sessions-index.json` 磁盘缓存与增量扫描。
- `src/tool/fold_artifact.rs`：处理大工具输出的内容哈希计算与文件折叠转存。
- `src/project_memory/store.rs`：实现基于 SQLite 的工作区项目记忆库初始化与数据持久化。
- `src/session/archive.rs`：实现闲置会话的后台归档压缩与恢复解压。
