# 测试指南

项目测试分为单元测试、进程内集成测试和进程端到端测试。

## 环境要求

运行测试需要以下基础环境。

- Rust stable 工具链，包含 `rustfmt` 与 `clippy` 组件。
- 系统 `PATH` 中需安装 `git` 与 `rg`（ripgrep）。在 Linux 上可通过包管理器安装，在 Windows 上可通过包管理器（如 Chocolatey）安装。

## 完整验证命令

```sh
# 代码格式检查
cargo fmt --check

# 代码静态检查
cargo clippy --locked --all-targets -- -D warnings

# 运行全量默认测试
cargo test --locked
```

## 测试分层与入口

项目在 `.cargo/config.toml` 中配置了三个测试别名。

| 层级 | 运行命令 | 实际执行与选集 | 验证范围 |
| --- | --- | --- | --- |
| 单元测试 | `cargo test-unit` | `test --locked --bin letcode -- --skip integration_tests::` | 解析、协议映射、状态迁移、渲染与局部单组件行为 |
| 进程内集成测试 | `cargo test-integration` | `test --locked --bin letcode integration_tests::` | 会话编排、工具调用、权限管控、日志提交与传输流协作 |
| 进程端到端测试 | `cargo test-e2e` | `test --locked --test cli` | 二进制命令行参数、配置加载、真实工具执行、进程生命周期与重启恢复 |

`test-unit` 与 `test-integration` 针对同一个二进制测试 target（`--bin letcode`），通过 libtest 名称是否包含 `integration_tests::` 子串划分选集，编译不隔离。`test-e2e` 执行 `tests/cli.rs` 编译生成的独立二进制。

## 代码组织

所有测试代码集中存放在根目录 `tests/` 下。

- `tests/internal/` 存放进程内测试源码，镜像 `src/` 模块层级。
- `tests/cli.rs` 存放进程端到端（E2E）测试。

进程内测试源码位于 `tests/internal/`，按 `src/` 的功能模块路径存放为 `tests.rs` 或专用测试文件。`src/` 内的模块通过 `#[cfg(test)] #[path = "..."]` 挂载对应测试。测试代码仍编译进 `--bin letcode`，不被 Cargo 自动识别为单独 target，不增加库目标，也不暴露额外公开接口，直接访问私有实现。单元测试和进程内集成测试未按目录物理拆开，统一通过测试名称是否包含 `integration_tests::` 区分。

进程端到端测试位于 `tests/cli.rs`，由 Cargo 编译为独立二进制 target（`--test cli`）。

## 定向运行

libtest 仅支持名称子串匹配，不支持通配符。

```sh
# 进程内测试（匹配完整名称中包含指定子串的测试）
cargo test --locked --bin letcode '<完整测试名中的连续片段>'

# 进程端到端测试
cargo test --locked --test cli '<完整测试名中的连续片段>'
```

## 确定性与隔离

端到端测试使用临时目录隔离 `HOME`、配置路径、工作区与会话存储，不读取用户本地配置。测试通过 Cargo 提供的环境变量 `CARGO_BIN_EXE_letcode` 获取已编译的可执行文件路径，拉起独立子进程。模型交互由本地 HTTP 或 TCP 模拟服务接管，不调用外部商业模型接口。

纯状态逻辑优先使用显式时间输入或 Tokio 虚拟时间。端到端测试中的真实 TCP 监听、子进程通信与操作系统文件锁依赖系统真实时钟，由显式超时防止挂起。

## 新增测试

- 单组件本地逻辑（含单组件本地文件操作）与私有逻辑的单元测试，补充在 `tests/internal/<模块>/` 下，由对应源码模块挂载。
- 跨模块协作与多组件持久化等进程内集成测试，同样补充在 `tests/internal/<模块>/` 下，测试名称需包含 `integration_tests::`。
- 真实二进制进程契约（如 CLI 参数、退出码、真实子进程拉起与重启恢复），补充在 `tests/cli.rs`。
- 基于现有可见性编写测试，不为测试暴露新的公开接口，不引入额外测试框架。
