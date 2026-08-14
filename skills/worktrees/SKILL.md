---
name: worktrees
description: 将 Git worktree 作为安全、隔离的开发通道，用于复杂、高风险或并行工作。
---

# Worktree 编排协议

使用 Git worktree 隔离并行代理任务、高风险实验、集成审查和清理工作。

## 核心约定

此工作流由编排代理负责。专家可以在隔离通道内工作，但父代理负责通道规划、分支与路径选择、文件归属、任务委派、差异检查、集成和清理。

worktree 统一放在：

```text
.letcode/worktrees/<slug>/
```

不要在主仓库的同级目录创建 worktree。

### 可选状态记录

多个通道需要长期协调时，可以使用 `.letcode/worktrees.json` 记录本地结构化状态：

```json
{
  "version": "1.0.0",
  "updatedAt": "2026-06-14T00:00:00.000Z",
  "lanes": [
    {
      "slug": "feature-auth-v2",
      "branch": "letcode/feature-auth-v2",
      "path": ".letcode/worktrees/feature-auth-v2",
      "base": "main",
      "purpose": "refactor authentication flow to use OAuth2",
      "owner": "orchestrator",
      "status": "active",
      "areas": ["src/auth", "src/config"],
      "createdAt": "2026-06-14T12:00:00.000Z"
    }
  ]
}
```

只在确有需要时创建该清单，并在通道状态变化、完成集成或清理后及时更新。默认将其视为本地工作流元数据；如果要纳入仓库约定并提交，应先征求用户同意。

## 安全规则

执行任何会改变 Git 状态的操作前：

- 确认当前目录位于目标仓库内。
- 检查当前分支、基础分支和未提交状态。
- 检查 `git worktree list`，避免路径或分支冲突。
- 确认计划使用的分支在本地和远程均不存在。
- 在仓库内创建 worktree 前，确认 `.letcode/worktrees/` 已被忽略。

执行以下操作前，必须取得用户明确确认：

- `git worktree add` 或 `git worktree remove`
- 创建、删除或重命名分支
- 合并、变基或 cherry-pick
- `git prune` 或 `git worktree prune`
- `git reset --hard`、`git clean`、强制推送、移除存在未提交变更的 worktree 等破坏性操作

未经用户针对具体操作的确认，不得删除分支、移除未清理的 worktree 或丢弃未提交变更。

## 忽略规则设置

创建通道前，先检查现有忽略文件。只添加缺少的条目，不要修改无关规则。

`.gitignore`：

```gitignore
# BEGIN letcode worktrees
.letcode/worktrees/
.letcode/worktrees.json
# END letcode worktrees
```

如果代理使用的忽略文件会隐藏 `.letcode`，应添加范围尽可能小的允许规则，确保代理能够读取工作通道中的文件。

## 工作流程

### 1. 规划与创建

1. 明确任务范围，并选择简短的 `<slug>`。
2. 按项目惯例选择分支名；没有明确惯例时，使用 `letcode/<slug>`。
3. 完成安全检查，并请求用户确认。
4. 确保忽略规则已配置。
5. 创建已获批准的通道：

   ```bash
   git worktree add -b <branch-name> .letcode/worktrees/<slug> <base>
   ```

6. 需要长期跟踪时，记录通道元数据。

### 2. 执行与委派

- 将每个受委派代理的工作目录设为对应通道路径。
- 通道内的编辑、构建和测试不得影响主工作区。
- 并行通道之间应分配互不重叠的文件范围。
- 只有用户要求提交，或已批准本地检查点提交时，才能创建提交。

### 3. 验证与集成

集成前：

1. 在通道内运行与项目匹配的格式化、静态检查、构建和测试。
2. 审查相对集成基础分支的完整差异。
3. 针对选定的集成操作请求用户确认。
4. 只执行已获批准的合并、变基或 cherry-pick。

### 4. 清理

1. 确认所有预期变更均已集成或归档。
2. 确认 worktree 中没有未提交变更。
3. 请求用户批准移除操作。
4. 执行已获批准的移除命令：

   ```bash
   git worktree remove .letcode/worktrees/<slug>
   ```

5. 更新或删除对应通道的元数据。

## 适用场景

适合在高风险重构、并行任务、探索性实验、复杂升级，或用户明确要求使用 worktree 时采用。

简单的单文件修改、纯文档改动、小型修复，或尚未弄清子模块与 worktree 限制的仓库，不应使用此工作流。
