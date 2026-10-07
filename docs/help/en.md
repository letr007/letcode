# letcode TUI Help

## Quick start

- Press `Tab` to switch between contents and text.
- Press `↑` / `↓` or `k` / `j` to select a chapter or scroll the text.
- Press `PgUp` / `PgDn` to page, or `Home` / `End` to reach the top or bottom.
- Press `Esc` or `q` to close help.

Wide terminals show both panes. Use `Tab` to switch panes in narrow terminals.

Start letcode from your project directory.

```sh
letcode
```

- Use `/model` to choose a configured model.
- Use `/permission` to check the permission mode.
- Type a task and press `Enter`. Include the goal and relevant files.
- Type `/help` or `/?` to open this manual.

Type `/` for command completion or `@` for expert completion.
Use `↑` / `↓` to choose an entry and `Tab` to insert it.
Add any arguments before submitting the completed command.

## Command reference

{{commands}}

## Keyboard shortcuts

**Main input**

These keys apply outside dialogs, questions, approval requests, and completion lists.

- `Enter` submits input. `Shift+Enter` inserts a newline.
- `←` / `→` move the cursor. `Backspace` / `Delete` remove text.
- `Home` / `Ctrl+A` move to the start of the input.
- `End` / `Ctrl+E` move to the end of the input.
- `↑` / `↓` recall previous or next input.
- `PgUp` / `PgDn` scroll the conversation. `Ctrl+End` returns to the bottom.
- `Ctrl+V` / `Cmd+V`, `Alt+V` or `Shift+Insert` paste from the clipboard.
- `Ctrl+Backspace` removes the last queued message and preserves the input draft.
- `Ctrl+T` cycles the reasoning effort supported by the current model.
- `Esc` clears a text selection. Otherwise, press it twice to interrupt active work.
- `Ctrl+C` copies selected text. Without a selection, it uses the same interrupt
  confirmation during a running turn and quits when the main turn is idle.

Terminals must report the key combination for it to work.
In particular, some terminals cannot distinguish `Ctrl+Backspace` from `Backspace`.

**Ctrl+X shortcuts**

Press `Ctrl+X`, release it, then press the next key.
These shortcuts apply outside dialogs, completion lists, questions, and approval requests.

- `b` toggles the session panel.
- `m`, `p`, `r`, `t` open model, permission, reasoning effort, and thinking-display options.
- `a`, `c`, `i`, `s` open expert models, context details, MCP tools, and skills.
- `h` or `?` opens help.
  In a child view with empty input, `h` browses the previous child; use `?` to open help.
- From the main view, `↓` opens the first child. `←` / `→` browse children.
- `↑` returns to the parent view.

**Dialogs and child views**

- In selection dialogs, press `↑` / `↓` to choose an item, `Enter` to confirm, and `Esc` to close.
- In searchable dialogs, type to filter the list and press `Backspace` to edit the filter.
  In non-searchable selection dialogs, press `k` / `j` to move between items.
- In the session picker, press `←` / `→` to switch between this workspace and all workspaces.
- In the configuration editor, press `Enter` to edit or expand an item.
  Outside a field editor, press `Ctrl+S` to save the configuration draft.
- In a child view with empty input, press `↑` to return to the parent.
  Press `←` / `→` or `h` / `l` to browse children; press `j` / `k` to scroll the conversation.
- From the main input or a child view, press `Alt+←` / `Alt+→` to browse children,
  and `Alt+↑` to return to the parent.
- In questions, press `Tab` to switch tabs and `↑` / `↓` to select an answer.
  Press `Enter` to select or edit an answer, then submit from the confirmation tab.

## Sessions and context

Conversation events are recorded in session logs.
`/resume` opens a picker; `/resume <session_id>` restores a session by ID or prefix.
The picker initially lists sessions from the current workspace.
Restoring a conversation does not restart unfinished tools or background processes.

`/new` starts a separate conversation.
Finish or interrupt current work before creating, restoring or navigating a session.
Use `/exit` or `/quit` to leave the TUI when no work is running.

Messages submitted during a running turn are queued and marked `QUEUED`.
`Ctrl+Backspace` removes the newest message that has not yet been dispatched.
It leaves the current execution and input draft unchanged.
Removed messages remain in input history and can be recalled with `↑`.

- `/context` opens details about the current context.
- `/compact` invokes the internal `historian` to summarize history and reduce active context.
- `/tree` opens conversation history for navigation.
- `/undo` moves to the previous completed turn. `/redo` moves forward again.

History navigation changes the conversation context and preserves recorded history.
It does not undo edits to your project files.
Compaction summarizes history; session logs retain the original records.

## Experts and child sessions

For direct delegation, start a message with an expert name such as `@explorer`, then a task.
Keep the task bounded and state the expected result.

- `@explorer` explores repository code without editing it.
- `@fixer` implements or repairs files within an assigned write scope.
- `@oracle` analyzes root causes, risks, and verification plans without editing files.
- `@designer` develops designs and interface decisions without editing files.
- `@librarian` gathers documentation and reference material without editing files.
- `@general` handles bounded read-only assistance.

```text
@explorer Find the authentication entry points without changing files.
```

`fixer` requires non-empty `owned_paths` to acquire write locks.
The `@fixer <task>` shorthand passes only task text and is rejected without assigned paths.
For file changes, specify the task and target paths to the main agent so it can delegate to `fixer`.

`/agents` configures expert model choices; it does not start a task.
`reviewer` handles permission review and `historian` handles context summaries.
They are internal experts and cannot be invoked with `@`.

`/child` toggles between parent and child views.
`/child first`, `/child next` and `/child prev` select a child to read.
Viewing a child session does not take over execution or change the active parent session.
Continuing an ended child session requires explicit takeover by the main agent.

The main agent can list, inspect, wait for or cancel tasks with
`agent__jobs`, `agent__status`, `agent__wait` and `agent__cancel`.
Read-only tasks can run concurrently. Overlapping read/write scopes conflict;
conflicting tasks are rejected rather than queued. Children cannot delegate again.

## Permissions and settings

Use `/permission` or `/perm` to inspect the mode.
Use a mode argument to change it, for example `/permission default`.

- `safe` requests approval for ordinary tool calls.
- `default` allows classified read and preview calls and asks about writes,
  commands and unknown calls. Web fetches and access outside the workspace also need approval.
- `auto` uses the same classification and sends approval requests to `reviewer`.
  `reviewer` can request justification, ask for user approval, or deny a call.
- `yolo` skips mode-based approval prompts. Tool scopes and child path limits still apply.

Tool classification is not a guarantee that a shell command has no side effects.

In an approval request, `←` / `→` choose an option and `Enter` confirms it.
`y` / `o` allow once; `n` / `d` deny. `a` grants session approval when offered.
`↑` / `↓` or `k` / `j` scroll the conversation while approval is pending.
Session approval covers the displayed resource or call scope, not every future operation.
Changing the mode clears those grants. A new session does not inherit them.

- `/model` chooses a configured model. `/reasoning` chooses a supported effort.
- `/thoughts` changes thinking display; `/tools` changes tool-output display.
- `/panel`, `/scrollbar` and `/theme` adjust the interface.
- `/language en` and `/language zh-CN` change the interface language.
- `/mcp` browses MCP services and tools. `/skill` browses local skills.
- `/config` opens the configuration editor. Field edits remain a draft until saved.

Model, permission and reasoning changes made during a turn take effect after it ends.
`Ctrl+S` in the configuration editor saves only configuration.
