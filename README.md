# letcode

`letcode` is a small Rust CLI AI-agent prototype. It launches a Ratatui/Crossterm TUI by default while keeping the original line-based CLI available as an explicit mode.

## Running

```sh
cargo run
```

That launches the Ratatui/Crossterm TUI by default. The old line-based CLI/REPL is still available explicitly:

```sh
cargo run -- --cli
# or
cargo run -- cli
```

The explicit CLI mode supports prompts plus built-in commands such as:

- `quit` / `exit` — leave the CLI
- `/permission` or `/perm` — show permission mode
- `/permission safe`, `/permission default`, `/permission solo` — change permission mode
- `/sessions` — list recorded sessions
- `/session resume <id-prefix>` or `/resume <id-prefix>` — resume a previous transcript

To start the TUI explicitly, if desired:

```sh
cargo run -- --tui
# or
cargo run -- tui
```

The OpenAI configuration uses:

- `OPENAI_API_KEY` — optional at startup; prompts that need the model will fail with a visible error if it is missing
- `OPENAI_BASE_URL` — optional, defaults to `https://api.openai.com/v1`

## TUI MVP

The TUI uses `ratatui` for rendering and `crossterm` for terminal input/raw-mode handling. It is visually inspired by the older abandoned reference implementation at `/Users/letr/Project/Rust/letcode-old`, but only borrows the dark theme direction and selected UX ideas. It is a reference for style and a few proven concepts, not a code structure to port forward wholesale. The new implementation keeps runtime, rendering, state, input, event, presentation, terminal, and agent-runner responsibilities split across focused modules under `src/tui/`.

The MVP includes:

- transcript/timeline area
- composer input area
- footer/status area with model, permission mode, phase, running tool, and key hints
- empty-state welcome wordmark
- streaming assistant text in a single evolving timeline item
- typed tool started/finished display
- compact permission prompt cards for default-mode approvals
- terminal RAII setup/restore for alternate screen, raw mode, and cursor visibility

The TUI intentionally does not include a model picker, session picker, file tree, full diff viewer, multi-tab workspace, mouse workflow, or plugin/component framework.

## TUI keybindings

- Type text normally in the composer
- `Enter` — submit the composer when no permission prompt is pending
- `Backspace` — edit the composer
- `Ctrl-C` — emergency quit
- `exit`, `quit`, `/exit`, or `/quit` then `Enter` — normal quit command
- `/help` — show available TUI commands
- `/permission`, `/permission safe`, `/permission default`, `/permission solo --yes` — inspect or change permission mode
- `q` is normal text; it does not quit by itself
- Transcript scrolling:
  - `↑` / `↓` — scroll transcript by 1 row
  - `PageUp` / `PageDown` — scroll transcript by 10 rows
  - `End` — jump back to the bottom (resume auto-follow)
- During permission prompts:
  - `y` / `Y` / `a` / `A` — approve
  - `n` / `N` / `d` / `D` / `Esc` — deny
  - `Enter` — no-op; it never approves a permission request

## Architecture boundaries

The maintainability goal for this MVP is a thin TUI entry with explicit boundaries so the explicit CLI stays intact and the new UI does not grow into a monolith:

- `src/tui/events.rs` defines typed UI events.
- `src/tui/state.rs` owns UI state transitions.
- `src/tui/timeline.rs` owns display/timeline item models and updates.
- `src/tui/presentation.rs` owns lightweight tool presentation policy.
- `src/tui/render.rs` is a pure Ratatui view layer over `TuiState`; it does not call OpenAI, execute tools, write transcripts, or decide permissions.
- `src/tui/input.rs` maps Crossterm keys to UI actions.
- `src/tui/runtime.rs` owns the terminal event loop and bridges input, render, and runner events.
- `src/tui/runner.rs` bridges the existing `Agent`, transcript recorder, and permission response channels into typed TUI events.
- `src/tui/terminal.rs` owns terminal setup/restore guards.

The executable entry remains in `src/main.rs`, where `cargo run` now starts the TUI by default and `cargo run -- --cli` or `cargo run -- cli` selects the old line-based CLI explicitly.

This split is intended to avoid recreating the old implementation’s monolithic TUI state and view/controller coupling.

## Verification

Automated checks used for the TUI work:

```sh
cargo fmt --check
cargo check
cargo test
cargo test tui::input
cargo test tui::runner
printf 'quit\n' | cargo run --quiet -- --cli
```

Recommended manual verification in a real terminal:

CLI explicit-path smoke check:

1. Run `cargo run -- --cli`.
2. Confirm the line-based CLI starts without entering the TUI.
3. Run a simple built-in command such as `/permission` or `/sessions`.
4. Exit with `quit` and confirm control returns cleanly to the shell.

TUI smoke check:

1. Run `cargo run`.
2. Confirm the welcome wordmark, composer, transcript area, and footer render.
3. Type `q` and confirm it appears as normal composer text rather than quitting.
4. Clear or submit an exit command: type `exit` or `/exit` then press `Enter`, and confirm the terminal returns to normal.
5. Run `cargo run -- tui` and confirm the subcommand-style entry reaches the same TUI flow.
6. Run again with `OPENAI_API_KEY` configured, submit a prompt, and confirm streaming assistant text appears.
7. In default permission mode, trigger a tool that requires approval and confirm `Enter` does nothing, `d`/`n` denies, and `a`/`y` approves.
8. Trigger a tool call and confirm started/finished states appear in the timeline.
9. Confirm the transcript files in `sessions/` continue to record user, assistant, tool, permission, and error events.
