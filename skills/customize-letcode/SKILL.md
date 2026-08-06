---
name: customize-letcode
description: >
  Use ONLY when the user is editing or creating letcode's own configuration:
  letcode.toml, files under ~/.config/letcode/, project .letcode/, AGENTS.md for
  letcode sessions, letcode skills, MCP servers, provider/model routes, permission
  mode, or TUI themes. Do not use for the user's application code, or for any
  project that is not configuring letcode itself.
---

<!--
  Built-in skill body registered in src/skills.rs before disk discovery.
-->

# Customizing letcode

letcode validates `letcode.toml` with `deny_unknown_fields` and refuses to start
when a field is wrong or a required provider/model is missing. The shapes below
are the accepted surface. Prefer reading `src/config.rs` (`RawAppConfig` and
builders) over guessing.

## Applying changes

`letcode.toml` is watched at runtime. A successful reload applies **supported
runtime fields only** (providers/models/routes, retry, compaction, tool timeout,
tool parallelism overrides). MCP, permission mode, Fast Mode, and
`max_iterations` / `max_tool_calls` still need a restart.

Bad config does not crash a running session (reload keeps the previous
in-memory state), but a broken on-disk file will block the next cold start.

After every `letcode.toml` edit, call `config__validate` (path `null` = default
`~/.config/letcode/letcode.toml`). If `valid` is false, fix from the returned
`error` and validate again until it passes. Prefer minimal diffs.

Outside a session: `letcode config validate [path]` (exit 1 when invalid).

## Where files live

| Scope | Path |
| --- | --- |
| Global config | `~/.config/letcode/letcode.toml` (required; missing file is a hard error) |
| Global skills | `~/.config/letcode/skills/<name>/SKILL.md` |
| Global AGENTS.md | `~/.config/letcode/AGENTS.md` (user-level instructions, if present) |
| Global themes | `~/.config/letcode/themes/<id>.toml` |
| Sessions / logs | under config dir via `global.sessions_dir` / `global.log_file` (defaults `sessions`, `logs/combined.log`) |
| Project skills | `.letcode/skills/<name>/SKILL.md` (also `.opencode/skills`, `.agents/skills`, `.claude/skills` from git root down to cwd) |
| Project AGENTS.md | `AGENTS.md` from repo root through cwd (later files append and take precedence) |

Skill discovery also scans `~/.config/opencode/skills`, `~/.agents/skills`, and
`~/.claude/skills`. Same skill name: later roots replace earlier ones.

## letcode.toml

Unknown top-level keys are rejected. At least one `[providers.<name>]` with at
least one model is required. `active_provider` defaults to the first provider key.

```toml
active_provider = "openai"
fast_mode = false

[global]
# max_iterations = 64
# max_tool_calls = 128
# tool_timeout_secs = 60
sessions_dir = "sessions"
log_file = "logs/combined.log"

[global.compaction]
# preserve_recent_tokens = 12000

[global.retry]
# enabled = true
# max_attempts = 50
# max_recovery_attempts = 3
# initial_delay_secs = 1
# backoff_multiplier = 2.0
# jitter_secs = 1

[permissions]
mode = "default" # safe | default | auto | yolo (solo is accepted as alias of yolo)

# Optional expert routes. Provider may be omitted to follow active_provider.
# [agents.explorer]
# provider = "openai"
# model = "gpt-5.5"
# Same keys for: fixer, oracle, designer, librarian, general, reviewer

[tools.parallelism]
# "fs__read" = "parallel"   # only narrow tools that already declare Parallel
# "web__fetch" = "exclusive"

[mcp.example_local]
type = "local"
command = ["npx", "-y", "some-mcp"]
# environment = { FOO = "bar" }   # alias: env
# enabled = true
# timeout = 5000                  # milliseconds

[mcp.example_remote]
type = "remote"
url = "https://example.com/mcp"
# headers = { Authorization = "Bearer ..." }
# enabled = true
# timeout = 10000
# oauth = false                   # true is rejected (OAuth not supported yet)

[providers.openai]
api_key = "YOUR_API_KEY"
base_url = "https://api.openai.com/v1"
protocol = "responses" # responses | completions (required unless provider name is openai)
default_model = "gpt-5.5"
# [providers.openai.retry]        # optional per-provider retry override

[providers.openai.models."gpt-5.5"]
display_name = "GPT-5.5"          # alias: name
# protocol = "completions"        # optional model override of provider protocol
# context_window = 400000
# effective_input_limit_tokens = 256000
# max_output_tokens = 128000
supports_tools = true             # omitted defaults to true
parallel_tool_calls = false
supports_reasoning = true         # omitted defaults to true
reasoning_effort = "medium"       # none|minimal|low|medium|high|xhigh|max|custom string
reasoning_efforts = ["none", "low", "medium", "high", "max"]
reasoning_summary = "auto"        # auto|concise|detailed
text_verbosity = "medium"         # low|medium|high
# temperature = 0.2
# top_p = 1.0

# [providers.openai.models."gpt-5.5".prompt_cache]
# enabled = true
# retention = "in_memory"         # in_memory | 24h
# namespace = "my-cache"
```

Shape notes:

- Relative `sessions_dir` / `log_file` resolve against the **config file directory**.
- Provider env overrides win over TOML: `<PROVIDER>_API_KEY`, `<PROVIDER>_BASE_URL`
  (provider name uppercased; non-alphanumeric → `_`). Example: `openai` →
  `OPENAI_API_KEY` / `OPENAI_BASE_URL`.
- `protocol` values are kebab-case in TOML: `responses`, `completions`.
- `[tools.parallelism]` may only **narrow** a tool that already declares
  `Parallel` support (e.g. force `exclusive`). Promoting exclusive tools to
  `parallel` is rejected.
- Local MCP `command` is an array of strings, never a single string. `type` is
  required. Remote servers must not set `command`/`environment`; local servers
  must not set `url`/`headers`/`oauth`.
- Built-in expert agent keys are fixed: `explorer`, `fixer`, `oracle`,
  `designer`, `librarian`, `general`, `reviewer`. There is no free-form agent
  map like opencode's `agent: { ... }`.
- `permissions.mode = "auto"` uses the same Ask matrix as `default`, but the
  sticky `reviewer` expert answers approvals.

## Skills

Skill folders contain exactly `SKILL.md` (regular file, not a symlink), with
YAML frontmatter:

```markdown
---
name: my-skill
description: One sentence covering what this skill does AND when to trigger it. Front-load concrete keywords/filenames; gate with "Use ONLY when..." if needed.
---

# My Skill

Instructions, examples, references.
```

- `name`: required, lowercase kebab-case, 1–64 chars, must match the folder name.
- `description`: required; this is the discovery signal shown to the model.
- Optional resource files may live beside `SKILL.md` and are readable via
  `skill__resource_list` / `skill__resource_read`.
- Prefer creating files under `~/.config/letcode/skills/` or
  `.letcode/skills/` over inventing config keys — letcode has no
  `skills.paths` / `skills.urls` config surface.
- A disk skill named `customize-letcode` overrides this built-in skill.

## AGENTS.md

letcode loads instruction markdown into the agent prompt via
`Agent::load_instruction_files_from`:

1. Global file: `<config_dir>/AGENTS.md` (typically `~/.config/letcode/AGENTS.md`).
2. Workspace chain: from the git root (or cwd if no `.git`) through the current
   directory, every `AGENTS.md` is appended after the global file. Deeper files
   are appended later so they take precedence for conflicts. Missing files are
   skipped; reloads are idempotent.

When editing these files for letcode behavior, keep them concise, actionable,
and free of secrets.

## Themes

Custom TUI themes are TOML files at `~/.config/letcode/themes/<id>.toml`.
Bundled ids such as `ocean`, `forest`, `rose`, `tokyonight` are seeded if
missing; reserved built-in theme ids (e.g. `dark`) cannot be shadowed by a
custom file of the same name.

Minimal theme file:

```toml
label = "Sunset"
description = "Warm accent"
accent = "#ff6600"
```

Unset color keys fall back to the dark palette. Select themes in TUI via
`/theme`.

## Escape hatches

- Missing config: create `~/.config/letcode/letcode.toml` (startup fails with a
  path hint if absent).
- Provider credentials: prefer env vars (`<PROVIDER>_API_KEY`) over committing
  keys into TOML.
- Tracing (optional): `LETCODE_LANGFUSE_ENABLED=true` plus Langfuse env keys;
  missing credentials keep tracing off.

## When proposing edits

- Preserve existing providers, models, MCP servers, and agent routes the user
  did not ask to change.
- Do not invent opencode-only keys (`$schema`, `plugin`, `command`,
  `permission.bash` pattern maps, `skills.urls`, free-form `agent` objects).
- Prefer new skill files under the correct skills directory over stuffing
  long instructions into `letcode.toml` (TOML has no skill body field).
- After writing `letcode.toml`, always run `config__validate` and keep editing
  until `valid` is true. Restart only for MCP / permissions / Fast Mode /
  max iteration/tool caps.
