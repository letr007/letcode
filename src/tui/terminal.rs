use std::io::{self, Stdout};

use crossterm::cursor::{Hide, Show};
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, SetTitle, disable_raw_mode, enable_raw_mode,
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io::Write;

pub type TuiTerminal = Terminal<CrosstermBackend<Stdout>>;

const RAW_MODE_BIT: u8 = 0b001;
const ALT_SCREEN_BIT: u8 = 0b010;
const MOUSE_CAPTURE_BIT: u8 = 0b100;
const KEYBOARD_ENHANCEMENT_BIT: u8 = 0b1000;
const BRACKETED_PASTE_BIT: u8 = 0b1_0000;

/// RAII guard for TUI terminal ownership.
///
/// Once entered, callers should avoid any direct stdout writes until the guard is dropped.
/// This keeps raw-mode and ratatui rendering ownership centralized and avoids screen
/// corruption from concurrent output.
#[derive(Debug)]
pub struct TerminalGuard {
    init_bits: u8,
}

impl TerminalGuard {
    pub fn enter() -> io::Result<Self> {
        let mut guard = Self { init_bits: 0 };

        enable_raw_mode()?;
        guard.init_bits |= RAW_MODE_BIT;

        crossterm::execute!(io::stdout(), EnterAlternateScreen)?;
        guard.init_bits |= ALT_SCREEN_BIT;

        crossterm::execute!(
            io::stdout(),
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )?;
        guard.init_bits |= KEYBOARD_ENHANCEMENT_BIT;

        crossterm::execute!(io::stdout(), EnableMouseCapture)?;
        guard.init_bits |= MOUSE_CAPTURE_BIT;

        crossterm::execute!(io::stdout(), EnableBracketedPaste)?;
        guard.init_bits |= BRACKETED_PASTE_BIT;

        // 启用鼠标拖拽/移动跟踪（SGR 1003 模式）
        // EnableMouseCapture 只启用基本点击，不包括拖拽事件
        write!(io::stdout(), "\x1b[?1003h")?;
        io::stdout().flush()?;

        crossterm::execute!(io::stdout(), Hide)?;

        Ok(guard)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = crossterm::execute!(io::stdout(), Show);

        if self.init_bits & MOUSE_CAPTURE_BIT != 0 {
            // 禁用鼠标移动跟踪
            let _ = write!(io::stdout(), "\x1b[?1003l");
            let _ = io::stdout().flush();
            let _ = crossterm::execute!(io::stdout(), DisableMouseCapture);
        }

        if self.init_bits & BRACKETED_PASTE_BIT != 0 {
            let _ = crossterm::execute!(io::stdout(), DisableBracketedPaste);
        }

        if self.init_bits & KEYBOARD_ENHANCEMENT_BIT != 0 {
            let _ = crossterm::execute!(io::stdout(), PopKeyboardEnhancementFlags);
        }

        if self.init_bits & ALT_SCREEN_BIT != 0 {
            let _ = crossterm::execute!(io::stdout(), LeaveAlternateScreen);
        }

        if self.init_bits & RAW_MODE_BIT != 0 {
            let _ = disable_raw_mode();
        }
    }
}

#[derive(Debug)]
pub struct OwnedTerminal {
    terminal: TuiTerminal,
    last_title: Option<String>,
    _guard: TerminalGuard,
}

impl OwnedTerminal {
    pub fn new() -> io::Result<Self> {
        let guard = TerminalGuard::enter()?;
        let backend = CrosstermBackend::new(io::stdout());
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;

        Ok(Self {
            terminal,
            last_title: None,
            _guard: guard,
        })
    }

    pub fn terminal_mut(&mut self) -> &mut TuiTerminal {
        &mut self.terminal
    }

    pub fn set_title(&mut self, title: &str) -> io::Result<()> {
        let title = sanitize_title(title);
        if self.last_title.as_deref() == Some(title.as_str()) {
            return Ok(());
        }

        let backend = self.terminal.backend_mut();
        crossterm::execute!(backend, SetTitle(&title))?;
        self.last_title = Some(title);
        Ok(())
    }
}

fn sanitize_title(title: &str) -> String {
    title
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
}
