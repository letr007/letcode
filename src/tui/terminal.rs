use std::io::{self, Stdout};

use crossterm::cursor::Show;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::{Terminal, backend::CrosstermBackend};

pub type TuiTerminal = Terminal<CrosstermBackend<Stdout>>;

const RAW_MODE_BIT: u8 = 0b001;
const ALT_SCREEN_BIT: u8 = 0b010;

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

        crossterm::execute!(io::stdout(), Show)?;

        Ok(guard)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = crossterm::execute!(io::stdout(), Show);

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
            _guard: guard,
        })
    }

    pub fn terminal(&self) -> &TuiTerminal {
        &self.terminal
    }

    pub fn terminal_mut(&mut self) -> &mut TuiTerminal {
        &mut self.terminal
    }
}
