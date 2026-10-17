use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};
use ratatui::style::Style;

use crate::command::command_metadata;

use super::{
    i18n::{Language, Translator},
    markdown::{MarkdownRenderOptions, render_markdown_document},
    theme::Theme,
    transcript_render::{Document, Line},
};

const EN: &str = include_str!("../../docs/help/en.md");
const ZH_CN: &str = include_str!("../../docs/help/zh-CN.md");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpAction {
    PreviousChapter,
    NextChapter,
    SelectChapter(usize),
    ScrollUp(usize),
    ScrollDown(usize),
    PageUp,
    PageDown,
    Top,
    Bottom,
    ToggleContents,
    FocusContents,
    FocusDocument,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelpChapter {
    pub title: String,
    pub markdown: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelpState {
    pub chapters: Vec<HelpChapter>,
    pub selected: usize,
    pub contents_focused: bool,
    pub scroll: usize,
    pub scroll_max: usize,
    pub viewport_rows: usize,
    language: Language,
    rendered: Document<Style>,
    rendered_width: usize,
    rendered_theme: Option<Theme>,
    rendered_chapter: usize,
}

impl HelpState {
    pub fn new(language: Language) -> Self {
        Self {
            chapters: chapters(&handbook(language)),
            selected: 0,
            contents_focused: false,
            scroll: 0,
            scroll_max: 0,
            viewport_rows: 0,
            language,
            rendered: Document::default(),
            rendered_width: 0,
            rendered_theme: None,
            rendered_chapter: 0,
        }
    }

    pub fn apply(&mut self, action: HelpAction) {
        match action {
            HelpAction::PreviousChapter => self.select(self.selected.saturating_sub(1)),
            HelpAction::NextChapter => self.select(self.selected.saturating_add(1)),
            HelpAction::SelectChapter(index) => self.select(index),
            HelpAction::ScrollUp(rows) => self.scroll = self.scroll.saturating_sub(rows),
            HelpAction::ScrollDown(rows) => {
                self.scroll = self.scroll.saturating_add(rows).min(self.scroll_max);
            }
            HelpAction::PageUp => {
                self.scroll = self
                    .scroll
                    .saturating_sub(self.viewport_rows.saturating_sub(1).max(1));
            }
            HelpAction::PageDown => {
                self.scroll = self
                    .scroll
                    .saturating_add(self.viewport_rows.saturating_sub(1).max(1))
                    .min(self.scroll_max);
            }
            HelpAction::Top => self.scroll = 0,
            HelpAction::Bottom => self.scroll = self.scroll_max,
            HelpAction::ToggleContents => self.contents_focused = !self.contents_focused,
            HelpAction::FocusContents => self.contents_focused = true,
            HelpAction::FocusDocument => self.contents_focused = false,
        }
    }

    fn select(&mut self, index: usize) {
        let index = index.min(self.chapters.len().saturating_sub(1));
        if index != self.selected {
            self.selected = index;
            self.scroll = 0;
            self.scroll_max = 0;
        }
    }

    pub fn prepare(&mut self, width: usize, rows: usize, theme: Theme, language: Language) {
        if self.language != language {
            self.chapters = chapters(&handbook(language));
            self.selected = self.selected.min(self.chapters.len().saturating_sub(1));
            self.language = language;
            self.scroll = 0;
            self.rendered_width = 0;
        }
        let width = width.max(1);
        if self.rendered_width != width
            || self.rendered_theme != Some(theme)
            || self.rendered_chapter != self.selected
        {
            self.rendered = render_markdown_document(
                &self.chapters[self.selected].markdown,
                theme,
                MarkdownRenderOptions::new(width),
            );
            self.rendered_width = width;
            self.rendered_theme = Some(theme);
            self.rendered_chapter = self.selected;
        }
        self.viewport_rows = rows;
        self.scroll_max = self.rendered.lines.len().saturating_sub(rows);
        self.scroll = self.scroll.min(self.scroll_max);
    }

    pub fn line_count(&self) -> usize {
        self.rendered.lines.len()
    }

    pub fn visible_lines(&self) -> &[Line<Style>] {
        let start = self.scroll.min(self.rendered.lines.len());
        let end = start
            .saturating_add(self.viewport_rows)
            .min(self.rendered.lines.len());
        &self.rendered.lines[start..end]
    }
}

pub fn handbook(language: Language) -> String {
    let source = match language {
        Language::En => EN,
        Language::ZhCn => ZH_CN,
    };
    let translator = Translator::new(language);
    let commands = command_metadata()
        .iter()
        .filter(|command| command.visible_in_help)
        .map(|command| {
            format!(
                "### `{}`\n\n{}\n\n`{}`\n\n",
                command.name,
                translator.t(command.description_key),
                command.usage,
            )
        })
        .collect::<String>();
    source.replace("{{commands}}", commands.trim_end())
}

fn chapters(markdown: &str) -> Vec<HelpChapter> {
    let mut headings = Vec::new();
    let mut heading = None;
    for (event, range) in Parser::new(markdown).into_offset_iter() {
        match event {
            Event::Start(Tag::Heading {
                level: HeadingLevel::H2,
                ..
            }) => {
                heading = Some((String::new(), range.start));
            }
            Event::Text(text) | Event::Code(text) => {
                if let Some((title, _)) = heading.as_mut() {
                    title.push_str(&text);
                }
            }
            Event::End(TagEnd::Heading(HeadingLevel::H2)) => {
                headings.push(heading.take().expect("help chapter heading"));
            }
            _ => {}
        }
    }
    headings
        .iter()
        .enumerate()
        .map(|(index, (title, start))| HelpChapter {
            title: title.clone(),
            markdown: markdown[*start
                ..headings
                    .get(index + 1)
                    .map_or(markdown.len(), |(_, start)| *start)]
                .trim_end()
                .to_string(),
        })
        .collect()
}

#[cfg(test)]
#[path = "../../tests/internal/tui/help/tests.rs"]
mod tests;
