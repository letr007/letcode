use super::*;

#[test]
fn localized_handbooks_cover_visible_commands() {
    for language in [Language::En, Language::ZhCn] {
        let handbook = handbook(language);
        assert!(!handbook.contains("{{commands}}"));
        let chapters = chapters(&handbook);
        for command in command_metadata()
            .iter()
            .filter(|command| command.visible_in_help)
        {
            assert!(chapters.iter().any(|chapter| {
                chapter
                    .markdown
                    .contains(&format!("### `{}`", command.name))
                    && chapter.markdown.contains(command.usage)
            }));
        }
    }
}

#[test]
fn chapter_headings_inside_code_fences_are_not_navigation_entries() {
    let chapters = chapters("# Guide\n\n## Start\n\n```text\n## Example\n```\n\n## Next\n\nBody\n");
    assert_eq!(chapters.len(), 2);
    assert_eq!(chapters[0].title, "Start");
    assert!(chapters[0].markdown.contains("## Example"));
    assert_eq!(chapters[1].title, "Next");
}

#[test]
fn chapter_navigation_resets_scroll_and_stays_in_bounds() {
    let mut help = HelpState::new(Language::En);
    help.prepare(40, 4, Theme::dark(), Language::En);
    help.apply(HelpAction::Bottom);
    assert!(help.scroll > 0);
    help.apply(HelpAction::NextChapter);
    assert_eq!(help.selected, 1);
    assert_eq!(help.scroll, 0);
    help.apply(HelpAction::SelectChapter(help.chapters.len() - 1));
    assert_eq!(help.selected, help.chapters.len() - 1);
    help.apply(HelpAction::NextChapter);
    assert_eq!(help.selected, help.chapters.len() - 1);
    help.apply(HelpAction::SelectChapter(0));
    help.apply(HelpAction::PreviousChapter);
    assert_eq!(help.selected, 0);
}

#[test]
fn paging_and_resize_keep_the_reading_offset_valid() {
    let mut help = HelpState::new(Language::En);
    help.apply(HelpAction::SelectChapter(1));
    help.prepare(32, 5, Theme::dark(), Language::En);
    help.apply(HelpAction::PageDown);
    assert_eq!(help.scroll, 4);
    help.apply(HelpAction::PageUp);
    assert_eq!(help.scroll, 0);
    help.apply(HelpAction::ScrollDown(usize::MAX));
    assert_eq!(help.scroll, help.scroll_max);
    help.prepare(100, 500, Theme::dark(), Language::En);
    assert_eq!(help.scroll, 0);
    assert_eq!(help.scroll_max, 0);
    assert!(help.visible_lines().len() <= 500);
}
