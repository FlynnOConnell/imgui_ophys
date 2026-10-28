"""
Small imgui widgets the pages share: section headings, a (?) with a tooltip, key / meaning tables, and operators
between pictures.
"""

from imgui_bundle import imgui

from imgui_ophys.plotting import ACCENT, DIM

KEY = imgui.ImVec4(1.00, 0.80, 0.20, 1.0)
CARD = imgui.ImVec4(0.17, 0.18, 0.21, 1.0)
CARD_HOVER = imgui.ImVec4(0.21, 0.22, 0.26, 1.0)
EDGE = imgui.ImVec4(0.35, 0.35, 0.37, 1.0)


def u32(color: imgui.ImVec4, alpha: float = 1.0) -> int:
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(color.x, color.y, color.z, alpha))


def section(title: str) -> None:
    """Accent heading with a rule under it."""
    imgui.dummy(imgui.ImVec2(0, imgui.get_font_size() * 0.3))
    imgui.text_colored(ACCENT, title)
    imgui.separator()
    imgui.dummy(imgui.ImVec2(0, imgui.get_font_size() * 0.2))


def help_mark(text: str) -> None:
    """A dim (?) after the last item, on its line, with text as its wrapped tooltip."""
    imgui.same_line(0, imgui.get_font_size() * 0.3)
    imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 30)
        imgui.text_unformatted(text)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()


def push_size(times: float) -> None:
    """Push the current font at times the current text size; pop with imgui.pop_font()."""
    # push_font takes a size before the main font scale, which get_font_size already includes
    imgui.push_font(None, times * imgui.get_font_size() / imgui.get_style().font_scale_main)


def operator(symbol: str, height: float) -> None:
    """symbol centered in a slot as tall as the pictures beside it, then stay on the line."""
    em = imgui.get_font_size()
    imgui.same_line(0, 0.4 * em)
    imgui.begin_group()
    imgui.dummy(imgui.ImVec2(0, (height - 1.4 * em) / 2))
    push_size(1.4)
    imgui.text_colored(DIM, symbol)
    imgui.pop_font()
    imgui.end_group()
    imgui.same_line(0, 0.4 * em)


def table(name: str, rows, width: float = 0.0) -> None:
    """Two columns: a key in KEY and its meaning, wrapped. width 0 takes the whole line."""
    em = imgui.get_font_size()
    flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h
    if not imgui.begin_table(f"##{name}", 2, flags, imgui.ImVec2(width, 0)):
        return
    imgui.table_setup_column("key", imgui.TableColumnFlags_.width_fixed, max(imgui.calc_text_size(k).x for k, _ in rows) + em)
    imgui.table_setup_column("meaning")
    for key, meaning in rows:
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text_colored(KEY, key)
        imgui.table_next_column()
        imgui.text_wrapped(meaning)
    imgui.end_table()
