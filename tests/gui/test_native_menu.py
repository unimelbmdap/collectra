import json
from unittest.mock import Mock

from webview.menu import Menu, MenuAction

from collectra.gui.native_menu import build_native_menu, put_file_menu_first


def test_file_precedes_default_edit_and_view_without_replacing_items():
    titles = [
        "Pipeline",
        "Edit",
        "View",
        "File",
        "Navigate",
        "Panels",
        "Layouts",
        "Tools",
    ]
    items = []
    for title in titles:
        item = Mock()
        item.submenu.return_value.title.return_value = title
        items.append(item)
    original = list(items)
    menu = Mock()
    menu.numberOfItems.side_effect = lambda: len(items)
    menu.itemAtIndex_.side_effect = lambda index: items[index]
    menu.removeItemAtIndex_.side_effect = lambda index: items.pop(index)
    menu.insertItem_atIndex_.side_effect = lambda item, index: items.insert(index, item)
    put_file_menu_first(menu)
    assert [item.submenu().title() for item in items] == [
        "Pipeline",
        "File",
        "Edit",
        "View",
        "Navigate",
        "Panels",
        "Layouts",
        "Tools",
    ]
    assert set(items) == set(original)
    put_file_menu_first(menu)
    assert menu.removeItemAtIndex_.call_count == 1


def actions(items):
    for item in items:
        if isinstance(item, Menu):
            yield from actions(item.items)
        elif isinstance(item, MenuAction):
            yield item


def test_native_menus_dispatch_distinct_viewer_actions():
    window = Mock()
    menus = build_native_menu(window)
    assert [menu.title for menu in menus] == [
        "File",
        "Navigate",
        "Panels",
        "Layouts",
        "Tools",
    ]
    names = []
    for item in actions(menus):
        item.function()
        script = window.evaluate_js.call_args.args[0]
        names.append(
            json.loads(script.removeprefix("runNativeMenuAction(").removesuffix(")"))
        )
    assert len(names) == len(set(names))
    assert {
        "open",
        "previous-file",
        "next-file",
        "choose-layout",
        "save-layout",
        "source-control",
        "split-left",
        "fit-image",
    } <= set(names)


def test_tools_menu_calls_version_control_history_and_changes():
    titles = [item.title for item in actions(build_native_menu(Mock()))]
    assert "History and Changes…" in titles
    assert not any("source control" in title.lower() for title in titles)
