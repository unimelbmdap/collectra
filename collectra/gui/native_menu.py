"""Native menu entries delegate to the same actions as the viewer toolbar."""

import json
import platform

from webview.menu import Menu, MenuAction, MenuSeparator


def put_file_menu_first(menu):
    """Keep the app menu at index zero, then File before Cocoa's Edit/View."""
    if menu is None:
        return
    for index in range(1, menu.numberOfItems()):
        item = menu.itemAtIndex_(index)
        submenu = item.submenu()
        if submenu is not None and submenu.title() == "File":
            if index != 1:
                menu.removeItemAtIndex_(index)
                menu.insertItem_atIndex_(item, 1)
            return


def arrange_native_menu():
    if platform.system() != "Darwin":
        return
    from AppKit import NSApplication
    from PyObjCTools import AppHelper

    # pywebview's startup callback runs on a worker; AppKit requires the UI thread.
    AppHelper.callAfter(lambda: put_file_menu_first(NSApplication.sharedApplication().mainMenu()))


def build_native_menu(window):
    def action(title, name):
        return MenuAction(title, lambda: window.evaluate_js(
            f"runNativeMenuAction({json.dumps(name)})"
        ))

    return [
        Menu("File", [
            action("Open Results…", "open"),
            action("Show Current Results in File Manager…", "reveal"),
            MenuSeparator(),
            action("Close Current Tab", "close-tab"),
        ]),
        Menu("Navigate", [
            action("Previous File", "previous-file"),
            action("Next File", "next-file"),
            MenuSeparator(),
            action("Find File…", "find-file"),
            action("Find Artefacts…", "find-artefacts"),
            action("Previous Search Match", "previous-match"),
            action("Next Search Match", "next-match"),
        ]),
        Menu("Panels", [
            action("Toggle Sidebar", "sidebar"),
            action("Open Pipeline Graph", "graph"),
            action("Show Artefacts", "artefacts"),
            action("Show Tasks", "tasks"),
            MenuSeparator(),
            action("Fit Pipeline Graph", "fit-graph"),
            action("Fit Image", "fit-image"),
            MenuSeparator(),
            Menu("Split Panel", [action(f"Split {direction.title()}", f"split-{direction}")
                                 for direction in ("left", "right", "above", "below")]),
            action("Toggle Current Panel", "collapse-panel"),
            action("Remove Current Panel", "remove-panel"),
        ]),
        Menu("Layouts", [
            action("Choose Layout…", "choose-layout"),
            action("Save Current Layout…", "save-layout"),
        ]),
        Menu("Tools", [
            action("Check for Updates…", "check-updates"),
            MenuSeparator(),
            action("Source Control…", "source-control"),
            action("Statistics…", "statistics"),
            action("Load Details…", "load-details"),
        ]),
    ]
