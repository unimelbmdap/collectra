def normalise_items(items: str | dict | list[str | dict] | None) -> list[str | dict]:
    """Convert items field to list (handles single string or list)."""
    if items is None:
        return []
    if isinstance(items, list):
        return items
    return [items]
