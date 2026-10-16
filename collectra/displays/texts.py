"""Text format selection and editor descriptors."""


def text(context, item):
    path = context.text_path(item.id)
    format = {".md": "markdown", ".markdown": "markdown", ".xml": "xml"}.get(
        path.suffix.lower() if path else "", "plain"
    )
    return {
        "kind": "text",
        "format": format,
        "text": str(item.data),
        "editable": True,
        "target_id": item.id,
    }
