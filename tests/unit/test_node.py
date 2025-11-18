from collectra import Text, Image, NodeStatus


def test_text():
    text = Text("text", data="Hello, World!")
    assert text() == "Hello, World!"
    text = Text("empty", "")


def test_image(raw_img_path):
    image = Image("image", data=raw_img_path)
    assert image() == str(raw_img_path)
