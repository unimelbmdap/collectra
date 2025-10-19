from collectra import Text, Image, NodeStatus


def test_text_node():
    text = Text("text", "Hello, World!")
    assert text() == "Hello, World!"    
    text = Text("empty", "")    


def test_image_node(raw_img_path):
    image = Image("image", raw_img_path)    
    assert image() == str(raw_img_path)
