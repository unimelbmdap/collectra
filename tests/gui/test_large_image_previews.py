import base64
import io
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import pytest

from collectra.display import DisplayContext


def decode(source):
    return Image.open(io.BytesIO(base64.b64decode(source.split(",", 1)[1])))


@pytest.mark.parametrize("extension", ["jpg", "png", "tif"])
@pytest.mark.parametrize("orientation", ["north", "west"])
def test_previews_are_bounded_and_editing_keeps_full_resolution(tmp_path, monkeypatch, extension, orientation):
    from collectra.displays import preview
    from collectra.gui.segmentation import paint_mask

    monkeypatch.setattr(preview, "MAX_PREVIEW_EDGE", 64)
    image_path = tmp_path / f"page.{extension}"
    Image.new("RGB", (200, 100), "white").save(image_path)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[20:80, 40:160] = 1
    Image.fromarray(mask).save(tmp_path / "mask.png")
    records = {
        "page": {"type": "collectra.Image", "id": "page", "data": image_path.name, "orientation": orientation},
        "region": {"type": "collectra.ImageSegmentation", "id": "region", "data": image_path.name,
                   "mask": "mask.png", "parents": ["page"], "orientation": orientation},
    }
    image_bytes = image_path.read_bytes()
    mask_bytes = (tmp_path / "mask.png").read_bytes()
    context = DisplayContext(records, tmp_path)
    view = context.artefact("page").display(context)
    assert max(decode(view["source"]).size) == 64
    annotation, = view["annotations"]
    assert max(decode(annotation["mask_source"]).size) <= 64
    assert annotation["mask_region"] == {"x": .2, "y": .2, "width": .6, "height": .6}
    region = context.artefact("region")
    view = region.display(context)
    segmentation = view["segmentation"]
    expected = (200, 100) if orientation == "north" else (100, 200)
    assert (segmentation["width"], segmentation["height"]) == expected
    mask_preview = decode(segmentation["mask_source"])
    assert mask_preview.size == decode(view["source"]).size
    assert mask_preview.size == (segmentation["preview_width"], segmentation["preview_height"])
    assert image_path.read_bytes() == image_bytes
    assert (tmp_path / "mask.png").read_bytes() == mask_bytes
    assert paint_mask(region, [{"tool": "brush", "radius": 2, "points": [[10, 10]]}], orientation)
    with Image.open(tmp_path / "mask.png") as saved:
        assert saved.size == (200, 100)
        assert saved.rotate(region.orientation.to_degree(), expand=True).getpixel((10, 10)) == 1
    assert image_path.read_bytes() == image_bytes


def test_browser_brush_coordinates_and_canvas_use_separate_resolutions():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for the browser coordinate test")
    directory = Path(__file__).resolve().parents[2] / "collectra/gui/displays"
    script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const buttons = [], scales = [], saved = [];
function element() { return {style:{}, addEventListener(){}, removeEventListener(){},
    setAttribute(){}, append(){}, appendChild(){}, remove(){},
    getBoundingClientRect(){return {left:0,top:0}}}; }
const drawing = {clearRect(){}, fillRect(){}, drawImage(){}, beginPath(){}, arc(){},
    fill(){}, moveTo(){}, lineTo(){}, stroke(){}, save(){}, restore(){},
    scale(x,y){scales.push([x,y])}};
let canvas;
global.document = {createElement(type){const e=element();if(type==='canvas'){canvas=e;e.getContext=()=>drawing;}return e;},
    createTextNode(){return {};},querySelector(){return null;},activeElement:null,
    addEventListener(){},removeEventListener(){}};
global.getComputedStyle=()=>({getPropertyValue:()=>''});
global.Image=class{decode(){return Promise.resolve();}};
global.OpenSeadragon = {Point:class {constructor(x,y){this.x=x;this.y=y;}},
    ControlAnchor:{TOP_LEFT:0,BOTTOM_LEFT:1},Button:class {
        constructor(args){buttons.push(args);this.element=element();this.imgRest={};this.imgGroup={};this.imgHover={};this.imgDown={};}
        destroy(){}
    }};
const size={width:640,height:480,preview_width:64,preview_height:48,orientation:'north',
    mask_source:'data:',bounds:{x:320,y:240,w:100,h:100}};
global.window={addEventListener(){},removeEventListener(){},pywebview:{api:{
    async update_segmentation_mask(id,strokes,path,orientation){saved.push({id,strokes,path,orientation});
        return {success:true,segmentation:size};}}}};
const surface=element();surface.listeners={};surface.addEventListener=(name,handler)=>surface.listeners[name]=handler;
surface.setPointerCapture=()=>{};surface.hasPointerCapture=()=>false;
const item={getContentSize:()=>({x:64,y:48}), imageToViewportRectangle:(...args)=>args,
    imageToViewportCoordinates:(x,y)=>({x,y}), viewportToImageCoordinates:p=>p};
const viewer={canvas:surface,container:surface,world:{getItemAt:()=>item},
    viewport:{pixelFromPoint:p=>p,pointFromPixel:p=>p},gestureSettingsMouse:{},
    addOverlay(){},addControl(){},addHandler(){},removeOverlay(){},removeControl(){},removeHandler(){}};
const token={};let zoom;
const controller={pageViewer:viewer,_artefactToken:token,syncDrawBoxButton(){},zoomToBoxGeometry:g=>zoom=g};
vm.runInThisContext(fs.readFileSync(process.argv[1]+'/imageSegmentation.js','utf8'));
vm.runInThisContext(fs.readFileSync(process.argv[1]+'/segmentationEditor.js','utf8'));
(async()=>{
    const editor=await mountSegmentationEditor(controller,{segmentation:size,target_id:'region'}, {token,resultsPath:'results.yaml'});
    assert.deepStrictEqual([canvas.width,canvas.height],[64,48]);
    assert.deepStrictEqual(zoom,{x:32,y:24,w:10,h:10});
    buttons[0].onClick();
    const event={button:0,clientX:32,clientY:24,pointerId:1,preventDefault(){},stopImmediatePropagation(){}};
    surface.listeners.pointerdown(event);surface.listeners.pointerup(event);
    await new Promise(resolve=>setImmediate(resolve));
    assert.deepStrictEqual(saved[0].strokes[0].points,[[320,240]]);
    assert.strictEqual(saved[0].strokes[0].radius,10);
    assert.deepStrictEqual(scales,[[.1,.1]]);
    editor.dispose();assert.strictEqual(canvas.width,0);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    result = subprocess.run([node, "-e", script, str(directory)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_preview_cache_and_mask_refresh_respect_session_limit(tmp_path):
    from types import SimpleNamespace
    import yaml
    from collectra.gui.backend import GUIBackend

    Image.new("RGB", (200, 100), "white").save(tmp_path / "page.jpg")
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[20:80, 40:160] = 1
    Image.fromarray(mask).save(tmp_path / "mask.png")
    records = {
        "page": {"type": "collectra.Image", "id": "page", "data": "page.jpg"},
        "region": {"type": "collectra.ImageSegmentation", "id": "region", "data": "page.jpg",
                   "mask": "mask.png", "parents": ["page"]},
    }
    results = tmp_path / "results.yaml"
    results.write_text(yaml.safe_dump(records))
    for limit in (64, 32, 64):
        api = GUIBackend(SimpleNamespace(ext="collectra"), max_pixels=limit)
        assert api.load_yaml(str(results))["success"]
        view = api.get_artefact_display("page")["view"]
        assert max(decode(view["source"]).size) == limit
        assert max(decode(view["annotations"][0]["mask_source"]).size) == limit
        view = api.get_artefact_display("region")["view"]
        assert max(decode(view["segmentation"]["mask_source"]).size) == limit
        response = api.update_segmentation_mask("region", [{"tool": "brush", "radius": 1, "points": [[10, 10]]}], str(results), "north")
        assert response["success"], response
        assert max(decode(response["segmentation"]["mask_source"]).size) == limit
        assert Image.open(tmp_path / "mask.png").size == (200, 100)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_preview_limits_are_rejected(tmp_path, limit):
    from types import SimpleNamespace
    from collectra.gui.backend import GUIBackend

    with pytest.raises(ValueError, match="positive integer"):
        GUIBackend(SimpleNamespace(ext="collectra"), max_pixels=limit)
    with pytest.raises(ValueError, match="positive integer"):
        DisplayContext({}, tmp_path, max_pixels=limit)
