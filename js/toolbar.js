// import { w2ui,w2toolbar,w2field,query,w2alert, w2utils,w2confirm} from "https://rawgit.com/vitmalina/w2ui/master/dist/w2ui.es6.min.js"
// import { w2ui,w2toolbar,w2field,query,w2alert, w2utils,w2confirm} from "https://cdn.jsdelivr.net/gh/vitmalina/w2ui@master/dist/w2ui.es6.min.js"

// https://stackoverflow.com/questions/36280818/how-to-convert-file-to-base64-in-javascript
function getBase64(file) {
   var reader = new FileReader();
   reader.readAsDataURL(file);
   reader.onload = function () {
    add_image(reader.result);
    //  console.log(reader.result);
   };
   reader.onerror = function (error) {
     console.log("Error: ", error);
   };
}

function getText(file) {
   var reader = new FileReader();
   reader.readAsText(file);
   reader.onload = function () {
    window.postMessage(["load",reader.result],"*")
    //  console.log(reader.result);
   };
   reader.onerror = function (error) {
     console.log("Error: ", error);
   };
}

document.querySelector("#upload_file").addEventListener("change", (event)=>{
    console.log(event);
    let file = document.querySelector("#upload_file").files[0];
    getBase64(file);
})

document.querySelector("#upload_state").addEventListener("change", (event)=>{
    console.log(event);
    let file = document.querySelector("#upload_state").files[0];
    getText(file);
})

open_setting = function() {
    if (!w2ui.foo) {
        new w2form({
            name: "foo",
            style: "border: 0px; background-color: transparent;",
            fields: [{
                    field: "canvas_width",
                    type: "int",
                    required: true,
                    html: {
                        label: "Canvas Width"
                    }
                },
                {
                    field: "canvas_height",
                    type: "int",
                    required: true,
                    html: {
                        label: "Canvas Height"
                    }
                },
            ],
            record: {
                canvas_width: 1200,
                canvas_height: 600,
            },
            actions: {
                Save() {
                    this.validate();
                    let record = this.getCleanRecord();
                    window.postMessage(["resize",record.canvas_width,record.canvas_height],"*");
                    w2popup.close();
                },
                custom: {
                    text: "Cancel",
                    style: "text-transform: uppercase",
                    onClick(event) {
                        w2popup.close();
                    }
                }
            }
        });
    }
    w2popup.open({
            title: "Form in a Popup",
            body: "<div id='form' style='width: 100%; height: 100%;''></div>",
            style: "padding: 15px 0px 0px 0px",
            width: 500,
            height: 280,
            showMax: true,
            async onToggle(event) {
                await event.complete
                w2ui.foo.resize();
            }
        })
        .then((event) => {
            w2ui.foo.render("#form")
        });
}

var button_lst=["clear", "load", "save", "export", "upload", "selection", "canvas", "eraser", "outpaint", "accept", "cancel", "retry", "prev", "current", "next", "eraser_size_btn", "eraser_size", "resize_selection", "scale", "zoom_in", "zoom_out", "help"];
var upload_button_lst=['clear', 'load', 'save', "upload", 'export', 'outpaint', 'resize_selection', 'help', "setting", "interrogate"];
var resize_button_lst=['clear', 'load', 'save', "upload", 'export', "selection", "canvas", "eraser", 'outpaint', 'resize_selection',"zoom_in", "zoom_out", 'help', "setting", "interrogate"];
var outpaint_button_lst=['clear', 'load', 'save', "canvas", "eraser", "upload", 'export', 'resize_selection', "zoom_in", "zoom_out",'help', "setting", "interrogate", "undo", "redo"];
var outpaint_result_lst=["accept", "cancel", "retry", "prev", "current", "next"];
var outpaint_result_func_lst=["accept", "retry", "prev", "current", "next"];

function check_button(id,text="",checked=true,tooltip="")
{
    return { type: "check",  id: id, text: text, icon: checked?"fa-solid fa-square-check":"fa-regular fa-square", checked: checked, tooltip: tooltip };
}

var toolbar=new w2toolbar({
    box: "#toolbar",
    name: "toolbar",
    tooltip: "top",
    items: [
        { type: "button", id: "clear", text: "Reset", tooltip: "Reset Canvas", icon: "fa-solid fa-rectangle-xmark" },
        { type: "break" },
        { type: "button", id: "load", tooltip: "Load Canvas", icon: "fa-solid fa-file-import" },
        { type: "button", id: "save", tooltip: "Save Canvas", icon: "fa-solid fa-file-export" },
        { type: "button", id: "export", tooltip: "Export Image", icon: "fa-solid fa-floppy-disk" },
        { type: "break" },
        { type: "button", id: "upload", text: "Upload Image", icon: "fa-solid fa-upload" },
        { type: "break" },
        { type: "radio", id: "selection", group: "1", tooltip: "Selection", icon: "fa-solid fa-arrows-up-down-left-right", checked: true },
        { type: "radio", id: "canvas", group: "1", tooltip: "Canvas", icon: "fa-solid fa-image" },
        { type: "radio", id: "eraser", group: "1", tooltip: "Eraser", icon: "fa-solid fa-eraser" },
        { type: "break" },
        { type: "button", id: "outpaint", text: "Outpaint", tooltip: "Run Outpainting", icon: "fa-solid fa-brush" },
        { type: "button", id: "interrogate", text: "Interrogate", tooltip: "Get a prompt with Clip Interrogator ", icon: "fa-solid fa-magnifying-glass" },
        { type: "break" },
        { type: "button", id: "accept", text: "Accept", tooltip: "Accept current result", icon: "fa-solid fa-check", hidden: true, disabled:true,},
        { type: "button", id: "cancel", text: "Cancel", tooltip: "Cancel current outpainting/error", icon: "fa-solid fa-ban", hidden: true},
        { type: "button", id: "retry", text: "Retry", tooltip: "Retry", icon: "fa-solid fa-rotate", hidden: true, disabled:true,},
        { type: "button", id: "prev", tooltip: "Prev Result", icon: "fa-solid fa-caret-left", hidden: true, disabled:true,},
        { type: "html", id: "current", hidden: true, disabled:true,
            async onRefresh(event) {
                await event.complete
                let fragment = query.html(`
                <div class="w2ui-tb-text">
                <div class="w2ui-tb-count">
                    <span>${this.sel_value ?? "1/1"}</span>
                </div> </div>`)
                query(this.box).find("#tb_toolbar_item_current").append(fragment)
            }
        },
        { type: "button", id: "next", tooltip: "Next Result", icon: "fa-solid fa-caret-right", hidden: true,disabled:true,},
        { type: "button", id: "add_image", text: "Add Image", icon: "fa-solid fa-file-circle-plus", hidden: true,disabled:true,},
        { type: "button", id: "delete_image", text: "Delete Image", icon: "fa-solid fa-trash-can", hidden: true,disabled:true,},
        { type: "button", id: "confirm", text: "Confirm", icon: "fa-solid fa-check", hidden: true,disabled:true,},
        { type: "button", id: "cancel_overlay", text: "Cancel", icon: "fa-solid fa-ban", hidden: true,disabled:true,},
        { type: "break" },
        { type: "spacer" },
        { type: "break" },
        { type: "button", id: "eraser_size_btn", tooltip: "Eraser Size", text:"Size", icon: "fa-solid fa-eraser", hidden: true, count: 32},
        { type: "html", id: "eraser_size", hidden: true,
            async onRefresh(event) {
                await event.complete
                // let fragment = query.html(`
                //     <input type="number" size="${this.eraser_size ? this.eraser_size.length:"2"}" style="margin: 0px 3px; padding: 4px;" min="8" max="${this.eraser_max ?? "256"}" value="${this.eraser_size ?? "32"}">
                //     <input type="range" style="margin: 0px 3px; padding: 4px;" min="8" max="${this.eraser_max ?? "256"}" value="${this.eraser_size ?? "32"}">`)
                let fragment = query.html(`
                    <input type="range" style="margin: 0px 3px; padding: 4px;" min="8" max="${this.eraser_max ?? "256"}" value="${this.eraser_size ?? "32"}">
                    `)
                fragment.filter("input").on("change", event => {
                    this.eraser_size = event.target.value;
                    window.overlay.freeDrawingBrush.width=this.eraser_size;
                    this.setCount("eraser_size_btn", event.target.value);
                    window.postMessage(["eraser_size", event.target.value],"*")
                    this.refresh();
                })
                query(this.box).find("#tb_toolbar_item_eraser_size").append(fragment)
            }
        },
        // { type: "button", id: "resize_eraser", tooltip: "Resize Eraser", icon: "fa-solid fa-sliders" },
        { type: "button", id: "resize_selection", text: "Resize Selection", tooltip: "Resize Selection", icon: "fa-solid fa-expand" },
        { type: "break" },
        { type: "html", id: "scale",
            async onRefresh(event) {
                await event.complete
                let fragment = query.html(`
                <div class="">
                <div style="padding: 4px; border: 1px solid silver">
                    <span>${this.scale_value ?? "100%"}</span>
                </div></div>`)
                query(this.box).find("#tb_toolbar_item_scale").append(fragment)
            }
        },
        { type: "button", id: "zoom_in", tooltip: "Zoom In", icon: "fa-solid fa-magnifying-glass-plus" },
        { type: "button", id: "zoom_out", tooltip: "Zoom Out", icon: "fa-solid fa-magnifying-glass-minus" },
        { type: "break" },
        { type: "button", id: "help", tooltip: "Help", icon: "fa-solid fa-circle-info" },
        { type: "new-line"},
        { type: "button", id: "setting", text: "Canvas Setting", tooltip: "Resize Canvas Here", icon: "fa-solid fa-sliders" },
        { type: "break" },
        check_button("enable_history","Enable History:",false, "Enable Canvas History"),
        { type: "button", id: "undo", tooltip: "Undo last erasing/last outpainting", icon: "fa-solid fa-rotate-left", disabled: true },
        { type: "button", id: "redo", tooltip: "Redo", icon: "fa-solid fa-rotate-right", disabled: true },
        { type: "break" },
        check_button("enable_img2img","Enable Img2Img",false),
        // check_button("use_correction","Photometric Correction",false),
        check_button("resize_check","Resize Small Input",true),
        check_button("enable_safety","Enable Safety Checker",true),
        check_button("square_selection","Square Selection Only",false),
        {type: "break"},
        check_button("use_seed","Use Seed:",false),
        { type: "html", id: "seed_val",
            async onRefresh(event) {
                await event.complete
                let fragment = query.html(`
                    <input type="number" style="margin: 0px 3px; padding: 4px; width:100px;" value="${this.config_obj.seed_val ?? "0"}">`)
                fragment.filter("input").on("change", event => {
                    this.config_obj.seed_val = event.target.value;
                    parent.config_obj=this.config_obj;
                    this.refresh();
                })
                query(this.box).find("#tb_toolbar_item_seed_val").append(fragment)
            }
        },
        { type: "button", id: "random_seed", tooltip: "Set a random seed", icon: "fa-solid fa-dice" },
    ],
    onClick(event) {
        switch(event.target){
            case "setting":
                open_setting();
                break;
            case "upload":
                this.upload_mode=true
                document.querySelector("#overlay_container").style.pointerEvents="auto";
                this.click("canvas");
                this.click("selection");
                this.show("confirm","cancel_overlay","add_image","delete_image");
                this.enable("confirm","cancel_overlay","add_image","delete_image");
                this.disable(...upload_button_lst);
                this.disable("undo","redo")
                query("#upload_file").click();
                if(this.upload_tip)
                {
                    this.upload_tip=false;
                    w2utils.notify("Note that only visible images will be added to canvas",{timeout:10000,where:query("#container")})
                }
                break;
            case "resize_selection":
                this.resize_mode=true;
                this.disable(...resize_button_lst);
                this.enable("confirm","cancel_overlay");
                this.show("confirm","cancel_overlay");
                window.postMessage(["resize_selection",""],"*");
                document.querySelector("#overlay_container").style.pointerEvents="auto";
                break;
            case "confirm":
                if(this.upload_mode)
                {
                    export_image();
                }
                else
                {
                    let sel_box=this.selection_box;
                    if(sel_box.width*sel_box.height>512*512)
                    {
                        w2utils.notify("Note that the outpainting will be much slower when the area of selection is larger than 512x512",{timeout:2000,where:query("#container")})
                    }
                    window.postMessage(["resize_selection",sel_box.x,sel_box.y,sel_box.width,sel_box.height],"*");
                }
            case "cancel_overlay":
                end_overlay();
                this.hide("confirm","cancel_overlay","add_image","delete_image");
                if(this.upload_mode){
                    this.enable(...upload_button_lst);
                }
                else
                {
                    this.enable(...resize_button_lst);
                    window.postMessage(["resize_selection","",""],"*");
                    if(event.target=="cancel_overlay")
                    {
                        this.selection_box=this.selection_box_bak;
                    }
                }
                if(this.selection_box)
                {
                    this.setCount("resize_selection",`${Math.floor(this.selection_box.width/8)*8}x${Math.floor(this.selection_box.height/8)*8}`);
                }
                this.disable("confirm","cancel_overlay","add_image","delete_image");
                this.upload_mode=false;
                this.resize_mode=false;
                this.click("selection");
                window.update_undo_redo(window.undo_redo_state.undo, window.undo_redo_state.redo);
                break;
            case "add_image":
                query("#upload_file").click();
                break;
            case "delete_image":
                let active_obj = window.overlay.getActiveObject();
                if(active_obj)
                {
                    window.overlay.remove(active_obj);
                    window.overlay.renderAll();
                }
                else
                {
                    w2utils.notify("You need to select an image first",{error:true,timeout:2000,where:query("#container")})
                }
                break;
            case "load":
                query("#upload_state").click();
                this.selection_box=null;
                this.setCount("resize_selection","");
                break;
            case "next":
            case "prev":
                window.postMessage(["outpaint", "", event.target], "*");
                break;
            case "outpaint":
                this.click("selection");
                this.disable(...outpaint_button_lst);
                this.show(...outpaint_result_lst);
                this.disable("undo","redo");
                if(this.outpaint_tip)
                {
                    this.outpaint_tip=false;
                    w2utils.notify("The canvas stays locked until you accept/cancel current outpainting. You can modify the 'sample number' to get multiple results; you can resize the canvas/selection with 'canvas setting'/'resize selection'; you can use 'photometric correction' to help preserve existing contents",{timeout:15000,where:query("#container")})
                }
                document.querySelector("#container").style.pointerEvents="none";
            case "retry":
                this.disable(...outpaint_result_func_lst);
                parent.config_obj["interrogate_mode"]=false;
                window.postMessage(["transfer",""],"*")
                break;
            case "interrogate":
                if(this.interrogate_tip)
                {
                    this.interrogate_tip=false;
                    w2utils.notify("ClipInterrogator v1 will be dynamically loaded when run at the first time, which may take a while",{timeout:10000,where:query("#container")})
                }
                parent.config_obj["interrogate_mode"]=true;
                window.postMessage(["transfer",""],"*")
                break
            case "accept":
            case "cancel":
                this.hide(...outpaint_result_lst);
                this.disable(...outpaint_result_func_lst);
                this.enable(...outpaint_button_lst);
                document.querySelector("#container").style.pointerEvents="auto";
                if(this.config_obj.enable_history)
                {
                    window.postMessage(["click", event.target, ""],"*");
                }
                else
                {
                    window.postMessage(["click", event.target],"*");
                }
                let app=parent.document.querySelector("gradio-app");
                app=app.shadowRoot??app;
                app.querySelector("#cancel").click();
                window.update_undo_redo(window.undo_redo_state.undo, window.undo_redo_state.redo);
                break;
            case "eraser":
            case "selection":
            case "canvas":
                if(event.target=="eraser")
                {
                    this.show("eraser_size","eraser_size_btn");
                    window.overlay.freeDrawingBrush.width=this.eraser_size;
                    window.overlay.isDrawingMode = true;
                }
                else
                {
                    this.hide("eraser_size","eraser_size_btn");
                    window.overlay.isDrawingMode = false;
                }
                if(this.upload_mode)
                {
                    if(event.target=="canvas")
                    {
                        window.postMessage(["mode", event.target],"*")
                        document.querySelector("#overlay_container").style.pointerEvents="none";
                        document.querySelector("#overlay_container").style.opacity = 0.5;
                    }
                    else
                    {
                        document.querySelector("#overlay_container").style.pointerEvents="auto";
                        document.querySelector("#overlay_container").style.opacity = 1.0;
                    }
                }
                else
                {
                    window.postMessage(["mode", event.target],"*")
                }
                break;
            case "help":
                w2popup.open({
                    title: "Document",
                    body: "Usage: <a href='https://github.com/lkwq007/stablediffusion-infinity/blob/master/docs/usage.md'  target='_blank'>https://github.com/lkwq007/stablediffusion-infinity/blob/master/docs/usage.md</a>"
                })
                break;
            case "clear":
                w2confirm("Reset canvas?").yes(() => {
                    window.postMessage(["click", event.target],"*");
                }).no(() => {})
                break;
            case "random_seed":
                this.config_obj.seed_val=Math.floor(Math.random() * 3000000000);
                parent.config_obj=this.config_obj;
                this.refresh();
                break;
            case "enable_history":
            case "enable_img2img":
            case "use_correction":
            case "resize_check":
            case "enable_safety":
            case "use_seed":
            case "square_selection":
                let target=this.get(event.target);
                if(event.target=="enable_history")
                {
                    if(!target.checked)
                    {
                        w2utils.notify("Enable canvas history might increase resource usage / slow down the canvas ", {error:true,timeout:3000,where:query("#container")})
                        window.postMessage(["click","history"],"*");
                    }
                    else
                    {
                        window.undo_redo_state.undo=false;
                        window.undo_redo_state.redo=false;
                        this.disable("undo","redo");
                    }
                }
                target.icon=target.checked?"fa-regular fa-square":"fa-solid fa-square-check";
                this.config_obj[event.target]=!target.checked;
                parent.config_obj=this.config_obj;
                this.refresh();
                break;
            case "save":
            case "export":
                ask_filename(event.target);
                break;
            default:
                // clear, save, export, outpaint, retry
                // break, save, export, accept, retry, outpaint
                window.postMessage(["click", event.target],"*")
        }
        console.log("Target: "+ event.target, event)
    }
})
window.w2ui=w2ui;
w2ui.toolbar.config_obj={
    resize_check: true,
    enable_safety: true,
    use_correction: false,
    enable_img2img: false,
    use_seed: false,
    seed_val: 0,
    square_selection: false,
    enable_history: false,
};
w2ui.toolbar.outpaint_tip=true;
w2ui.toolbar.upload_tip=true;
w2ui.toolbar.interrogate_tip=true;
window.update_count=function(cur,total){
  w2ui.toolbar.sel_value=`${cur}/${total}`;
  w2ui.toolbar.refresh();
}
window.update_eraser=function(val,max_val){
  w2ui.toolbar.eraser_size=`${val}`;
  w2ui.toolbar.eraser_max=`${max_val}`;
  w2ui.toolbar.setCount("eraser_size_btn", `${val}`);
  w2ui.toolbar.refresh();
}
window.update_scale=function(val){
  w2ui.toolbar.scale_value=`${val}`;
  w2ui.toolbar.refresh();
}
window.enable_result_lst=function(){
  w2ui.toolbar.enable(...outpaint_result_lst);
}
function onObjectScaled(e)
{
    let object = e.target;
    if(object.isType("rect"))
    {
        let width=object.getScaledWidth();
        let height=object.getScaledHeight();
        object.scale(1);
        width=Math.max(Math.min(width,window.overlay.width-object.left),256);
        height=Math.max(Math.min(height,window.overlay.height-object.top),256);
        let l=Math.max(Math.min(object.left,window.overlay.width-width-object.strokeWidth),0);
        let t=Math.max(Math.min(object.top,window.overlay.height-height-object.strokeWidth),0);
        if(window.w2ui.toolbar.config_obj.square_selection)
        {
            let max_val = Math.min(Math.max(width,height),window.overlay.width,window.overlay.height);
            width=max_val;
            height=max_val;
        }
        object.set({ width: width, height: height, left:l,top:t})
        window.w2ui.toolbar.selection_box={width: width, height: height, x:object.left, y:object.top};
        window.w2ui.toolbar.setCount("resize_selection",`${Math.floor(width/8)*8}x${Math.floor(height/8)*8}`);
        window.w2ui.toolbar.refresh();
    }
}
function onObjectMoved(e)
{
    let object = e.target;
    if(object.isType("rect"))
    {   
        let l=Math.max(Math.min(object.left,window.overlay.width-object.width-object.strokeWidth),0);
        let t=Math.max(Math.min(object.top,window.overlay.height-object.height-object.strokeWidth),0);
        object.set({left:l,top:t});
        window.w2ui.toolbar.selection_box={width: object.width, height: object.height, x:object.left, y:object.top};
    }
}
window.setup_overlay=function(width,height)
{
    if(window.overlay)
    {
        window.overlay.setDimensions({width:width,height:height});
        let app=parent.document.querySelector("gradio-app");
        app=app.shadowRoot??app;
        app.querySelector("#sdinfframe").style.height=80+Number(height)+"px";
        document.querySelector("#container").style.height= height+"px";
        document.querySelector("#container").style.width = width+"px";
    }
    else
    {
        canvas=new fabric.Canvas("overlay_canvas");
        canvas.setDimensions({width:width,height:height});
        let app=parent.document.querySelector("gradio-app");
        app=app.shadowRoot??app;
        app.querySelector("#sdinfframe").style.height=80+Number(height)+"px";
        canvas.freeDrawingBrush = new fabric.EraserBrush(canvas);
        canvas.on("object:scaling", onObjectScaled);
        canvas.on("object:moving", onObjectMoved);
        window.overlay=canvas;
    }
    document.querySelector("#overlay_container").style.pointerEvents="none";
}
window.update_overlay=function(width,height)
{
    window.overlay.setDimensions({width:width,height:height},{backstoreOnly:true});
    // document.querySelector("#overlay_container").style.pointerEvents="none";
}
window.adjust_selection=function(x,y,width,height)
{
    var rect = new fabric.Rect({
        left: x,
        top: y,
        fill: "rgba(0,0,0,0)",
        strokeWidth: 3, 
        stroke: "rgba(0,0,0,0.7)",
        cornerColor: "red",
        cornerStrokeColor: "red",
        borderColor: "rgba(255, 0, 0, 1.0)",
        width: width,
        height: height,
        lockRotation: true,
    });
    rect.setControlsVisibility({ mtr: false });
    window.overlay.add(rect);
    window.overlay.setActiveObject(window.overlay.item(0));
    window.w2ui.toolbar.selection_box={width: width, height: height, x:x, y:y};
    window.w2ui.toolbar.selection_box_bak={width: width, height: height, x:x, y:y};
}
function add_image(url)
{
    fabric.Image.fromURL(url,function(img){
        window.overlay.add(img);
        window.overlay.setActiveObject(img);
    },{left:100,top:100});
}
function export_image()
{
    data=window.overlay.toDataURL();
    document.querySelector("#upload_content").value=data;
    if(window.w2ui.toolbar.config_obj.enable_history)
    {
        window.postMessage(["upload","",""],"*");
        window.w2ui.toolbar.enable("undo");
        window.w2ui.toolbar.disable("redo");
    }
    else
    {
        window.postMessage(["upload",""],"*");
    }
    end_overlay();
}
function end_overlay()
{
    window.overlay.clear();
    document.querySelector("#overlay_container").style.opacity = 1.0;
    document.querySelector("#overlay_container").style.pointerEvents="none";
}
function ask_filename(target)
{
    w2prompt({
        label: "Enter filename",
        value: `outpaint_${((new Date(Date.now() -(new Date()).getTimezoneOffset() * 60000))).toISOString().replace("T","_").replace(/[^0-9_]/g, "").substring(0,15)}`,
    })
    .change((event) => {
        console.log("change", event.detail.originalEvent.target.value);
    })
    .ok((event) => {
        console.log("value=", event.detail.value);
        window.postMessage(["click",target,event.detail.value],"*");
    })
    .cancel((event) => {
        console.log("cancel");
    });
}

document.querySelector("#container").addEventListener("wheel",(e)=>{e.preventDefault()})
window.setup_shortcut=function(json)
{
    var config=JSON.parse(json);
    var key_map={};
    Object.keys(config.shortcut).forEach(k=>{
        key_map[config.shortcut[k]]=k;
    })
    document.addEventListener("keydown",(e)=>{
        if(e.target.tagName!="INPUT")
        {
            let key=e.key;
            if(e.ctrlKey)
            {
                key="Ctrl+"+e.key;
                if(key in key_map)
                {
                    e.preventDefault();
                }
            }
            if(key in key_map)
            {
                w2ui.toolbar.click(key_map[key]);
            }
        }
    })
}
window.undo_redo_state={undo:false,redo:false};
window.update_undo_redo=function(s0,s1)
{
    if(s0)
    {
        w2ui.toolbar.enable("undo");
    }
    else
    {
        w2ui.toolbar.disable("undo");
    }
    if(s1)
    {
        w2ui.toolbar.enable("redo");
    }
    else
    {
        w2ui.toolbar.disable("redo");
    }
    window.undo_redo_state.undo=s0;
    window.undo_redo_state.redo=s1;
}