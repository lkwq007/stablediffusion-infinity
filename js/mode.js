function(mode){
    let app=document.querySelector("gradio-app");
    let frame=app.querySelector("#sdinfframe").contentWindow;
    frame.postMessage(["mode", mode], "*");
    return mode;
}