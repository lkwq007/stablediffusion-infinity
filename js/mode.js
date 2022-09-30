function(mode){
    let app=document.querySelector("gradio-app").shadowRoot;
    let frame=app.querySelector("#sdinfframe").contentWindow.document;
    frame.querySelector("#mode").value=mode;
    return mode;
}