
let app=document.querySelector("gradio-app");
app=app.shadowRoot??app;
let frame=app.querySelector("#sdinfframe").contentWindow;
frame.setup_shortcut(json);
var config=JSON.parse(json);
var key_map={};
Object.keys(config.shortcut).forEach(k=>{
    key_map[config.shortcut[k]]=k;
});
document.addEventListener("keydown", e => {
    if(e.target.tagName!="INPUT"&&e.target.tagName!="GRADIO-APP"&&e.target.tagName!="TEXTAREA")
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
        let app=document.querySelector("gradio-app");
        app=app.shadowRoot??app;
        let frame=app.querySelector("#sdinfframe").contentDocument;
        frame.dispatchEvent(
          new KeyboardEvent("keydown", {key: e.key, ctrlKey: e.ctrlKey})
        );
    }
})