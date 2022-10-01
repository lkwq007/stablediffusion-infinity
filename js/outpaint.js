function(a){
    if(!window.my_observe_outpaint)
    {
        console.log("setup outpaint here");
        window.my_observe_outpaint = new MutationObserver(function (event) {
            console.log(event);
            let app=document.querySelector("gradio-app");
            let frame=app.querySelector("#sdinfframe").contentWindow;
            var str=document.querySelector("gradio-app").querySelector("#output textarea").value;
            frame.postMessage(["outpaint", str], "*");
        });
        window.my_observe_outpaint_target=document.querySelector("gradio-app").querySelector("#output span")
        window.my_observe_outpaint.observe(window.my_observe_outpaint_target, {
            attributes: false, 
            subtree: true,
            childList: true, 
            characterData: true
        });
        window.addEventListener("message", function(e){
            if(e.data[0]=="transfer")
            {
                document.querySelector("gradio-app").querySelector("#input textarea").value=e.data[1];
                document.querySelector("gradio-app").querySelector("#proceed").click();
            }
        });
    }
    let app=document.querySelector("gradio-app");
    let frame=app.querySelector("#sdinfframe").contentWindow;
    frame.postMessage(["transfer"],"*")
    return a;
}