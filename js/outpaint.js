function(a){
    if(!window.my_observe_outpaint)
    {
        console.log("setup outpaint here");
        window.my_observe_outpaint = new MutationObserver(function (event) {
            console.log(event);
            let app=document.querySelector("gradio-app").shadowRoot;
            let frame=app.querySelector("#sdinfframe").contentWindow.document;
            frame.querySelector("#outpaint").click();
        });
        window.my_observe_outpaint_target=document.querySelector("gradio-app").shadowRoot.querySelector("#output span")
        window.my_observe_outpaint.observe(window.my_observe_outpaint_target, {
            attributes: false, 
            subtree: true,
            childList: true, 
            characterData: true
        });
    }
    let app=document.querySelector("gradio-app").shadowRoot;
    let frame=app.querySelector("#sdinfframe").contentWindow.document;
    let button=frame.querySelector("#transfer");
    button.click();
    return a;
}