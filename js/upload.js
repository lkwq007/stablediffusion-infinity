function(a,b){
    if(!window.my_observe_upload)
    {
        console.log("setup upload here");
        window.my_observe_upload = new MutationObserver(function (event) {
            console.log(event);
            var frame=document.querySelector("gradio-app").shadowRoot.querySelector("#sdinfframe").contentWindow.document;
            frame.querySelector("#upload").click();
        });
        window.my_observe_upload_target = document.querySelector("gradio-app").shadowRoot.querySelector("#upload span");
        window.my_observe_upload.observe(window.my_observe_upload_target, {
            attributes: false, 
            subtree: true,
            childList: true, 
            characterData: true
        });
    }
    return [a,b];
}