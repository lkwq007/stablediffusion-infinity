var setup_outpaint=function(){
    if(!window.my_observe_outpaint)
    {
        console.log("setup outpaint here");
        window.my_observe_outpaint = new MutationObserver(function (event) {
            console.log(event);
            let app=document.querySelector("gradio-app");
            app=app.shadowRoot??app;
            let frame=app.querySelector("#sdinfframe").contentWindow;
            frame.postMessage(["outpaint", ""], "*");
        });
        var app=document.querySelector("gradio-app");
        app=app.shadowRoot??app;
        window.my_observe_outpaint_target=app.querySelector("#output span");
        window.my_observe_outpaint.observe(window.my_observe_outpaint_target, {
            attributes: false, 
            subtree: true,
            childList: true, 
            characterData: true
        });
    }
}; 
window.config_obj={
    resize_check: true,
    enable_safety: true,
    use_correction: false,
    enable_img2img: false,
    use_seed: false,
    seed_val: 0,
    interrogate_mode: false,
};
setup_outpaint();