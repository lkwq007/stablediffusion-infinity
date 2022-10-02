function(token_val, width, height, size, model_choice){
    let app=document.querySelector("gradio-app").shadowRoot;
    app.querySelector("#sdinfframe").style.height=height+"px";
    let frame=app.querySelector("#sdinfframe").contentWindow.document;
    if(frame.querySelector("#setup").value=="0")
    {
        window.my_setup=setInterval(function(){
            let frame=document.querySelector("gradio-app").shadowRoot.querySelector("#sdinfframe").contentWindow.document;
            console.log("Check PyScript...")
            if(frame.querySelector("#setup").value=="1")
            {
                frame.querySelector("#draw").click();
                clearInterval(window.my_setup);
            }
        },100)
    }
    else
    {
        frame.querySelector("#draw").click();
    }
    return [token_val, width, height, size, model_choice];
}