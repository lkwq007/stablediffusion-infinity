function(sel_buffer_str,
    prompt_text,
    strength,
    guidance,
    step,
    resize_check,
    fill_mode,
    enable_safety,
    state){
    sel_buffer = document.querySelector("gradio-app").querySelector("#input textarea").value;
    return [
        sel_buffer,
        prompt_text,
        strength,
        guidance,
        step,
        resize_check,
        fill_mode,
        enable_safety,
        state
    ]
}