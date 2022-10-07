function(sel_buffer_str,
    prompt_text,
    negative_prompt_text,
    strength,
    guidance,
    step,
    resize_check,
    fill_mode,
    enable_safety,
    use_correction,
    state){
    sel_buffer = document.querySelector("gradio-app").shadowRoot.querySelector("#input textarea").value;
    return [
        sel_buffer,
        prompt_text,
        negative_prompt_text,
        strength,
        guidance,
        step,
        resize_check,
        fill_mode,
        enable_safety,
        use_correction,
        state
    ]
}