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
    enable_img2img,
    use_seed,
    seed_val,
    generate_num,
    scheduler,
    scheduler_eta,
    state){
    let app=document.querySelector("gradio-app");
    app=app.shadowRoot??app;
    sel_buffer=app.querySelector("#input textarea").value;
    let use_correction_bak=false;
    ({resize_check,enable_safety,use_correction_bak,enable_img2img,use_seed,seed_val}=window.config_obj);
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
        enable_img2img,
        use_seed,
        seed_val,
        generate_num,
        scheduler,
        scheduler_eta,
        state,
    ]
}