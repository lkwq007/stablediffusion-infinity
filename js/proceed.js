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
    interrogate_mode,
    state){
    let app=document.querySelector("gradio-app");
    app=app.shadowRoot??app;
    sel_buffer=app.querySelector("#input textarea").value;
    let use_correction_bak=false;
    ({resize_check,enable_safety,enable_img2img,use_seed,seed_val,interrogate_mode}=window.config_obj);
    seed_val=Number(seed_val);
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
        interrogate_mode,
        state,
    ]
}