from peft import LoraConfig, PromptTuningInit, PromptTuningConfig, get_peft_model, prepare_model_for_int8_training, TaskType, IA3Config


def create_peft_config(model, method='lora'):

    if method == 'lora':
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules = ["q_proj", "v_proj"]
        )
    
    elif method == 'prefix':
        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS, # TaskType.TOKEN_CLS
            inference_mode=False, 
            num_virtual_tokens=20)
    elif method == 'prompt':
        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False, 
            num_virtual_tokens=20,
        )
    elif method == 'ia3':
        peft_config = IA3Config(task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            target_modules = ["q_proj", "v_proj"])
    else:
        raise NotImplementedError(f'{method} is not implemented!')

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    # add LoRA adaptor
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config