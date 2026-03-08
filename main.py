import threading
import torch
import yaml
import gc
import re

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)

from tools import (
    get_current_datetime, 
    get_world_clock, 
    perform_web_search, 
    save_to_file, 
    apply_formatting_filter, 
    read_local_document, 
    analyze_local_image
)

from error import KillSwitchTriggeredError
from lora import apply_lora_adapter
from log import agent_logger


class ThreadKillSwitch(StoppingCriteria):
    def __init__(self):
        self.halt_generation = False
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.halt_generation


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()


def initialize_engine():
    agent_logger.info("Initializing Granite engine...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(config['agent_config']['model']['repo_id'])
    
    model = AutoModelForCausalLM.from_pretrained(
        config['agent_config']['model']['repo_id'],
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,        
        torch_dtype=torch.float16     
    )
    
    lora_repo = config['agent_config']['model'].get('lora_repo_id', 'none')
    if lora_repo and str(lora_repo).strip().lower() != 'none':
        model = apply_lora_adapter(model, lora_repo)
    
    model.config.max_position_embeddings = 4096
    model.config.pad_token_id = tokenizer.eos_token_id
    torch.cuda.empty_cache()                    
    
    return tokenizer, model


def execute_single_task(user_input, tokenizer, model, message_history):
    if len(message_history) > 5:
        message_history[:] = [message_history[0]] + message_history[-4:]
        
    current_time = get_current_datetime()

    doc_data = ""
    if 'read' in user_input.lower():
        match = re.search(r'([\w\.\-\\/]+\.(?:pdf|docx|txt|html|pptx|md|log))', user_input, flags=re.IGNORECASE)
        if match:
            file_path = match.group(1)
            doc_data = read_local_document(file_path)
        else:
            agent_logger.warning("Read command detected, but no valid file extension found in prompt.")

    vision_data = ""
    image_match = re.search(r'([\w\.\-\\/]+\.(?:png|jpg|jpeg|webp))', user_input, flags=re.IGNORECASE)
    if image_match:
        image_path = image_match.group(1)
        vision_data = analyze_local_image(image_path, user_input)

    clock_data = ""
    if any(kw in user_input.lower() for kw in ['time', 'clock']):
        agent_logger.info("Synchronizing temporal context across global zones...")
        clock_data = get_world_clock()

    search_cfg = config['agent_config'].get('search_config', {})
    max_res = search_cfg.get('max_results', 5)
    
    if 'whitelist' in user_input.lower():
        trusted_domains = search_cfg.get('trusted_sites', [])
    else:
        trusted_domains = None
    
    if 'search' in user_input.lower():
        search_data = perform_web_search(
            user_input, 
            max_results=max_res,
            trusted_sites=trusted_domains
        )
    else:
        search_data = ""

    volatile_prompt = f"### Input:\n[SYSTEM_TIME: {current_time}]\n\n"
    if clock_data:
        volatile_prompt += f"{clock_data}\n\n"
    if doc_data:
        volatile_prompt += f"{doc_data}\n"
    if vision_data:
        volatile_prompt += f"{vision_data}\n"
    if search_data:
        volatile_prompt += f"[SEARCH_DATA]:\n{search_data}\n\n"
        
    volatile_prompt += f"### Instruction:\n{user_input}"

    message_history.append({"role": "user", "content": volatile_prompt})

    prompt_string = tokenizer.apply_chat_template(
        message_history, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    prompt_string += "<think>\n"
    
    inputs = tokenizer(prompt_string, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    kill_switch = ThreadKillSwitch()

    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "streamer": streamer,
        "stopping_criteria": StoppingCriteriaList([kill_switch]),
        "max_new_tokens": config['agent_config']['generation'].get('max_new_tokens', 1024),
        "temperature": config['agent_config']['generation'].get('temperature', 0.15),
        "repetition_penalty": config['agent_config']['generation'].get('repetition_penalty', 1.12),
        "do_sample": config['agent_config']['generation'].get('do_sample', True),
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }

    gen_thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    gen_thread.start()
    
    print()

    generated_text = ""
    try:
        for new_text in streamer:
            generated_text += new_text
            print(new_text, end="", flush=True)
            
            if "User:" in generated_text:
                kill_switch.halt_generation = True 
                raise KillSwitchTriggeredError("User:", generated_text)

        gen_thread.join()
        print() 

    except KillSwitchTriggeredError:
        gen_thread.join()
        generated_text = generated_text.replace("User:", "").strip()
        print("\n[SYSTEM] Generation severed to prevent hallucination.")

    if "<think>" in generated_text and "</think>" in generated_text:
        final_clean_answer = generated_text.split("</think>")[-1].strip()
    else:
        final_clean_answer = generated_text.strip()

    target = None
    user_lower = user_input.lower()
    
    if 'bullet point' in user_lower:
        target = "bullets"
    elif 'numbered' in user_lower:
        target = "numbers"
    elif 'clean' in user_lower or 'paragraph' in user_lower:
        target = "clean"
    elif 'markdown' in user_lower:
        target = "markdown"
        
    if target:
        final_clean_answer = apply_formatting_filter(final_clean_answer, target)

    if 'save' in user_lower:
        ext = ".md" if target == "markdown" else ".txt"
        file_saved_path = save_to_file(final_clean_answer, user_input, extension=ext)
        if file_saved_path:
            print(f"\n[SYSTEM] Output successfully saved to: {file_saved_path}")

    message_history[-1]["content"] = user_input 
    message_history.append({"role": "assistant", "content": final_clean_answer})

    del inputs
    gc.collect()
    torch.cuda.empty_cache()

    return final_clean_answer
