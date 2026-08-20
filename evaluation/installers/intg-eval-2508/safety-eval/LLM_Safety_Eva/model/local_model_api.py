from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import traceback
import inspect

app = FastAPI(title="Dynamic LLM Loader")


# ===== 请求体定义 =====
class PromptRequest(BaseModel):
    input_prompt: str
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    think: Optional[bool] = False


class ModelLoadRequest(BaseModel):
    model_name: str


# ===== 全局状态 =====
current_model = None
current_tokenizer = None
current_model_name = None
current_has_think = False
current_has_chat_template = False


def unload_current():
    """释放当前模型，防止显存占满"""
    global current_model, current_tokenizer, current_model_name
    global current_has_think, current_has_chat_template

    if current_model is not None:
        try:
            del current_model
        except Exception:
            pass
    if current_tokenizer is not None:
        try:
            del current_tokenizer
        except Exception:
            pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    current_model = None
    current_tokenizer = None
    current_model_name = None
    current_has_think = False
    current_has_chat_template = False


@app.post("/load_model")
def load_model(req: ModelLoadRequest):
    """加载指定模型"""
    global current_model, current_tokenizer, current_model_name
    global current_has_think, current_has_chat_template

    unload_current()

    try:
        # ---- 加载 tokenizer ----
        tokenizer = AutoTokenizer.from_pretrained(
            req.model_name,
            trust_remote_code=True
        )

        # 部分模型没有 pad_token，手动设成 eos_token
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        # 对部分 chat 模型，开启默认 system prompt
        if hasattr(tokenizer, "use_default_system_prompt"):
            tokenizer.use_default_system_prompt = True

        # ---- 加载模型 ----
        model = AutoModelForCausalLM.from_pretrained(
            req.model_name,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )

        # ---- 检测是否有 <think> token ----
        try:
            vocab = tokenizer.get_vocab()
            has_think = "<think>" in vocab if vocab is not None else False
        except Exception:
            has_think = False

        # ---- 检测是否有 chat_template ----
        chat_template = getattr(tokenizer, "chat_template", None)
        has_chat_template = bool(chat_template and chat_template.strip())

        # ---- 更新全局状态 ----
        current_model = model
        current_tokenizer = tokenizer
        current_model_name = req.model_name
        current_has_think = has_think
        current_has_chat_template = has_chat_template

        return {
            "status": "success",
            "message": f"Model {req.model_name} loaded.",
            "has_think": has_think,
            "has_chat_template": has_chat_template,
        }

    except Exception as e:
        unload_current()
        # 把异常栈也打出来方便 debug
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model {req.model_name}: {repr(e)}\n{traceback.format_exc()}",
        )


# ===== fallback prompt 模板 =====
def build_prompt_fallback(model_name: str, user_prompt: str) -> str:
    """
    对没有 chat_template 的模型，构造一个尽量合理的指令风格 prompt。
    根据模型名字做一点 heuristic 区分：
      - llm-jp: 用 Alpaca/Dolly 日文风格
      - plamo / Fugaku-LLM: 简单的日文对话风格
      - 其它: 英文 User/Assistant 风格
    """
    name_lower = (model_name or "").lower()

    # ---- llm-jp: 参考官方文档的 Alpaca 日文模板 ----
    if "llm-jp" in name_lower:
        return (
            "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n"
            "### 指示:\n"
            f"{user_prompt}\n\n"
            "### 応答:\n"
        )

    # ---- plamo / Fugaku / 日本系模型：简单日文对话 ----
    if "plamo" in name_lower or "fugaku-llm" in name_lower or "sarashina" in name_lower:
        return f"ユーザー: {user_prompt}\nアシスタント:"

    # ---- 默认英文指令风格 ----
    return f"User: {user_prompt}\nAssistant:"


def chat_generate(
        model,
        tokenizer,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        think: bool = False,
        model_name: Optional[str] = None,
        has_chat_template: bool = False,
):
    """
    统一的生成函数：
      1. 能用 chat_template + apply_chat_template 就优先用
      2. 如果没有模板 / 失败 / 输出为空 → 回退到 build_prompt_fallback
    """
    # ---- 尝试使用 chat_template ----
    text = None
    if has_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "あなたは有能なAIアシスタントです。"},
            {"role": "user", "content": user_prompt},
        ]

        try:
            extra_kwargs = {}

            # 一些 tokenizer 有 enable_thinking 参数，一些没有
            try:
                sig = inspect.signature(tokenizer.apply_chat_template)
                if "enable_thinking" in sig.parameters:
                    extra_kwargs["enable_thinking"] = think
            except Exception:
                pass

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **extra_kwargs,
            )
        except Exception:
            # 模板用不了，回退
            text = None

    # ---- 如果没有模板 / 模板失败 / 模板输出为空：回退 ----
    if not text or not text.strip():
        text = build_prompt_fallback(model_name or "", user_prompt)

    # ---- 正常 tokenize + generate ----
    inputs = tokenizer(text, return_tensors="pt")

    # 对很多日文模型，token_type_ids 没用，删掉避免报错
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    # 移到模型所在设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,

            # ===== 防复读机制 =====
            repetition_penalty=1.2,  # ❶ 强制避免无限复读
            no_repeat_ngram_size=3,  # ❷ 避免局部重复循环（如“解答の説明:”）
        )

    # 只截取新生成部分
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return response.strip()


@app.post("/generate")
def generate(req: PromptRequest):
    """生成接口"""
    if current_model is None or current_tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded. Call /load_model first.")

    # 只有当模型真的有 <think> 且模板支持时才开启 think
    think_flag = bool(req.think and current_has_think)

    try:
        reply = chat_generate(
            model=current_model,
            tokenizer=current_tokenizer,
            user_prompt=req.input_prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            think=think_flag,
            model_name=current_model_name,
            has_chat_template=current_has_chat_template,
        )

        return {
            "model": current_model_name,
            "generated_text": reply,
            "think_mode": think_flag,
            "has_think": current_has_think,
            "has_chat_template": current_has_chat_template,
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Generation failed:\n" + traceback.format_exc(),
        )
