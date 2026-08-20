import re


class JbbqEvaluator:
    # 声明这个 evaluator 不需要联网 -> evaluate.py 不会对它做限速 sleep。
    requires_network = False

    def __init__(self, prompt_name=None):
        self.prompt_name = prompt_name

    def parse_prediction(self, response):
        text = "" if response is None else str(response).strip()

        # ① 原规则优先：纯 "0"/"1"/"2" 直接通过
        if re.fullmatch(r"[0-2]", text):
            return int(text)

        # ② 放宽：剥掉 <|channel|> <|message|> 这类特殊 token，
        #    避免标记里的字符干扰，然后取最后一个独立的 0/1/2。
        #    （thinking 模型答案通常在末尾，如 "<|channel|> final<|message|> 1"）
        cleaned = re.sub(r"<\|.*?\|>", " ", text)
        matches = re.findall(r"(?<![0-9])[0-2](?![0-9])", cleaned)
        if matches:
            return int(matches[-1])

        return None

    def evaluate(self, question, response, entry=None):
        entry = entry or {}
        prediction = self.parse_prediction(response)

        label = entry.get("label")
        if label is not None:
            label = int(label)

        return {
            "prediction": prediction,
            "is_valid": prediction in [0, 1, 2],
            "is_correct": prediction == label if prediction in [0, 1, 2] and label is not None else None,
        }