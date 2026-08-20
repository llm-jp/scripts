
# import json
# import os



# def append_result_to_json(result: dict, folder_path: str, json_name="results.json", reset=False):

#     os.makedirs(folder_path, exist_ok=True)
#     json_path = os.path.join(folder_path, json_name)

#     # 如果需要清空
#     if reset and os.path.exists(json_path):
#         os.remove(json_path)

#     # 读取已有 JSON 或初始化空列表
#     data = []
#     if os.path.exists(json_path):
#         try:
#             with open(json_path, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#                 if not isinstance(data, list):
#                     data = []
#         except json.JSONDecodeError:
#             data = []

#     # 追加新结果
#     data.append(result)

#     # 写回 JSON
#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
import json
import os
import re

def _clean_unicode(obj):
    """
    递归清理非法 Unicode 代理项、隐藏字符、截断 emoji
    """
    if isinstance(obj, str):
        # 去掉 UTF-16 代理项（\ud800-\udfff）以及控制符
        obj = re.sub(r'[\ud800-\udfff]', '', obj)
        obj = obj.replace('\u202e', '').replace('\u200b', '')
        # 去掉无法编码的部分
        obj = obj.encode("utf-8", "ignore").decode("utf-8", "ignore")
        return obj
    elif isinstance(obj, list):
        return [_clean_unicode(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: _clean_unicode(v) for k, v in obj.items()}
    else:
        return obj


def append_result_to_json(result: dict, folder_path: str, json_name="results.json", reset=False):
    """
    向 JSON 文件追加结果（线程安全 + UTF-8 安全版）
    - 自动清理非法字符
    - 出错不会破坏已有文件
    """
    os.makedirs(folder_path, exist_ok=True)
    json_path = os.path.join(folder_path, json_name)

    # 如果需要清空
    if reset and os.path.exists(json_path):
        os.remove(json_path)

    # 读取已有 JSON
    data = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except (json.JSONDecodeError, UnicodeDecodeError):
            # 出错就初始化为空
            data = []

    # 清理输入与已有数据
    result = _clean_unicode(result)
    data = _clean_unicode(data)

    # 追加新结果
    data.append(result)

    # 写回 JSON
    tmp_path = json_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, json_path)  # 原子替换，防止中断损坏
    except Exception as e:
        print(f"[WARN] JSON 写入失败: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
