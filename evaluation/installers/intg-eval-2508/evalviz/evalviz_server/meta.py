"""UI に渡す定義 (サンプルの状態とフラグ、カテゴリの説明)。UI はこれを見て凡例・絞り込みを作る。"""
from __future__ import annotations

from . import ljeval

STATUSES = [
    {"key": "correct", "label": "正解", "short": "正解", "desc": "抽出した回答が正解と一致"},
    {"key": "wrong", "label": "不正解", "short": "誤答", "desc": "回答は抽出でき、選択肢の範囲内だが正解と一致しない"},
    {"key": "out_of_label", "label": "選択肢外", "short": "選択肢外", "desc": "抽出した回答が選択肢 (label_list) に無い"},
    {"key": "unparsed", "label": "回答の抽出不可", "short": "抽出不可", "desc": "回答形式 (<answer> タグ・\\boxed{} 等) に沿った回答が見つからない"},
    {"key": "empty", "label": "出力が空", "short": "空", "desc": "最終出力が空 (thinking モデルで推論が出力長の上限で打ち切られた場合など)"},
    {"key": "scored", "label": "正誤なし", "short": "—", "desc": "BLEU・F1 等の連続値の指標だけのデータセット (サンプル単位の正誤は出さない)"},
]

FLAGS = [
    {"key": "reasoning_cut", "label": "推論の打ち切り", "desc": "推論 (reasoning_content) はあるが最終出力が空。推論が出力長の上限内に終わっていない"},
    {"key": "repetition", "label": "反復", "desc": "出力か推論の末尾 4000 バイトが同じ文字列の繰り返し (圧縮率 0.15 未満)"},
    {"key": "continued", "label": "回答後の続き", "desc": "回答の区切り (</answer>・1 行目・空行) のあとに 30 字を超える生成が続く。ベースモデルが次の例を書き始める等"},
    {"key": "zh_mix", "label": "中国語の混入", "desc": "出力か推論の末尾に簡体字にしかない字が 3 字以上"},
    {"key": "ja_in_en", "label": "英語タスクで日本語", "desc": "英語のデータセットの出力に仮名が 5 字以上"},
    {"key": "other_script", "label": "想定外の文字種", "desc": "キリル・アラビア・インド系・タイ・ハングル等の文字が出力に 10 字以上"},
    {"key": "special_token", "label": "特殊トークンの漏れ", "desc": "<|...|> や Harmony のチャネル名 (analysis / assistantfinal) が出力に残る"},
]


def meta() -> dict:
    return {
        "statuses": STATUSES,
        "flags": FLAGS,
        "en_datasets": sorted(ljeval.EN_DATASETS),
        "category_desc": {
            "NLI": "自然言語推論", "QA": "質問応答", "RC": "読解", "CR": "常識推論", "MC": "多肢選択", "HE": "人間の試験",
            "HE-JA": "人間の試験 (日本語)", "HE-EN": "人間の試験 (英語)", "EL": "エンティティリンキング", "FA": "基礎解析",
            "MR": "数学的推論", "MT": "機械翻訳", "CG": "コード生成", "SUM": "要約", "IF": "指示追従", "BBH": "BIG-Bench Hard",
            "LM": "言語モデリング (容認性)", "SP": "構造化出力",
        },
    }
