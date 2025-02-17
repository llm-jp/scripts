"""
Reads JSON input from standard input, processes each line, and prints structured JSON output.
"""
    
import json
import re
import sys
from collections import defaultdict

BUFFER_SIZE=10**6

def parse_data(data: str) -> dict:
    """
    Parses structured text data into a dictionary format.
    
    The function extracts labeled sections from the input text using regex,
    organizing them into a nested dictionary format. Each section is identified
    by a key, and the corresponding values are stored in lists.
    
    Args:
        data (str): Input text data containing labeled sections.
    
    Returns:
        dict: A dictionary containing extracted key-value pairs from the text.
    """
    result: dict[str, list] = defaultdict(list)
    result[""] = [""]
    pattern = r"([（\()]\d+?[）\)])?【(.+?)】(.*)"

    _key = None
    _prev_key = ""
    for line in data.split("\n"):
        line += "\n"
        matches = re.match(pattern, line)
        if matches:
            match_groups = matches.groups()
            if match_groups[0] or _key is None:
                # iniialize if '^（Number）'appear or after fill value
                _prev_key = ""
            elif _key is not None:
                # Nested dictionary handling
                _key = _prev_key + "-" + _key
            _key = match_groups[1]
            if match_groups[2]:
                result[_key].append(match_groups[2])

        else:
            assert _key is not None or _prev_key is not None
            if _key is None:
                # Concatenating values across multiple lines
                result[_prev_key][-1] += line
                continue
            result[_key].append(line)
            # Prepare for multi-line values
            _prev_key = _key
            _key = None

    return result


def process_line(line: str) -> None:
    """
    Processes a single line of JSON input by parsing its text content.
    
    Args:
        line (str): A single line of JSON string.
    """
    data = json.loads(line)
    parsed = parse_data(data["text"])
    parsed["meta"]=data["meta"]
    print(json.dumps(parsed, ensure_ascii=False))


if __name__ == "__main__":
    for line in sys.stdin:
        try:
            process_line(line)
        except Exception as e:
            raise e
