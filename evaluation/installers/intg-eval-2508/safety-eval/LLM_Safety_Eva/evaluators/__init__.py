import yaml

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
evaluation_items = config.get('evaluation_items', [])

print(evaluation_items)

__all__ = evaluation_items