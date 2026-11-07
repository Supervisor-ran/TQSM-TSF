"""Validate collection records (no Web application dependency)."""
import re

def validate_json_structure(obj, file_name, idx):
    required_keys = ['abstract', 'created_at', 'source', 'view_count']

    for key in required_keys:
        if key not in obj:
            return False, f"[{file_name}] JSON No.{idx}: missing key '{key}'"

    if not obj['abstract']:
        return False, f"[{file_name}] JSON No.{idx}: 'abstract' is empty"

    if not re.match(r'^\d{4}-\d{2}-\d{2}$', obj['created_at']):
        return False, f"[{file_name}] JSON No.{idx}: 'created_at' format is invalid"

    if not obj['source']:
        return False, f"[{file_name}] JSON No.{idx}: 'source' is empty"

    if not obj['view_count']:
        return False, f"[{file_name}] JSON No.{idx}: 'view_count' is empty"

    return True, ""
