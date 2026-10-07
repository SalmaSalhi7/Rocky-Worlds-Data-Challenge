from __future__ import annotations

from pathlib import Path
import re


_KEY_PATTERN = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_]+)(?P<sep>\s+)(?P<value>.*?)(?P<comment>\s+#.*)?$")


def render_ecf_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(render_ecf_value(v) for v in value) + "]"
    if isinstance(value, tuple):
        return "[" + ", ".join(render_ecf_value(v) for v in value) + "]"
    return str(value)


def render_ecf_template(template_path: str | Path, output_path: str | Path, updates: dict) -> Path:
    template_path = Path(template_path)
    output_path = Path(output_path)
    print('template path: ', template_path)
    print('output path: ', output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_keys: set[str] = set()
    rendered_lines: list[str] = []

    for line in template_path.read_text(encoding="utf-8").splitlines():
        match = _KEY_PATTERN.match(line)
        if match and match.group("key") in updates:
            key = match.group("key")
            seen_keys.add(key)
            comment = match.group("comment") or ""
            rendered_lines.append(
                f"{match.group('indent')}{key}{match.group('sep')}{render_ecf_value(updates[key])}{comment}"
            )
        else:
            rendered_lines.append(line)

    for key, value in updates.items():
        if key not in seen_keys:
            rendered_lines.append(f"{key}    {render_ecf_value(value)}")

    output_path.write_text("\n".join(rendered_lines) + "\n", encoding="utf-8")
    return output_path

