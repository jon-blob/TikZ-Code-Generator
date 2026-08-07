"""Small deterministic TikZ node-content cleaner."""

from typing import List


def skip_whitespace_and_comments(tex: str, start: int) -> int:
    """Return the next non-whitespace character outside a LaTeX comment."""

    i = start
    n = len(tex)

    while i < n:
        if tex[i].isspace():
            i += 1
            continue

        if tex[i] == "%":
            backslashes = 0
            j = i - 1
            while j >= 0 and tex[j] == "\\":
                backslashes += 1
                j -= 1

            if backslashes % 2 == 1:  # escaped \%
                break

            while i < n and tex[i] != "\n":
                i += 1
            continue

        break

    return i


def _is_escaped(tex: str, pos: int) -> bool:
    backslashes = 0
    pos -= 1
    while pos >= 0 and tex[pos] == "\\":
        backslashes += 1
        pos -= 1
    return backslashes % 2 == 1


def find_matching(tex: str, start: int, left: str, right: str) -> int:
    """Find the matching closing delimiter while ignoring comments."""

    if start >= len(tex) or tex[start] != left:
        return -1

    depth = 1
    i = start + 1
    n = len(tex)

    while i < n:
        i = skip_whitespace_and_comments(tex, i)
        if i >= n:
            break

        if tex[i] == left and not _is_escaped(tex, i):
            depth += 1
        elif tex[i] == right and not _is_escaped(tex, i):
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return -1


def is_path_node(tex: str, pos: int) -> bool:
    """Return whether ``pos`` starts a standalone TikZ path ``node``."""

    if pos > 0 and (tex[pos - 1].isalnum() or tex[pos - 1] == "_"):
        return False

    end = pos + len("node")
    if end < len(tex) and (tex[end].isalnum() or tex[end] == "_"):
        return False

    end = skip_whitespace_and_comments(tex, end)
    return end < len(tex) and tex[end] in "([{"


def find_next_path_node(tex: str, start: int) -> int:
    pos = start
    while True:
        pos = tex.find("node", pos)
        if pos == -1:
            return -1
        if is_path_node(tex, pos):
            return pos
        pos += len("node")


def replace_all_nodes(tex: str, replacement: str = "Text") -> str:
    """Replace non-empty TikZ node contents while preserving node structure."""

    result: List[str] = []
    i = 0
    n = len(tex)

    while i < n:
        command_node = tex.find(r"\node", i)
        path_node = find_next_path_node(tex, i)
        candidates = [p for p in (command_node, path_node) if p != -1]

        if not candidates:
            result.append(tex[i:])
            break

        node_pos = min(candidates)
        result.append(tex[i:node_pos])
        j = node_pos + (len(r"\node") if tex[node_pos] == "\\" else len("node"))
        j = skip_whitespace_and_comments(tex, j)

        while j < n and tex[j] != "{":
            if tex[j] == "[":
                end = find_matching(tex, j, "[", "]")
                if end == -1:
                    break
                j = end + 1
                j = skip_whitespace_and_comments(tex, j)
                continue

            if tex[j] == "(":
                end = find_matching(tex, j, "(", ")")
                if end == -1:
                    break
                j = end + 1
                j = skip_whitespace_and_comments(tex, j)
                continue

            next_pos = skip_whitespace_and_comments(tex, j)
            j = j + 1 if next_pos == j else next_pos

        if j >= n or tex[j] != "{":
            result.append(tex[node_pos:])
            break

        end = find_matching(tex, j, "{", "}")
        if end == -1:
            result.append(tex[node_pos:])
            break

        result.append(tex[node_pos:j])
        result.append("{")

        if skip_whitespace_and_comments(tex, j + 1) != end:
            result.append(replacement)

        result.append("}")
        i = end + 1

    return "".join(result)


def process_latex(tex: str, mode: str, replacement: str = "Text") -> str:
    if mode == "clean_all_text":
        if r"\begin{tikzpicture}" in tex:
            tex = tex.replace(
                r"\begin{tikzpicture}",
                r"\tikzset{every node/.append style={text opacity=0}}"
                "\n"
                r"\begin{tikzpicture}",
            )
        return tex

    if mode == "replace_all":
        return replace_all_nodes(tex=tex, replacement=replacement)

    raise ValueError(f"Unknown mode: {mode}")
