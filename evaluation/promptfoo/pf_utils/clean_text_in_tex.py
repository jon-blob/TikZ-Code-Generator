"""Deterministic TikZ node-text replacement."""


def skip_whitespace_and_comments(tex: str, start: int) -> int:
    i = start
    while i < len(tex):
        if tex[i].isspace():
            i += 1
            continue
        if tex[i] == "%" and not is_escaped(tex, i):
            while i < len(tex) and tex[i] != "\n":
                i += 1
            continue
        break
    return i


def is_escaped(tex: str, pos: int) -> bool:
    backslashes = 0
    pos -= 1
    while pos >= 0 and tex[pos] == "\\":
        backslashes += 1
        pos -= 1
    return backslashes % 2 == 1


def find_matching(tex: str, start: int, left: str, right: str) -> int:
    if start >= len(tex) or tex[start] != left:
        return -1

    depth = 1
    i = start + 1
    while i < len(tex):
        i = skip_whitespace_and_comments(tex, i)
        if i >= len(tex):
            break
        if tex[i] == left and not is_escaped(tex, i):
            depth += 1
        elif tex[i] == right and not is_escaped(tex, i):
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def is_path_node(tex: str, pos: int) -> bool:
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
        if pos == -1 or is_path_node(tex, pos):
            return pos
        pos += len("node")


def placeholder(index: int) -> str:
    """Return aaaa, aaab, ..., zzzz for a zero-based index."""
    if not 0 <= index < 26**4:
        raise ValueError("Too many TikZ text nodes")

    chars = ["a"] * 4
    for pos in range(3, -1, -1):
        chars[pos] = chr(ord("a") + index % 26)
        index //= 26
    return "".join(chars)


def replace_node_text(tex: str, sequential: bool = True, replacement: str = "Text") -> str:
    result: list[str] = []
    i = count = 0

    while i < len(tex):
        command_node = tex.find(r"\node", i)
        path_node = find_next_path_node(tex, i)
        nodes = [pos for pos in (command_node, path_node) if pos != -1]
        if not nodes:
            result.append(tex[i:])
            break

        node = min(nodes)
        result.append(tex[i:node])
        j = node + (len(r"\node") if tex[node] == "\\" else len("node"))
        j = skip_whitespace_and_comments(tex, j)

        while j < len(tex) and tex[j] != "{":
            if tex[j] in "[(":
                end = find_matching(tex, j, tex[j], "]" if tex[j] == "[" else ")")
                if end == -1:
                    break
                j = skip_whitespace_and_comments(tex, end + 1)
            else:
                j += 1

        if j >= len(tex) or tex[j] != "{":
            result.append(tex[node:])
            break

        end = find_matching(tex, j, "{", "}")
        if end == -1:
            result.append(tex[node:])
            break

        result.append(tex[node : j + 1])
        if skip_whitespace_and_comments(tex, j + 1) != end:
            result.append(placeholder(count) if sequential else replacement)
            count += 1
        result.append("}")
        i = end + 1

    return "".join(result)


def replace_text_with_placeholders(tex: str) -> str:
    return replace_node_text(tex, sequential=True)


def replace_all_nodes(tex: str, replacement: str = "Text") -> str:
    return replace_node_text(tex, sequential=False, replacement=replacement)


def process_latex(tex: str, mode: str, replacement: str = "Text") -> str:
    if mode == "clean_all_text":
        marker = r"\begin{tikzpicture}"
        if marker in tex:
            return tex.replace(
                marker,
                r"\tikzset{every node/.append style={text opacity=0}}" + "\n" + marker,
            )
        return tex
    if mode == "replace_all":
        return replace_all_nodes(tex, replacement)
    if mode == "replace_sequential":
        return replace_text_with_placeholders(tex)
    raise ValueError(f"Unknown mode: {mode}")
