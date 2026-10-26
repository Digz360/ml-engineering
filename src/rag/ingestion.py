from pathlib import Path
import hashlib


MAX_CHUNK_CHARS = 1200

def load_markdown(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def create_chunk_id(source: str, section: str, text: str) -> str:
    content = f"{source}:{section}:{text}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def build_chunk(source: str, section: str, text: str) -> dict:
    return {
        "id": create_chunk_id(source, section, text),
        "text": text,
        "metadata": {
            "source": source,
            "section": section,
            "version": 1,
        },
    }


def split_section(
    text: str,
    source: str,
    section: str,
) -> list[dict]:

    chunks = []

    # Divide por palavras, garantindo um limite real de tamanho
    words = text.split()
    current_words = []
    current_length = 0

    for word in words:

        word_length = len(word) + 1

        if current_length + word_length > MAX_CHUNK_CHARS:

            chunks.append(
                build_chunk(
                    source=source,
                    section=section,
                    text=" ".join(current_words),
                )
            )

            current_words = []
            current_length = 0

        current_words.append(word)
        current_length += word_length

    if current_words:
        chunks.append(
            build_chunk(
                source=source,
                section=section,
                text=" ".join(current_words),
            )
        )

    return chunks


def chunk_text(text: str, source: str) -> list[dict]:

    chunks = []

    current_section = "Introduction"
    current_lines = []

    def flush_section():
        if not current_lines:
            return

        content = "\n".join(current_lines).strip()

        if content:
            chunks.extend(
                split_section(
                    text=content,
                    source=source,
                    section=current_section,
                )
            )

    for line in text.splitlines():

        if line.startswith("## "):
            flush_section()

            current_section = line[3:].strip()
            current_lines = []

        else:
            current_lines.append(line)

    flush_section()

    return chunks


def chunk_markdown(path: str) -> list[dict]:
    path_obj = Path(path)
    text = load_markdown(path)

    return chunk_text(
        text=text,
        source=path_obj.name,
    )