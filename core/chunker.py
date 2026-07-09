def split_text_into_chunks(text:str , chunk_size:int=150 , overlap:int = 30) -> list:
    """Take the text wall from scraper and slice into bite-sized paragraphs."""

    words = text.split(" ")
    chunks = []

    for i in range(0 , len(words)  , chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size]).strip()

        if len(chunk) > 40:
            chunks.append(chunk)

    return chunks