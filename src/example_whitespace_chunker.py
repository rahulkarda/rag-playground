from src.chunker import chunk_whitespace

if __name__ == "__main__":
    text = """
This is the first paragraph.

This is the second paragraph, with  more   spaces.

This	is	the third with	tabs.
"""
    chunks = chunk_whitespace(text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: [{chunk.start}:{chunk.end}] '{chunk.text}'")
