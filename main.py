import dataclasses
from typing import List, Sequence, Iterator
import pyarrow as pa
import ollama
from ollama import EmbedResponse, ChatResponse

import lancedb


@dataclasses.dataclass
class TableContent:
    content: str
    vectors: List[float]

    def serialise(self) -> dict:
        return {"vectors": self.vectors, "content": self.content}


class LocalLanceRaG:
    table_name: str = "sample_data_table"
    vector_cols_count = 1024

    def __init__(self, db_path: str = "data/sample-lancedb"):
        self.db = lancedb.connect(db_path)

    def initialise(self) -> None:
        """
        1. creates table if it doesn't exist.
        2.
        :return: None
        """
        self.create_table()

    def create_table(self):
        schema = pa.schema(
            [
                pa.field("vectors", pa.list_(pa.float32(), list_size=self.vector_cols_count)),
                pa.field("content", pa.string())
            ])
        self.db.create_table(self.table_name, schema=schema, exist_ok=True)

    def insert_data(self, data: TableContent | List[TableContent]) -> None:
        """
        Inserts provided data in the database.
        :param data: Either single or list of TableContent data to insert in database.
        :return: None
        """
        tbl = self.db.open_table(self.table_name)
        items = [item.serialise() for item in data] if (type(data).__name__ == "list") else [data.serialise()]
        tbl.add(items)

    def compute_vectors(self, content: str) -> List[float] | Sequence[float]:
        """
        Computes vectors for provided query.
        :param content: Content to use for vectorisation
        :return: vectorised list of float values are returned.
        """
        response: EmbedResponse = ollama.embed(
            model='bge-m3',
            input=content,
            truncate=False
        )
        return response.embeddings[0]

    def prepare_prompt(self, context: str, query: str) -> str:
        """
        Prepares prompt based on provided context.

        :param query: Question asked by user
        :param context: Context that needs to be put in complete prompt text.
        :return: prompt prepared from provided context.
        """
        return (f"You are an expert in the field relevant to provided context. This query: '{query}' must be answered "
                f"with context of '{context}'.")

    def complete(self, prompt: str, stream: bool = False) -> str | Iterator[str]:
        """
        Answers the provided prompt.

        :param prompt: Prompt taken by ML model to generate output
        :param stream: should the response be streamed?
        :return: Answer to the provided prompt.
        """
        response: ChatResponse | Iterator[ChatResponse] = ollama.chat(
            model='qwen2.5:3b',
            messages=[{'role': 'user', 'content': prompt}],
            stream=stream
        )

        if not stream:
            yield response.message.content
        else:
            for chunk in response:
                yield chunk.message.content

    def search(self, query: str, stream: bool = False) -> str | Iterator[str]:
        """
        1. Perform vector search based for query
        2. Generates the output via AI model using results from step 1 as context.

        :param stream: Should the response be streamed or plain text response must be returned?
        :param query: Ask a question that needs to be answered based on RAG
        :return: Answer to the query post RAG
        """

        # compute vectors and search in database
        query_vectors = self.compute_vectors(query)
        table = self.db.open_table(self.table_name)
        context = table.search(query_vectors).limit(5).select(["content"]).to_list()
        context = "\n".join([record["content"] for record in context])

        # Generate model output based on context
        result = self.complete(prompt=self.prepare_prompt(context, query), stream=stream)
        return result if stream else list(result)[0]


if __name__ == '__main__':
    # initial setup
    instance = LocalLanceRaG()
    instance.initialise()

    # insert data to perform rag.
    content = "Electricity was never a superpower owned by Thomas Edition?"
    instance.insert_data(TableContent(content=content, vectors=instance.compute_vectors(content)))

    # search for the query
    res = instance.search("Which superpower was owned by Thomas?", stream=False)
    print(f"Result: {res}")
