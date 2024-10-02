import asyncio
import logging
from pdf_loader import PDFLoader
from vector_database import VectorDatabase

class PDFRAG:
    def __init__(self, pdf_path):
        self.pdf_loader = PDFLoader(pdf_path)
        self.vector_db = VectorDatabase(max_concurrent_calls=3)
        logging.info(f"Initialized PDFRAG with PDF: {pdf_path}")

    async def build_database(self):
        logging.info("Starting to build database...")
        chunks = self.pdf_loader.load_and_split()
        logging.info(f"PDF split into {len(chunks)} chunks")
        try:
            await self.vector_db.build_from_chunks(chunks)
            logging.info("Database built successfully")
        except Exception as e:
            logging.error(f"Error building database: {str(e)}")
            logging.info("Partial database may have been built")

    async def query(self, question, k=3):
        logging.info(f"Querying with question: {question}")
        query_embedding = (await self.vector_db.get_embeddings_with_retry([question]))[0]
        logging.info("Query embedding generated")
        relevant_chunks = self.vector_db.search(query_embedding, k)
        logging.info(f"Found {len(relevant_chunks)} relevant chunks")
        
        context = "\n".join([chunk[0] for chunk in relevant_chunks])
        
        prompt = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

        logging.info("Sending request to OpenAI API")
        response = await self.vector_db.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        logging.info("Received response from OpenAI API")
        return response.choices[0].message.content

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting PDF RAG application")
    rag = PDFRAG("Workout2.pdf")
    await rag.build_database()
    
    while True:
        question = input("Ask a question (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        answer = await rag.query(question)
        print(f"Answer: {answer}\n")

    logging.info("PDF RAG application finished")

if __name__ == "__main__":
    asyncio.run(main())