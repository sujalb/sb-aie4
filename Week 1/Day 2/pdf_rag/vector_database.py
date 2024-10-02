import numpy as np
from collections import defaultdict
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import logging
import time
import random
import asyncio

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class RateLimiter:
    def __init__(self, rate=3, per=60):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()

    def check(self):
        now = time.time()
        time_passed = now - self.last_check
        self.last_check = now
        self.allowance += time_passed * (self.rate / self.per)
        if self.allowance > self.rate:
            self.allowance = self.rate
        if self.allowance < 1:
            return False
        self.allowance -= 1
        return True

rate_limiter = RateLimiter(rate=3, per=60)  # 3 requests per 60 seconds

class VectorDatabase:
    def __init__(self, max_concurrent_calls=3):
        self.vectors = defaultdict(np.array)
        self.client = AsyncOpenAI()
        self.semaphore = asyncio.Semaphore(max_concurrent_calls)
        logging.info(f"VectorDatabase initialized with AsyncOpenAI client (max {max_concurrent_calls} concurrent calls)")

    def insert(self, key, vector):
        self.vectors[key] = vector
        logging.info(f"Inserted vector for key: {key[:50]}...")

    def search(self, query_vector, k=5):
        logging.info(f"Searching for top {k} similar vectors")
        scores = [(key, cosine_similarity(query_vector, vector)) 
                  for key, vector in self.vectors.items()]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    async def get_embedding_with_rate_limit(self, text):
        async with self.semaphore:
            try:
                response = await self.client.embeddings.create(input=text, model="text-embedding-3-small")
                return response.data[0].embedding
            except Exception as e:
                logging.error(f"Error generating embedding: {str(e)}")
                raise
            finally:
                await asyncio.sleep(1)  # Ensure at least 1 second between calls

    async def get_embeddings_with_retry(self, texts, max_retries=5):
        all_embeddings = []
        for i, text in enumerate(texts):
            for attempt in range(max_retries):
                try:
                    embedding = await self.get_embedding_with_rate_limit(text)
                    all_embeddings.append(embedding)
                    logging.info(f"Successfully generated embedding for chunk {i+1}/{len(texts)}")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logging.warning(f"Attempt {attempt+1} failed. Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        logging.error(f"Failed to generate embedding for chunk {i+1} after {max_retries} attempts")
                        raise
        return all_embeddings

    async def build_from_chunks(self, chunks):
        logging.info(f"Building database from {len(chunks)} chunks")
        embeddings = await self.get_embeddings_with_retry(chunks)
        for chunk, embedding in zip(chunks, embeddings):
            self.insert(chunk, embedding)
        logging.info("Database built successfully")