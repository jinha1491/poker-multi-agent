from datasets import load_dataset
import chromadb
from dotenv import load_dotenv

load_dotenv()

def load_pokerbench():
    print("Loading PokerBench dataset...")
    dataset = load_dataset("RZ412/PokerBench", split="train")
    
    client = chromadb.PersistentClient(path="./chroma_db")
    
    collection = client.get_or_create_collection(
        name="poker_strategy",
        metadata={"hnsw:space": "cosine"}
    )

    # check how many already loaded
    existing = collection.count()
    print(f"Already loaded: {existing} scenarios, resuming from there...")

    batch_size = 100
    for i in range(existing, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        
        collection.add(
            documents=batch["instruction"],
            metadatas=[{"optimal_action": action} for action in batch["output"]],
            ids=[f"hand_{i+j}" for j in range(len(batch["instruction"]))]
        )
        
        if i % 1000 == 0:
            print(f"Loaded {i} scenarios...")
    
    print("Done! PokerBench loaded into ChromaDB.")

if __name__ == "__main__":
    load_pokerbench()
