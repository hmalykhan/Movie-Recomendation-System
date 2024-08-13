import os
from dotenv import load_dotenv, dotenv_values
import pymongo
import requests
load_dotenv()
client=pymongo.MongoClient(os.getenv("MONGO_CLIENT"))
db=client.sample_mflix
collection=db.movies
# print(collection.find().limit(5))
hf_token = os.getenv("TOKEN")
embedding_url="https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        embedding_url,
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": text}
    )
    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
    return response.json()

# for doc in collection.find({'plot': {"$exists": True},'pilot_embedding_hf': {"$exists": True}}).limit(50):
#     del doc['pilot_embedding_hf']
#     doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
#     collection.replace_one({'_id': doc['_id']}, doc)
# for doc in collection.find({'plot': {"$exists": True},'pilot_embedding_hf': {"$exists": True}}).limit(50):
#      del doc['pilot_embedding_hf']

query = "imaginary characters from outer space at war."
results = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": generate_embedding(query),
            "path": "plot_embedding_hf",
            "numCandidates": 100,
            "limit": 4,
            "index": "PlotSemanticSearch"
        }
    }
]);

for document in results:
    print(f"Movie Name: {document['title']},\nMovie Plot: {document['plot']}\n")