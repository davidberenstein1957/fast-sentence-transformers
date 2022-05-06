from fast_sentence_transformers import FastSentenceTransformer

encoder = FastSentenceTransformer("all-MiniLM-L6-v2", quantize=False)

encoder.encode("Hello hello, hey, hello hello")
encoder.encode(["Life is too short to eat bad food!"] * 2)
