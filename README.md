# Hybrid retrieval demo

BM25 + MiniLM (ONNX) over 20,958 SQuAD paragraphs. Evaluation, limits and code: https://github.com/dmatharv25-cmd/rag-retrieval-eval

## Live demo

https://rag-demo-7iru.onrender.com (free Render instance, so the first load can take a minute).

## Known limit

In the live demo, "Who was the first emperor of Rome?" returns a Constantine paragraph at rank 1 and the Augustus paragraph at rank 2. Lexical overlap can outrank the true answer. This is one example, not a measured rate. Full evaluation and limits are in the main repo.

## Resources

- **Memory:** about 355 MB after loading and about 400 MB after a query, measured locally on Windows. Render's free tier has 512 MB. On Render's free tier, the service stayed live after test queries, with no restart or out-of-memory event in Render's Events tab. I did not read the memory peak there.
- **Latency:** on Render's free tier (0.15 CPU, 512 MB), 12 consecutive queries on the live demo all succeeded and returned in 1.6 to 2.0 s each (mean 1.8 s), timed from a local machine, so this includes the network round trip.

## Back to the evaluation

Method, results and code: https://github.com/dmatharv25-cmd/rag-retrieval-eval
