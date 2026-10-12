import json
import os
import re
import numpy as np
import psutil
import scipy.sparse as sp
import onnxruntime as ort
import gradio as gr
from tokenizers import Tokenizer

W = 0.6


def tok(s):
    return re.findall(r"\w+", s.lower())


with open("data/corpus.json", encoding="utf-8") as f:
    corpus = json.load(f)
C = np.load("data/minilm_corpus_emb.npy")
N = len(corpus)
M = sp.load_npz("data/bm25_M.npz").tocsc()
idf = np.load("data/bm25_idf.npy")
with open("data/bm25_vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)

tk = Tokenizer.from_file("minilm_onnx/tokenizer.json")
tk.enable_truncation(max_length=256)
tk.enable_padding()
so = ort.SessionOptions()
so.enable_cpu_mem_arena = False
sess = ort.InferenceSession("minilm_onnx/model.onnx", so, providers=["CPUExecutionProvider"])


def bm25_scores(q):
    s = np.zeros(N, dtype=np.float32)
    for t in tok(q):
        i = vocab.get(t)
        if i is not None:
            col = M[:, i]
            s[col.indices] += idf[i] * col.data
    return s


def embed(text):
    e = tk.encode(text)
    ids = np.array([e.ids], dtype=np.int64)
    mask = np.array([e.attention_mask], dtype=np.int64)
    out = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    v = (out * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
    return (v / np.linalg.norm(v, axis=1, keepdims=True))[0]


def z(s):
    return (s - s.mean()) / (s.std() + 1e-9)


def mem():
    return round(psutil.Process(os.getpid()).memory_info().rss / 1e6)


print("memory after loading (MB):", mem())


def search(question):
    if not question or not question.strip():
        return "Type a question."
    h = (1 - W) * z(bm25_scores(question)) + W * z(C @ embed(question))
    top = np.argsort(-h)[:3]
    out = [f"**Result {r}** (hybrid score {h[i]:.2f})\n\n{corpus[i]}" for r, i in enumerate(top, 1)]
    print("memory after query (MB):", mem())
    return "\n\n---\n\n".join(out)


demo = gr.Interface(
    fn=search,
    inputs=gr.Textbox(label="Ask a question about Wikipedia topics (SQuAD)", lines=2),
    outputs=gr.Markdown(label="Top 3 paragraphs"),
    title="Hybrid Retrieval Demo: BM25 + MiniLM",
    description="Finds the most relevant paragraph among 20,958 SQuAD paragraphs. See the repo for evaluation and limits: https://github.com/dmatharv25-cmd/rag-retrieval-eval",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
