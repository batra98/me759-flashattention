#!/usr/bin/env python3
"""NumPy reference: scaled dot-product attention."""
import sys, numpy as np

def sdp(Q, K, V):
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)
    S -= S.max(axis=-1, keepdims=True)
    P  = np.exp(S); P /= P.sum(axis=-1, keepdims=True)
    return (P @ V).astype(np.float32)


def sdp_causal(Q, K, V):
    """Lower-triangular causal mask (autoregressive attention)."""
    d = Q.shape[-1]
    N = Q.shape[0]
    S = Q @ K.T / np.sqrt(d)
    mask = np.triu(np.ones((N, N), dtype=np.float32) * (-np.inf), k=1)
    S = S + mask
    S -= S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=-1, keepdims=True)
    return (P @ V).astype(np.float32)

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv)>1 else 1024
    d = int(sys.argv[2]) if len(sys.argv)>2 else 64
    rng = np.random.default_rng(42)
    Q = (rng.random((N,d))*0.1).astype(np.float32)
    K = (rng.random((N,d))*0.1).astype(np.float32)
    V = (rng.random((N,d))*0.1).astype(np.float32)
    O = sdp(Q, K, V)
    np.save("ref_output.npy", O)
    print(f"Saved reference → ref_output.npy  shape={O.shape}")
