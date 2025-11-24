#  FPGA-Accelerated Self-Attention — Code Explanation

This project demonstrates how the Transformer’s **self-attention block** is accelerated using a custom **Vitis HLS kernel** deployed on a **PYNQ-Z2 FPGA**.  
Python handles the rest of the Transformer inference, while the FPGA overlay computes the **self-attention scores and value mixing**.

---

## 1. HLS Kernel: `self_attention`

### **Function**
```cpp
void self_attention(const int *Q_flat,
                    const int *K_flat,
                    const int *V_flat,
                    float *OUT_flat);
```
### What the Hardware Does

- Receives flattened **Q, K, V** tensors over **AXI-Lite**.
- Reconstructs them into 3D arrays:
  - `Q[SEQ_LEN][HEADS][HEAD_DIM]`
  - `K[SEQ_LEN][HEADS][HEAD_DIM]`
  - `V[SEQ_LEN][HEADS][HEAD_DIM]`
- Computes **QKᵀ attention scores** for every `(i, j, head)`.
- Performs **weighted sum with V** to produce context vectors.
- Writes the flattened output tensor back via **AXI-Lite**.

### Key Optimizations

- `#pragma HLS unroll` used to parallelize small loops.
- On-chip **BRAM arrays** for fast access.
- Scaling by `1 / HEAD_DIM` implemented in hardware.
- Clean **AXI-Lite control interface** for simple Python integration.
