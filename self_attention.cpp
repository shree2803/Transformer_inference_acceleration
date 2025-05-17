//#include <ap_int.h>
#include <hls_stream.h>
//#include <hls_math.h>
#include <iostream>
#include "self_attention.h"


#define TENSOR_SIZE (SEQ_LEN * HEADS * HEAD_DIM)


extern "C" {
void self_attention(
    const int *Q_flat,  // [SEQ_LEN][HEADS][HEAD_DIM] -> flattened
    const int *K_flat,
    const int *V_flat,
    float *OUT_flat       // output: [SEQ_LEN][HEADS][HEAD_DIM]
) {




#pragma HLS INTERFACE s_axilite port=Q_flat     bundle=control
#pragma HLS INTERFACE s_axilite port=K_flat     bundle=control
#pragma HLS INTERFACE s_axilite port=V_flat     bundle=control
#pragma HLS INTERFACE s_axilite port=OUT_flat   bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control


    float Q[SEQ_LEN][HEADS][HEAD_DIM];
    float K[SEQ_LEN][HEADS][HEAD_DIM];
    float V[SEQ_LEN][HEADS][HEAD_DIM];
    float OUT[SEQ_LEN][HEADS][HEAD_DIM] = {0};
    float attn[SEQ_LEN][SEQ_LEN][HEADS];


    // Load inputs
    int idx = 0;
    for (int i = 0; i < SEQ_LEN; ++i)
        for (int h = 0; h < HEADS; ++h)
	#pragma HLS unroll
            for (int d = 0; d < HEAD_DIM; ++d, ++idx) {
                #pragma HLS unroll
                Q[i][h][d] = Q_flat[idx];
                K[i][h][d] = K_flat[idx];
                V[i][h][d] = V_flat[idx];
            }


    // Compute attention scores
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int j = 0; j < SEQ_LEN; ++j) {
		#pragma HLS unroll
            for (int h = 0; h < HEADS; ++h) {
			#pragma HLS unroll
                float score = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q[i][h][d] * K[j][h][d];
                }
                attn[i][j][h] = score / HEAD_DIM;
            }
        }
    }


    // Weighted sum with V
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int h = 0; h < HEADS; ++h) {
            for (int d = 0; d < HEAD_DIM; ++d) {
			#pragma HLS unroll
                float val = 0.0f;
                for (int j = 0; j < SEQ_LEN; ++j) {
				#pragma HLS unroll
                    val += attn[i][j][h] * V[j][h][d];
                }
                OUT[i][h][d] = val;
            }
        }
    }


    // Store output
    idx = 0;
    for (int i = 0; i < SEQ_LEN; ++i)
        for (int h = 0; h < HEADS; ++h)
            for (int d = 0; d < HEAD_DIM; ++d, ++idx) {
                OUT_flat[idx] = OUT[i][h][d];
            }
}
}
