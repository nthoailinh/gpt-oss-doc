// getp_run.cpp
// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"

// Comment following include if not profiling
#include "../util/profile.cpp"

// HIP GPU Optimization - Integrated Implementation
#include <hip/hip_runtime.h>
#include <iostream>
#include <cassert>
#include <cstring>

// HIP Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Global GPU memory buffers - persistent across inference calls
static struct GPUMemory {
    // Model weights on GPU (persistent)
    float *d_token_embedding;
    float *d_out_weights;          // final classification weights
    float *d_rms_attn_w;
    float *d_rms_ffn_w;
    float *d_rms_out_w;
    
    // Intermediate computation buffers
    float *d_x;                    // current hidden state
    float *d_logits;              // output logits
    float *d_temp_buffer;         // temporary computations
    
    // Configuration
    int hidden_dim;
    int vocab_size;
    int n_layers;
    bool weights_loaded;
    bool initialized;
    
    GPUMemory() : d_token_embedding(nullptr), d_out_weights(nullptr), 
                  d_rms_attn_w(nullptr), d_rms_ffn_w(nullptr), d_rms_out_w(nullptr),
                  d_x(nullptr), d_logits(nullptr), d_temp_buffer(nullptr),
                  hidden_dim(0), vocab_size(0), n_layers(0), 
                  weights_loaded(false), initialized(false) {}
} g_gpu_mem;

// GPU Kernels for Neural Network Operations

// Optimized matrix multiplication kernel
__global__ void matmul_kernel_fixed(float *out, const float *x, const float *w,
                                   int n, int d) {
    const int TILE_SIZE = 256;
    __shared__ float shared_x[TILE_SIZE];
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (row >= d) return;
    
    // Use double precision for accumulation to reduce numerical error
    double sum = 0.0;  // ✅ Double precision accumulator
    
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        int x_idx = tile * TILE_SIZE + tid;
        
        if (x_idx < n) {
            shared_x[tid] = x[x_idx];
        } else {
            shared_x[tid] = 0.0f;
        }
        __syncthreads();
        
        int tile_end = min(TILE_SIZE, n - tile * TILE_SIZE);
        for (int k = 0; k < tile_end; k++) {
            int col = tile * TILE_SIZE + k;
            int w_idx = row * n + col;
            
            // Double precision multiply-add
            sum += (double)shared_x[k] * (double)w[w_idx];
        }
        __syncthreads();
    }
    
    // Convert back to float for output
    out[row] = (float)sum;
}

// RMSNorm kernel
__global__ void rmsnorm_kernel(float *out, const float *x, const float *weight, int size) {
  extern __shared__ float sdata[];
  
  int tid = threadIdx.x;
  int stride = blockDim.x;
  
  // Each thread accumulates multiple elements
  float thread_sum = 0.0f;
  for (int i = tid; i < size; i += stride) {
      thread_sum += x[i] * x[i];
  }
  sdata[tid] = thread_sum;
  __syncthreads();
  
  // Block-wide reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
          sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
  }
  
  // Broadcast RMS factor to all threads
  __shared__ float rms_factor;
  if (tid == 0) {
      float variance = sdata[0] / size;
      rms_factor = rsqrtf(variance + 1e-5f);
  }
  __syncthreads();
  
  // Apply normalization - each thread handles multiple elements
  for (int i = tid; i < size; i += stride) {
      out[i] = weight[i] * rms_factor * x[i];
  }
}

// Token embedding lookup kernel
__global__ void token_embedding_kernel(float *out, const float *embedding_table, 
                                     int token, int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;
    
    out[idx] = embedding_table[token * hidden_dim + idx];
}

// Element-wise operations kernel
__global__ void elementwise_add_kernel(float *out, const float *a, const float *b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    out[idx] = a[idx] + b[idx];
}

#ifndef TEAM05_PROFILING
#define PROFILE_FUNCTION()
void start_timer() {}
void start_timer(const char *msg) {}
void stop_timer(const char *msg) {}
void stop_timer_and_print(const char *msg) {}
void reset_timing_summary() {}
void print_timing_summary() {}
#endif

#ifndef GETP_RUN
#define GETP_RUN

// GPU Memory Management Functions
void init_gpu_memory(int hidden_dim, int vocab_size, int n_layers) {
    if (g_gpu_mem.initialized) {
        return; // Already initialized
    }
    
    g_gpu_mem.hidden_dim = hidden_dim;
    g_gpu_mem.vocab_size = vocab_size;
    g_gpu_mem.n_layers = n_layers;
    
    // Allocate GPU memory for computation buffers
    HIP_CHECK(hipMalloc(&g_gpu_mem.d_x, hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_mem.d_logits, vocab_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_gpu_mem.d_temp_buffer, hidden_dim * sizeof(float)));
    
    g_gpu_mem.initialized = true;
    
    printf("GPU computation buffers allocated: hidden_dim=%d, vocab_size=%d\n", 
           hidden_dim, vocab_size);
}

void load_model_weights_to_gpu(TransformerWeights *weights, Config *config) {
    if (g_gpu_mem.weights_loaded) {
        return; // Already loaded
    }
    
    int hidden_dim = config->hidden_dim;
    int vocab_size = config->vocab_size;
    int n_layers = config->n_layers;
    
    printf("Loading model weights to GPU...\n");
    
    // Allocate and copy token embedding table
    size_t embedding_size = (size_t)vocab_size * hidden_dim * sizeof(float);
    HIP_CHECK(hipMalloc(&g_gpu_mem.d_token_embedding, embedding_size));
    HIP_CHECK(hipMemcpy(g_gpu_mem.d_token_embedding, weights->token_embedding_table, 
                        embedding_size, hipMemcpyHostToDevice));
    
    // Allocate and copy final output weights
    size_t out_weights_size = (size_t)hidden_dim * vocab_size * sizeof(float);
    HIP_CHECK(hipMalloc(&g_gpu_mem.d_out_weights, out_weights_size));
    HIP_CHECK(hipMemcpy(g_gpu_mem.d_out_weights, weights->out, 
                        out_weights_size, hipMemcpyHostToDevice));
    
    // Allocate and copy RMSNorm weights
    size_t rms_size = n_layers * hidden_dim * sizeof(float);
    HIP_CHECK(hipMalloc(&g_gpu_mem.d_rms_attn_w, rms_size));
    HIP_CHECK(hipMemcpy(g_gpu_mem.d_rms_attn_w, weights->rms_attn_w, 
                        rms_size, hipMemcpyHostToDevice));
    
    HIP_CHECK(hipMalloc(&g_gpu_mem.d_rms_ffn_w, rms_size));
    HIP_CHECK(hipMemcpy(g_gpu_mem.d_rms_ffn_w, weights->rms_ffn_w, 
                        rms_size, hipMemcpyHostToDevice));
    
    // Final RMSNorm
    HIP_CHECK(hipMalloc(&g_gpu_mem.d_rms_out_w, hidden_dim * sizeof(float)));
    HIP_CHECK(hipMemcpy(g_gpu_mem.d_rms_out_w, weights->rms_out_w, 
                        hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    
    g_gpu_mem.weights_loaded = true;
    
    printf("Model weights loaded to GPU successfully!\n");
    printf("- Token embedding: %.1f MB\n", embedding_size / (1024.0 * 1024.0));
    printf("- Output weights: %.1f MB\n", out_weights_size / (1024.0 * 1024.0));
    printf("- RMSNorm weights: %.1f MB\n", (rms_size * 2 + hidden_dim * sizeof(float)) / (1024.0 * 1024.0));
}

void cleanup_gpu_memory() {
    if (g_gpu_mem.initialized) {
        // Clean up computation buffers
        if (g_gpu_mem.d_x) HIP_CHECK(hipFree(g_gpu_mem.d_x));
        if (g_gpu_mem.d_logits) HIP_CHECK(hipFree(g_gpu_mem.d_logits));
        if (g_gpu_mem.d_temp_buffer) HIP_CHECK(hipFree(g_gpu_mem.d_temp_buffer));
        
        // Clean up model weights
        if (g_gpu_mem.d_token_embedding) HIP_CHECK(hipFree(g_gpu_mem.d_token_embedding));
        if (g_gpu_mem.d_out_weights) HIP_CHECK(hipFree(g_gpu_mem.d_out_weights));
        if (g_gpu_mem.d_rms_attn_w) HIP_CHECK(hipFree(g_gpu_mem.d_rms_attn_w));
        if (g_gpu_mem.d_rms_ffn_w) HIP_CHECK(hipFree(g_gpu_mem.d_rms_ffn_w));
        if (g_gpu_mem.d_rms_out_w) HIP_CHECK(hipFree(g_gpu_mem.d_rms_out_w));
        
        // Reset pointers
        g_gpu_mem.d_x = g_gpu_mem.d_logits = g_gpu_mem.d_temp_buffer = nullptr;
        g_gpu_mem.d_token_embedding = g_gpu_mem.d_out_weights = nullptr;
        g_gpu_mem.d_rms_attn_w = g_gpu_mem.d_rms_ffn_w = g_gpu_mem.d_rms_out_w = nullptr;
        
        g_gpu_mem.initialized = false;
        g_gpu_mem.weights_loaded = false;
        printf("GPU memory cleaned up\n");
    }
}

// GPU-accelerated final classification (RMSNorm → matmul)
void gpu_final_classification(float *h_logits, const float *h_x, int hidden_dim, int vocab_size) {
  const int BLOCK_SIZE = 256;
  
  // Copy input to GPU
  HIP_CHECK(hipMemcpy(g_gpu_mem.d_x, h_x, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  
  // RMSNorm: Single block with proper shared memory
  int rmsnorm_threads = min(BLOCK_SIZE, hidden_dim);
  size_t shared_mem_size = rmsnorm_threads * sizeof(float);
  
  hipLaunchKernelGGL(rmsnorm_kernel,
                     dim3(1),                    // Single block for reduction
                     dim3(rmsnorm_threads),      // Threads per block
                     shared_mem_size,            // Shared memory size
                     0,                          // Stream
                     g_gpu_mem.d_temp_buffer, g_gpu_mem.d_x, g_gpu_mem.d_rms_out_w, hidden_dim);
  
  HIP_CHECK(hipGetLastError());
  
  // MatMul: Grid covers output vocabulary
  int matmul_grid = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  hipLaunchKernelGGL(matmul_kernel_fixed,
                     dim3(matmul_grid), dim3(BLOCK_SIZE), 0, 0,
                     g_gpu_mem.d_logits, g_gpu_mem.d_temp_buffer, g_gpu_mem.d_out_weights, 
                     hidden_dim, vocab_size);
  
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  
  // Copy results back
  HIP_CHECK(hipMemcpy(h_logits, g_gpu_mem.d_logits, vocab_size * sizeof(float), hipMemcpyDeviceToHost));
}

void debug_gpu_kernels(Transformer *transformer) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  
  printf("🔧 Debugging GPU kernels...\n");
  
  // Test with small synthetic data first
  const int test_size = 1024;  // Small test
  float *test_input = (float*)malloc(test_size * sizeof(float));
  float *test_weight = (float*)malloc(test_size * sizeof(float));
  float *cpu_result = (float*)malloc(test_size * sizeof(float));
  float *gpu_result = (float*)malloc(test_size * sizeof(float));
  
  // Initialize test data
  for (int i = 0; i < test_size; i++) {
      test_input[i] = sinf(i * 0.01f);
      test_weight[i] = 1.0f;  // Unit weights
  }
  
  // CPU RMSNorm reference
  memcpy(cpu_result, test_input, test_size * sizeof(float));
  rmsnorm(cpu_result, cpu_result, test_weight, test_size);
  
  // GPU RMSNorm test
  float *d_test_in, *d_test_weight, *d_test_out;
  HIP_CHECK(hipMalloc(&d_test_in, test_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_test_weight, test_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_test_out, test_size * sizeof(float)));
  
  HIP_CHECK(hipMemcpy(d_test_in, test_input, test_size * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_test_weight, test_weight, test_size * sizeof(float), hipMemcpyHostToDevice));
  
  // Launch RMSNorm kernel
  const int BLOCK_SIZE = 256;
  int threads = min(BLOCK_SIZE, test_size);
  size_t shared_size = threads * sizeof(float);
  
  hipLaunchKernelGGL(rmsnorm_kernel,
                     dim3(1), dim3(threads), shared_size, 0,
                     d_test_out, d_test_in, d_test_weight, test_size);
  
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  
  HIP_CHECK(hipMemcpy(gpu_result, d_test_out, test_size * sizeof(float), hipMemcpyDeviceToHost));
  
  // Compare results
  float max_diff = 0.0f;
  for (int i = 0; i < test_size; i++) {
      float diff = fabsf(cpu_result[i] - gpu_result[i]);
      max_diff = fmaxf(max_diff, diff);
      
      if (i < 5) {  // Print first few elements
          printf("  [%d] CPU: %.6f, GPU: %.6f, diff: %.2e\n", 
                 i, cpu_result[i], gpu_result[i], diff);
      }
  }
  
  printf("RMSNorm test - Max difference: %.2e %s\n", 
         max_diff, (max_diff < 1e-5f) ? "✅ PASS" : "❌ FAIL");
  
  // Cleanup
  HIP_CHECK(hipFree(d_test_in));
  HIP_CHECK(hipFree(d_test_weight));
  HIP_CHECK(hipFree(d_test_out));
  free(test_input);
  free(test_weight);
  free(cpu_result);
  free(gpu_result);
}

void verify_matmul_kernel(Transformer *transformer) {
  printf("🔍 Verifying MatMul kernel with detailed debug...\n");
  
  Config *p = &transformer->config;
  int hidden_dim = p->hidden_dim;      // n = 32 (input dim)
  int vocab_size = p->vocab_size;      // d = 201088 (output dim)
  
  printf("Matrix dimensions: W[%d,%d] * x[%d] -> out[%d]\n", 
         vocab_size, hidden_dim, hidden_dim, vocab_size);
  
  // Create simple test vector
  float *test_x = (float*)malloc(hidden_dim * sizeof(float));
  for (int i = 0; i < hidden_dim; i++) {
      test_x[i] = (float)(i + 1);  // [1, 2, 3, ..., 32]
  }
  
  printf("Test input: x[0:5] = [%.1f, %.1f, %.1f, %.1f, %.1f]\n",
         test_x[0], test_x[1], test_x[2], test_x[3], test_x[4]);
  
  // Check a few weight values for debugging
  float *weights = transformer->weights.out;
  printf("Weight samples: W[0,0:3] = [%.4f, %.4f, %.4f]\n",
         weights[0], weights[1], weights[2]);
  printf("Weight samples: W[1,0:3] = [%.4f, %.4f, %.4f]\n", 
         weights[hidden_dim], weights[hidden_dim + 1], weights[hidden_dim + 2]);
  
  // CPU and GPU computation...
  float *cpu_out = (float*)malloc(vocab_size * sizeof(float));
  float *gpu_out = (float*)malloc(vocab_size * sizeof(float));
  
  // CPU reference
  matmul(cpu_out, test_x, weights, hidden_dim, vocab_size);
  
  // GPU test
  float *d_x, *d_out;
  HIP_CHECK(hipMalloc(&d_x, hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, vocab_size * sizeof(float)));
  
  HIP_CHECK(hipMemcpy(d_x, test_x, hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  
  const int BLOCK_SIZE = 256;
  int grid_size = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  hipLaunchKernelGGL(matmul_kernel_fixed,
                     dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                     d_out, d_x, g_gpu_mem.d_out_weights, 
                     hidden_dim, vocab_size);
  
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  
  HIP_CHECK(hipMemcpy(gpu_out, d_out, vocab_size * sizeof(float), hipMemcpyDeviceToHost));
  
  // Detailed comparison
  float max_diff = 0.0f;
  int mismatch_count = 0;
  
  printf("First 10 elements comparison:\n");
  for (int i = 0; i < min(10, vocab_size); i++) {
      float diff = fabsf(cpu_out[i] - gpu_out[i]);
      max_diff = fmaxf(max_diff, diff);
      
      if (diff > 1e-4f) mismatch_count++;
      
      printf("  [%d] CPU: %10.4f, GPU: %10.4f, diff: %.2e %s\n", 
             i, cpu_out[i], gpu_out[i], diff, (diff > 1e-4f) ? "❌" : "✅");
  }
  
  printf("MatMul verification: max_diff=%.2e, mismatches=%d/10 %s\n", 
         max_diff, mismatch_count, (mismatch_count == 0) ? "✅ PASS" : "❌ FAIL");
  
  // Manual verification of first element
  float manual_result = 0.0f;
  for (int j = 0; j < hidden_dim; j++) {
      manual_result += weights[0 * hidden_dim + j] * test_x[j];
  }
  printf("Manual check for out[0]: %.4f (should match CPU: %.4f)\n", 
         manual_result, cpu_out[0]);
  
  // Cleanup
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_out));
  free(test_x);
  free(cpu_out);
  free(gpu_out);
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the warm-up process
  // TODO:
  // - Memory allocation
  // - Load model
  // - ...
  
  // Initialize HIP context
  int device_count;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  
  if (device_count == 0) {
      std::cerr << "No HIP devices found!" << std::endl;
      exit(1);
  }
  
  HIP_CHECK(hipSetDevice(0)); // Use first GPU
  
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  
  printf("Using GPU: %s\n", prop.name);
  printf("GPU Memory: %ld MB\n", prop.totalGlobalMem / (1024*1024));
  
  // Initialize GPU memory and load model weights
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  
  printf("Model config: hidden_dim=%d, vocab_size=%d, n_layers=%d\n", 
         p->hidden_dim, p->vocab_size, p->n_layers);
  
  // Step 1: Allocate GPU computation buffers
  init_gpu_memory(p->hidden_dim, p->vocab_size, p->n_layers);
  
  // Step 2: Load model weights to GPU (one-time cost)
  load_model_weights_to_gpu(w, p);
  
  printf("🚀 Full GPU pipeline initialized - weights loaded to GPU!\n");

  // Add kernel debugging
  printf("\n=== GPU Kernel Debugging ===\n");
  debug_gpu_kernels(transformer);
  verify_matmul_kernel(transformer);
  printf("============================\n\n");
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the finish process
  // TODO:
  // - Memory deallocation
  // - Unload model
  // - ...
  
  // Cleanup GPU resources
  cleanup_gpu_memory();
}

float *forward_hip(Transformer *transformer, int token, int pos) {
  PROFILE_FUNCTION();

  start_timer();

  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  float *x = s->x;
  int head_dim = p->head_dim;
  int hidden_dim = p->hidden_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int kv_mul =
      p->n_attn_heads /
      p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int intermediate_dim = p->intermediate_dim;
  int n_experts = p->n_experts;

  // copy the token embedding into x
  float *content_row = w->token_embedding_table + token * hidden_dim;
  memcpy(x, content_row, hidden_dim * sizeof(*x));

  stop_timer("init");

  start_timer("forward_layers");

  // forward all the layers
  for (unsigned long long l = 0; l < p->n_layers; l++) {
    // s->t (hidden_dim, )
    rmsnorm(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, hidden_dim);

    // key and value point to the kv cache
    int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    // s->qkv = w->w_qkv * s->t = (head_dim * (n_attn_heads + 2 * n_kv_heads),
    // hidden_dim) * (hidden_dim, ) = head_dim * (n_attn_heads + 2 * n_kv_heads)
    float *w_qkv = w->w_qkv + 1ll * l * hidden_dim *
                                  (head_dim * p->n_attn_heads +
                                   2 * head_dim * p->n_kv_heads);
    float *b_qkv =
        w->b_qkv +
        1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    matmul(s->qkv, s->t, w_qkv, hidden_dim,
           (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
    // add bias
    for (int i = 0; i < (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim; ++i) {
      s->qkv[i] += b_qkv[i];
    }
    // Separate q, k, v
    memcpy(s->q, s->qkv, head_dim * p->n_attn_heads * sizeof(float)); // gate
    memcpy(s->k, s->qkv + head_dim * p->n_attn_heads,
           head_dim * p->n_kv_heads * sizeof(float)); // gate
    memcpy(s->v, s->qkv + head_dim * p->n_attn_heads + head_dim * p->n_kv_heads,
           head_dim * p->n_kv_heads * sizeof(float)); // gate

    // RoPE relative positional encoding: complex-valued rotate q and k in each
    // head Adapted from
    // https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py#L85
    // RoPE with YaRN scaling adapted from Python code
    float ntk_beta = 32.0f;
    float ntk_alpha = 1.0f;
    float *cos_vals =
        reinterpret_cast<float *>(malloc((head_dim / 2) * sizeof(float)));
    float *sin_vals =
        reinterpret_cast<float *>(malloc((head_dim / 2) * sizeof(float)));
    compute_cos_sin(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                    p->initial_context_length, ntk_beta, ntk_alpha, cos_vals,
                    sin_vals);
    apply_rotary_emb(s->q, cos_vals, sin_vals, p->n_attn_heads, head_dim);
    apply_rotary_emb(s->k, cos_vals, sin_vals, p->n_kv_heads, head_dim);

    free(cos_vals);
    free(sin_vals);

    // multihead attention. iterate over all heads
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < p->n_attn_heads; h++) {
      // get the query vector for this head
      float *q = s->q + h * head_dim;
      // attention scores for this head
      float *att = s->att + h * p->seq_len;
      // iterate over all timesteps, including the current one
      for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        // GQA
        float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
        // calculate the attention score as the dot product of q and k
        double score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
          score += q[i] * k[i];
        }
        score /= sqrtf(head_dim);
        // Apply sliding window mask if enabled
        if (p->sliding_window > 0 && (l % 2 == 0)) {
          score += s->mask[pos * p->seq_len + t];
        }
        // save the score to the attention buffer
        att[t] = score;
      }
      // Add attention sink score
      att[pos + 1] = w->attn_sinks[l * p->n_attn_heads + h];
      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax(att, pos + 2);

      // weighted sum of the values
      float *tb = s->tb + h * head_dim;
      memset(tb, 0, head_dim * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        // GQA
        float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
        // get the attention weight for this timestep
        float a = att[t];
        // accumulate the weighted value into xb
        for (int i = 0; i < head_dim; i++) {
          tb[i] += a * v[i];
        }
      }
    }
    // final matmul to get the output of the attention
    float *w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    float *b_o = w->b_o + 1ll * l * hidden_dim;
    matmul(s->tb2, s->tb, w_o, head_dim * p->n_attn_heads, hidden_dim);
    // add bias b_o
    for (int i = 0; i < hidden_dim; i++) {
      s->tb2[i] += b_o[i];
    }

    // residual connection back into x
    for (int i = 0; i < hidden_dim; i++) {
      x[i] += s->tb2[i];
    }

    // ffn rmsnorm
    rmsnorm(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, hidden_dim);

    // MoE
    // Compute router_score
    float *w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
    float *b_router = w->b_router + 1ll * l * n_experts;
    matmul(s->router_score, s->t, w_router, hidden_dim,
           n_experts); // s->router_score now stores router_score (n_experts, )
    // add bias b_router
    for (int i = 0; i < n_experts; i++) {
      s->router_score[i] += b_router[i];
    }
    // Select top-k experts
    topk(s->topk_v, s->topk_i, s->router_score, n_experts,
         p->experts_per_token);
    // Normalize selected experts using softmax or sigmoid
    softmax(s->topk_v, p->experts_per_token); // expert

    // Route the tokens to their corresponding top-k experts
    memset(s->e_agg, 0, hidden_dim * sizeof(float));
    for (int e = 0; e < n_experts; e++) {
      float expert_w = 0;
      int in_topk = 0;
      // Check if expert i is in top-k experts
      for (int idx = 0; idx < p->experts_per_token; idx++) {
        if (s->topk_i[idx] == e) {
          in_topk = 1;
          expert_w = s->topk_v[idx];
          break;
        }
      }

      if (in_topk) {
        float *w_mlp1 = w->w_mlp1 + 1ll * (l * n_experts + e) *
                                        (2 * p->intermediate_dim) * hidden_dim;
        float *b_mlp1 =
            w->b_mlp1 + 1ll * (l * n_experts + e) * (2 * p->intermediate_dim);
        matmul(s->mlp1_out, s->t, w_mlp1, hidden_dim,
               2 * p->intermediate_dim); // (2 * intermediate_dim, )
        for (int i = 0; i < 2 * p->intermediate_dim; i++) {
          s->mlp1_out[i] += b_mlp1[i];
        }
        // Split mlp1_out into gate and up
        for (int j = 0; j < p->intermediate_dim; j++) {
          s->gate[j] = s->mlp1_out[2 * j];
          s->up[j] = s->mlp1_out[2 * j + 1];
        }

        // SwiGLU non-linearity
        const float alpha = 1.702f;
        for (int i = 0; i < p->intermediate_dim; i++) {
          float val = s->gate[i];
          float up_val = s->up[i];
          // Clamping
          if (val > p->swiglu_limit)
            val = p->swiglu_limit;
          if (up_val > p->swiglu_limit)
            up_val = p->swiglu_limit;
          if (up_val < -p->swiglu_limit)
            up_val = -p->swiglu_limit;
          // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
          val *= (1.0f / (1.0f + expf(-alpha * val)));
          // elementwise multiply with w_gate(x)
          val *= (up_val +
                  1.0f); // gpt-oss adds an extra bias of 1 to the up layer
          s->gate_up[i] = val;
        }

        // final matmul to get the output of the ffn
        float *w_mlp2 =
            w->w_mlp2 +
            1ll * (l * n_experts + e) * hidden_dim *
                p->intermediate_dim; // (out: hidden_dim, in: intermediate_dim)
        float *b_mlp2 = w->b_mlp2 + 1ll * (l * n_experts + e) * hidden_dim;
        matmul(s->tb2, s->gate_up, w_mlp2, p->intermediate_dim,
               hidden_dim); // (hidden_dim, )
        for (int i = 0; i < hidden_dim; i++) {
          s->tb2[i] += b_mlp2[i];
        }

        // aggregate topk experts using weighted sum
        for (int i = 0; i < hidden_dim; i++) {
          s->e_agg[i] += s->tb2[i] * expert_w;
        }
      }
    }

    // residual connection
    for (int i = 0; i < hidden_dim; i++) {
      x[i] += s->e_agg[i];
    }
  }

  stop_timer("forward_layers");

  start_timer("final_matmul");

  // GPU-accelerated final classification pipeline (rmsnorm → matmul)
  // Skip CPU rmsnorm - do both rmsnorm and matmul on GPU!
  gpu_final_classification(s->logits, x, hidden_dim, p->vocab_size);

  stop_timer("final_matmul");
  return s->logits;
}

// write File
std::ofstream writeFile_val("data/logits.txt");
bool print_logit = true;

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
  // <|start|>: 200006
  // <|end|>: 200007
  // <|return|>: 200002
  // <|message|>: 200008
  // <|channel|>: 200005
  // <|constrain|>: 200003
  // <|endoftext|>: 199999
  
  // Inference here

  const char *empty_prompt = "";
  if (input_seq == NULL) {
    input_seq = empty_prompt;
  }

  start_timer("malloc_prompt_tokens");

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) *
                                     sizeof(int)); // +3 for '\0', ?BOS, ?EOS

  stop_timer("malloc_prompt_tokens");

  start_timer("encode_prompt");

  encode(tokenizer, input_seq, 1, 0, prompt_tokens, &num_prompt_tokens,
         transformer->config.initial_context_length);

  stop_timer("encode_prompt");

  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence
  while (pos < steps) {

    // forward the transformer to get logits for the next token
    float *logits = forward_hip(transformer, token, pos);

    if (print_logit && pos % 100 == 0 && writeFile_val.is_open()) {
      for (int q = 0; q < sampler->vocab_size; q += 25) {
        writeFile_val << logits[q] << ' ';
      }
      writeFile_val << '\n';
    }

    // advance the state machine
    pos++;
    if (pos < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt
      // token
      next = prompt_tokens[pos];
    } else {
      start_timer("sample_next_token");

      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
      // save the output token, it will be printed to file
      output_tokens[pos - num_prompt_tokens] = next;

      stop_timer("sample_next_token");
    }

    // data-dependent terminating condition: the EOS (=199999 or =200002) token
    // delimits sequences
    if (next == 199999 || next == 200002) {
      break;
    }

    start_timer("decode_piece");

    // print the token as string, decode it with the Tokenizer object
    // should be removed
    const char *piece = decode_piece(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes

    stop_timer("decode_piece");
    fflush(stdout);

    token = next;
  }

  // should be removed
  printf("\n");

  // Marker for end of sequence
  output_tokens[pos - num_prompt_tokens + 1] = -1;

  free(prompt_tokens);

  return pos - num_prompt_tokens + 1;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {

  reset_timing_summary();

  long long num_token_out = 0;
  for (int idx = 0; idx < requests->num_reqs; ++idx) {
    const char *input_seq = get_str_req_ptr(requests, idx);
    int *output_tokens = get_tok_gen_ptr(requests, idx);
    num_token_out +=
        simple_getp_generate(transformer, tokenizer, sampler, input_seq,
                             output_tokens, requests->max_seq_len);
  }

  writeFile_val.close();
  print_timing_summary();
  return num_token_out;
}

#endif // GETP_RUN
