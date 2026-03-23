#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMacros.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ y,
    int64_t rows,
    int64_t cols,
    float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float shared_sum[];

    if (row >= rows) {
        return;
    }

    const scalar_t* row_x = x + row * cols;
    scalar_t* row_y = y + row * cols;

    float local_sum = 0.0f;
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        const float value = static_cast<float>(row_x[col]);
        local_sum += value * value;
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    const float rstd = rsqrtf(shared_sum[0] / static_cast<float>(cols) + eps);
    for (int64_t col = tid; col < cols; col += blockDim.x) {
        const float value = static_cast<float>(row_x[col]);
        const float scale = static_cast<float>(weight[col]);
        row_y[col] = static_cast<scalar_t>(value * rstd * scale);
    }
}

template <typename scalar_t>
__global__ void silu_mul_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t* __restrict__ out,
    int64_t total_elements) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    const float gate_value = static_cast<float>(gate[idx]);
    const float up_value = static_cast<float>(up[idx]);
    const float silu = gate_value / (1.0f + expf(-gate_value));
    out[idx] = static_cast<scalar_t>(silu * up_value);
}

template <typename scalar_t>
__global__ void store_kvcache_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    const int64_t* __restrict__ slot_mapping,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = num_tokens * num_kv_heads * head_dim;
    if (idx >= total) {
        return;
    }

    const int64_t token_idx = idx / (num_kv_heads * head_dim);
    const int64_t rem = idx % (num_kv_heads * head_dim);
    const int64_t head_idx = rem / head_dim;
    const int64_t dim_idx = rem % head_dim;

    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
        return;
    }

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    const int64_t input_offset = idx;
    const int64_t k_offset = (((block_idx * num_kv_heads + head_idx) * head_dim + dim_idx) * block_size) + block_offset;
    const int64_t v_offset = (((block_idx * num_kv_heads + head_idx) * block_size + block_offset) * head_dim) + dim_idx;

    k_cache[k_offset] = key[input_offset];
    v_cache[v_offset] = value[input_offset];
}

template <typename scalar_t>
__global__ void paged_attention_decode_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ k_cache,
    const scalar_t* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables,
    const int64_t* __restrict__ context_lens,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t max_num_blocks,
    float scale) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= num_heads) {
        return;
    }

    const int64_t ctx_len = context_lens[batch_idx];
    if (ctx_len <= 0) {
        for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
            output[(static_cast<int64_t>(batch_idx) * num_heads + head_idx) * head_dim + dim] = 0.0f;
        }
        return;
    }

    const int num_kv_groups = static_cast<int>(num_heads / num_kv_heads);
    const int kv_head_idx = head_idx / num_kv_groups;

    extern __shared__ float shared_memory[];
    float* shared_scalars = shared_memory;
    float* shared_probs = shared_memory + 2;

    if (tid == 0) {
        float max_score = -FLT_MAX;
        for (int64_t block_num = 0; block_num < max_num_blocks; ++block_num) {
            const int64_t start = block_num * block_size;
            if (start >= ctx_len) {
                break;
            }
            const int32_t phys_block = block_tables[batch_idx * max_num_blocks + block_num];
            if (phys_block < 0) {
                continue;
            }
            const int64_t valid_tokens = (ctx_len - start) < block_size ? (ctx_len - start) : block_size;
            for (int64_t token_offset = 0; token_offset < valid_tokens; ++token_offset) {
                float score = 0.0f;
                for (int64_t dim = 0; dim < head_dim; ++dim) {
                    const float qv = static_cast<float>(query[((static_cast<int64_t>(batch_idx) * num_heads + head_idx) * head_dim) + dim]);
                    const float kv = static_cast<float>(k_cache[(((static_cast<int64_t>(phys_block) * num_kv_heads + kv_head_idx) * head_dim + dim) * block_size) + token_offset]);
                    score += qv * kv;
                }
                score *= scale;
                if (score > max_score) {
                    max_score = score;
                }
            }
        }

        float denom = 0.0f;
        for (int64_t block_num = 0; block_num < max_num_blocks; ++block_num) {
            const int64_t start = block_num * block_size;
            if (start >= ctx_len) {
                break;
            }
            const int32_t phys_block = block_tables[batch_idx * max_num_blocks + block_num];
            if (phys_block < 0) {
                continue;
            }
            const int64_t valid_tokens = (ctx_len - start) < block_size ? (ctx_len - start) : block_size;
            for (int64_t token_offset = 0; token_offset < valid_tokens; ++token_offset) {
                float score = 0.0f;
                for (int64_t dim = 0; dim < head_dim; ++dim) {
                    const float qv = static_cast<float>(query[((static_cast<int64_t>(batch_idx) * num_heads + head_idx) * head_dim) + dim]);
                    const float kv = static_cast<float>(k_cache[(((static_cast<int64_t>(phys_block) * num_kv_heads + kv_head_idx) * head_dim + dim) * block_size) + token_offset]);
                    score += qv * kv;
                }
                denom += expf(score * scale - max_score);
            }
        }

        shared_scalars[0] = max_score;
        shared_scalars[1] = denom > 0.0f ? denom : 1.0f;
    }
    __syncthreads();

    const float max_score = shared_scalars[0];
    const float denom = shared_scalars[1];

    for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
        float acc = 0.0f;
        for (int64_t block_num = 0; block_num < max_num_blocks; ++block_num) {
            const int64_t start = block_num * block_size;
            if (start >= ctx_len) {
                break;
            }
            const int32_t phys_block = block_tables[batch_idx * max_num_blocks + block_num];
            if (phys_block < 0) {
                continue;
            }
            const int64_t valid_tokens = (ctx_len - start) < block_size ? (ctx_len - start) : block_size;

            if (tid == 0) {
                for (int64_t token_offset = 0; token_offset < valid_tokens; ++token_offset) {
                    float score = 0.0f;
                    for (int64_t inner_dim = 0; inner_dim < head_dim; ++inner_dim) {
                        const float qv = static_cast<float>(query[((static_cast<int64_t>(batch_idx) * num_heads + head_idx) * head_dim) + inner_dim]);
                        const float kv = static_cast<float>(k_cache[(((static_cast<int64_t>(phys_block) * num_kv_heads + kv_head_idx) * head_dim + inner_dim) * block_size) + token_offset]);
                        score += qv * kv;
                    }
                    shared_probs[token_offset] = expf(score * scale - max_score) / denom;
                }
            }
            __syncthreads();

            for (int64_t token_offset = 0; token_offset < valid_tokens; ++token_offset) {
                const float vv = static_cast<float>(v_cache[(((static_cast<int64_t>(phys_block) * num_kv_heads + kv_head_idx) * block_size + token_offset) * head_dim) + dim]);
                acc += shared_probs[token_offset] * vv;
            }
            __syncthreads();
        }
        output[(static_cast<int64_t>(batch_idx) * num_heads + head_idx) * head_dim + dim] = acc;
    }
}

torch::Tensor rms_norm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps) {
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    auto output = torch::empty_like(x_contig);

    const int64_t cols = x_contig.size(-1);
    const int64_t rows = x_contig.numel() / cols;
    const int threads = 256;
    const size_t shared_mem = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_contig.scalar_type(), "rms_norm_forward_cuda", ([&] {
        rms_norm_kernel<scalar_t><<<rows, threads, shared_mem, at::cuda::getDefaultCUDAStream()>>>(
            x_contig.data_ptr<scalar_t>(),
            weight_contig.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            rows,
            cols,
            static_cast<float>(eps));
    }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor silu_mul_forward_cuda(
    torch::Tensor gate,
    torch::Tensor up) {
    auto gate_contig = gate.contiguous();
    auto up_contig = up.contiguous();
    auto output = torch::empty_like(gate_contig);

    const int64_t total = gate_contig.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(gate_contig.scalar_type(), "silu_mul_forward_cuda", ([&] {
        silu_mul_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            gate_contig.data_ptr<scalar_t>(),
            up_contig.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total);
    }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void store_kvcache_forward_cuda(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping,
    int64_t block_size) {
    auto key_contig = key.contiguous();
    auto value_contig = value.contiguous();
    auto slot_contig = slot_mapping.contiguous();

    const int64_t num_tokens = key_contig.size(0);
    const int64_t num_kv_heads = key_contig.size(1);
    const int64_t head_dim = key_contig.size(2);
    const int64_t total = num_tokens * num_kv_heads * head_dim;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(key_contig.scalar_type(), "store_kvcache_forward_cuda", ([&] {
        store_kvcache_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            key_contig.data_ptr<scalar_t>(),
            value_contig.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            slot_contig.data_ptr<int64_t>(),
            num_tokens,
            num_kv_heads,
            head_dim,
            block_size);
    }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor paged_attention_decode_forward_cuda(
    torch::Tensor query,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    double scale,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size) {
    auto query_contig = query.contiguous();
    auto k_cache_contig = k_cache.contiguous();
    auto v_cache_contig = v_cache.contiguous();
    auto block_tables_contig = block_tables.contiguous();
    auto context_lens_contig = context_lens.contiguous();

    const int64_t batch_size = query_contig.size(0);
    const int64_t max_num_blocks = block_tables_contig.size(1);
    auto output = torch::zeros({batch_size, num_heads, head_dim}, query_contig.options().dtype(torch::kFloat32));

    const int threads = static_cast<int>(std::min<int64_t>(256, std::max<int64_t>(32, head_dim)));
    const dim3 grid(batch_size, num_heads);
    const size_t shared_mem = (2 + block_size) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query_contig.scalar_type(), "paged_attention_decode_forward_cuda", ([&] {
        paged_attention_decode_kernel<scalar_t><<<grid, threads, shared_mem, at::cuda::getDefaultCUDAStream()>>>(
            query_contig.data_ptr<scalar_t>(),
            k_cache_contig.data_ptr<scalar_t>(),
            v_cache_contig.data_ptr<scalar_t>(),
            block_tables_contig.data_ptr<int32_t>(),
            context_lens_contig.data_ptr<int64_t>(),
            output.data_ptr<float>(),
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_num_blocks,
            static_cast<float>(scale));
    }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}