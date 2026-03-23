#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);      \
    CHECK_CONTIGUOUS(x)

torch::Tensor rms_norm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps);

torch::Tensor silu_mul_forward_cuda(
    torch::Tensor gate,
    torch::Tensor up);

void store_kvcache_forward_cuda(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping,
    int64_t block_size);

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
    int64_t block_size);

torch::Tensor rms_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    double eps) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(x.size(-1) == weight.size(0), "weight size must match hidden dim");
    return rms_norm_forward_cuda(x, weight, eps);
}

torch::Tensor silu_mul_forward(
    torch::Tensor gate,
    torch::Tensor up) {
    CHECK_INPUT(gate);
    CHECK_INPUT(up);
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have the same shape");
    return silu_mul_forward_cuda(gate, up);
}

void store_kvcache_forward(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping,
    int64_t block_size) {
    CHECK_INPUT(key);
    CHECK_INPUT(value);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(slot_mapping);
    TORCH_CHECK(key.dim() == 3, "key must be [num_tokens, num_kv_heads, head_dim]");
    TORCH_CHECK(value.sizes() == key.sizes(), "value must match key shape");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be 4D");
    TORCH_CHECK(v_cache.dim() == 4, "v_cache must be 4D");
    store_kvcache_forward_cuda(key, value, k_cache, v_cache, slot_mapping, block_size);
}

torch::Tensor paged_attention_decode_forward(
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
    CHECK_INPUT(query);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(block_tables);
    CHECK_INPUT(context_lens);
    TORCH_CHECK(query.dim() == 3, "query must be [batch, num_heads, head_dim]");
    TORCH_CHECK(block_tables.dim() == 2, "block_tables must be [batch, max_num_blocks]");
    TORCH_CHECK(context_lens.dim() == 1, "context_lens must be [batch]");
    return paged_attention_decode_forward_cuda(
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_forward", &rms_norm_forward, "RMSNorm forward (CUDA)");
    m.def("silu_mul_forward", &silu_mul_forward, "SiLU*Mul forward (CUDA)");
    m.def("store_kvcache_forward", &store_kvcache_forward, "KV cache store (CUDA)");
    m.def("paged_attention_decode_forward", &paged_attention_decode_forward, "Paged attention decode (CUDA)");
}