import torch
import triton
import triton.language as tl
from math import ceil
from einops import einsum, rearrange


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Args:
            - Q: Query tensor of shape (batch_size, seq_q, dim)
            - K: Key tensor of shape (batch_size, seq_kv, dim)
            - V: Value tensor of shape (batch_size, seq_kv, dim)
            - is_causal: whether to apply causal masking, ignored for this naive implementation
        Returns:
            - O: Output tensor of shape (batch_size, seq_q, dim)
        """
        ctx.is_causal = is_causal

        B = Q.shape[0]
        seq_q = Q.shape[1]
        seq_kv = K.shape[1]
        dim = Q.shape[2]

        Br, Bc = 32, 32  # Block sizes for query and key/value
        Tr, Tc = ceil(seq_q / Br), ceil(seq_kv / Bc)  # Number of blocks

        softmax_scale = 1.0 / (dim ** 0.5)

        O = torch.zeros_like(Q, dtype=torch.float32)
        L = torch.zeros((B, seq_q), device=Q.device, dtype=Q.dtype)

        for i in range(Tr):
            start_q = i * Br
            end_q = min((i + 1) * Br, seq_q)

            Q_i = Q[:, start_q:end_q, :]  # (B, Br, dim)
            O_i = torch.zeros_like(Q_i) # (B, Br, dim)

            row_m_prev = torch.full((B, end_q - start_q), float('-inf'), device=Q.device, dtype=Q.dtype)
            row_l = torch.zeros((B, end_q - start_q), device=Q.device, dtype=Q.dtype)
            for j in range(Tc):
                start_kv = j * Bc
                end_kv = min((j + 1) * Bc, seq_kv)

                K_j = K[:, start_kv:end_kv, :]  # (B, Bc, dim)
                V_j = V[:, start_kv:end_kv, :]  # (B, Bc, dim)

                S_ij = einsum(Q_i, K_j, 'b seq_q d,b seq_kv d->b seq_q seq_kv') * softmax_scale  # (B, Br, Bc)

                row_m = torch.maximum(row_m_prev, S_ij.max(dim=-1).values)  # (B, Br)
                P_ij = torch.exp(S_ij - row_m.unsqueeze(-1))  # (B, Br, Bc)

                factor = torch.exp(row_m_prev - row_m)  # (B, Br)
                row_l = factor * row_l + P_ij.sum(dim=-1)  # (B, Br)

                D_ij = einsum(P_ij, V_j, 'b seq_q seq_kv,b seq_kv d->b seq_q d')  # (B, Br, dim)
                O_i = factor.unsqueeze(-1) * O_i + D_ij  # (B, Br, dim)

                row_m_prev = row_m
            
            # matrix inverse of diag(row_l) @ O_i ==> O_i / row_l
            O[:, start_q:end_q, :] = O_i / row_l.unsqueeze(-1).clamp(min=1e-8)
            L[:, start_q:end_q] = torch.log(row_l.clamp(min=1e-8)) + row_m
        
        ctx.save_for_backward(Q, K, V, O, L)
        return O
    
    @staticmethod
    @torch.compile(fullgraph=True)
    def backward(ctx, dO):
        """
        Args:
            - dO: Gradient of output tensor of shape (batch_size, seq_q, dim)
        Returns:
            - dQ: Gradient of query tensor of shape (batch_size, seq_q, dim)
            - dK: Gradient of key tensor of shape (batch_size, seq_kv, dim)
            - dV: Gradient of value tensor of shape (batch_size, seq_kv, dim)
            - None: Placeholder for is_causal argument
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        B = Q.shape[0]
        seq_q = Q.shape[1]
        seq_kv = K.shape[1]
        dim = Q.shape[2]

        Br, Bc = 32, 32  # Block sizes for query and key/value
        Tr, Tc = ceil(seq_q / Br), ceil(seq_kv / Bc)  # Number of blocks

        softmax_scale = 1.0 / (dim ** 0.5)

        dQ = torch.zeros_like(Q, dtype=torch.float32)
        dK = torch.zeros_like(K, dtype=torch.float32)
        dV = torch.zeros_like(V, dtype=torch.float32)

        D = torch.sum(O * dO, dim=-1)  # (B, seq_q)

        for j in range(Tc):
            start_kv = j * Bc
            end_kv = min((j + 1) * Bc, seq_kv)

            K_j = K[:, start_kv:end_kv, :]  # (B, Bc, dim)
            V_j = V[:, start_kv:end_kv, :]  # (B, Bc, dim)

            dK_j = torch.zeros_like(K_j)
            dV_j = torch.zeros_like(V_j)

            for i in range(Tr):
                start_q = i * Br
                end_q = min((i + 1) * Br, seq_q)

                Q_i = Q[:, start_q:end_q, :]  # (B, Br, dim)
                dO_i = dO[:, start_q:end_q, :]  # (B, Br, dim)
                L_i = L[:, start_q:end_q]  # (B, Br)
                D_i = D[:, start_q:end_q]  # (B, Br)

                S_ij = einsum(Q_i, K_j, 'b seq_q d,b seq_kv d->b seq_q seq_kv') * softmax_scale  # (B, Br, Bc)
                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))  # (B, Br, Bc)

                dV_j += einsum(P_ij, dO_i, 'b seq_q seq_kv,b seq_q d->b seq_kv d')  # (B, Bc, dim)
                dP_ij = einsum(dO_i, V_j, 'b seq_q d,b seq_kv d->b seq_q seq_kv')  # (B, Br, Bc)
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))  # (B, Br, Bc)
                
                dQ[:, start_q:end_q, :] += einsum(dS_ij, K_j, 'b seq_q seq_kv,b seq_kv d->b seq_q d') * softmax_scale  # (B, Br, dim)
                dK_j += einsum(dS_ij, Q_i, 'b seq_q seq_kv,b seq_q d->b seq_kv d') * softmax_scale  # (B, Bc, dim)
            
            dK[:, start_kv:end_kv, :] = dK_j
            dV[:, start_kv:end_kv, :] = dV_j
        return dQ, dK, dV, None # is_causal has no gradient


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,  # input matrix pointers
    O_ptr, L_ptr,         # output matrix pointers
    # Stride parameters for each tensor
    stride_qb, stride_qq, stride_qd,  # Q batch/query/feature strides
    stride_kb, stride_kd, stride_kk,  # K batch/feature/key strides  
    stride_vb, stride_vk, stride_vd,  # V batch/key/feature strides
    stride_ob, stride_oq, stride_od,  # O batch/query/feature strides
    stride_lb, stride_lq,             # L batch/query strides
    N_QUERIES, N_KEYS,                # Number of queries and keys
    scale,                            # Scaling factor 1/sqrt(d)
    D: tl.constexpr,                  # Feature dimension (compile-time constant)
    Q_TILE_SIZE: tl.constexpr,        # Query tile size B_q
    K_TILE_SIZE: tl.constexpr,        # Key/Value tile size B_k 
    is_causal: tl.constexpr=False     # Whether to apply causal masking
):
    # `tl.program_id` gives us a way to check which thread block we're running in
    query_tile_index = tl.program_id(0)  # Query tile index
    batch_index = tl.program_id(1)       # Batch index
    
    # Block pointers give us a way to select from an ND region of memory
    # and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor
    # - The overall shape of the tensor to handle out-of-bounds access
    # - The strides of each dimension to use the memory layout properly
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to use load/store at a time
    # - The order of the dimensions in memory from major to minor
    # axes (= np.argsort(strides)) for optimizations, especially useful on H100
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)                      # Memory layout order (row-major)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),     # !! K.T !!
        order=(0, 1)                      # column-major
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)                      # row-major
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')

    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    L_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    M_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

    n_keys = (query_tile_index + 1) * Q_TILE_SIZE if is_causal else N_KEYS
    for j in range(tl.cdiv(n_keys, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')

        S_ij = tl.dot(Q_i, K_j) * scale

        if is_causal:
            query_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            key_indices = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            # (Q_TILE_SIZE, 1) x (1, K_TILE_SIZE) ==> (Q_TILE_SIZE, K_TILE_SIZE)
            mask = query_indices[:, None] < key_indices[None, :]
            # if mask is True, set S_ij to -inf
            S_ij = tl.where(mask, float('-inf'), S_ij)
        
        row_m = tl.maximum(M_i, tl.max(S_ij, axis=-1))
        P_ij = tl.exp(S_ij - row_m[:, None])

        factor = tl.exp(M_i - row_m)
        L_i = factor * L_i + tl.sum(P_ij, axis=-1)

        O_i = factor[:, None] * O_i
        O_i = tl.dot(P_ij.to(V_j.dtype), V_j, acc=O_i)

        M_i = row_m

        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    L_i = tl.maximum(L_i, 1e-8)
    O_i = O_i / L_i[:, None]
    L_i = tl.log(L_i) + M_i

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    tl.store(O_block_ptr, O_i.to(O_block_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))


@triton.jit
def flash_bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr,    # input matrix pointers
    L_ptr,                  # Logsumexp matrix pointers
    dO_ptr,                 # gradient of output matrix pointer
    D_ptr,                  # output * gradient matrix pointer
    dQ_ptr,                 # gradient of Q matrix pointers
    # Stride parameters for each tensor
    stride_qb, stride_qq, stride_qd,    # Q batch/query/feature strides
    stride_kb, stride_kk, stride_kd,    # K batch/key/feature strides  
    stride_vb, stride_vk, stride_vd,    # V batch/key/feature strides
    stride_lb, stride_lq,               # L batch/query strides
    stride_dob, stride_doq, stride_dod, # dO batch/query/feature strides
    stride_db, stride_dq,               # D batch/query strides
    N_QUERIES, N_KEYS,                  # Number of queries and keys
    scale,                              # Scaling factor 1/sqrt(d)
    D: tl.constexpr,                    # Feature dimension (compile-time constant)
    Q_TILE_SIZE: tl.constexpr,          # Query tile size B_q
    K_TILE_SIZE: tl.constexpr,          # Key/Value tile size B_k 
    is_causal: tl.constexpr=False       # Whether to apply causal masking
):
    key_tile_index = tl.program_id(0)  # Query tile index
    batch_index = tl.program_id(1)       # Batch index

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0, )
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
    L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option='zero')
    dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option='zero')
    D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option='zero')

    dQ_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    n_keys = (key_tile_index + 1) * K_TILE_SIZE if is_causal else N_KEYS
    for j in range(tl.cdiv(n_keys, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')

        S_ij = tl.dot(Q_i, K_j.T) * scale

        if is_causal:
            query_indices = tl.arange(0, Q_TILE_SIZE)
            key_indices = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            # (Q_TILE_SIZE, 1) x (1, K_TILE_SIZE) ==> (Q_TILE_SIZE, K_TILE_SIZE)
            mask = query_indices[:, None] < key_indices[None, :]
            # if mask is True, set S_ij to -inf
            S_ij = tl.where(mask, float('-inf'), S_ij)
        
        P_ij = tl.exp(S_ij - L_i[:, None])

        dP_ij = tl.dot(dO_i, V_j.T)

        dS_ij = P_ij * (dP_ij - D_i[:, None])

        dQ_i += tl.dot(dS_ij, K_j) * scale

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    tl.store(dQ_block_ptr, dQ_i.to(dO_block_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_bwd_dKdV_kernel(
    Q_ptr, K_ptr, V_ptr,    # input matrix pointers
    L_ptr,                  # Logsumexp matrix pointers
    dO_ptr,                 # gradient of output matrix pointer
    D_ptr,                  # output * gradient matrix pointer
    dK_ptr, dV_ptr,         # gradient of K/V matrix pointers
    # Stride parameters for each tensor
    stride_qb, stride_qq, stride_qd,    # Q batch/query/feature strides
    stride_kb, stride_kk, stride_kd,    # K batch/key/feature strides  
    stride_vb, stride_vk, stride_vd,    # V batch/key/feature strides
    stride_lb, stride_lq,               # L batch/query strides
    stride_dob, stride_doq, stride_dod, # dO batch/query/feature strides
    stride_db, stride_dq,               # D batch/query strides
    N_QUERIES, N_KEYS,                  # Number of queries and keys
    scale,                              # Scaling factor 1/sqrt(d)
    D: tl.constexpr,                    # Feature dimension (compile-time constant)
    Q_TILE_SIZE: tl.constexpr,          # Query tile size B_q
    K_TILE_SIZE: tl.constexpr,          # Key/Value tile size B_k 
    is_causal: tl.constexpr=False       # Whether to apply causal masking
):
    key_tile_index = tl.program_id(0)  # Key/Value tile index
    batch_index = tl.program_id(1)       # Batch index

    query_tile_index = key_tile_index * K_TILE_SIZE // Q_TILE_SIZE if is_causal else 0

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0, )
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
    V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')

    dK_i = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_i = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    for i in range(query_tile_index, tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option='zero')
        dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option='zero')
        D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option='zero')

        S_ij = tl.dot(Q_i, K_j.T) * scale

        if is_causal:
            query_indices = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            key_indices = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            # (Q_TILE_SIZE, 1) x (1, K_TILE_SIZE) ==> (Q_TILE_SIZE, K_TILE_SIZE)
            mask = query_indices[:, None] < key_indices[None, :]
            # if mask is True, set S_ij to -inf
            S_ij = tl.where(mask, float('-inf'), S_ij)
        
        P_ij = tl.exp(S_ij - L_i[:, None])

        dV_i += tl.dot(P_ij.to(dO_i.dtype).T, dO_i)

        dP_ij = tl.dot(dO_i, V_j.T)

        dS_ij = P_ij * (dP_ij - D_i[:, None])

        dK_i += tl.dot(dS_ij.T, Q_i) * scale

        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    tl.store(dK_block_ptr, dK_i.to(dO_block_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_i.to(dO_block_ptr.dtype.element_ty), boundary_check=(0, 1))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B = Q.shape[0]
        seq_q = Q.shape[1]
        seq_kv = K.shape[1]
        dim = Q.shape[2]
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "FlashAttentionTriton only supports CUDA tensors."
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "FlashAttentionTriton only supports contiguous tensors."

        ctx.is_causal = is_causal
        
        Br, Bc = 32, 32
        Tr = ceil(seq_q / Br)

        softmax_scale = 1.0 / (dim ** 0.5)

        O = torch.empty_like(Q, dtype=torch.float32)
        L = torch.empty((B, seq_q), device=Q.device, dtype=torch.float32)

        K = rearrange(K, 'b s d -> b d s')
        grid = (Tr, B)
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            seq_q, seq_kv,
            softmax_scale,
            D=dim, Q_TILE_SIZE=Br, K_TILE_SIZE=Bc, is_causal=is_causal
        )
        K = rearrange(K, 'b d s -> b s d')
        ctx.save_for_backward(Q, K, V, O, L)
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        B = Q.shape[0]
        seq_q = Q.shape[1]
        seq_kv = K.shape[1]
        dim = Q.shape[2]

        assert dO.is_cuda, "FlashAttentionTriton only supports CUDA tensors."
        assert dO.is_contiguous(), "FlashAttentionTriton only supports contiguous tensors."

        Br, Bc = 32, 32
        Tr, Tc = ceil(seq_q / Br), ceil(seq_kv / Bc)

        softmax_scale = 1.0 / (dim ** 0.5)

        dQ = torch.zeros_like(Q, dtype=torch.float32)
        dK = torch.zeros_like(K, dtype=torch.float32)
        dV = torch.zeros_like(V, dtype=torch.float32)

        D = torch.sum(O * dO, dim=-1)

        # compute dQ
        grid = (Tr, B)
        flash_bwd_dQ_kernel[grid](
            Q, K, V,
            L,
            dO,
            D,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D.stride(0), D.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            seq_q, seq_kv,
            softmax_scale,
            D=dim, Q_TILE_SIZE=Br, K_TILE_SIZE=Bc, is_causal=is_causal
        )

        # compute dK, dV
        gird = (Tc, B)
        flash_bwd_dKdV_kernel(
            Q, K, V,
            L, 
            dO,
            D,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D.stride(0), D.stride(1),
            seq_q, seq_kv,
            softmax_scale,
            D=dim, Q_TILE_SIZE=Br, K_TILE_SIZE=Bc, is_causal=is_causal
        )

        return dQ, dK, dV, None