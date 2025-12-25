import torch
from math import ceil
from einops import einsum


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

        O = torch.zeros_like(Q)
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
        Tr, Tc = ceil(Br / seq_q), ceil(Bc / seq_kv)  # Number of blocks

        sofmax_scale = 1.0 / (dim ** 0.5)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

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
                O_i = O[:, start_q:end_q, :]  # (B, Br, dim)
                dO_i = dO[:, start_q:end_q, :]  # (B, Br, dim)
                L_i = L[:, start_q:end_q]  # (B, Br)
                D_i = D[:, start_q:end_q]  # (B, Br)

                S_ij = einsum(Q_i, K_j, 'b seq_q d,b seq_kv d->b seq_q seq_kv') * sofmax_scale  # (B, Br, Bc)
                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))  # (B, Br, Bc)

                dV_j += einsum(P_ij, dO_i, 'b seq_q seq_kv,b seq_q d->b seq_kv d')  # (B, Bc, dim)
                dP_ij = einsum(dO_i, V_j, 'b seq_q d,b seq_kv d->b seq_q seq_kv')  # (B, Br, Bc)
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))  # (B, Br, Bc)
                
                dQ[:, start_q:end_q, :] += einsum(dS_ij, K_j, 'b seq_q seq_kv,b seq_kv d->b seq_q d') * sofmax_scale  # (B, Br, dim)
                dK_j += einsum(dS_ij, Q_i, 'b seq_q seq_kv,b seq_q d->b seq_kv d') * sofmax_scale  # (B, Bc, dim)
            
            dK[:, start_kv:end_kv, :] = dK_j
            dV[:, start_kv:end_kv, :] = dV_j
        
        return dQ, dK, dV, None # is_causal has no gradient