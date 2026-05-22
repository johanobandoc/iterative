import dataclasses

import jax
import jax.numpy as jnp

from utils import ParamInitializer, jax_pytree_struct
from utils import layer_repr

import model as base_model


@jax_pytree_struct
class GPT(ParamInitializer):
    core: base_model.GPT
    num_experts: int = dataclasses.field(metadata=dict(static=True))
    depth: int = dataclasses.field(metadata=dict(static=True))
    res_stream_width: int = dataclasses.field(metadata=dict(static=True))
    memory_len: int = dataclasses.field(metadata=dict(static=True))
    ticks: int = dataclasses.field(metadata=dict(static=True))
    checkpoint_token_step: bool = dataclasses.field(metadata=dict(static=True))
    segment_local_kv_cache: bool = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        validate_recurrent_model_config(cfg)
        core = base_model.GPT.param_specs(cfg)
        depth_blocks = base_model._stack_param_specs(
            core.blocks,
            cfg.depth,
        )
        depth_router = core.aggregation_router
        if depth_router is not None:
            depth_router = base_model._stack_param_specs(
                depth_router,
                cfg.depth,
            )
        core = dataclasses.replace(
            core,
            blocks=depth_blocks,
            aggregation_router=depth_router,
        )
        return GPT(
            core=core,
            num_experts=core.num_experts,
            depth=cfg.depth,
            res_stream_width=core.res_stream_width,
            memory_len=core.memory_len,
            ticks=core.ticks,
            checkpoint_token_step=core.checkpoint_token_step,
            segment_local_kv_cache=core.segment_local_kv_cache,
        )

    @classmethod
    def init(cls, key, cfg):
        return cls._init_fn(key, cfg.mesh, cfg.rules, cfg.model)

    def __repr__(self):
        return layer_repr(self)


def count_params(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(model))


def validate_recurrent_model_config(cfg):
    if (
        getattr(cfg, "recurrent_architecture", "rec_shared_kv")
        != "rec_shared_kv"
    ):
        raise ValueError(
            "`model_shared_kv` requires "
            "`recurrent_architecture='rec_shared_kv'`."
        )
    base_model.validate_recurrent_model_config(cfg)
    if cfg.depth < 1:
        raise ValueError("`depth` must be >= 1.")


def init_recurrent_state(params, batch_size: int):
    dtype = params.core.embed.weight.dtype
    d_emb = params.core.embed.d_emb
    kv_heads = params.core.attn_kv_heads
    head_dim = params.core.attn_head_dim
    res_stream = jnp.zeros((batch_size, params.res_stream_width), dtype=dtype)
    prev_layer_out = jnp.zeros((1, batch_size, d_emb), dtype=dtype)
    depth_k_cache = jnp.zeros(
        (
            params.depth,
            batch_size,
            kv_heads,
            params.memory_len,
            head_dim,
        ),
        dtype=dtype,
    )
    depth_v_cache = jnp.zeros_like(depth_k_cache)
    depth_cache_fill = jnp.zeros(
        (params.depth, batch_size),
        dtype=jnp.int32,
    )
    depth_abs_pos = jnp.zeros(
        (params.depth, batch_size),
        dtype=jnp.int32,
    )
    return (
        res_stream,
        prev_layer_out,
        depth_k_cache,
        depth_v_cache,
        depth_cache_fill,
        depth_abs_pos,
    )


def reset_recurrent_state_slots(params, recurrent_state, slot_reset_mask):
    del params
    if slot_reset_mask is None:
        return recurrent_state
    reset_mask = jnp.asarray(slot_reset_mask, dtype=jnp.bool_)
    if reset_mask.ndim == 0:
        reset_mask = jnp.broadcast_to(reset_mask[None], (recurrent_state[0].shape[0],))

    (
        res_stream,
        prev_layer_out,
        depth_k_cache,
        depth_v_cache,
        depth_cache_fill,
        depth_abs_pos,
    ) = recurrent_state
    return (
        jnp.where(reset_mask[:, None], jnp.zeros_like(res_stream), res_stream),
        jnp.where(
            reset_mask[None, :, None],
            jnp.zeros_like(prev_layer_out),
            prev_layer_out,
        ),
        jnp.where(
            reset_mask[None, :, None, None, None],
            jnp.zeros_like(depth_k_cache),
            depth_k_cache,
        ),
        jnp.where(
            reset_mask[None, :, None, None, None],
            jnp.zeros_like(depth_v_cache),
            depth_v_cache,
        ),
        jnp.where(
            reset_mask[None, :],
            jnp.zeros_like(depth_cache_fill),
            depth_cache_fill,
        ),
        jnp.where(
            reset_mask[None, :],
            jnp.zeros_like(depth_abs_pos),
            depth_abs_pos,
        ),
    )


@jax.jit
def reset_kv_cache_state(params, recurrent_state):
    del params
    return (
        recurrent_state[0],
        recurrent_state[1],
        jnp.zeros_like(recurrent_state[2]),
        jnp.zeros_like(recurrent_state[3]),
        jnp.zeros_like(recurrent_state[4]),
        jnp.zeros_like(recurrent_state[5]),
    )


def _run_depth_stage(
    params,
    stage_blocks,
    stage_router,
    current_embed,
    incoming_res_stream,
    stage_k_cache,
    stage_v_cache,
    stage_cache_fill,
    stage_abs_pos,
    write_kv_cache,
):
    aggregation_gates = base_model._compute_router_gates(
        stage_router,
        params.core.softmax_expert_routing,
        current_embed,
        incoming_res_stream,
    )
    step_freqs = base_model.precompute_frequencies(
        stage_abs_pos[:, None],
        features=params.core.attn_head_dim,
        dtype=current_embed.dtype,
    )
    x_stack, q_stack, k_stack, v_stack, gate_logits_stack = jax.vmap(
        base_model._prepare_recurrent_layer_attention,
        in_axes=(0, None, None, None, None),
        out_axes=(0, 0, 0, 0, 0),
    )(
        stage_blocks,
        current_embed,
        incoming_res_stream,
        params.core.direct_qkv_from_input,
        step_freqs,
    )
    merged_k, merged_v = base_model._merge_layer_kv_candidates(
        params.core,
        k_stack,
        v_stack,
    )
    next_k_cache, next_v_cache, key_seq, value_seq, valid_mask = (
        base_model._build_attention_cache_sequence(
            stage_k_cache,
            stage_v_cache,
            merged_k,
            merged_v,
            stage_cache_fill,
            write_kv_cache,
        )
    )
    layer_outs, _ = jax.vmap(
        base_model._finish_recurrent_layer_attention,
        in_axes=(0, 0, 0, 0, None, None, None),
        out_axes=(0, 0),
    )(
        stage_blocks,
        x_stack,
        q_stack,
        gate_logits_stack,
        key_seq,
        value_seq,
        valid_mask,
    )
    next_res_stream = base_model._aggregate_layer_outputs(
        params.core,
        layer_outs,
        aggregation_gates if params.core.soft_routed_experts_aggregation else None,
    )
    layer_pairwise_stats = base_model._layer_pairwise_cosine_matrix_sum_and_count(
        layer_outs
    )
    write_kv_cache = jnp.asarray(write_kv_cache)
    if params.memory_len > 0:
        stage_cache_fill = jnp.minimum(
            stage_cache_fill + write_kv_cache.astype(stage_cache_fill.dtype),
            params.memory_len,
        )
    stage_abs_pos = stage_abs_pos + write_kv_cache.astype(stage_abs_pos.dtype)
    return (
        next_res_stream,
        (next_k_cache, next_v_cache, stage_cache_fill, stage_abs_pos),
        layer_pairwise_stats,
    )


def _run_depth_stage_segment_local(
    params,
    stage_blocks,
    stage_router,
    current_embed,
    incoming_res_stream,
    stage_k_cache,
    stage_v_cache,
    stage_cache_fill,
    stage_abs_pos,
    stage_local_k_cache,
    stage_local_v_cache,
    local_pos,
):
    aggregation_gates = base_model._compute_router_gates(
        stage_router,
        params.core.softmax_expert_routing,
        current_embed,
        incoming_res_stream,
    )
    step_freqs = base_model.precompute_frequencies(
        stage_abs_pos[:, None],
        features=params.core.attn_head_dim,
        dtype=current_embed.dtype,
    )
    x_stack, q_stack, k_stack, v_stack, gate_logits_stack = jax.vmap(
        base_model._prepare_recurrent_layer_attention,
        in_axes=(0, None, None, None, None),
        out_axes=(0, 0, 0, 0, 0),
    )(
        stage_blocks,
        current_embed,
        incoming_res_stream,
        params.core.direct_qkv_from_input,
        step_freqs,
    )
    merged_k, merged_v = base_model._merge_layer_kv_candidates(
        params.core,
        k_stack,
        v_stack,
    )
    stage_k_cache = jax.lax.stop_gradient(stage_k_cache)
    stage_v_cache = jax.lax.stop_gradient(stage_v_cache)
    next_stage_local_k_cache, next_stage_local_v_cache, key_seq, value_seq, valid_mask = (
        base_model._build_segment_local_attention_cache_sequence(
            stage_k_cache,
            stage_v_cache,
            stage_local_k_cache,
            stage_local_v_cache,
            merged_k,
            merged_v,
            stage_cache_fill,
            local_pos,
        )
    )
    layer_outs, _ = jax.vmap(
        base_model._finish_recurrent_layer_attention,
        in_axes=(0, 0, 0, 0, None, None, None),
        out_axes=(0, 0),
    )(
        stage_blocks,
        x_stack,
        q_stack,
        gate_logits_stack,
        key_seq,
        value_seq,
        valid_mask,
    )
    next_res_stream = base_model._aggregate_layer_outputs(
        params.core,
        layer_outs,
        aggregation_gates if params.core.soft_routed_experts_aggregation else None,
    )
    layer_pairwise_stats = base_model._layer_pairwise_cosine_matrix_sum_and_count(
        layer_outs
    )
    return (
        next_res_stream,
        (
            next_stage_local_k_cache,
            next_stage_local_v_cache,
            stage_abs_pos + jnp.asarray(1, dtype=stage_abs_pos.dtype),
        ),
        layer_pairwise_stats,
    )


def _run_depth_sweep(
    params,
    current_embed,
    res_stream,
    depth_k_cache,
    depth_v_cache,
    depth_cache_fill,
    depth_abs_pos,
    write_kv_cache,
):
    def depth_body(carry, depth_inputs):
        incoming_res_stream, pairwise_matrix_sum, pairwise_count = carry
        (
            stage_blocks,
            stage_router,
            stage_k_cache,
            stage_v_cache,
            stage_cache_fill,
            stage_abs_pos,
        ) = depth_inputs
        next_res_stream, stage_state, layer_pairwise_stats = _run_depth_stage(
            params,
            stage_blocks,
            stage_router,
            current_embed,
            incoming_res_stream,
            stage_k_cache,
            stage_v_cache,
            stage_cache_fill,
            stage_abs_pos,
            write_kv_cache,
        )
        stage_pairwise_matrix_sum, stage_pairwise_count = layer_pairwise_stats
        return (
            (
                next_res_stream,
                pairwise_matrix_sum + stage_pairwise_matrix_sum,
                pairwise_count + stage_pairwise_count,
            ),
            stage_state,
        )

    def depth_body_no_router(carry, depth_inputs):
        stage_blocks, stage_k_cache, stage_v_cache, stage_cache_fill, stage_abs_pos = depth_inputs
        return depth_body(
            carry,
            (
                stage_blocks,
                None,
                stage_k_cache,
                stage_v_cache,
                stage_cache_fill,
                stage_abs_pos,
            ),
        )

    scan_xs = (
        params.core.blocks,
        depth_k_cache,
        depth_v_cache,
        depth_cache_fill,
        depth_abs_pos,
    )
    if params.core.aggregation_router is None:
        scan_fn = depth_body_no_router
    else:
        scan_fn = depth_body
        scan_xs = (
            params.core.blocks,
            params.core.aggregation_router,
            depth_k_cache,
            depth_v_cache,
            depth_cache_fill,
            depth_abs_pos,
        )
    (
        next_res_stream,
        layer_pairwise_matrix_sum,
        layer_pairwise_count,
    ), (next_k_cache, next_v_cache, next_cache_fill, next_abs_pos) = jax.lax.scan(
        scan_fn,
        (res_stream, *base_model._empty_layer_pairwise_stats(params.num_experts)),
        scan_xs,
    )
    return (
        next_res_stream,
        next_k_cache,
        next_v_cache,
        next_cache_fill,
        next_abs_pos,
        (layer_pairwise_matrix_sum, layer_pairwise_count),
    )


def _run_depth_sweep_segment_local(
    params,
    current_embed,
    res_stream,
    depth_k_cache,
    depth_v_cache,
    depth_cache_fill,
    depth_abs_pos,
    depth_local_k_cache,
    depth_local_v_cache,
    local_pos,
):
    def depth_body(carry, depth_inputs):
        incoming_res_stream, pairwise_matrix_sum, pairwise_count = carry
        (
            stage_blocks,
            stage_router,
            stage_k_cache,
            stage_v_cache,
            stage_cache_fill,
            stage_abs_pos,
            stage_local_k_cache,
            stage_local_v_cache,
        ) = depth_inputs
        next_res_stream, stage_state, layer_pairwise_stats = (
            _run_depth_stage_segment_local(
                params,
                stage_blocks,
                stage_router,
                current_embed,
                incoming_res_stream,
                stage_k_cache,
                stage_v_cache,
                stage_cache_fill,
                stage_abs_pos,
                stage_local_k_cache,
                stage_local_v_cache,
                local_pos,
            )
        )
        stage_pairwise_matrix_sum, stage_pairwise_count = layer_pairwise_stats
        return (
            (
                next_res_stream,
                pairwise_matrix_sum + stage_pairwise_matrix_sum,
                pairwise_count + stage_pairwise_count,
            ),
            stage_state,
        )

    def depth_body_no_router(carry, depth_inputs):
        (
            stage_blocks,
            stage_k_cache,
            stage_v_cache,
            stage_cache_fill,
            stage_abs_pos,
            stage_local_k_cache,
            stage_local_v_cache,
        ) = depth_inputs
        return depth_body(
            carry,
            (
                stage_blocks,
                None,
                stage_k_cache,
                stage_v_cache,
                stage_cache_fill,
                stage_abs_pos,
                stage_local_k_cache,
                stage_local_v_cache,
            ),
        )

    scan_xs = (
        params.core.blocks,
        depth_k_cache,
        depth_v_cache,
        depth_cache_fill,
        depth_abs_pos,
        depth_local_k_cache,
        depth_local_v_cache,
    )
    if params.core.aggregation_router is None:
        scan_fn = depth_body_no_router
    else:
        scan_fn = depth_body
        scan_xs = (
            params.core.blocks,
            params.core.aggregation_router,
            depth_k_cache,
            depth_v_cache,
            depth_cache_fill,
            depth_abs_pos,
            depth_local_k_cache,
            depth_local_v_cache,
        )
    (
        next_res_stream,
        layer_pairwise_matrix_sum,
        layer_pairwise_count,
    ), (next_local_k_cache, next_local_v_cache, next_abs_pos) = jax.lax.scan(
        scan_fn,
        (res_stream, *base_model._empty_layer_pairwise_stats(params.num_experts)),
        scan_xs,
    )
    return (
        next_res_stream,
        next_abs_pos,
        next_local_k_cache,
        next_local_v_cache,
        (layer_pairwise_matrix_sum, layer_pairwise_count),
    )


def _append_depth_segment_to_cache(
    depth_k_cache,
    depth_v_cache,
    depth_local_k_cache,
    depth_local_v_cache,
    depth_cache_fill,
):
    return jax.vmap(base_model._append_segment_to_cache)(
        depth_k_cache,
        depth_v_cache,
        depth_local_k_cache,
        depth_local_v_cache,
        depth_cache_fill,
    )


def _init_depth_local_cache(depth_k_cache, segment_len):
    local_shape = depth_k_cache.shape[:3] + (segment_len, depth_k_cache.shape[-1])
    return jnp.zeros(local_shape, dtype=depth_k_cache.dtype)


def _segment_local_token_step(
    params,
    carry,
    token_ids,
    local_pos,
    depth_k_cache,
    depth_v_cache,
    depth_cache_fill,
):
    (
        res_stream,
        prev_layer_out,
        depth_abs_pos,
        depth_local_k_cache,
        depth_local_v_cache,
    ) = carry
    token_embed = base_model.embedding_forward(params.core.embed, token_ids)
    (
        res_stream,
        depth_abs_pos,
        depth_local_k_cache,
        depth_local_v_cache,
        layer_pairwise_stats,
    ) = _run_depth_sweep_segment_local(
        params,
        token_embed,
        res_stream,
        depth_k_cache,
        depth_v_cache,
        depth_cache_fill,
        depth_abs_pos,
        depth_local_k_cache,
        depth_local_v_cache,
        local_pos,
    )
    layer_pairwise_matrix_sum, layer_pairwise_count = layer_pairwise_stats
    next_carry = (
        res_stream,
        prev_layer_out,
        depth_abs_pos,
        depth_local_k_cache,
        depth_local_v_cache,
    )
    return next_carry, (
        base_model._logits_from_res_stream(params.core, res_stream),
        layer_pairwise_matrix_sum,
        layer_pairwise_count,
    )


def _run_token_steps(params, recurrent_state, token_embed):
    (
        res_stream,
        prev_layer_out,
        depth_k_cache,
        depth_v_cache,
        depth_cache_fill,
        depth_abs_pos,
    ) = recurrent_state

    def recurrent_body(step_idx, loop_state):
        (
            current_res_stream,
            current_depth_k_cache,
            current_depth_v_cache,
            current_depth_cache_fill,
            current_depth_abs_pos,
            layer_pairwise_matrix_sum,
            layer_pairwise_count,
        ) = loop_state
        write_kv_cache = step_idx == (params.ticks - 1)
        (
            next_res_stream,
            next_depth_k_cache,
            next_depth_v_cache,
            next_depth_cache_fill,
            next_depth_abs_pos,
            depth_pairwise_stats,
        ) = _run_depth_sweep(
            params,
            token_embed,
            current_res_stream,
            current_depth_k_cache,
            current_depth_v_cache,
            current_depth_cache_fill,
            current_depth_abs_pos,
            write_kv_cache,
        )
        depth_pairwise_matrix_sum, depth_pairwise_count = depth_pairwise_stats
        return (
            next_res_stream,
            next_depth_k_cache,
            next_depth_v_cache,
            next_depth_cache_fill,
            next_depth_abs_pos,
            layer_pairwise_matrix_sum + depth_pairwise_matrix_sum,
            layer_pairwise_count + depth_pairwise_count,
        )

    (
        res_stream,
        depth_k_cache,
        depth_v_cache,
        depth_cache_fill,
        depth_abs_pos,
        layer_pairwise_matrix_sum,
        layer_pairwise_count,
    ) = jax.lax.fori_loop(
        0,
        params.ticks,
        recurrent_body,
        (
            res_stream,
            depth_k_cache,
            depth_v_cache,
            depth_cache_fill,
            depth_abs_pos,
            *base_model._empty_layer_pairwise_stats(params.num_experts),
        ),
    )
    return (
        (
            res_stream,
            prev_layer_out,
            depth_k_cache,
            depth_v_cache,
            depth_cache_fill,
            depth_abs_pos,
        ),
        layer_pairwise_matrix_sum,
        layer_pairwise_count,
    )


def _token_step(params, carry, token_ids):
    token_embed = base_model.embedding_forward(params.core.embed, token_ids)
    carry, layer_pairwise_matrix_sum, layer_pairwise_count = _run_token_steps(
        params,
        carry,
        token_embed,
    )
    res_stream = carry[0]
    return carry, (
        base_model._logits_from_res_stream(params.core, res_stream),
        layer_pairwise_matrix_sum,
        layer_pairwise_count,
    )


def _finalize_segment_local_state(scan_carry, depth_k_cache, depth_v_cache, depth_cache_fill):
    (
        res_stream,
        prev_layer_out,
        depth_abs_pos,
        depth_local_k_cache,
        depth_local_v_cache,
    ) = scan_carry
    depth_k_cache, depth_v_cache, depth_cache_fill = _append_depth_segment_to_cache(
        depth_k_cache,
        depth_v_cache,
        depth_local_k_cache,
        depth_local_v_cache,
        depth_cache_fill,
    )
    return (
        res_stream,
        prev_layer_out,
        depth_k_cache,
        depth_v_cache,
        depth_cache_fill,
        depth_abs_pos,
    )


def _forward_with_state_segment_local(params, x, recurrent_state, slot_reset_mask=None):
    recurrent_state = reset_recurrent_state_slots(params, recurrent_state, slot_reset_mask)
    (
        res_stream,
        prev_layer_out,
        depth_k_cache,
        depth_v_cache,
        depth_cache_fill,
        depth_abs_pos,
    ) = recurrent_state
    x_tokens = jnp.swapaxes(x, 0, 1)
    segment_len = x_tokens.shape[0]
    depth_local_k_cache = _init_depth_local_cache(depth_k_cache, segment_len)
    depth_local_v_cache = _init_depth_local_cache(depth_v_cache, segment_len)

    def token_step_logits_only(carry, inputs):
        local_pos, token_ids = inputs
        carry, (logits, _, _) = _segment_local_token_step(
            params,
            carry,
            token_ids,
            local_pos,
            depth_k_cache,
            depth_v_cache,
            depth_cache_fill,
        )
        return carry, logits

    token_step_impl = (
        jax.checkpoint(token_step_logits_only)
        if params.checkpoint_token_step
        else token_step_logits_only
    )
    scan_carry, logits = jax.lax.scan(
        token_step_impl,
        (
            res_stream,
            prev_layer_out,
            depth_abs_pos,
            depth_local_k_cache,
            depth_local_v_cache,
        ),
        (jnp.arange(segment_len, dtype=jnp.int32), x_tokens),
        _split_transpose=params.checkpoint_token_step,
    )
    final_state = _finalize_segment_local_state(
        scan_carry,
        depth_k_cache,
        depth_v_cache,
        depth_cache_fill,
    )
    return jnp.swapaxes(logits, 0, 1), final_state


def forward_with_state(params, x, recurrent_state, slot_reset_mask=None):
    if params.segment_local_kv_cache:
        return _forward_with_state_segment_local(
            params,
            x,
            recurrent_state,
            slot_reset_mask=slot_reset_mask,
        )

    recurrent_state = reset_recurrent_state_slots(params, recurrent_state, slot_reset_mask)

    def token_step_logits_only(carry, token_ids):
        carry, (logits, _, _) = _token_step(params, carry, token_ids)
        return carry, logits

    token_step_impl = (
        jax.checkpoint(token_step_logits_only)
        if params.checkpoint_token_step
        else token_step_logits_only
    )

    final_state, logits = jax.lax.scan(
        token_step_impl,
        recurrent_state,
        jnp.swapaxes(x, 0, 1),
        _split_transpose=params.checkpoint_token_step,
    )
    return jnp.swapaxes(logits, 0, 1), final_state


def forward(params, x, segment_ids, freqs):
    del segment_ids, freqs
    batch_size = x.shape[0]
    logits, _ = forward_with_state(
        params,
        x,
        init_recurrent_state(params, batch_size),
        slot_reset_mask=None,
    )
    return logits


def forward_loss_with_state(
    params,
    x,
    y,
    recurrent_state,
    slot_reset_mask=None,
    loss_mask=None,
    pre_output_reg_cost=0.0,
):
    loss, final_state, _, _, _ = forward_loss_with_state_and_stats(
        params,
        x,
        y,
        recurrent_state,
        slot_reset_mask=slot_reset_mask,
        loss_mask=loss_mask,
        pre_output_reg_cost=pre_output_reg_cost,
    )
    return loss, final_state


def _forward_loss_with_state_and_stats_segment_local(
    params,
    x,
    y,
    recurrent_state,
    slot_reset_mask=None,
    loss_mask=None,
    pre_output_reg_cost=0.0,
):
    recurrent_state = reset_recurrent_state_slots(params, recurrent_state, slot_reset_mask)
    (
        res_stream,
        prev_layer_out,
        depth_k_cache,
        depth_v_cache,
        depth_cache_fill,
        depth_abs_pos,
    ) = recurrent_state
    x_tokens = jnp.swapaxes(x, 0, 1)
    y_tokens = jnp.swapaxes(y, 0, 1)
    mask_tokens = None if loss_mask is None else jnp.swapaxes(loss_mask, 0, 1)
    segment_len = x_tokens.shape[0]
    depth_local_k_cache = _init_depth_local_cache(depth_k_cache, segment_len)
    depth_local_v_cache = _init_depth_local_cache(depth_v_cache, segment_len)

    def token_step(carry, token_ids, local_pos):
        return _segment_local_token_step(
            params,
            carry,
            token_ids,
            local_pos,
            depth_k_cache,
            depth_v_cache,
            depth_cache_fill,
        )

    token_step_impl = (
        jax.checkpoint(token_step)
        if params.checkpoint_token_step
        else token_step
    )

    def loss_step(carry, inputs):
        (
            scan_carry,
            ce_loss_sum,
            reg_loss_sum,
            weight_sum,
            layer_pairwise_matrix_sum,
            layer_pairwise_count,
        ) = carry
        if mask_tokens is None:
            local_pos, token_ids, target_ids = inputs
            token_mask = jnp.ones_like(target_ids, dtype=jnp.float32)
        else:
            local_pos, token_ids, target_ids, token_mask = inputs
            token_mask = token_mask.astype(jnp.float32)

        scan_carry, (
            logits,
            token_pairwise_matrix_sum,
            token_pairwise_count,
        ) = token_step_impl(scan_carry, token_ids, local_pos)
        res_stream = scan_carry[0]
        res_stats = base_model._res_stream_stats(res_stream)
        token_reg_loss = base_model._pre_output_activation_l2(res_stream)
        token_log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_loss = -jnp.take_along_axis(
            token_log_probs,
            target_ids[:, None],
            axis=-1,
        ).squeeze(-1)
        ce_loss_sum = ce_loss_sum + jnp.sum(token_loss * token_mask)
        reg_loss_sum = reg_loss_sum + jnp.sum(token_reg_loss * token_mask)
        weight_sum = weight_sum + jnp.sum(token_mask)
        layer_pairwise_matrix_sum = (
            layer_pairwise_matrix_sum + token_pairwise_matrix_sum
        )
        layer_pairwise_count = layer_pairwise_count + token_pairwise_count
        return (
            (
                scan_carry,
                ce_loss_sum,
                reg_loss_sum,
                weight_sum,
                layer_pairwise_matrix_sum,
                layer_pairwise_count,
            ),
            res_stats,
        )

    positions = jnp.arange(segment_len, dtype=jnp.int32)
    scan_inputs = (
        (positions, x_tokens, y_tokens)
        if mask_tokens is None
        else (positions, x_tokens, y_tokens, mask_tokens)
    )
    carry0 = (
        (
            res_stream,
            prev_layer_out,
            depth_abs_pos,
            depth_local_k_cache,
            depth_local_v_cache,
        ),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.zeros((params.num_experts, params.num_experts), dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )
    (
        scan_carry,
        ce_loss_sum,
        reg_loss_sum,
        weight_sum,
        layer_pairwise_matrix_sum,
        layer_pairwise_count,
    ), res_stats_seq = jax.lax.scan(
        loss_step,
        carry0,
        scan_inputs,
        _split_transpose=params.checkpoint_token_step,
    )
    final_state = _finalize_segment_local_state(
        scan_carry,
        depth_k_cache,
        depth_v_cache,
        depth_cache_fill,
    )
    ce_loss = ce_loss_sum / jnp.maximum(weight_sum, 1.0)
    reg_loss = reg_loss_sum / jnp.maximum(weight_sum, 1.0)
    loss = ce_loss + pre_output_reg_cost * reg_loss
    return (
        loss,
        final_state,
        res_stats_seq,
        jnp.stack([ce_loss, reg_loss]),
        (layer_pairwise_matrix_sum, layer_pairwise_count),
    )


def forward_loss_with_state_and_stats(
    params,
    x,
    y,
    recurrent_state,
    slot_reset_mask=None,
    loss_mask=None,
    pre_output_reg_cost=0.0,
):
    if params.segment_local_kv_cache:
        return _forward_loss_with_state_and_stats_segment_local(
            params,
            x,
            y,
            recurrent_state,
            slot_reset_mask=slot_reset_mask,
            loss_mask=loss_mask,
            pre_output_reg_cost=pre_output_reg_cost,
        )

    recurrent_state = reset_recurrent_state_slots(params, recurrent_state, slot_reset_mask)
    token_step_impl = (
        jax.checkpoint(
            lambda carry, token_ids: _token_step(params, carry, token_ids)
        )
        if params.checkpoint_token_step
        else lambda carry, token_ids: _token_step(params, carry, token_ids)
    )
    x_tokens = jnp.swapaxes(x, 0, 1)
    y_tokens = jnp.swapaxes(y, 0, 1)
    mask_tokens = None if loss_mask is None else jnp.swapaxes(loss_mask, 0, 1)

    def loss_step(carry, inputs):
        (
            recurrent_state,
            ce_loss_sum,
            reg_loss_sum,
            weight_sum,
            layer_pairwise_matrix_sum,
            layer_pairwise_count,
        ) = carry
        if mask_tokens is None:
            token_ids, target_ids = inputs
            token_mask = jnp.ones_like(target_ids, dtype=jnp.float32)
        else:
            token_ids, target_ids, token_mask = inputs
            token_mask = token_mask.astype(jnp.float32)

        recurrent_state, (
            logits,
            token_pairwise_matrix_sum,
            token_pairwise_count,
        ) = token_step_impl(recurrent_state, token_ids)
        res_stats = base_model._res_stream_stats(recurrent_state[0])
        token_reg_loss = base_model._pre_output_activation_l2(recurrent_state[0])
        token_log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_loss = -jnp.take_along_axis(
            token_log_probs,
            target_ids[:, None],
            axis=-1,
        ).squeeze(-1)
        ce_loss_sum = ce_loss_sum + jnp.sum(token_loss * token_mask)
        reg_loss_sum = reg_loss_sum + jnp.sum(token_reg_loss * token_mask)
        weight_sum = weight_sum + jnp.sum(token_mask)
        layer_pairwise_matrix_sum = (
            layer_pairwise_matrix_sum + token_pairwise_matrix_sum
        )
        layer_pairwise_count = layer_pairwise_count + token_pairwise_count
        return (
            (
                recurrent_state,
                ce_loss_sum,
                reg_loss_sum,
                weight_sum,
                layer_pairwise_matrix_sum,
                layer_pairwise_count,
            ),
            res_stats,
        )

    scan_inputs = (
        (x_tokens, y_tokens)
        if mask_tokens is None
        else (x_tokens, y_tokens, mask_tokens)
    )
    carry0 = (
        recurrent_state,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.zeros((params.num_experts, params.num_experts), dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )
    (
        final_state,
        ce_loss_sum,
        reg_loss_sum,
        weight_sum,
        layer_pairwise_matrix_sum,
        layer_pairwise_count,
    ), res_stats_seq = jax.lax.scan(
        loss_step,
        carry0,
        scan_inputs,
        _split_transpose=params.checkpoint_token_step,
    )
    ce_loss = ce_loss_sum / jnp.maximum(weight_sum, 1.0)
    reg_loss = reg_loss_sum / jnp.maximum(weight_sum, 1.0)
    loss = ce_loss + pre_output_reg_cost * reg_loss
    return (
        loss,
        final_state,
        res_stats_seq,
        jnp.stack([ce_loss, reg_loss]),
        (layer_pairwise_matrix_sum, layer_pairwise_count),
    )


def forward_loss(
    params,
    x,
    y,
    segment_ids=None,
    freqs=None,
    loss_mask=None,
    pre_output_reg_cost=0.0,
):
    del segment_ids, freqs
    batch_size = x.shape[0]
    loss, _ = forward_loss_with_state(
        params,
        x,
        y,
        init_recurrent_state(params, batch_size),
        slot_reset_mask=None,
        loss_mask=loss_mask,
        pre_output_reg_cost=pre_output_reg_cost,
    )
    return loss


def forward_v2(params, x, segment_ids, cache, head_dim):
    del params, x, segment_ids, cache, head_dim
    raise NotImplementedError(
        "`recnanogpt` inference is not implemented yet for `shared_kv`."
    )
