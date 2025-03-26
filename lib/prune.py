import time
import torch
import torch.nn as nn
import transformers
import torch.distributed as dist
from .save_results import save_time_result
from .data import get_loaders
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT

def compute_delta_h_target_sum(model, tokenizer, n_tokens=10, sequence_length=128, device=None, tolerance_factor=1.0, oversample_factor=5):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_samples = oversample_factor * n_tokens
    input_ids = torch.randint(0, tokenizer.vocab_size, (total_samples, sequence_length), device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        hidden_states = model(input_ids, attention_mask=attention_mask)[0]  # [total_samples, seq_len, hidden_size]
    token_hidden = hidden_states[:, 0, :]  
    token_norms = torch.norm(token_hidden, dim=-1)  # shape: [total_samples]

    low_quantile = torch.quantile(token_norms.float(), 0.05)
    high_quantile = torch.quantile(token_norms.float(), 0.95)
    valid_mask = (token_norms >= low_quantile) & (token_norms <= high_quantile)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze()

    if valid_indices.numel() < n_tokens:
        repeat_factor = (n_tokens // valid_indices.numel()) + 1 if valid_indices.numel() > 0 else 1
        valid_indices = valid_indices.repeat(repeat_factor)
    selected_indices = valid_indices[:n_tokens]
    
    calib_input_ids = input_ids[selected_indices]
    calib_attention_mask = attention_mask[selected_indices]
    calib_token_norms = token_norms[selected_indices]

    delta_h_list = tolerance_factor * calib_token_norms
    delta_h_target_sum = delta_h_list.sum().item()

    print(f"Selected calibration δh_list: {delta_h_list.tolist()}")
    print(f"Calibration δh target sum: {delta_h_target_sum:.6f}")

    return delta_h_target_sum, delta_h_list, calib_input_ids, calib_attention_mask

def estimate_total_token_perturbation(model, layer_index, original_weights, current_weights,
                                      calib_input_ids, calib_attention_mask, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    for name, weight in original_weights.items():
        if name.startswith(f"model.layers.{layer_index}"):
            param_path = name.split('.')
            module = model
            for p in param_path[:-1]:
                module = module[int(p)] if p.isdigit() else getattr(module, p)
            setattr(module, param_path[-1], weight)

    hidden_states_original = []
    def hook_fn_orig(module, inp, out):
        hidden_states_original.append(out[0].detach())
    if layer_index < len(model.model.layers) - 1:
        handle = model.model.layers[layer_index + 1].register_forward_hook(hook_fn_orig)
    else:
        handle = model.model.norm.register_forward_hook(hook_fn_orig)

    model(calib_input_ids, attention_mask=calib_attention_mask)
    handle.remove()

    for name, weight in current_weights.items():
        if name.startswith(f"model.layers.{layer_index}"):
            param_path = name.split('.')
            module = model
            for p in param_path[:-1]:
                module = module[int(p)] if p.isdigit() else getattr(module, p)
            setattr(module, param_path[-1], weight)

    hidden_states_perturbed = []
    def hook_fn_pert(module, inp, out):
        hidden_states_perturbed.append(out[0].detach())
    if layer_index < len(model.model.layers) - 1:
        handle = model.model.layers[layer_index + 1].register_forward_hook(hook_fn_pert)
    else:
        handle = model.model.norm.register_forward_hook(hook_fn_pert)

    model(calib_input_ids, attention_mask=calib_attention_mask)
    handle.remove()
    
    perturbation = hidden_states_original[0] - hidden_states_perturbed[0]  # [n_tokens, seq_len, hidden_size]
    token_perturb_norms = torch.norm(perturbation[:, 0, :], dim=-1)  # shape: [n_tokens]
    total_sum = token_perturb_norms.sum().item()
    num_tokens_total = token_perturb_norms.numel()
    avg = total_sum / num_tokens_total if num_tokens_total > 0 else 0

    return total_sum, avg, token_perturb_norms

def get_adaptive_sparsity(layer_index, total_layers, base_sparsity, delta_h_current, delta_h_target_sum, 
                         min_factor=0.5, max_factor=1.5, sensitivity_weight=0.5):
    base_layer_sparsity = base_sparsity * (min_factor + (max_factor - min_factor) * (layer_index / (total_layers - 1)))
    if delta_h_current == 0 or delta_h_target_sum == 0:
        return base_layer_sparsity
    delta_ratio = delta_h_target_sum / delta_h_current
    adjustment_factor = torch.clamp(torch.tensor(delta_ratio ** sensitivity_weight), min_factor, max_factor).item()
    adjusted_sparsity = base_layer_sparsity * adjustment_factor
    adjusted_sparsity = max(0.0, min(adjusted_sparsity, 1.0))
    return adjusted_sparsity

def adjust_mask_based_on_total_perturbation(model, layer_index, subset, original_weights, current_weights, weight_mask,
                                              tokenizer, delta_h_target_sum, device=None, max_iterations=5, restore_ratio_base=0.1):
    if device is None:
        device = next(model.parameters()).device

    adjusted_mask = {name: mask.clone() for name, mask in weight_mask.items()}

    total_perturbation, avg_perturbation = estimate_total_token_perturbation(
        model, layer_index, original_weights, current_weights, calib_input_ids, calib_attention_mask, device=device
    )

    print(f"Layer {layer_index}: Initial total token perturbation: {total_perturbation:.6f} (target: {delta_h_target_sum:.6f})")

    if total_perturbation <= delta_h_target_sum:
        return adjusted_mask, total_perturbation

    for iteration in range(max_iterations):
        print(f"Layer {layer_index}: Iteration {iteration+1}, total perturbation: {total_perturbation:.6f}")
        restore_ratio = min(restore_ratio_base * (iteration + 1), 0.5)
        for name in subset:
            orig_mask = adjusted_mask[name]
            orig_weight = original_weights[name]
            importance = torch.abs(orig_weight) * orig_mask.float()

            pruned_indices = torch.where(orig_mask)
            if len(pruned_indices[0]) == 0:
                continue

            pruned_importance = importance[pruned_indices]
            num_to_restore = max(1, int(len(pruned_indices[0]) * restore_ratio))
            _, top_indices = torch.topk(pruned_importance, num_to_restore)

            for idx in top_indices:
                row, col = pruned_indices[0][idx], pruned_indices[1][idx]
                adjusted_mask[name][row, col] = False

            current_weights[name] = orig_weight.clone()
            current_weights[name][adjusted_mask[name]] = 0

        total_perturbation, avg_perturbation = estimate_total_token_perturbation(
            model, layer_index, original_weights, current_weights, calib_input_ids, calib_attention_mask, device=device
        )
        if total_perturbation <= delta_h_target_sum:
            print(f"Layer {layer_index}: Total perturbation reduced to {total_perturbation:.6f}")
            break

    print(f"Layer {layer_index}: Final total token perturbation after adjustment: {total_perturbation:.6f}")
    return adjusted_mask, total_perturbation

def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()
            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params

def prepare_calibration_input(args, model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids

def return_reorder_indice(input_tensor):
    positive_tensor = input_tensor.clone()
    negative_tensor = input_tensor.clone()

    positive_mask = positive_tensor > 0
    negative_mask = negative_tensor < 0

    positive_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )
    negative_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )

    positive_indices[~positive_mask] = float("inf")
    negative_indices[~negative_mask] = float("inf")

    positive_value, _ = torch.sort(positive_indices, dim=1)
    negative_value, _ = torch.sort(negative_indices, dim=1)

    positive_value = torch.flip(positive_value, dims=[1])

    negative_value[negative_value == float("inf")] = 0
    positive_value[positive_value == float("inf")] = 0

    reorder_indice = (positive_value + negative_value).to(torch.int64)

    return reorder_indice

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers

    total_time = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            start_time = time.time()
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = torch.zeros_like(W) == 1
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * args.sparsity_ratio)].cpu()
                W_mask = W_metric <= thresh
            W[W_mask] = 0
            end_time = time.time()
            total_time += end_time - start_time

    if args.get_time_overhead:
        save_time_result(args, args.output_results_file, total_time)

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device)

    total_time = 0
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev, non_blocking=True)
            outs = outs.to(dev, non_blocking=True)
            attention_mask = attention_mask.to(dev, non_blocking=True)
            position_ids = position_ids.to(dev, non_blocking=True)
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        for name in subset:
            print(f"pruning layer {i} name {name}")
            start_time = time.time()
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_mask = torch.zeros_like(W_metric) == 1
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, : int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)
            subset[name].weight.data[W_mask] = 0
            end_time = time.time()
            total_time += end_time - start_time
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    if args.get_time_overhead:
        save_time_result(args, args.output_results_file, total_time)
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    print("Starting ...")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    print("Ready.")
    total_time = 0
    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps = inps.to(dev, non_blocking=True)
            outs = outs.to(dev, non_blocking=True)
            attention_mask = attention_mask.to(dev, non_blocking=True)
            position_ids = position_ids.to(dev, non_blocking=True)
        subset = find_layers(layer)
        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            print(f"pruning layer {i} name {name}")
            start_time = time.time()
            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            end_time = time.time()
            total_time += end_time - start_time
            gpts[name].free()
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    if args.get_time_overhead:
        save_time_result(args, args.output_results_file, total_time)
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def prune_DSnoT(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    is_distributed = hasattr(args, 'distributed') and args.distributed
    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend='nccl')
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    delta_h_target_sum, calib_delta_h_list, calib_input_ids, calib_attention_mask = compute_delta_h_target_sum(
        model, tokenizer, n_tokens=10, sequence_length=128, device=device
    )
    print(f"Target global δh Sum: {delta_h_target_sum:.6f}")
    print("Loading calibration data...")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("Calibration data loaded")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device)
    layers = model.model.layers
    total_layers = len(layers)
    if is_distributed:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        layers_per_rank = (total_layers + world_size - 1) // world_size
        start_layer = min(rank * layers_per_rank, total_layers)
        end_layer = min(start_layer + layers_per_rank, total_layers)
        layer_indices = list(range(start_layer, end_layer))
    else:
        layer_indices = list(range(total_layers))
    original_weights_all = {}
    current_weights_all = {}
    weight_masks_all = {}
    delta_h_history = {}
    total_time = 0
    global_objective_weight = getattr(args, 'global_objective_weight', 0.5)
    max_iterations = getattr(args, 'max_global_iterations', 3)
    enable_feedback = getattr(args, 'enable_layer_feedback', False)
    for i_idx, i in enumerate(layer_indices):
        layer = layers[i]
        subset = find_layers(layer)
        if hasattr(model, 'hf_device_map') and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev, non_blocking=True)
            outs = outs.to(dev, non_blocking=True)
            attention_mask = attention_mask.to(dev, non_blocking=True)
            position_ids = position_ids.to(dev, non_blocking=True)
            device = dev
        if i_idx > 0 and hasattr(args, 'adaptive_sparsity') and args.adaptive_sparsity:
            layer_sparsity = args.sparsity_ratio * (0.8 + 0.4 * (i / (total_layers - 1)))
            layer_sparsity = min(0.95, max(0.1, layer_sparsity))
            print(f"Layer {i}: using adaptive sparsity {layer_sparsity:.4f}")
        else:
            layer_sparsity = args.sparsity_ratio
            print(f"Layer {i}: using global sparsity {layer_sparsity:.4f}")
        active_sublayers = []
        for sub_idx, name in enumerate(subset):
            start_time = time.time()
            active_sublayers.append(name)
            print(f"Layer {i}: Adding sublayer {name}, total active sublayers: {len(active_sublayers)}")
            optimize_sublayer_group(
                args, model, i, active_sublayers, subset,
                original_weights_all, current_weights_all, weight_masks_all,
                tokenizer, delta_h_target_sum, calib_input_ids, calib_attention_mask, inps, outs, attention_mask, position_ids,
                device, layer_sparsity, global_objective_weight, max_iterations,
                prune_n, prune_m
            )
            end_time = time.time()
            total_time += end_time - start_time
        for name in subset:
            key = f"{i}_{name}"
            if key in current_weights_all:
                subset[name].weight.data = current_weights_all[key]
        original_weights_dict = {name: original_weights_all[f"{i}_{name}"] for name in active_sublayers}
        current_weights_dict = {name: current_weights_all[f"{i}_{name}"] for name in active_sublayers}
        total_delta, avg_delta, current_token_perturb_norms = estimate_total_token_perturbation(
            model, i, original_weights_dict, current_weights_dict, calib_input_ids, calib_attention_mask, device=device
        )
        delta_h_history[i] = total_delta
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
        # 如果启用层间反馈，使用 total_delta 判断
        if enable_feedback and i_idx > 0:
            total_delta, _, _ = estimate_total_token_perturbation(model, i, original_weights_dict, current_weights_dict, calib_input_ids, calib_attention_mask, device=device)
        if total_delta > delta_h_target_sum * 1.2:
             print(f"Layer {i}: Total δh {total_delta:.6f} exceeds target, considering feedback optimization")
             # 反馈优化逻辑待实现
    if is_distributed:
        dist.barrier()
        if dist.get_rank() == 0:
            print("Aggregating results from all processes...")
            # 聚合代码
    if args.get_time_overhead:
        save_time_result(args, args.output_results_file, total_time)
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    return check_sparsity(model)

def optimize_sublayer_group(args, model, layer_idx, active_sublayers, layer_subset,
                            original_weights_all, current_weights_all, weight_masks_all,
                            tokenizer, delta_h_target_sum, calib_input_ids, calib_attention_mask, inps, outs, attention_mask, position_ids,
                            device, layer_sparsity, global_objective_weight, max_iterations,
                            prune_n=0, prune_m=0):
    wrapped_layers = {}
    temp_cache = {}
    for name in active_sublayers:
        key = f"{layer_idx}_{name}"
        if key not in original_weights_all:
            original_weights_all[key] = layer_subset[name].weight.data.clone()
        wrapped_layers[name] = WrappedGPT(layer_subset[name], initial_method=args.initial_method)
    def add_batch(name):
        def tmp(_, inp, out):
            wrapped_layers[name].add_batch(inp[0].data, out.data)
        return tmp
    handles = []
    for name in active_sublayers:
        handles.append(layer_subset[name].register_forward_hook(add_batch(name)))
    for j in range(args.nsamples):
        try:
            with torch.no_grad():
                outs[j] = model.model.layers[layer_idx](inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("WARNING: OOM during forward pass. Reducing batch size.")
                torch.cuda.empty_cache()
                for k in range(j, args.nsamples):
                    with torch.no_grad():
                        outs[k] = model.model.layers[layer_idx](inps[k].unsqueeze(0), attention_mask=attention_mask[0:1], position_ids=position_ids[0:1])[0]
            else:
                raise e
    for h in handles:
        h.remove()
    # 初始掩码计算
    for name in active_sublayers:
        key = f"{layer_idx}_{name}"
        if key not in weight_masks_all:
            if args.initial_method == "wanda":
                initial_metric = torch.abs(layer_subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            elif args.initial_method == "magnitude":
                initial_metric = torch.abs(layer_subset[name].weight.data)
            elif args.initial_method == "sparsegpt":
                W = layer_subset[name].weight.data.clone()
                if isinstance(layer_subset[name], nn.Conv2d):
                    W = W.flatten(1)
                if isinstance(layer_subset[name], transformers.Conv1D):
                    W = W.t()
                W = W.float()
                H = wrapped_layers[name].H
                dead = torch.diag(H) == 0
                H[dead, dead] = 1
                W[:, dead] = 0
                percdamp = 0.01
                damp = percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(wrapped_layers[name].columns, device=wrapped_layers[name].dev)
                H[diag, diag] += damp
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H
                initial_metric = W**2 / (torch.diag(Hinv).reshape((1, -1))) ** 2
            if prune_n != 0:
                weight_mask = torch.zeros_like(initial_metric) == 1
                for ii in range(initial_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = initial_metric[:, ii : (ii + prune_m)].float()
                        weight_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(initial_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(initial_metric.shape[1] * layer_sparsity)]
                weight_mask = torch.zeros_like(initial_metric) == 1
                weight_mask.scatter_(1, indices, True)
            weight_masks_all[key] = weight_mask
            current_weights = original_weights_all[key].clone()
            current_weights[weight_mask] = 0
            current_weights_all[key] = current_weights
            layer_subset[name].weight.data = current_weights
            temp_cache[f"{key}_importance"] = initial_metric
    original_weights_dict = {name: original_weights_all[f"{layer_idx}_{name}"] for name in active_sublayers}
    current_weights_dict = {name: current_weights_all[f"{layer_idx}_{name}"] for name in active_sublayers}
    total_delta, _, _ = estimate_total_token_perturbation(
        model, layer_idx, original_weights_dict, current_weights_dict, calib_input_ids, calib_attention_mask, device=device
    )
    initial_recon_error = compute_total_reconstruction_error(active_sublayers, wrapped_layers, current_weights_dict)
    initial_objective = compute_global_objective(total_delta, initial_recon_error, delta_h_target_sum, global_objective_weight)
    print(f"Layer {layer_idx}: Initial total δh = {total_delta:.6f}, target = {delta_h_target_sum:.6f}, recon_error = {initial_recon_error:.6f}")
    print(f"Layer {layer_idx}: Initial global objective = {initial_objective:.6f}")
    need_optimization = (
        len(active_sublayers) > 1 and
        (total_delta > delta_h_target_sum * 0.9 or initial_objective > 0.1 or getattr(args, 'always_optimize', False))
    )
    if need_optimization:
        print(f"Layer {layer_idx}: Performing joint optimization for {len(active_sublayers)} sublayers")
        importance_scores = {}
        for name in active_sublayers:
            key = f"{layer_idx}_{name}"
            DSnoT_metric = original_weights_all[key] * wrapped_layers[name].sum_metric_row.reshape((1, -1))
            if hasattr(args, 'pow_of_var_regrowing') and args.pow_of_var_regrowing:
                DSnoT_metric /= torch.pow(wrapped_layers[name].var.reshape((1, -1)), args.pow_of_var_regrowing)
            importance_scores[name] = torch.abs(DSnoT_metric)
        optimize_group_with_unified_objective(
            args, model, layer_idx, active_sublayers, layer_subset,
            wrapped_layers, original_weights_all, current_weights_all, weight_masks_all,
            importance_scores, tokenizer, delta_h_target_sum, calib_input_ids, calib_attention_mask, layer_sparsity, global_objective_weight, device, max_iterations
        )
    for key in list(temp_cache.keys()):
        del temp_cache[key]
    final_weights_dict = {name: current_weights_all[f"{layer_idx}_{name}"] for name in active_sublayers}
    total_delta_final, _, _ = estimate_total_token_perturbation(
        model, layer_idx, original_weights_dict, final_weights_dict, calib_input_ids, calib_attention_mask, device=device
    )
    final_recon_error = compute_total_reconstruction_error(active_sublayers, wrapped_layers, final_weights_dict)
    final_objective = compute_global_objective(total_delta_final, final_recon_error, delta_h_target_sum, global_objective_weight)
    print(f"Layer {layer_idx}: Final optimization results for {len(active_sublayers)} sublayers:")
    print(f"  Total δh: {total_delta_final:.6f} (target: {delta_h_target_sum:.6f})")
    print(f"  Reconstruction error: {final_recon_error:.6f}")
    print(f"  Global objective: {final_objective:.6f}")
    total_params = 0
    pruned_params = 0
    for name in active_sublayers:
        key = f"{layer_idx}_{name}"
        mask = weight_masks_all[key]
        total_params += mask.numel()
        pruned_params += mask.sum().item()
    actual_sparsity = pruned_params / total_params
    print(f"  Actual sparsity: {actual_sparsity:.4f} (target: {layer_sparsity:.4f})")
    del wrapped_layers
    torch.cuda.empty_cache()

def optimize_group_with_unified_objective(args, model, layer_idx, active_sublayers, layer_subset,
                                          wrapped_layers, original_weights_all, current_weights_all, weight_masks_all,
                                          importance_scores, tokenizer, delta_h_target_sum, calib_input_ids, calib_attention_mask, layer_sparsity,
                                          global_objective_weight, device, max_iterations=3):
    total_params = 0
    pruned_params = 0
    for name in active_sublayers:
        key = f"{layer_idx}_{name}"
        mask = weight_masks_all[key]
        total_params += mask.numel()
        pruned_params += mask.sum().item()
    current_sparsity = pruned_params / total_params
    original_weights_dict = {name: original_weights_all[f"{layer_idx}_{name}"] for name in active_sublayers}
    best_objective = float('inf')
    best_masks = {name: weight_masks_all[f"{layer_idx}_{name}"].clone() for name in active_sublayers}
    for iteration in range(max_iterations):
        print(f"Layer {layer_idx}: Joint optimization iteration {iteration+1}/{max_iterations}")
        if abs(current_sparsity - layer_sparsity) > 0.005:
            adjust_group_sparsity(active_sublayers, layer_idx, importance_scores, weight_masks_all, layer_sparsity, current_sparsity)
            for name in active_sublayers:
                key = f"{layer_idx}_{name}"
                current_weight = original_weights_all[key].clone()
                current_weight[weight_masks_all[key]] = 0
                current_weights_all[key] = current_weight
                layer_subset[name].weight.data = current_weight
            pruned_params = 0
            for name in active_sublayers:
                key = f"{layer_idx}_{name}"
                pruned_params += weight_masks_all[key].sum().item()
            current_sparsity = pruned_params / total_params
        current_weights_dict = {name: current_weights_all[f"{layer_idx}_{name}"] for name in active_sublayers}
        total_delta, _, _ = estimate_total_token_perturbation(
            model, layer_idx, original_weights_dict, current_weights_dict, calib_input_ids, calib_attention_mask, device=device
        )
        recon_error = compute_total_reconstruction_error(active_sublayers, wrapped_layers, current_weights_dict)
        current_objective = compute_global_objective(total_delta, recon_error, delta_h_target_sum, global_objective_weight)
        print(f"  Current metrics - Total δh: {total_delta:.6f}, recon_error: {recon_error:.6f}, objective: {current_objective:.6f}")
        if current_objective < best_objective:
            best_objective = current_objective
            for name in active_sublayers:
                key = f"{layer_idx}_{name}"
                best_masks[name] = weight_masks_all[key].clone()
            print(f"  New best objective: {best_objective:.6f}")
        if total_delta > delta_h_target_sum:
            print(f"  Total δh {total_delta:.6f} exceeds target {delta_h_target_sum:.6f}, reducing δh")
            improve_delta_h(args, model, layer_idx, active_sublayers, layer_subset,
                            wrapped_layers, original_weights_all, current_weights_all, weight_masks_all,
                            importance_scores, tokenizer, delta_h_target_sum, device)
        else:
            print("  Total δh is within target, optimizing reconstruction error")
            improve_reconstruction_error(args, model, layer_idx, active_sublayers, layer_subset,
                                 wrapped_layers, original_weights_all, current_weights_all, weight_masks_all,
                                 importance_scores, tokenizer, delta_h_target_sum, calib_input_ids, calib_attention_mask, device)
        if iteration < max_iterations - 1:
            pruned_params = 0
            for name in active_sublayers:
                key = f"{layer_idx}_{name}"
                pruned_params += weight_masks_all[key].sum().item()
            current_sparsity = pruned_params / total_params
            if abs(current_sparsity - layer_sparsity) > 0.01:
                print(f"  Sparsity drift detected: {current_sparsity:.4f} vs target {layer_sparsity:.4f}, readjusting")
                adjust_group_sparsity(active_sublayers, layer_idx, importance_scores, weight_masks_all, layer_sparsity, current_sparsity)
                for name in active_sublayers:
                    key = f"{layer_idx}_{name}"
                    current_weight = original_weights_all[key].clone()
                    current_weight[weight_masks_all[key]] = 0
                    current_weights_all[key] = current_weight
                    layer_subset[name].weight.data = current_weight
    for name in active_sublayers:
        key = f"{layer_idx}_{name}"
        weight_masks_all[key] = best_masks[name].clone()
        current_weight = original_weights_all[key].clone()
        current_weight[weight_masks_all[key]] = 0
        current_weights_all[key] = current_weight
        layer_subset[name].weight.data = current_weight
    current_weights_dict = {name: current_weights_all[f"{layer_idx}_{name}"] for name in active_sublayers}
    total_delta, _, _ = estimate_total_token_perturbation(
            model, layer_idx, original_weights_dict, current_weights_dict, calib_input_ids, calib_attention_mask, device=device
        )
    recon_error = compute_total_reconstruction_error(active_sublayers, wrapped_layers, current_weights_dict)
    final_objective = compute_global_objective(total_delta, recon_error, delta_h_target_sum, global_objective_weight)
    print(f"  Final joint optimization: Total δh = {total_delta:.6f}, recon_error = {recon_error:.6f}, objective = {final_objective:.6f}")

def adjust_group_sparsity(active_sublayers, layer_idx, importance_scores, weight_masks_all, target_sparsity, current_sparsity):
    all_importance = []
    all_mask_indices = []
    for name in active_sublayers:
        key = f"{layer_idx}_{name}"
        mask = weight_masks_all[key]
        importance = importance_scores[name]
        if current_sparsity < target_sparsity:
            unpruned_indices = torch.nonzero(~mask, as_tuple=True)
            if len(unpruned_indices[0]) > 0:
                unpruned_importance = importance[unpruned_indices]
                for i, imp in enumerate(unpruned_importance):
                    row, col = unpruned_indices[0][i].item(), unpruned_indices[1][i].item()
                    all_importance.append(imp.item())
                    all_mask_indices.append((name, row, col, False))
        else:
            pruned_indices = torch.nonzero(mask, as_tuple=True)
            if len(pruned_indices[0]) > 0:
                pruned_importance = importance[pruned_indices]
                for i, imp in enumerate(pruned_importance):
                    row, col = pruned_indices[0][i].item(), pruned_indices[1][i].item()
                    all_importance.append(imp.item())
                    all_mask_indices.append((name, row, col, True))
    if not all_importance:
        return
    importance_with_indices = list(zip(all_importance, all_mask_indices))
    if current_sparsity < target_sparsity:
        importance_with_indices.sort()
    else:
        importance_with_indices.sort(reverse=True)
    total_params = 0
    for name in active_sublayers:
        key = f"{layer_idx}_{name}"
        total_params += weight_masks_all[key].numel()
    params_to_adjust = int(abs(target_sparsity - current_sparsity) * total_params)
    params_to_adjust = min(params_to_adjust, len(importance_with_indices))
    for i in range(params_to_adjust):
        _, (name, row, col, is_pruned) = importance_with_indices[i]
        key = f"{layer_idx}_{name}"
        if current_sparsity < target_sparsity:
            weight_masks_all[key][row, col] = True
        else:
            weight_masks_all[key][row, col] = False

def improve_delta_h(args, model, layer_idx, active_sublayers, layer_subset,
                    wrapped_layers, original_weights_all, current_weights_all, weight_masks_all,
                    importance_scores, tokenizer, delta_h_target_sum, calib_input_ids, calib_attention_mask, device):
    original_weights_dict = {name: original_weights_all[f"{layer_idx}_{name}"] for name in active_sublayers}
    pruned_params = []
    for name in active_sublayers:
        key = f"{layer_idx}_{name}"
        mask = weight_masks_all[key]
        importance = importance_scores[name]
        pruned_indices = torch.nonzero(mask, as_tuple=True)
        if len(pruned_indices[0]) > 0:
            pruned_importance = importance[pruned_indices]
            for i, imp in enumerate(pruned_importance):
                row, col = pruned_indices[0][i].item(), pruned_indices[1][i].item()
                pruned_params.append((imp.item(), name, row, col))
    pruned_params.sort(reverse=True)
    total_delta, _, _ = estimate_total_token_perturbation(
        model, layer_idx, original_weights_dict,
        {name: current_weights_all[f"{layer_idx}_{name}"] for name in active_sublayers},
        calib_input_ids, calib_attention_mask, device=device
    )
    params_restored = 0
    max_restore = min(500, len(pruned_params))
    for idx, (imp, name, row, col) in enumerate(pruned_params[:max_restore]):
        key = f"{layer_idx}_{name}"
        weight_masks_all[key][row, col] = False
        current_weight = original_weights_all[key].clone()
        current_weight[weight_masks_all[key]] = 0
        current_weights_all[key] = current_weight
        layer_subset[name].weight.data = current_weight
        params_restored += 1
        if (params_restored % 10 == 0) or (idx == max_restore - 1):
            total_delta, _, _ = estimate_total_token_perturbation(
                model, layer_idx, original_weights_dict,
                {name: current_weights_all[f"{layer_idx}_{name}"] for name in active_sublayers},
                calib_input_ids, calib_attention_mask, device=device
            )
            if total_delta <= delta_h_target_sum:
                print(f"  Total token perturbation reduced to {total_delta:.6f} after restoring {params_restored} parameters")
                break
    print(f"  Total parameters restored: {params_restored}")

def improve_reconstruction_error(args, model, layer_idx, active_sublayers, layer_subset,
                                 wrapped_layers, original_weights_all, current_weights_all, weight_masks_all,
                                 importance_scores, tokenizer, delta_h_target_sum, calib_input_ids, calib_attention_mask, device):
    original_weights_dict = {name: original_weights_all[f"{layer_idx}_{name}"] for name in active_sublayers}
    current_weights_dict = {name: current_weights_all[f"{layer_idx}_{name}"] for name in active_sublayers}
    initial_recon_error = compute_total_reconstruction_error(active_sublayers, wrapped_layers, current_weights_dict)
    init_mean, init_max, init_p95 = estimate_combined_perturbation(
        model, layer_idx, original_weights_dict, current_weights_dict, tokenizer, calib_input_ids, calib_attention_mask, device=device
    )
    print(f"  Initial reconstruction error: {initial_recon_error:.2f}, δh (p95): {init_p95:.6f}")
    successful_swaps = []
    best_recon_error = initial_recon_error
    max_iterations = 10
    top_k_per_layer = 5
    for iteration in range(max_iterations):
        print(f"  Iteration {iteration+1}/{max_iterations}")
        found_improvement = False
        for name in active_sublayers:
            key = f"{layer_idx}_{name}"
            mask = weight_masks_all[key]
            W = original_weights_all[key]
            if hasattr(wrapped_layers[name], 'H') and wrapped_layers[name].H is not None:
                H = wrapped_layers[name].H.float()
                diag_H = torch.diag(H)
                diag_H = torch.where(diag_H > 0, diag_H, torch.ones_like(diag_H)*1e-6)
                recon_impact = W.pow(2) / diag_H.reshape(1, -1)
            else:
                recon_impact = torch.abs(W * wrapped_layers[name].sum_metric_row.reshape(1, -1))
            recon_impact_norm = recon_impact / (torch.max(recon_impact) + 1e-10)
            unpruned_indices = torch.nonzero(~mask, as_tuple=True)
            pruned_indices = torch.nonzero(mask, as_tuple=True)
            if len(unpruned_indices[0]) == 0 or len(pruned_indices[0]) == 0:
                continue
            unpruned_importance = recon_impact_norm[unpruned_indices]
            pruned_importance = recon_impact_norm[pruned_indices]
            max_params = min(10, len(unpruned_importance))
            unpruned_values, unpruned_indices_sorted = torch.topk(unpruned_importance, max_params, largest=False)
            max_params = min(10, len(pruned_importance))
            pruned_values, pruned_indices_sorted = torch.topk(pruned_importance, max_params, largest=True)
            layer_candidates = []
            for p_i in range(min(max_params, 20)):
                p_idx = pruned_indices_sorted[p_i]
                p_row, p_col = pruned_indices[0][p_idx], pruned_indices[1][p_idx]
                p_val = pruned_values[p_i].item()
                for u_i in range(min(max_params, 20)):
                    u_idx = unpruned_indices_sorted[u_i]
                    u_row, u_col = unpruned_indices[0][u_idx], unpruned_indices[1][u_idx]
                    u_val = unpruned_values[u_i].item()
                    if p_val > u_val * 2:
                        gain = p_val - u_val
                        layer_candidates.append((gain, p_row.item(), p_col.item(), u_row.item(), u_col.item()))
            layer_candidates.sort(reverse=True, key=lambda x: x[0])
            layer_candidates = layer_candidates[:top_k_per_layer]
            for _, p_row, p_col, u_row, u_col in layer_candidates:
                temp_mask = weight_masks_all[key].clone()
                temp_mask[p_row, p_col] = False  # 恢复被剪枝参数
                temp_mask[u_row, u_col] = True   # 剪枝不重要参数
                temp_weights = {n: current_weights_all[f"{layer_idx}_{n}"].clone() for n in active_sublayers}
                temp_weight = original_weights_all[key].clone()
                temp_weight[temp_mask] = 0
                temp_weights[name] = temp_weight
                for n, w in temp_weights.items():
                    layer_subset[n].weight.data = w
                _, _, p95_p = estimate_combined_perturbation(
                    model, layer_idx, original_weights_dict, temp_weights, tokenizer, calib_input_ids, calib_attention_mask, device=device
                )
                if p95_p <= delta_h_target_sum * 1.5:
                    temp_recon_error = compute_total_reconstruction_error(active_sublayers, wrapped_layers, temp_weights)
                    improvement = best_recon_error - temp_recon_error
                    if improvement > 0:
                        print(f"    Found better swap in {name}: δh (p95)={p95_p:.6f}, recon_error={temp_recon_error:.2f}, improvement={improvement:.2f}")
                        best_recon_error = temp_recon_error
                        weight_masks_all[key] = temp_mask.clone()
                        current_weights_all[key] = temp_weight.clone()
                        successful_swaps.append((name, p_row, p_col, u_row, u_col))
                        found_improvement = True
                        for n in active_sublayers:
                            k = f"{layer_idx}_{n}"
                            if n == name:
                                continue
                            layer_subset[n].weight.data = current_weights_all[k]
                        break
                for n in active_sublayers:
                    k = f"{layer_idx}_{n}"
                    layer_subset[n].weight.data = current_weights_all[k]
        if not found_improvement:
            print(f"  No further improvements found in iteration {iteration+1}")
            break
        if best_recon_error < initial_recon_error * 0.5:
            print(f"  Sufficient improvement achieved: {initial_recon_error:.2f} -> {best_recon_error:.2f}")
            break
    for name in active_sublayers:
        key = f"{layer_idx}_{name}"
        layer_subset[name].weight.data = current_weights_all[key]
    final_weights_dict = {name: current_weights_all[f"{layer_idx}_{name}"] for name in active_sublayers}
    final_mean, final_max, final_p95 = estimate_combined_perturbation(model, layer_idx, original_weights_dict, temp_weights, tokenizer, calib_input_ids, calib_attention_mask, device=device)
    final_recon_error = compute_total_reconstruction_error(active_sublayers, wrapped_layers, final_weights_dict)
    print(f"  Final results: δh (p95) = {final_p95:.6f}, recon_error = {final_recon_error:.2f}")
    print(f"  Total successful swaps: {len(successful_swaps)}")
    return final_recon_error

def compute_global_objective(total_delta_h, recon_error, delta_h_target_sum, weight=0.5):
    if total_delta_h > delta_h_target_sum:
        delta_h_norm = 1.0 + (total_delta_h - delta_h_target_sum) / delta_h_target_sum
    else:
        delta_h_norm = total_delta_h / delta_h_target_sum
    recon_norm = recon_error / 1e6  
    return weight * delta_h_norm + (1 - weight) * recon_norm

def estimate_combined_perturbation(model, layer_idx, original_weights, current_weights, tokenizer, 
                                   calib_input_ids, calib_attention_mask, device=None):
    if device is None:
        device = next(model.parameters()).device

    temp_originals = {}
    for name, weight in original_weights.items():
        module = get_module_by_name(model.model.layers[layer_idx], name)
        if module is not None:
            temp_originals[name] = module.weight.data.clone()
            module.weight.data = weight.clone()

    hidden_states_original = []
    def hook_fn(module, inp, out):
        hidden_states_original.append(out[0].detach())
    if layer_idx < len(model.model.layers) - 1:
        handle = model.model.layers[layer_idx + 1].register_forward_hook(hook_fn)
    else:
        if hasattr(model.model, "norm"):
            handle = model.model.norm.register_forward_hook(hook_fn)
        else:
            last_module = list(model.model.children())[-1]
            handle = last_module.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(calib_input_ids, attention_mask=calib_attention_mask)
    handle.remove()

    for name, weight in current_weights.items():
        module = get_module_by_name(model.model.layers[layer_idx], name)
        if module is not None:
            module.weight.data = weight.clone()

    hidden_states_perturbed = []
    def hook_fn(module, inp, out):
        hidden_states_perturbed.append(out[0].detach())
    if layer_idx < len(model.model.layers) - 1:
        handle = model.model.layers[layer_idx + 1].register_forward_hook(hook_fn)
    else:
        if hasattr(model.model, "norm"):
            handle = model.model.norm.register_forward_hook(hook_fn)
        else:
            last_module = list(model.model.children())[-1]
            handle = last_module.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(calib_input_ids, attention_mask=calib_attention_mask)
    handle.remove()

    perturbation = hidden_states_original[0] - hidden_states_perturbed[0]
    perturb_norms = torch.norm(perturbation[:, 0, :], dim=-1)

    for name, weight in temp_originals.items():
        module = get_module_by_name(model.model.layers[layer_idx], name)
        if module is not None:
            module.weight.data = weight

    mean_perturbation = torch.mean(perturb_norms).item()
    max_perturbation = torch.max(perturb_norms).item()
    p95_perturbation = torch.quantile(perturb_norms.float(), 0.95).item()
    return mean_perturbation, max_perturbation, p95_perturbation

def compute_total_reconstruction_error(active_sublayers, wrapped_layers, current_weights_dict):
    total_error = 0.0
    for name in active_sublayers:
        if name in current_weights_dict and name in wrapped_layers:
            weight = current_weights_dict[name]
            if hasattr(wrapped_layers[name], 'sum_metric_row'):
                DSnoT_metric = weight * wrapped_layers[name].sum_metric_row.reshape((1, -1))
                layer_error = torch.sum(torch.abs(torch.sum(DSnoT_metric, dim=1)))
                total_error += layer_error.item()
    return total_error

def get_module_by_name(model, name):
    name_parts = name.split(".")
    current_module = model
    for part in name_parts:
        if part.isdigit():
            current_module = current_module[int(part)]
        elif hasattr(current_module, part):
            current_module = getattr(current_module, part)
        else:
            return None
    return current_module