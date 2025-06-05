class MirrorWorkerWarp:
    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl", use_ray=False
    ):
        
        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        """Init torch process group for model weights update"""
        import torch
        from openrlhf.utils.distributed_util import init_process_group

        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=rank, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
        self._model_update_with_ray = use_ray
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        import torch

        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        """主进程调试"""
        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        # 创建一个空的张量，用于存储权重
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        import torch
        from openrlhf.trainer.ray.utils import get_physical_gpu_id

        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        func, args = handle
        list_args = list(args)
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
        weight = func(*list_args)
        self.model_runner.model.load_weights(weights=[(name, weight)])
        torch.cuda.synchronize()

    def sync_grads(self, name, dtype, shape, empty_cache = False):
        """
        sync model grads
        """
        import torch
        if torch.distributed.get_rank() == 0:
            print(f"sync grads: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        grads = torch.empty(shape, dtype=dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective
            collective.broadcast(grads, 0, group_name=self._model_update_group)
        else:
            torch.distributed.broadcast(grads, 0, group=self._model_update_group)

        param = self.model_runner.model.get_parameter(name)
        param.grad = grads

        del grads

    def step(self):
        """
        mirror update weight
        """
        if not hasattr(self, "optimizer"):
            print("Warning: optimizer not initialized, skipping step")
            return False
        self.optimizer.step()
        self.optimizer.zero_grad()
        return True

        
    def create_ops(
        self, name, lr = 1e-5, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0, adamw_mode = True
    ):
        from deepspeed.ops.adam import FusedAdam
        import torch

        model_params = self.model_runner.model.parameters()
        optimizer = FusedAdam(
            model_params,
            lr = 1e-3,
            betas = (0.9, 0.999),
            eps = 1e-8,
            weight_decay = 0,
            adamw_mode = True
        )
        self.optimizer = optimizer
        print(f"Create Mirror Optimizer with {type(optimizer).__name__}")
        print(f"Mirror Optimizer init lr: {lr}, betas: {betas}, eps: {eps}, weight_decay: {weight_decay}, adamw_mode: {adamw_mode}")
        return optimizer
    