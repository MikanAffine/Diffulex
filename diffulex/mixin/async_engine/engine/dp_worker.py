import asyncio

from diffulex.sampling_params import SamplingParams


class DiffulexDPWorkerAsyncMixin:
    """Async RPC helpers extracted from DiffulexDPWorker."""

    async def _ask_async(self, replica: int, cmd: str, *args):
        """Async version of _ask that runs in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._ask, replica, cmd, *args)

    async def add_request_async(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Async version of add_request."""
        target = self._rr
        self._rr = (self._rr + 1) % self.dp_size
        local_id = await self._ask_async(target, "add_request", prompt, sampling_params)
        gid = self._gid_counter
        self._gid_counter += 1
        self._gid_map[(target, local_id)] = gid
        self._rev_gid_map[gid] = (target, local_id)
        return gid

    async def step_async(self):
        """Async version of step that runs all DP replicas concurrently."""
        all_outputs = []
        total_tokens = 0
        any_prefill = False
        merged_diff_steps = {}
        merged_deltas = []

        # Check all replicas in parallel
        tasks = [self._ask_async(i, "is_finished") for i in range(self.dp_size)]
        done_flags = await asyncio.gather(*tasks)

        # Step all non-finished replicas in parallel
        step_tasks = []
        for i, done in enumerate(done_flags):
            if not done:
                step_tasks.append((i, self._ask_async(i, "step")))

        if step_tasks:
            step_results = await asyncio.gather(*[task for _, task in step_tasks])
            for (i, _), (outputs, num_tokens, is_prefill, n_diff_steps, deltas) in zip(step_tasks, step_results):
                if outputs:
                    # remap local seq_ids to global ids
                    for sid, toks in outputs:
                        gid = self._gid_map.get((i, sid), None)
                        if gid is not None:
                            all_outputs.append((gid, toks))
                total_tokens += num_tokens
                any_prefill = any_prefill or is_prefill
                if n_diff_steps:
                    merged_diff_steps.update(n_diff_steps)
                if deltas:
                    for sid, toks, fin in deltas:
                        gid = self._gid_map.get((i, sid), None)
                        if gid is not None:
                            merged_deltas.append((gid, toks, fin))

        return all_outputs, total_tokens, any_prefill, merged_diff_steps, merged_deltas

    async def is_finished_async(self):
        """Async version of is_finished that checks all replicas in parallel."""
        tasks = [self._ask_async(i, "is_finished") for i in range(self.dp_size)]
        results = await asyncio.gather(*tasks)
        return all(results)

    async def generate_async(self, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | list[SamplingParams], use_tqdm: bool = True):
        """Async version of generate that allows concurrent request handling."""
        import random
        n = len(prompts)
        idxs = list(range(n))
        random.shuffle(idxs)
        shuffled_prompts = [prompts[i] for i in idxs]
        # Align sampling params with shuffled prompts
        if isinstance(sampling_params, list):
            if len(sampling_params) == n:
                shuffled_sps = [sampling_params[i] for i in idxs]
            elif len(sampling_params) == self.dp_size:
                # per-shard SP; keep as-is and broadcast per-shard below
                shuffled_sps = sampling_params
            else:
                shuffled_sps = [sampling_params[0]] * n
        else:
            shuffled_sps = sampling_params

        # Even partition of shuffled inputs
        base = n // self.dp_size
        rem = n % self.dp_size
        slices = {}
        start = 0
        for i in range(self.dp_size):
            add = base + (1 if i < rem else 0)
            end = start + add
            if start < end:
                slices[i] = (start, end)
            start = end

        # Send generate requests to all replicas concurrently
        async def send_generate(replica_idx: int, start_idx: int, end_idx: int):
            if isinstance(shuffled_sps, list):
                if len(shuffled_sps) == n:
                    sp_arg = shuffled_sps[start_idx:end_idx]
                elif len(shuffled_sps) == self.dp_size:
                    sp_arg = shuffled_sps[replica_idx]
                else:
                    sp_arg = shuffled_sps[0]
            else:
                sp_arg = shuffled_sps
            conn = self.conns[replica_idx]
            # Send in executor to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor,
                conn.send,
                ("generate", shuffled_prompts[start_idx:end_idx], sp_arg, use_tqdm)
            )
            return replica_idx

        # Send all requests concurrently
        send_tasks = [
            send_generate(i, s, e) for i, (s, e) in slices.items()
        ]
        await asyncio.gather(*send_tasks)

        # Collect results asynchronously
        collected = {}

        async def wait_for_result(replica_idx: int):
            conn = self.conns[replica_idx]
            loop = asyncio.get_running_loop()
            try:
                # Poll for data availability in executor, then recv
                def check_and_recv():
                    # Poll is non-blocking, but we run it in executor to be safe
                    if conn.poll():
                        return conn.recv()
                    return None

                # Poll until data is available
                while True:
                    result = await loop.run_in_executor(self._executor, check_and_recv)
                    if result is not None:
                        tag, payload = result
                        break
                    await asyncio.sleep(0.001)  # Small sleep to yield control
            except EOFError:
                p = self.ps[replica_idx]
                exitcode = p.exitcode
                raise RuntimeError(
                    f"DP child #{replica_idx} terminated unexpectedly during generate (exitcode={exitcode}). "
                    f"Enable envs: PYTHONFAULTHANDLER=1 CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 for more info."
                )
            if tag == "ok":
                collected[replica_idx] = payload
            else:
                raise RuntimeError(f"DP child #{replica_idx} error: {payload}")

        # Wait for all results concurrently
        await asyncio.gather(*[wait_for_result(i) for i in slices.keys()])

        # Restore to original order
        restored = [None] * n
        for i, (s, e) in slices.items():
            outs = collected.get(i, [])
            # outs are aligned with shuffled order s:e
            for local_k, out in enumerate(outs):
                global_pos = s + local_k
                orig_idx = idxs[global_pos]
                restored[orig_idx] = out
        assert all(x is not None for x in restored), "Mismatch in outputs after DP collection"
        return restored