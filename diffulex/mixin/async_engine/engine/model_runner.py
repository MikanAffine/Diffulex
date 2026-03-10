import asyncio

from concurrent.futures import ThreadPoolExecutor


class ModelRunnerAsyncMixin:
    async def call_async(self, method_name, *args):
        """Async version of call that runs in a thread pool executor."""
        loop = asyncio.get_event_loop()
        # Use default executor or create one if needed
        executor = getattr(self, "_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1)
            self._executor = executor
        return await loop.run_in_executor(executor, self.call, method_name, *args)
