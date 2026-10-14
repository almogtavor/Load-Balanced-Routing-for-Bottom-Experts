"""What the router kernel itself costs, against the stock one it replaces.

The rule is slower in every serve measurement so far, and there are only two candidate
explanations: the selection kernel costs more than the one it displaces, or the selection it
produces costs more downstream. This separates them, because it times the selection alone.

Stock vLLM spends one `topk_softmax` launch per MoE layer. `placement_router.kernel.select` replaces it, so
the arms here are launch-for-launch comparable and the difference is the whole overhead the
rule can possibly add at the router. Kernel launches are counted from the profiler rather than
asserted, since "one launch" is the load-bearing claim and a stray `.contiguous()` or dtype
cast would break it silently.

`keep=8` asks for no substitution: same kernel, same launch, same `K` argmax passes, different
answer only in that there is none. If it matches stock, the kernel is not the overhead.

    python3 experiments/routerlat.py
"""
import torch

from placement_router.kernel import select
from placement_router.vendor.fused_topk_router import vllm_topk_softmax

E, K = 128, 8
ITERS = 200


def timed(fn, *a):
    for _ in range(20):
        fn(*a)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(ITERS):
        fn(*a)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / ITERS                             # us per call


def launches(fn, *a):
    """CUDA kernels the call actually enqueues."""
    fn(*a)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        fn(*a)
        torch.cuda.synchronize()
    return sum(ev.count for ev in p.key_averages() if ev.self_device_time_total > 0)


def main():
    print(f"E={E} K={K} ranks=4, {ITERS} iters, us per router call")
    for t in (256, 2048, 8192):
        lg = torch.randn(t, E, device="cuda")
        w = torch.empty(t, K, dtype=torch.float32, device="cuda")
        i = torch.empty(t, K, dtype=torch.int32, device="cuda")
        x = torch.empty(t, K, dtype=torch.int32, device="cuda")

        def stock():
            return vllm_topk_softmax(w, i, x, lg, True)

        arms = [("stock topk_softmax", stock)]
        for keep, load_penalty in ((8, 0.0), (4, 0.0), (4, 1.0)):
            arms.append((f"select keep={keep} load_penalty={load_penalty}",
                         lambda keep=keep, load_penalty=load_penalty: select(lg, K, keep, 4, load_penalty=load_penalty)))
        b = None
        for name, fn in arms:
            us, n = timed(fn), launches(fn)
            b = b or us
            print(f"  T={t:<5} {name:<26} {us:7.1f} us  {us / b:5.2f}x stock  {n} kernel(s)")


if __name__ == "__main__":
    main()
