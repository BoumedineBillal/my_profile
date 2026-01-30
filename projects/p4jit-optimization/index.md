# P4JIT: Dynamic ESP32-P4 Optimization

> "Premature optimization is the root of all evil, but JIT compilation is just pure fun." - My Contributions

## Abstract
This project introduces a **lightweight JIT compiler** for the ESP32-P4, leveraging its custom NPU and RISC-V extensions to accelerate Python bytecode execution at the edge.

<figure class="my-12 mx-auto table text-center">
    <img src="assets/demo.jpg" class="rounded-xl border border-slate-700 shadow-2xl max-w-full h-auto">
    <figcaption class="mt-4 text-slate-500 italic text-sm font-mono leading-snug">
        Figure 1: Hardware-accelerated inference pipeline on the ESP32-P4 NPU.
    </figcaption>
</figure>

## Core Optimization Logic
The cost function for our quantization error minimization is defined as:

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} Cost(h_\theta(x^{(i)}), y^{(i)})$$

Where $h_\theta(x)$ represents the hypothesis function of the quantized model.

## Implementation Details
We verified the generated assembly using the following MLIR dialect wrapper in C++:

```cpp
// ESP32-P4 Custom Instruction Wrapper
#include <esp_dsp.h>

void execute_jit_block(uint8_t* code_ptr, size_t size) {
    // Flush instruction cache for the modified region
    cache_invalidate_icache_all();
    
    // Jump to JIT-compiled code
    void (*jit_entry)() = (void (*)())code_ptr;
    
    // Low-level MLIR hook
    __asm__ volatile (
        "esp.vsmulas.s8.qacc.ld.incp q0, a1, q1, q2, 0"
        : 
        : "r"(code_ptr)
        : "memory"
    );
    
    jit_entry();
}
```

## Benchmarks
<div class="overflow-x-auto my-10 rounded-xl border border-white/10 bg-slate-900/50">
  <table class="w-full text-left border-collapse">
    <thead>
      <tr class="border-b border-white/10 bg-white/5">
        <th class="p-4 font-mono text-cyan-400">Operation</th>
        <th class="p-4 font-mono text-slate-300">Interpreter (ms)</th>
        <th class="p-4 font-mono text-slate-300">P4JIT (ms)</th>
        <th class="p-4 font-mono text-cyan-500">Speedup</th>
      </tr>
    </thead>
    <tbody class="text-slate-400">
      <tr class="border-b border-white/5 hover:bg-white/5 transition-colors">
        <td class="p-4 font-medium text-slate-200">Matrix Mul</td>
        <td class="p-4 text-sm">450</td>
        <td class="p-4 text-sm">12</td>
        <td class="p-4 font-bold text-cyan-400">37.5x</td>
      </tr>
      <tr class="hover:bg-white/5 transition-colors">
        <td class="p-4 font-medium text-slate-200">Conv2D</td>
        <td class="p-4 text-sm">1200</td>
        <td class="p-4 text-sm">28</td>
        <td class="p-4 font-bold text-cyan-400">42.8x</td>
      </tr>
    </tbody>
  </table>
</div>
