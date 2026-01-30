# P4JIT: Dynamic ESP32-P4 Optimization

> "Premature optimization is the root of all evil, but JIT compilation is just pure fun." - My Contributions

## Abstract
This project introduces a **lightweight JIT compiler** for the ESP32-P4, leveraging its custom NPU and RISC-V extensions to accelerate Python bytecode execution at the edge.

![Demo Image](assets/demo.jpg)

## Core optimization Logic
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
| Operation | Intepreter (ms) | P4JIT (ms) | Speedup |
|-----------|-----------------|------------|---------|
| Matrix Mul| 450             | 12         | **37.5x**|
| Conv2D    | 1200            | 28         | **42.8x**|
