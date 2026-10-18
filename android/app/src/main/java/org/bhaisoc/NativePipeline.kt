package org.bhaisoc

/**
 * NativePipeline.kt
 * Kotlin JNI interface to native C++ POSIX mmap loader and core affinity.
 */
object NativePipeline {
    init {
        System.loadLibrary("bhaisoc_native")
    }

    /**
     * Pins current inference thread to Exynos 9611 Cortex-A73 Big Cores (Cores 4-7)
     */
    external fun pinToBigCores()

    /**
     * Memory-maps 1.6GB ExecuTorch model directly into C++ virtual memory,
     * completely bypassing the 256MB Android Java Heap.
     */
    external fun mmapLlamaModel(modelPath: String): Boolean

    /**
     * Unloads mapped memory on app shutdown.
     */
    external fun unloadModels()
}
