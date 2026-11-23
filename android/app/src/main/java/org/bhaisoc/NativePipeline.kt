package org.bhaisoc

/**
 * NativePipeline.kt
 * Kotlin JNI interface to native C++ POSIX mmap loader, core affinity,
 * fast anomaly pre-filter, and temporal debounce hysteresis.
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
     * Temporal debounce hysteresis filter:
     * Enforces continuous frame detection (>=2 frames) and a 3-second thermal cooldown.
     */
    external fun shouldTriggerTier2(maxConfidence: Float): Boolean

    /**
     * Sub-5ms fast variance test to quickly reject non-PCB or empty camera regions
     * without invoking heavier neural network inference.
     */
    external fun fastAnomalyPreFilter(frameData: ByteArray, width: Int, height: Int): Boolean

    /**
     * Unloads mapped memory on app shutdown.
     */
    external fun unloadModels()
}
