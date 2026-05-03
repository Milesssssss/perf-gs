// Entry-point aggregator. Each compute kernel lives in its own .slang file
// and is loaded directly by Python. This file currently exists only as a
// placeholder and documentation marker — when kernels share common
// resource names (e.g. output_image) you cannot naively #include them into
// one module. We'll revisit aggregation if/when it becomes useful.
