void triggerCudaCreation() {
	cudaFree(0); // manually trigger creation of the context
}