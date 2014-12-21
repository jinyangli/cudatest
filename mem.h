void launch_kernel(int n_tblk, int nt_tblk, float *device, int n);
void alloc_device_mem(float **device_array, int n);
void free_device_mem(float *device);
void transfer_mem(float *device, float *host, int n, bool host2dev);


