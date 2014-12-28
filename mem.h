void launch_kernel(int n_tblk, int nt_tblk, float *device, int n);
void alloc_mem(float **host_array, float **device_array, int n);
void free_mem(float *host, float *device);
void transfer_mem(float *device, float *host, int n, bool host2dev);
void copy_mem(float *dst, float *src, int n);


