// TODO

__kernel void sum(__global unsigned int * in,
                  __global unsigned int * result, unsigned int n,
                  __local  unsigned int * buffer)

{
    int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);
    int local_size = get_local_size(0);
    if (global_id >= n) buffer[local_id] = 0;
    else buffer[local_id] = in[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = local_size; i > 1; i/=2)
    {
        if (2 * local_id < i)
        {
            unsigned int a = buffer[local_id];
            unsigned int b = buffer[local_id + i /2];
            buffer[local_id] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0)
    {
        atomic_add(result, buffer[0]);
    }
}