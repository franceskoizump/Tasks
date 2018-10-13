// TODO

__kernel void sum_blocks(__global int * in, unsigned int n, __local int * buffer)
{
    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);
    unsigned int local_size = get_local_size(0);
    if (n <= global_id) {
        buffer[local_id] = 0;
    }
    else { 
        buffer[local_id] = in[global_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < 7; i+=1)//7 because of log(local_size(0)) == 7
    {
        unsigned int off = (1 << i);
        if (local_id >= off)
        {
            buffer[local_id] = buffer[local_id - off] + buffer[local_id];
        }
    barrier(CLK_LOCAL_MEM_FENCE);
    }
    in[global_id] = buffer[local_id];
    
}

__kernel void add(__global int * in, unsigned int n, __local int * buffer)
{
    unsigned int global_id = get_global_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int local_size = get_local_size(0);
    unsigned int local_id = get_local_id(0);
    if (n <= global_id) {
        buffer[local_id] = 0;
    }
    else { 
        buffer[local_id] = in[global_id];
    }
    for (int i = 0; i < group_id; i++)
    {
        buffer[local_id] += in[(i+1) * local_size - 1];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    in[global_id] = buffer[local_id];
}


__kernel void max_prefix_sum(__global unsigned int * in,
                  __global unsigned int * result,
                  __global unsigned int * index,
                   unsigned int n,
                  __local  unsigned int * buffer, __local unsigned int * ind_b)

{
    int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);
    int local_size = get_local_size(0);
    if (global_id >= n) buffer[local_id] = -1000;
    else buffer[local_id] = in[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = local_size; i > 1; i/=2)
    {
        if (2 * local_id < i && global_id < n)
        {
            unsigned int a = buffer[local_id];
            unsigned int b = buffer[local_id + i /2];
            buffer[local_id] = a < b ? b : a;
            ind_b[local_id] = a < b ? local_id + i/2 : local_id ;

        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0)
    {
        atom_max(result, buffer[0]);
    }
}