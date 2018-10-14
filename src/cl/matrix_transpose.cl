__kernel void matrix_transpose(__global float * a,
                               __global float * at,
                               unsigned int m,
                               unsigned int k,
                               __local float * buffer)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    unsigned int tile = get_local_size(0);

    if (i < m && j < k)
        buffer[tile * local_j + local_i] = a[k * j + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    at[m * i + j] = buffer[tile * local_j + local_i];
}