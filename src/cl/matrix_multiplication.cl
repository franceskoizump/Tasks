__kernel void matrix_multiplication(__global float * as, 
                                    __global float * bs, 
                                    __global float * cs, 
                                    unsigned int m,
                                    unsigned int k,
                                    unsigned int n)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    unsigned int tile = get_local_size(0);
    __local float tileA[16][16];
    __local float tileB[16][16];
    float sum = 0.f;

    for (int tileK = 0; tileK * 16  < k; tileK+=1)
    {

        tileA[local_j][local_i] = as[j * k + local_i + tileK * 16];   
        tileB[local_j][local_i] = bs[(local_j + tileK * 16) * k + i];   
        
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int l = 0; l < tile; ++l) 
            sum += tileA[local_j][l] * tileB[l][local_i];
        barrier(CLK_LOCAL_MEM_FENCE);

    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < m && j < n) {
        cs[n*j + i] = sum;
}

}