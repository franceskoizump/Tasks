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

        if (j < m && tileK + local_i < k) tileA[local_i][local_j] = as[j * k + local_i + tileK];   
        else tileA[local_i][local_j] = 0.f;
        if (i < n && tileK + local_j < k) tileA[local_i][local_j] = bs[(local_j + tileK) * k + i];   
        else tileA[local_i][local_j] = 0.f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int l = 0; l < tile; ++l) 
            sum += tileA[l][local_j] * tileB[local_i][l];


    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < m && j < n) {
        cs[n*i + j] = sum;
}

}