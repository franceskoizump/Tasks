#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            // TODO: implement on OpenCL
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();
            ocl::Kernel kernel_sum(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "sum_blocks");
            ocl::Kernel kernel_add(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "add");
            ocl::Kernel kernel_max(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
            kernel_sum.compile();
            kernel_add.compile();
            kernel_max.compile();
            gpu::gpu_mem_32i as_gpu;
            gpu::gpu_mem_32i res;
            gpu::gpu_mem_32i ind;
        
        
            as_gpu.resizeN(n);
            as_gpu.writeN(as.data(), n);
            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            res.resizeN(1);
            ind.resizeN(1);
            ocl::LocalMem buffer(sizeof(int) * workGroupSize);
            ocl::LocalMem buf_id(sizeof(int) * workGroupSize);
            timer t;
            std::vector<int> a = as;
            std::vector<int> r = as;
            for (int i = 1; i < a.size(); i++)
            {
                a[i] = a[i] + a[i-1];
            }
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int pref = -100000;
                int index = 0;
                res.writeN(&pref, 1);
                ind.writeN(&index, 1);
                kernel_sum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, n, buffer);
            
                
                std::cout << "WOW" << std::endl;
                kernel_add.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, n, buffer);
                as_gpu.readN(r.data(), n);
              std::cout << n << std::endl;    
                for (int i = 0; i < n; i++)
            {
                std::cout << a[i] << "   ==== " << r[i] << std::endl;
            }
                kernel_max.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, res, ind, n, buffer, buf_id);
                res.readN(&pref, 1);
                ind.readN(&index, 1);
                EXPECT_THE_SAME(pref, reference_max_sum, "GPU result should be consistent!");
                EXPECT_THE_SAME(index, reference_result, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
