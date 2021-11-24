
#include "vkFFT.h"
#include "client_utils.h"
#include "utils_VkFFT.h"

#include <iostream>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

class vkfft_params : public fft_params
{
};

// mostly lifed from VkFFT
VkFFTResult launch_vkfft(vkfft_params params) {

    VkGPU vkGPU = {};
    VkFFTResult resFFT = VKFFT_SUCCESS;
    VkResult res = VK_SUCCESS;

    vkGPU.enableValidationLayers = 0;
    vkGPU.device_id = 0;

    res = createInstance(&vkGPU, 0); // XXX sample_id; 2 and 102 are special here
    if (res != 0)
        return VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE;

    res = setupDebugMessenger(&vkGPU);
    if (res != 0)
        return VKFFT_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER;

    res = findPhysicalDevice(&vkGPU);
    if (res != 0)
        return VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE;

    res = createDevice(&vkGPU, 0); // XXX sample_id
    if (res != 0)
        return VKFFT_ERROR_FAILED_TO_CREATE_DEVICE;

    res = createFence(&vkGPU);
    if (res != 0)
        return VKFFT_ERROR_FAILED_TO_CREATE_FENCE;

    res = createCommandPool(&vkGPU);
    if (res != 0)
        return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_POOL;

    vkGetPhysicalDeviceProperties(vkGPU.physicalDevice, &vkGPU.physicalDeviceProperties);
    vkGetPhysicalDeviceMemoryProperties(vkGPU.physicalDevice, &vkGPU.physicalDeviceMemoryProperties);

    glslang_initialize_process();

    // in loop:
    VkFFTConfiguration configuration = {};
    VkFFTApplication app = {};

    configuration.FFTdim = params.length.size();
    for (int i = 0; i< params.length.size(); ++i) {
        configuration.size[i] = params.length[i];
    }
    configuration.numberBatches = params.nbatch;
    configuration.device = &vkGPU.device;
    configuration.queue = &vkGPU.queue;
    configuration.fence = &vkGPU.fence;
    configuration.commandPool = &vkGPU.commandPool;
    configuration.physicalDevice = &vkGPU.physicalDevice;
    configuration.isCompilerInitialized = 1;
    if(params.transform_type == rocfft_transform_type_real_forward ||
       params.transform_type == rocfft_transform_type_real_inverse)
        configuration.performR2C = true;

    uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.numberBatches;

    VkBuffer buffer = {};
    VkDeviceMemory bufferDeviceMemory = {};
    resFFT = allocateBuffer(&vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
    if (resFFT != VKFFT_SUCCESS) return resFFT;
    configuration.buffer = &buffer;
    configuration.bufferSize = &bufferSize;

    // Input data:
    const auto gpu_input = compute_input(params);
    resFFT = transferDataFromCPU(&vkGPU, (void*)gpu_input[0].data(), &buffer, bufferSize);

    if (resFFT != VKFFT_SUCCESS) return resFFT;

    resFFT = initializeVkFFT(&app, configuration);
    if (resFFT != VKFFT_SUCCESS) return resFFT;

    VkFFTLaunchParams launchParams = {};

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    commandBufferAllocateInfo.commandPool = vkGPU.commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = {};
    res = vkAllocateCommandBuffers(vkGPU.device, &commandBufferAllocateInfo, &commandBuffer);
    if (res != 0) return VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
    VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
    if (res != 0) return VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;

    launchParams.commandBuffer = &commandBuffer;

    int num_iter = 1;
    int inverse = 1;
    for (uint64_t i = 0; i < num_iter; i++) {
        resFFT = VkFFTAppend(&app, inverse, &launchParams);
        if (resFFT != VKFFT_SUCCESS) return resFFT;
    }

    res = vkEndCommandBuffer(commandBuffer);
    if (res != 0) return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
    res = vkQueueSubmit(vkGPU.queue, 1, &submitInfo, vkGPU.fence);
    if (res != 0) return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
    res = vkWaitForFences(vkGPU.device, 1, &vkGPU.fence, VK_TRUE, 100000000000);
    if (res != 0) return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
    std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
    double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;

    std::cout << "Total time: " << totTime << std::endl;

    res = vkResetFences(vkGPU.device, 1, &vkGPU.fence);
    if (res != 0) return VKFFT_ERROR_FAILED_TO_RESET_FENCES;

    // vkFreeCommandBuffers(vkGPU.device, vkGPU.commandPool, 1, &commandBuffer);

    // vkDestroyBuffer(vkGPU.device, buffer, NULL);
    // vkFreeMemory(vkGPU.device, bufferDeviceMemory, NULL);
    // deleteVkFFT(&app);

    return VKFFT_SUCCESS;
}


int main(int argc, char* argv[])
{
    // Control output verbosity:
    int verbose;

    // Device number for running tests:
    int deviceId;

    // Number of performance trial samples
    int ntrial;

    // // FFT parameters:
    vkfft_params params;

    // Declare the supported options.

    // clang-format doesn't handle boost program options very well:
    // clang-format off
    po::options_description opdesc("rocfft rider command line options");
    opdesc.add_options()("help,h", "produces this help message")
        ("device", po::value<int>(&deviceId)->default_value(0), "Select a specific device id")
        ("verbose", po::value<int>(&verbose)->default_value(0), "Control output verbosity")
        ("ntrial,N", po::value<int>(&ntrial)->default_value(1), "Trial size for the problem")
        ("notInPlace,o", "Not in-place FFT transform (default: in-place)")
        ("double", "Double precision transform (default: single)")
        ("transformType,t", po::value<rocfft_transform_type>(&params.transform_type)
         ->default_value(rocfft_transform_type_complex_forward),
         "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
         "forward\n3) real inverse")
        ( "batchSize,b", po::value<size_t>(&params.nbatch)->default_value(1),
          "If this value is greater than one, arrays will be used ")
        ( "itype", po::value<rocfft_array_type>(&params.itype)
          ->default_value(rocfft_array_type_unset),
          "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ( "otype", po::value<rocfft_array_type>(&params.otype)
          ->default_value(rocfft_array_type_unset),
          "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ("length",  po::value<std::vector<size_t>>(&params.length)->multitoken(), "Lengths.")
        ("istride", po::value<std::vector<size_t>>(&params.istride)->multitoken(), "Input strides.")
        ("ostride", po::value<std::vector<size_t>>(&params.ostride)->multitoken(), "Output strides.")
        ("idist", po::value<size_t>(&params.idist)->default_value(0),
         "Logical distance between input batches.")
        ("odist", po::value<size_t>(&params.odist)->default_value(0),
         "Logical distance between output batches.")
        ("isize", po::value<std::vector<size_t>>(&params.isize)->multitoken(),
         "Logical size of input buffer.")
        ("osize", po::value<std::vector<size_t>>(&params.osize)->multitoken(),
         "Logical size of output buffer.")
        ("ioffset", po::value<std::vector<size_t>>(&params.ioffset)->multitoken(), "Input offsets.")
        ("ooffset", po::value<std::vector<size_t>>(&params.ooffset)->multitoken(), "Output offsets.");
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opdesc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << opdesc << std::endl;
        return 0;
    }

    params.placement
        = vm.count("notInPlace") ? rocfft_placement_notinplace : rocfft_placement_inplace;
    params.precision = vm.count("double") ? rocfft_precision_double : rocfft_precision_single;

    if(!vm.count("length"))
    {
        std::cout << "Please specify transform length!" << std::endl;
        std::cout << opdesc << std::endl;
        return 0;
    }

    params.validate();
    params.valid(true);
    std::cout << params.str() << std::endl;

    auto ret = launch_vkfft(params);

    std::cout << ret << std::endl;
}
