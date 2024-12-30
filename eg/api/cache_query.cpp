#include <array>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <iostream>
#include <vector>

#define HIP_CHECK(condition)                                                           \
    {                                                                                  \
        hipError_t error = condition;                                                  \
        if(error != hipSuccess)                                                        \
        {                                                                              \
            std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
            exit(error);                                                               \
        }                                                                              \
    }

#define HSA_CHECK(func)                                       \
    {                                                         \
        hsa_status_t error = func;                            \
        if(error != HSA_STATUS_SUCCESS)                       \
        {                                                     \
            std::cout << "HSA error: " << error << std::endl; \
            exit(error);                                      \
        }                                                     \
    }

struct Agents
{
    std::vector<hsa_agent_t> cpus;
    std::vector<hsa_agent_t> gpus;
};

// Callback function to be called during hsa_iterate_agents
hsa_status_t populate_device_type(hsa_agent_t agent, void* data)
{
    Agents*           res = (Agents*)data;
    hsa_device_type_t device_type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, (void*)&device_type);
    if(device_type == HSA_DEVICE_TYPE_CPU)
    {
        res->cpus.push_back(agent);
    }
    else if(device_type == HSA_DEVICE_TYPE_GPU)
    {
        res->gpus.push_back(agent);
    }
    return HSA_STATUS_SUCCESS;
}

int main()
{
    hsa_status_t hsa_status;
    uint32_t     agent_ind = 0;

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    std::cout << "Current device: " << device_id << "\n";

    // Use hsa_iterate_agents() to run populate_device_type() on each agent
    // , and update Agents struct passed in
    Agents agents;
    HSA_CHECK(hsa_iterate_agents(populate_device_type, (void*)&agents));
    std::cout << "Number of CPUs: " << agents.cpus.size()
              << "\nNumber of GPUs: " << agents.gpus.size() << std::endl;

    // Try matching agent # with device # by indexing into agents.gpus
    // with device # (assuming they have same ordering)
    hsa_agent_t             agent = agents.gpus.at(device_id);
    std::array<uint32_t, 4> cache_sizes;
    HSA_CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_CACHE_SIZE, (void*)&cache_sizes));

    for(int i = 0; i < 4; i++)
    {
        std::cout << "L" << (i + 1) << " size = " << cache_sizes[i] << " bytes.\n";
    }

    return 0;
}
