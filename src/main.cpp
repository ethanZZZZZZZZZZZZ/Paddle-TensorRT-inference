#include <exception>
#include <iostream>
#include <string>

#include "common/config.h"
#include "common/logging.h"
#include "pipeline/pipeline.h"

namespace {

void PrintUsage(const char* program) {
    std::cout << "Usage: " << program << " --config <path>\n"
              << "\n"
              << "Current stage supports the mock backend, optional Paddle backend, synthetic input, "
              << "optional OpenCV video_file input, dynamic batch, and optional predictor pool.\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        }
        if (arg == "--config") {
            if (i + 1 >= argc) {
                EDGE_LOG_ERROR("--config requires a path");
                PrintUsage(argv[0]);
                return 1;
            }
            config_path = argv[++i];
            continue;
        }

        EDGE_LOG_ERROR("Unknown argument: " << arg);
        PrintUsage(argv[0]);
        return 1;
    }

    if (config_path.empty()) {
        EDGE_LOG_ERROR("Missing required --config argument");
        PrintUsage(argv[0]);
        return 1;
    }

    try {
        EDGE_LOG_INFO("Loading config: " << config_path);
        const edge::Config config = edge::Config::LoadFromFile(config_path);
        EDGE_LOG_INFO("Effective config:");
        config.Print(std::cout);

        edge::Pipeline pipeline(config);
        if (!pipeline.Init()) {
            EDGE_LOG_ERROR("Pipeline init failed; see previous [ERROR] lines for the failing stage");
            return 1;
        }
        if (!pipeline.Run()) {
            EDGE_LOG_ERROR("Pipeline run failed; see previous [ERROR] lines for the failing stage");
            return 1;
        }

        EDGE_LOG_INFO("edge_infer finished successfully");
        return 0;
    } catch (const std::exception& ex) {
        EDGE_LOG_ERROR(ex.what());
        return 1;
    }
}
