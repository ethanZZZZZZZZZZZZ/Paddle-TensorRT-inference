#pragma once

#include <iostream>

#define EDGE_LOG_INFO(message) \
    do { std::cout << "[INFO] " << message << std::endl; } while (0)

#define EDGE_LOG_WARN(message) \
    do { std::cerr << "[WARN] " << message << std::endl; } while (0)

#define EDGE_LOG_ERROR(message) \
    do { std::cerr << "[ERROR] " << message << std::endl; } while (0)
