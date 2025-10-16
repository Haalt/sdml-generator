#include <algorithm>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>

#include "generator/generator_thread.hpp"
#include "models/tag_dict.h"

// global flag for signal handling
volatile sig_atomic_t running = 1;

// signal handler for graceful shutdown
void signal_handler(int signal)
{
    std::cout << "Received signal " << signal << ", shutting down..." << std::endl;
    running = 0;
}

int main(int argc, char *argv[])
{
    // register signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "Prompt Generator with Integrated Scoring Starting..." << std::endl;

    // default paths
    std::string model_path = "best_model.onnx";
    std::string tokenizer_path = "tokenizer/weighted_tokenizer.json";
    std::string output_file = "prompts.txt";
    std::string dict_path = "dict.json";
    std::string loras_json_path = "loras.json";
    float score_threshold = 0.8f;
    std::string generator_config_path = "generator_config.json";
    int num_generator_threads = 4;

    // parse CLI args
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)
        {
            model_path = argv[++i];
        }
        else if (arg == "--tokenizer" && i + 1 < argc)
        {
            tokenizer_path = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            output_file = argv[++i];
        }
        else if (arg == "--threshold" && i + 1 < argc)
        {
            score_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--threads" && i + 1 < argc)
        {
            num_generator_threads = std::stoi(argv[++i]);
        }
        else if (arg == "--dict" && i + 1 < argc)
        {
            dict_path = argv[++i];
        }
        else if (arg == "--loras" && i + 1 < argc)
        {
            loras_json_path = argv[++i];
        }
        else if (arg == "--generator_config" && i + 1 < argc)
        {
            generator_config_path = argv[++i];
        }

        // usage
        else if (arg == "--help")
        {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --dict <path>            Path to tag dictionary JSON (default: dict.json)" << std::endl;
            std::cout << "  --model <path>           Path to ONNX model file (default: best_model.onnx)" << std::endl;
            std::cout << "  --tokenizer <path>       Path to scoring tokenizer file (default: tokenizer/weighted_tokenizer.json)" << std::endl;
            std::cout << "  --loras <path>           Path to LoRAs JSON file (default: loras.json)" << std::endl;
            std::cout << "  --output <path>          Path to output file (default: prompts.txt)" << std::endl;
            std::cout << "  --threshold <float>      Score threshold (default: 0.8)" << std::endl;
            std::cout << "  --threads <int>          Number of generator threads (default: 4)" << std::endl;
            std::cout << "  --generator_config <path> Path to generator config JSON (default: generator_config.json)" << std::endl;
            std::cout << "  --help                   Show this help message" << std::endl;
            return 0;
        }
    }

    if (!std::filesystem::exists(model_path))
    {
        std::cerr << "Error: Model file not found: " << model_path << std::endl;
        return 1;
    }

    if (!std::filesystem::exists(tokenizer_path))
    {
        std::cerr << "Error: Scoring tokenizer file not found: " << tokenizer_path << std::endl;
        return 1;
    }

    if (!std::filesystem::exists(dict_path))
    {
        std::cerr << "Error: Dict file not found: " << dict_path << std::endl;
        return 1;
    }

    if (!std::filesystem::exists(loras_json_path))
    {
        std::cerr << "Error: LoRAs JSON file not found: " << loras_json_path << std::endl;
        return 1;
    }

    if (!std::filesystem::exists(generator_config_path))
    {
        std::cerr << "Error: Generator config JSON file not found: " << generator_config_path << std::endl;
        return 1;
    }

    if (!Tag_dict::loadFromJson(dict_path))
    {
        std::cerr << "Error: Failed to load tag dictionary: " << dict_path << std::endl;
        return 1;
    }

    // generator configuration
    generator::GeneratorConfig config;
    config.modelPath = model_path;
    config.tokenizerPath = tokenizer_path;
    config.lorasJsonPath = loras_json_path;
    config.generatorConfigPath = generator_config_path;
    config.scoreThreshold = score_threshold;
    config.numGeneratorThreads = num_generator_threads;

    try
    {
        // Create generator thread
        std::cout << "Initializing generator with " << num_generator_threads << " threads..." << std::endl;
        generator::GeneratorThread generator_thread(config);

        std::cout << "Generator ready!" << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;

        // open output file (maybe open only when writing)
        // -> lost 7 hours on a dangling FD once
        std::ofstream out_file(output_file, std::ios::app);
        if (!out_file.is_open())
        {
            std::cerr << "Error: Failed to open output file: " << output_file << std::endl;
            generator_thread.stopAndJoin();
            return 1;
        }

        std::cout << "Starting main loop, writing to " << output_file << std::endl;
        std::cout << "Using score threshold: " << score_threshold << std::endl;

        // track time for queue refill and status
        auto start_time = std::chrono::steady_clock::now();

        // Print the current configuration
        std::cout << "===== Configuration =====" << std::endl;
        std::cout << "Model path: " << model_path << std::endl;
        std::cout << "Tokenizer path: " << tokenizer_path << std::endl;
        std::cout << "LoRAs JSON path: " << loras_json_path << std::endl;
        std::cout << "Generator config path: " << generator_config_path << std::endl;
        std::cout << "Output file: " << output_file << std::endl;
        std::cout << "Score threshold: " << score_threshold << std::endl;
        std::cout << "Generator threads: " << num_generator_threads << std::endl;
        std::cout << "========================" << std::endl;

        // main loop: periodic status logging, real writing happens after loop
        while (running)
        {
            // TODO: remove that
            auto now = std::chrono::steady_clock::now();
            (void)std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

            static int statCounter = 0;
            if (statCounter++ % 10 == 0)
            {
                std::cout << "[Main] Currently accepted prompts: " << generator_thread.getFilteredPromptsSize() << std::endl;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (!running)
            {
                break;
            }
        }

        // generation stopped by signal, write final prompts
        generator_thread.stopAndJoin();

        auto filteredPrompts = generator_thread.getFilteredPrompts();
        std::vector<std::string> formattedPrompts = generator_thread.formatPrompts(filteredPrompts);
        std::cout << "Writing final " << formattedPrompts.size() << " prompts to output file." << std::endl;

        std::random_device rd;
        std::mt19937 gen(rd());

        // shuffle prompts
        std::shuffle(formattedPrompts.begin(), formattedPrompts.end(), gen);

        for (const auto &prompt : formattedPrompts)
        {
            out_file << prompt << std::endl;
        }

        out_file.close();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in main: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Generator stopped" << std::endl;
    return 0;
}
