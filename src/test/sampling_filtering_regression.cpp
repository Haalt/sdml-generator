#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "evaluation/filtering_thread.hpp"
#include "models/Lora_controller.h"
#include "models/Tag_list.h"
#include "models/json_tag_dict.h"
#include "tokenizer/tokenizer.hpp"

namespace
{
    using generator::FilteringThread;
    using generator::Prompt;
    using generator::ScoredPrompt;
    using generator::Tokenizer;

    std::vector<std::string> decodePromptTokens(const Prompt &prompt, const Tokenizer &tokenizer)
    {
        std::vector<std::string> tags;
        for (int64_t tokenId : prompt.tokens)
        {
            if (tokenId == 0)
            {
                break;
            }
            std::optional<std::string> tag = tokenizer.tokenToText(tokenId);
            if (tag.has_value())
            {
                tags.push_back(*tag);
            }
        }
        return tags;
    }

    std::vector<std::string> decodeGeneratedTags(const Tag_list &tagList, const Tokenizer &tokenizer)
    {
        std::vector<std::string> tags;
        for (const auto &tokenVector : tagList.get_precomputed_tokens())
        {
            for (int64_t tokenId : tokenVector)
            {
                std::optional<std::string> tag = tokenizer.tokenToText(tokenId);
                if (tag.has_value())
                {
                    tags.push_back(*tag);
                }
            }
        }
        return tags;
    }

    size_t countCategoryMatches(const std::vector<std::string> &promptTags, const std::unordered_set<std::string> &categoryTags)
    {
        size_t count = 0;
        for (const auto &tag : promptTags)
        {
            if (categoryTags.count(tag) != 0)
            {
                ++count;
            }
        }
        return count;
    }

    std::vector<std::string> pickSingleTokenTags(const std::vector<std::string> &tags, const Tokenizer &tokenizer, size_t count)
    {
        std::vector<std::string> result;
        std::unordered_set<std::string> seen;
        for (const auto &tag : tags)
        {
            const std::vector<std::string> parts = Tokenizer::splitTagString(tag);
            if (parts.size() != 1)
            {
                continue;
            }
            if (tokenizer.tokenToIndex(parts.front()).has_value())
            {
                if (!seen.insert(parts.front()).second)
                {
                    continue;
                }
                result.push_back(parts.front());
                if (result.size() == count)
                {
                    return result;
                }
            }
        }
        return result;
    }

    bool hasTokenId(const std::vector<std::vector<int64_t>> &tokenVectors, int64_t tokenId)
    {
        for (const auto &tokenVector : tokenVectors)
        {
            if (std::find(tokenVector.begin(), tokenVector.end(), tokenId) != tokenVector.end())
            {
                return true;
            }
        }
        return false;
    }

    bool observedTokenId(const Lora &lora, int64_t tokenId, int attempts)
    {
        for (int attempt = 0; attempt < attempts; ++attempt)
        {
            auto [tokenVectors, _] = lora.get_precomputed_tokens();
            if (hasTokenId(tokenVectors, tokenId))
            {
                return true;
            }
        }
        return false;
    }

    std::string pickSingleTokenClothingTag(const Tag_dict &tagDict, const Tokenizer &tokenizer)
    {
        const std::vector<std::string> tags = pickSingleTokenTags(tagDict.getListNode("clothes.clothing"), tokenizer, 1);
        return tags.empty() ? std::string{} : tags.front();
    }

    std::filesystem::path writeSplitFilterDictionary(const std::string &dictPath)
    {
        std::ifstream in(dictPath);
        nlohmann::json data;
        in >> data;

        nlohmann::json filterCategories = nlohmann::json::array();
        filterCategories.push_back("clothes.clothing");
        data["rules"]["filter_categories"] = std::move(filterCategories);

        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        std::filesystem::path outputPath = std::filesystem::temp_directory_path() /
                                           ("sdml_split_filter_dict_" + std::to_string(now) + ".json");
        std::ofstream out(outputPath);
        out << data.dump(2) << '\n';
        return outputPath;
    }

    std::string joinTags(const std::vector<std::string> &tags)
    {
        std::string result;
        for (size_t i = 0; i < tags.size(); ++i)
        {
            if (i != 0)
            {
                result += ", ";
            }
            result += tags[i];
        }
        return result;
    }

    std::filesystem::path writePackedLeafDictionary(const std::vector<std::string> &packedTags,
                                                    bool includeFilterCategories)
    {
        nlohmann::json data;
        data["tags"]["packed_direct"] = joinTags(packedTags);
        if (includeFilterCategories)
        {
            data["tags"]["packed_filter"]["bundle"] = joinTags(packedTags);
            data["rules"]["filter_categories"] = nlohmann::json::array({"packed_filter"});
        }

        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        std::filesystem::path outputPath = std::filesystem::temp_directory_path() /
                                           ("sdml_packed_leaf_dict_" + std::to_string(now) + ".json");
        std::ofstream out(outputPath);
        out << data.dump(2) << '\n';
        return outputPath;
    }

    bool containsAllTags(const std::vector<std::string> &actualTags,
                         const std::vector<std::string> &expectedTags)
    {
        for (const std::string &tag : expectedTags)
        {
            if (std::find(actualTags.begin(), actualTags.end(), tag) == actualTags.end())
            {
                return false;
            }
        }
        return true;
    }

    std::string pickSemanticAnchorLeaf(const Tag_dict &tagDict, const std::unordered_set<std::string> &categoryTags)
    {
        for (const auto &tag : tagDict.flatten())
        {
            if (tag.find(',') == std::string::npos)
            {
                continue;
            }

            size_t matchCount = 0;
            for (const auto &part : Tokenizer::splitTagString(tag))
            {
                if (categoryTags.count(part) != 0)
                {
                    ++matchCount;
                }
            }

            if (matchCount == 1)
            {
                return tag;
            }
        }
        return {};
    }

    bool waitForAcceptedCount(FilteringThread &filterThread, FilteringThread::CandidateQueue &queue, size_t expectedCount)
    {
        for (int attempt = 0; attempt < 50; ++attempt)
        {
            if (queue.sizeApprox() == 0 && filterThread.getAcceptedPrompts().size() == expectedCount)
            {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return queue.sizeApprox() == 0 && filterThread.getAcceptedPrompts().size() == expectedCount;
    }

    size_t runFilteringScenario(const std::shared_ptr<Tag_dict> &tagDict,
                                const std::string &tokenizerPath,
                                const std::vector<std::vector<std::string>> &promptTags,
                                int loraCap)
    {
        Tokenizer tokenizer(tokenizerPath, ',', true);

        FilteringThread::Config config;
        config.scoreThreshold = 0.0f;
        config.loraCap = loraCap;
        config.maxAcceptedPrompts = 32;
        config.tagDict = tagDict;
        config.useTagsAsLoras = true;
        config.scoreOnly = false;

        FilteringThread filterThread(config);
        filterThread.setTokenizer(Tokenizer(tokenizerPath, ',', true));
        FilteringThread::CandidateQueue queue(16);
        filterThread.registerQueue(&queue);

        for (size_t scoreIndex = 0; scoreIndex < promptTags.size(); ++scoreIndex)
        {
            auto prompt = std::make_shared<Prompt>(tokenizer.processPrompt(promptTags[scoreIndex], 7.0f));
            ScoredPrompt scoredPrompt;
            scoredPrompt.score = static_cast<float>(scoreIndex + 1);
            scoredPrompt.prompt = std::move(prompt);
            if (!queue.enqueue(std::move(scoredPrompt)))
            {
                filterThread.stopAndJoin();
                return 0;
            }
            filterThread.notifyWorkAvailable();
        }

        for (int attempt = 0; attempt < 50 && queue.sizeApprox() != 0; ++attempt)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        const size_t acceptedCount = filterThread.getAcceptedPrompts().size();
        filterThread.stopAndJoin();
        return acceptedCount;
    }
} // namespace

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <dict.json> <tokenizer.json> [iterations]" << std::endl;
        return 2;
    }

    const std::string dictPath = argv[1];
    const std::string tokenizerPath = argv[2];
    const int iterations = (argc >= 4) ? std::max(1, std::atoi(argv[3])) : 2000;

    const std::shared_ptr<Tag_dict> tagDict = create_tag_dict(dictPath);
    if (!tagDict)
    {
        std::cerr << "Failed to load dictionary: " << dictPath << std::endl;
        return 1;
    }

    Tokenizer tokenizer(tokenizerPath, ',', true);
    Tag_list tagList(&tokenizer, tagDict.get());

    const std::vector<std::string> eyeColorTagList = tagDict->getListNode("body.eye_color");
    const std::vector<std::string> clothingTagList = tagDict->getListNode("clothes.clothing");
    const std::unordered_set<std::string> eyeColorTags(eyeColorTagList.begin(), eyeColorTagList.end());
    const std::unordered_set<std::string> clothingTags(clothingTagList.begin(), clothingTagList.end());
    const std::string eyeColorAnchor = pickSemanticAnchorLeaf(*tagDict, eyeColorTags);
    const std::string clothingAnchor = pickSemanticAnchorLeaf(*tagDict, clothingTags);

    if (eyeColorAnchor.empty() || clothingAnchor.empty())
    {
        std::cerr << "Failed to identify packed-leaf regression anchors from the dictionary" << std::endl;
        return 1;
    }

    const std::vector<std::string> rerollTokenCandidates = pickSingleTokenTags(tagDict->flatten(), tokenizer, 4);
    if (rerollTokenCandidates.size() < 4)
    {
        std::cerr << "Failed to pick single-token candidates for LoRA reroll precompute regression" << std::endl;
        return 1;
    }

    const std::optional<int64_t> rerollTagAId = tokenizer.tokenToIndex(rerollTokenCandidates[0]);
    const std::optional<int64_t> rerollTagBId = tokenizer.tokenToIndex(rerollTokenCandidates[1]);
    const std::optional<int64_t> rerollVariationAId = tokenizer.tokenToIndex(rerollTokenCandidates[2]);
    const std::optional<int64_t> rerollVariationBId = tokenizer.tokenToIndex(rerollTokenCandidates[3]);
    if (!rerollTagAId.has_value() || !rerollTagBId.has_value() ||
        !rerollVariationAId.has_value() || !rerollVariationBId.has_value())
    {
        std::cerr << "Failed to resolve token IDs for LoRA reroll precompute regression" << std::endl;
        return 1;
    }

    bool useFirstRerollSet = true;
    auto *rerollMain = new MainLora({}, {}, "badquality", {});
    rerollMain->set_reroll_tags([&]() -> std::vector<std::string>
                                { return {useFirstRerollSet ? rerollTokenCandidates[0] : rerollTokenCandidates[1]}; });
    rerollMain->set_reroll_variations([&]() -> std::vector<std::string>
                                      { return {useFirstRerollSet ? rerollTokenCandidates[2] : rerollTokenCandidates[3]}; });

    Lora_controller rerollController({rerollMain}, {}, {}, {}, {});

    useFirstRerollSet = true;
    rerollController.reroll_loras();
    rerollController.precompute_all_tokens(&tokenizer);
    if (!hasTokenId(rerollMain->get_precomputed_tokens().first, *rerollTagAId) ||
        hasTokenId(rerollMain->get_precomputed_tokens().first, *rerollTagBId))
    {
        std::cerr << "Regression: reroll_tags tokens were not reflected after initial precompute refresh" << std::endl;
        return 1;
    }
    if (!observedTokenId(*rerollMain, *rerollVariationAId, 200) ||
        observedTokenId(*rerollMain, *rerollVariationBId, 200))
    {
        std::cerr << "Regression: reroll_variations tokens were not reflected after initial precompute refresh" << std::endl;
        return 1;
    }

    const size_t firstTagCacheSize = rerollController.get_tag_precompute_cache_size();
    const size_t firstVariationCacheSize = rerollController.get_variation_precompute_cache_size();

    useFirstRerollSet = false;
    rerollController.reroll_loras();
    rerollController.precompute_all_tokens(&tokenizer);
    if (!hasTokenId(rerollMain->get_precomputed_tokens().first, *rerollTagBId) ||
        hasTokenId(rerollMain->get_precomputed_tokens().first, *rerollTagAId))
    {
        std::cerr << "Regression: reroll_tags tokens did not switch after reroll precompute refresh" << std::endl;
        return 1;
    }
    if (!observedTokenId(*rerollMain, *rerollVariationBId, 200) ||
        observedTokenId(*rerollMain, *rerollVariationAId, 200))
    {
        std::cerr << "Regression: reroll_variations tokens did not switch after reroll precompute refresh" << std::endl;
        return 1;
    }

    const size_t secondTagCacheSize = rerollController.get_tag_precompute_cache_size();
    const size_t secondVariationCacheSize = rerollController.get_variation_precompute_cache_size();

    rerollController.reroll_loras();
    rerollController.precompute_all_tokens(&tokenizer);
    if (rerollController.get_tag_precompute_cache_size() != secondTagCacheSize ||
        rerollController.get_variation_precompute_cache_size() != secondVariationCacheSize)
    {
        std::cerr << "Regression: reroll precompute cache size changed unexpectedly for repeated values" << std::endl;
        return 1;
    }
    if (secondTagCacheSize < firstTagCacheSize || secondVariationCacheSize < firstVariationCacheSize)
    {
        std::cerr << "Regression: reroll precompute cache shrank unexpectedly" << std::endl;
        return 1;
    }

    const std::vector<std::string> packedLeafTags = {
        rerollTokenCandidates[0],
        rerollTokenCandidates[1],
        rerollTokenCandidates[2]};
    const std::filesystem::path packedSamplingDictPath = writePackedLeafDictionary(packedLeafTags, false);
    const std::shared_ptr<Tag_dict> packedSamplingDict = create_tag_dict(packedSamplingDictPath.string());
    std::filesystem::remove(packedSamplingDictPath);
    if (!packedSamplingDict)
    {
        std::cerr << "Failed to load packed leaf sampling regression dictionary" << std::endl;
        return 1;
    }

    const std::vector<std::string> packedSamplingLeaves = packedSamplingDict->flatten();
    if (packedSamplingLeaves.size() != 1 || packedSamplingLeaves.front() != joinTags(packedLeafTags))
    {
        std::cerr << "Regression: packed direct string leaf was not preserved as one sampling unit" << std::endl;
        return 1;
    }

    Tag_list packedSamplingTagList(&tokenizer, packedSamplingDict.get());
    packedSamplingTagList.generate_tags(1);
    const std::vector<std::string> packedPromptTags = decodeGeneratedTags(packedSamplingTagList, tokenizer);
    if (!containsAllTags(packedPromptTags, packedLeafTags))
    {
        std::cerr << "Regression: packed direct string leaf did not emit all subtags" << std::endl;
        return 1;
    }

    const std::filesystem::path packedFilteringDictPath = writePackedLeafDictionary(packedLeafTags, true);
    const std::shared_ptr<Tag_dict> packedFilteringDict = create_tag_dict(packedFilteringDictPath.string());
    std::filesystem::remove(packedFilteringDictPath);
    if (!packedFilteringDict)
    {
        std::cerr << "Failed to load packed leaf filtering regression dictionary" << std::endl;
        return 1;
    }

    const size_t packedAccepted = runFilteringScenario(
        packedFilteringDict,
        tokenizerPath,
        {{packedLeafTags[0]}, {packedLeafTags[1]}},
        1);
    if (packedAccepted != 1)
    {
        std::cerr << "Regression: packed direct string leaf subtags did not share one filtering family" << std::endl;
        return 1;
    }

    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        tagList.reset();
        tagList.generate_tags(20);
        const std::vector<std::string> promptTags = decodeGeneratedTags(tagList, tokenizer);

        const size_t eyeColorCount = countCategoryMatches(promptTags, eyeColorTags);
        if (eyeColorCount > 1)
        {
            std::cerr << "Regression: sampled prompt contains multiple eye colors" << std::endl;
            return 1;
        }
    }

    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        tagList.set_tags({eyeColorAnchor});
        tagList.generate_tags(20);
        const std::vector<std::string> promptTags = decodeGeneratedTags(tagList, tokenizer);
        const size_t eyeColorCount = countCategoryMatches(promptTags, eyeColorTags);
        if (eyeColorCount > 1)
        {
            std::cerr << "Regression: packed leaf with semantic eye color allowed a second eye color" << std::endl;
            return 1;
        }
    }

    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        tagList.set_tags({clothingAnchor});
        tagList.generate_tags(20);
        const std::vector<std::string> promptTags = decodeGeneratedTags(tagList, tokenizer);
        const size_t clothingCount = countCategoryMatches(promptTags, clothingTags);
        if (clothingCount > 1)
        {
            std::cerr << "Regression: packed leaf with semantic clothing allowed a second clothes.clothing element" << std::endl;
            return 1;
        }
    }

    const std::string clothingTag = pickSingleTokenClothingTag(*tagDict, tokenizer);
    if (clothingTag.empty())
    {
        std::cerr << "Failed to pick a single-token clothes.clothing tag for filtering regression" << std::endl;
        return 1;
    }

    FilteringThread::Config config;
    config.scoreThreshold = 0.0f;
    config.loraCap = 2;
    config.maxAcceptedPrompts = 32;
    config.tagDict = tagDict;
    config.useTagsAsLoras = true;
    config.scoreOnly = false;

    FilteringThread filterThread(config);
    filterThread.setTokenizer(Tokenizer(tokenizerPath, ',', true));
    FilteringThread::CandidateQueue queue(16);
    filterThread.registerQueue(&queue);

    for (int scoreIndex = 0; scoreIndex < 5; ++scoreIndex)
    {
        auto prompt = std::make_shared<Prompt>(tokenizer.processPrompt({clothingTag}, 7.0f));
        ScoredPrompt scoredPrompt;
        scoredPrompt.score = static_cast<float>(scoreIndex + 1);
        scoredPrompt.prompt = std::move(prompt);
        if (!queue.enqueue(std::move(scoredPrompt)))
        {
            std::cerr << "Failed to enqueue regression prompt" << std::endl;
            filterThread.stopAndJoin();
            return 1;
        }
        filterThread.notifyWorkAvailable();
    }

    const bool reachedExpectedCount = waitForAcceptedCount(filterThread, queue, 2);
    const std::vector<ScoredPrompt> acceptedPrompts = filterThread.getAcceptedPrompts();
    filterThread.stopAndJoin();

    if (!reachedExpectedCount || acceptedPrompts.size() != 2)
    {
        std::cerr << "Regression: filtering did not cap repeated clothes.clothing prompts at loraCap" << std::endl;
        return 1;
    }

    for (const auto &accepted : acceptedPrompts)
    {
        if (!accepted.prompt)
        {
            std::cerr << "Regression: filtering accepted a null prompt" << std::endl;
            return 1;
        }
        const std::vector<std::string> promptTags = decodePromptTokens(*accepted.prompt, tokenizer);
        const size_t clothingCount = countCategoryMatches(promptTags, clothingTags);
        const bool containsExpectedTag =
            std::find(promptTags.begin(), promptTags.end(), clothingTag) != promptTags.end();
        if (clothingCount != 1 || !containsExpectedTag)
        {
            std::cerr << "Regression: filtering accepted prompt does not decode to the expected clothing family" << std::endl;
            return 1;
        }
    }

    const std::vector<std::string> singleTokenEyeColors = pickSingleTokenTags(eyeColorTagList, tokenizer, 1);
    const std::vector<std::string> singleTokenClothingTags = pickSingleTokenTags(clothingTagList, tokenizer, 5);
    if (singleTokenEyeColors.empty() || singleTokenClothingTags.size() < 5)
    {
        std::cerr << "Failed to pick split filter regression tags" << std::endl;
        return 1;
    }

    std::vector<std::vector<std::string>> splitFilterPrompts;
    splitFilterPrompts.reserve(singleTokenClothingTags.size());
    for (const auto &variantClothingTag : singleTokenClothingTags)
    {
        splitFilterPrompts.push_back({singleTokenEyeColors.front(), variantClothingTag});
    }

    const size_t fallbackAccepted = runFilteringScenario(tagDict, tokenizerPath, splitFilterPrompts, 2);
    if (fallbackAccepted != 2)
    {
        std::cerr << "Regression: fallback filtering did not cap prompts by shared eye color" << std::endl;
        return 1;
    }

    const std::filesystem::path splitDictPath = writeSplitFilterDictionary(dictPath);
    const std::shared_ptr<Tag_dict> splitFilterDict = create_tag_dict(splitDictPath.string());
    std::filesystem::remove(splitDictPath);
    if (!splitFilterDict)
    {
        std::cerr << "Failed to load split filter regression dictionary" << std::endl;
        return 1;
    }

    Tag_list splitTagList(&tokenizer, splitFilterDict.get());
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        splitTagList.set_tags({eyeColorAnchor});
        splitTagList.generate_tags(20);
        const std::vector<std::string> promptTags = decodeGeneratedTags(splitTagList, tokenizer);
        const size_t eyeColorCount = countCategoryMatches(promptTags, eyeColorTags);
        if (eyeColorCount > 1)
        {
            std::cerr << "Regression: split filter dictionary stopped enforcing eye color sampling uniqueness" << std::endl;
            return 1;
        }
    }

    const size_t splitAccepted = runFilteringScenario(splitFilterDict, tokenizerPath, splitFilterPrompts, 2);
    if (splitAccepted != splitFilterPrompts.size())
    {
        std::cerr << "Regression: filter_categories exclusion still capped prompts by shared eye color" << std::endl;
        return 1;
    }

    std::cout << "Sampling/filtering regression checks passed (" << iterations
              << " sampling iterations, clothing tag: " << clothingTag << ")" << std::endl;
    return 0;
}
